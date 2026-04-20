from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
from constants import CIFAR100_FINE_TO_COARSE
from src.dhvt import build_dhvt_t
from src.data import build_dataloaders, MixupCutmix
from src.losses import (
    build_sc_soft_labels, build_superclass_indices,
    compute_superclass_logits, CombinedLoss,
)
from src import schedule as sch

# ---------------------------------------------------------------------------
# 재현성 도우미
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 체크포인트 저장/복원
# ---------------------------------------------------------------------------
def save_ckpt(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def make_state(*, epoch: int, stage: int, global_epoch: int, model, optimizer,
               scheduler, scaler, best_top1: float, best_sc: float,
               seed: int, args) -> dict:
    return {
        "epoch": epoch,                     # stage 내 local epoch (0-based 완료 직후 값)
        "stage": stage,
        "global_epoch": global_epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_top1": best_top1,
        "best_sc": best_sc,
        "seed": seed,
        "args": vars(args),
    }

# ---------------------------------------------------------------------------
# 학습 / 검증 루프
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader, optimizer, scaler: GradScaler,
                    mixup: MixupCutmix, combined: CombinedLoss,
                    fine_w: float, sc_w: float, device: torch.device,
                    epoch_desc: str) -> dict:
    """1 epoch 학습. 리턴 dict: {loss, fine, sc, acc_approx}."""
    model.train()
    n = 0
    sum_loss = 0.0
    sum_fine = 0.0
    sum_sc = 0.0
    correct_approx = 0.0   # mixup 환경에서 "targets_a 와 일치하는 top-1" 의 비례값

    bar = tqdm(loader, desc=epoch_desc, leave=False, dynamic_ncols=True)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, ta, tb, lam = mixup(images, labels)
        optimizer.zero_grad(set_to_none=True)

        # FP16 AMP — A5000 에서 속도/메모리 모두 크게 절감
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            fine_loss, sc_loss = combined(logits, ta, tb, lam)
            loss = fine_w * fine_loss + sc_w * sc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        n += bs
        sum_loss += loss.item() * bs
        sum_fine += fine_loss.item() * bs
        sum_sc += sc_loss.item() * bs
        pred = logits.argmax(dim=1)
        if lam == 1.0:
            correct_approx += (pred == ta).sum().item()
        else:
            correct_approx += (lam * (pred == ta).float().sum()
                               + (1 - lam) * (pred == tb).float().sum()).item()

        bar.set_postfix(loss=f"{sum_loss/n:.4f}", fine=f"{sum_fine/n:.4f}",
                        sc=f"{sum_sc/n:.4f}", acc=f"{100*correct_approx/n:.2f}%")

    return {"loss": sum_loss / n, "fine": sum_fine / n, "sc": sum_sc / n,
            "acc_approx": 100.0 * correct_approx / n}

@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device,
             fine_to_coarse: torch.Tensor, sc_indices,
             combined: CombinedLoss) -> dict:
    model.eval()
    n = 0
    correct1 = 0
    sc_sum = 0.0
    loss_sum = 0.0
    fine_sum = 0.0
    sc_loss_sum = 0.0

    for images, labels in tqdm(loader, desc="[Val]", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()
        
        _, top5 = logits.topk(5, dim=1)                      # [B, 5]
        top5_coarse = fine_to_coarse[top5]                   # [B, 5]
        target_coarse = fine_to_coarse[labels]               # [B]
        match = (top5_coarse == target_coarse.unsqueeze(1)).float().mean(dim=1)
        sc_sum += match.sum().item()

        fine_loss, sc_loss = combined(logits, labels, labels, 1.0)
        fine_sum += fine_loss.item() * images.size(0)
        sc_loss_sum += sc_loss.item() * images.size(0)
        loss_sum += (fine_loss.item() + sc_loss.item()) * images.size(0)
        n += images.size(0)

    return {
        "top1": 100.0 * correct1 / n,
        "sc": 100.0 * sc_sum / n,
        "loss": loss_sum / n,
        "fine": fine_sum / n,
        "sc_loss": sc_loss_sum / n,
    }

# ---------------------------------------------------------------------------
# 한 stage 실행
# ---------------------------------------------------------------------------
def run_stage(stage: int, *, model, optimizer, scheduler, scaler,
              train_loader, test_loader, train_sampler,
              mixup: MixupCutmix, combined: CombinedLoss,
              fine_to_coarse_t: torch.Tensor, sc_indices,
              device: torch.device, save_dir: Path,
              total_epochs: int, start_epoch: int,
              best_top1: float, best_sc: float,
              seed: int, args) -> tuple[float, float]:
    stage_tag = f"stage{stage}"
    for epoch in range(start_epoch, total_epochs):
        if stage == 1:
            fine_w, sc_w = sch.stage1_loss_weights(epoch)
            mixup.enabled = True
        else:
            fine_w, sc_w = sch.stage2_loss_weights(epoch)
            mixup.enabled = sch.stage2_mixup_enabled(epoch)

        train_sampler.set_epoch(epoch)
        t0 = time.time()
        desc = f"[{stage_tag} ep{epoch+1}/{total_epochs} fw={fine_w:.2f} sw={sc_w:.2f} mix={mixup.enabled}]"
        tr = train_one_epoch(model, train_loader, optimizer, scaler, mixup, combined, fine_w, sc_w, device, desc)
        scheduler.step()
        val = validate(model, test_loader, device, fine_to_coarse_t, sc_indices, combined)
        dt = time.time() - t0

        global_epoch = epoch + 1 if stage == 1 else sch.STAGE1_EPOCHS + epoch + 1
        lr = optimizer.param_groups[0]["lr"]
        print(f"{stage_tag} ep{epoch+1}/{total_epochs} (global {global_epoch}) "
              f"lr={lr:.2e} | train loss={tr['loss']:.4f} fine={tr['fine']:.4f} sc={tr['sc']:.4f} "
              f"| val top1={val['top1']:.2f} sc={val['sc']:.2f} loss={val['loss']:.4f} "
              f"| {dt:.1f}s", flush=True)

        # best 갱신
        improved_top1 = val["top1"] > best_top1
        if improved_top1:
            best_top1 = val["top1"]
            best_sc = val["sc"]       # best_top1 시점의 sc (평가 기준은 best_top1 의 ckpt)

        # ── 체크포인트 저장 ──
        state = make_state(epoch=epoch, stage=stage, global_epoch=global_epoch,
                           model=model, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, best_top1=best_top1, best_sc=best_sc,
                           seed=seed, args=args)

        save_ckpt(str(save_dir / f"{stage_tag}_last.pt"), state)
        if improved_top1:
            save_ckpt(str(save_dir / f"{stage_tag}_best.pt"), state)
        # Stage 1 마지막 epoch: Stage 2 시작점으로 고정 저장
        if stage == 1 and epoch + 1 == total_epochs:
            save_ckpt(str(save_dir / "stage1_epoch250.pt"), state)
        # Stage 2 마지막 epoch: 최종 제출용 저장
        if stage == 2 and epoch + 1 == total_epochs:
            save_ckpt(str(save_dir / f"stage2_epoch{global_epoch}.pt"), state)

        history_path = save_dir / f"{stage_tag}_history.jsonl"
        with history_path.open("a") as f:
            rec = {
                "epoch": epoch + 1, "global_epoch": global_epoch, "lr": lr,
                "fine_w": fine_w, "sc_w": sc_w, "mixup": mixup.enabled,
                "train": tr, "val": val, "time_sec": dt,
            }
            f.write(json.dumps(rec) + "\n")

    return best_top1, best_sc

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def str2bool(v: str) -> bool:
    return v.lower() in ("y", "yes", "true", "t", "1")

def parse_args():
    p = argparse.ArgumentParser(description="DHVT-T CIFAR-100 2-stage trainer (scratch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=["1", "2", "all"], default="all",
                   help="'1' Stage 1 만, '2' Stage 2 만 (--resume 필요), 'all' Stage1→Stage2 연속 (기본).")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./checkpoints/seed0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--stage1_epochs", type=int, default=sch.STAGE1_EPOCHS)
    p.add_argument("--stage2_epochs", type=int, default=sch.STAGE2_EPOCHS)
    p.add_argument("--ra_repeats", type=int, default=3)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--download", type=str2bool, default=True, help="CIFAR-100 없으면 자동 다운로드 여부.")
    p.add_argument("--resume", type=str, default="", help="체크포인트 경로. 해당 stage/epoch 부터 이어서 진행.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU 가 필요합니다 (A5000 단일 GPU 전제).")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 ──
    train_loader, test_loader, train_sampler = build_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        ra_repeats=args.ra_repeats, download=args.download)

    # ── fine_to_coarse / Soft70 / superclass 인덱스 준비 (GPU) ──
    fine_to_coarse_t = torch.as_tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)
    soft_matrix = build_sc_soft_labels(CIFAR100_FINE_TO_COARSE, num_classes=100, device=device)
    sc_groups = build_superclass_indices(CIFAR100_FINE_TO_COARSE)
    sc_indices = [torch.as_tensor(g, dtype=torch.long, device=device) for g in sc_groups]

    combined = CombinedLoss(soft_matrix, sc_indices, fine_to_coarse_t)
    mixup = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5, enabled=True)

    # ── 모델 ──
    model = build_dhvt_t(num_classes=100, drop_path_rate=args.drop_path_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] DHVT-T params: {n_params:,}  (expect ≈ 5.8M)")

    # ── Resume 분기 (어떤 stage/epoch 부터인지 결정) ──
    resume_ckpt: Optional[dict] = None
    start_stage = 1 if args.stage in ("1", "all") else 2
    start_epoch_s1 = 0
    start_epoch_s2 = 0
    best_top1 = -1.0
    best_sc = -1.0

    if args.resume and os.path.isfile(args.resume):
        print(f"[resume] loading {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["model"])
        best_top1 = resume_ckpt.get("best_top1", -1.0)
        best_sc = resume_ckpt.get("best_sc", -1.0)
        if resume_ckpt["stage"] == 1:
            # Stage 1 도중/끝 체크포인트.
            start_stage = 1
            # `epoch` 필드는 "방금 끝낸 epoch index (0-based)". 다음 반복은 +1 부터.
            start_epoch_s1 = resume_ckpt["epoch"] + 1
            if start_epoch_s1 >= args.stage1_epochs and args.stage == "all":
                # Stage 1 은 이미 끝났음 → Stage 2 부터 시작
                start_stage = 2
        else:  # stage 2
            start_stage = 2
            start_epoch_s2 = resume_ckpt["epoch"] + 1

    # ── Optimizer / Scheduler / Scaler (Stage 1 기준) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=sch.STAGE1_LR,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sch.stage1_lr_lambda)
    scaler = GradScaler("cuda")

    # Stage 1 resume 상태 복원 (모델은 위에서 이미 로드)
    if resume_ckpt is not None and resume_ckpt["stage"] == 1 and start_stage == 1:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        scaler.load_state_dict(resume_ckpt["scaler"])

    # ── Stage 1 ──
    if start_stage == 1 and args.stage in ("1", "all"):
        print(f"\n═══════ Stage 1 시작 (epoch {start_epoch_s1}..{args.stage1_epochs-1}) ═══════")
        best_top1, best_sc = run_stage(
            stage=1, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train_loader=train_loader, test_loader=test_loader,
            train_sampler=train_sampler, mixup=mixup, combined=combined,
            fine_to_coarse_t=fine_to_coarse_t, sc_indices=sc_indices,
            device=device, save_dir=save_dir,
            total_epochs=args.stage1_epochs, start_epoch=start_epoch_s1,
            best_top1=best_top1, best_sc=best_sc, seed=args.seed, args=args)
        print(f"[Stage 1 완료] best top1={best_top1:.2f} sc={best_sc:.2f}")

    # ── Stage 2 전환 ──
    if args.stage in ("2", "all"):
        if args.stage == "all" and (resume_ckpt is None or resume_ckpt["stage"] != 2):
            s1_end = save_dir / "stage1_epoch250.pt"
            if not s1_end.exists():
                raise FileNotFoundError(
                    f"Stage 2 시작점 {s1_end} 가 없습니다. Stage 1 을 먼저 완주해야 함.")
            print(f"[stage2] loading Stage 1 final: {s1_end}")
            s1_ckpt = torch.load(str(s1_end), map_location="cpu")
            model.load_state_dict(s1_ckpt["model"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=sch.STAGE2_LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sch.stage2_lr_lambda)
        scaler = GradScaler("cuda")

        if resume_ckpt is not None and resume_ckpt["stage"] == 2:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            best_top1 = resume_ckpt.get("best_top1", -1.0)
            best_sc = resume_ckpt.get("best_sc", -1.0)
        else:
            best_top1 = -1.0
            best_sc = -1.0

        print(f"\n═══════ Stage 2 시작 (epoch {start_epoch_s2}..{args.stage2_epochs-1}, "
              f"global {sch.STAGE1_EPOCHS+start_epoch_s2+1}..{sch.STAGE1_EPOCHS+args.stage2_epochs}) ═══════")
        best_top1, best_sc = run_stage(
            stage=2, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train_loader=train_loader, test_loader=test_loader,
            train_sampler=train_sampler, mixup=mixup, combined=combined,
            fine_to_coarse_t=fine_to_coarse_t, sc_indices=sc_indices,
            device=device, save_dir=save_dir,
            total_epochs=args.stage2_epochs, start_epoch=start_epoch_s2,
            best_top1=best_top1, best_sc=best_sc, seed=args.seed, args=args)
        print(f"[Stage 2 완료] best top1={best_top1:.2f} sc={best_sc:.2f}")

    print("\n학습 종료. 최종 제출용 체크포인트:")
    print(f"  {save_dir/'stage2_best.pt'}")
    print(f"  평가:  python evaluate.py --checkpoint {save_dir/'stage2_best.pt'} --data_root {args.data_root}")

if __name__ == "__main__":
    main()