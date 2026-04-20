from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from constants import CIFAR100_FINE_TO_COARSE, IMAGENET_MEAN, IMAGENET_STD
from src.dhvt import build_dhvt_t

def build_test_loader(data_root: str, batch_size: int = 256, num_workers: int = 4) -> DataLoader:
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, data_root: str, batch_size: int = 256, num_workers: int = 4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_test_loader(data_root, batch_size, num_workers)
    model = build_dhvt_t(num_classes=100, drop_path_rate=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    if missing or unexpected:
        print(f"[warn] missing={missing}  unexpected={unexpected}")
    model.eval()

    fine_to_coarse = torch.as_tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)

    n = 0
    correct1 = 0
    sc_sum = 0.0
    
    for images, labels in tqdm(loader, desc=f"[Eval {Path(checkpoint_path).name}]", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        # Top-1
        pred = logits.argmax(dim=1)
        correct1 += (pred == labels).sum().item()

        # Super-Class (Top-5): Top-5 예측의 coarse 가 GT coarse 와 일치하는 비율
        _, top5 = logits.topk(5, dim=1)               # [B, 5]
        top5_coarse = fine_to_coarse[top5]             # [B, 5]
        target_coarse = fine_to_coarse[labels]         # [B]
        match_ratio = (top5_coarse == target_coarse.unsqueeze(1)).float().mean(dim=1)
        sc_sum += match_ratio.sum().item()
        n += labels.size(0)

    top1 = 100.0 * correct1 / n
    sc = 100.0 * sc_sum / n
    
    return {"top1": top1, "sc": sc, "n_test": n,
            "checkpoint": str(checkpoint_path),
            "stage": ckpt.get("stage") if isinstance(ckpt, dict) else None,
            "global_epoch": ckpt.get("global_epoch") if isinstance(ckpt, dict) else None,
            "seed": ckpt.get("seed") if isinstance(ckpt, dict) else None}

def aggregate(results: List[dict]) -> dict:
    tops = np.array([r["top1"] for r in results])
    scs = np.array([r["sc"] for r in results])
    
    return {
        "n_runs": len(results),
        "top1_mean": float(tops.mean()),
        "top1_std": float(tops.std(ddof=1)) if len(tops) > 1 else 0.0,
        "sc_mean": float(scs.mean()),
        "sc_std": float(scs.std(ddof=1)) if len(scs) > 1 else 0.0,
        "per_run": results,
    }

def main():
    p = argparse.ArgumentParser(description="DHVT-T CIFAR-100 evaluation (Top-1 + SC Top-5)")
    p.add_argument("--checkpoint", type=str, default=None, help="단일 체크포인트 평가 (--aggregate 와 상호 배타).")
    p.add_argument("--aggregate", type=str, nargs="+", default=None, help="여러 체크포인트를 받아 mean±std 로 요약.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_json", type=str, default="", help="결과를 json 으로 덤프할 경로 (비우면 stdout 만).")
    args = p.parse_args()

    if not args.checkpoint and not args.aggregate:
        p.error("--checkpoint 또는 --aggregate 중 하나는 필수")
    if args.checkpoint and args.aggregate:
        p.error("--checkpoint 와 --aggregate 는 동시 사용 불가")

    if args.checkpoint:
        r = evaluate_checkpoint(args.checkpoint, args.data_root, args.batch_size, args.num_workers)
        print(f"\n{r['checkpoint']}")
        print(f"  Top-1 acc:        {r['top1']:.2f}%")
        print(f"  Super-Class acc:  {r['sc']:.2f}%")
        print(f"  (n_test = {r['n_test']})")
        out = r
    else:
        results = [evaluate_checkpoint(c, args.data_root, args.batch_size, args.num_workers)
                   for c in args.aggregate]
        summary = aggregate(results)
        print(f"\n{summary['n_runs']} runs aggregated:")
        print(f"  Top-1:        {summary['top1_mean']:.2f} ± {summary['top1_std']:.2f}")
        print(f"  Super-Class:  {summary['sc_mean']:.2f} ± {summary['sc_std']:.2f}")
        for r in results:
            print(f"    seed={r['seed']}  top1={r['top1']:.2f}  sc={r['sc']:.2f}  "
                  f"({Path(r['checkpoint']).name})")
        out = summary

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"\n[saved] {args.save_json}")

if __name__ == "__main__":
    main()