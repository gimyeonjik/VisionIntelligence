from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Stage 1 LR — cosine + warmup + cooldown
# ---------------------------------------------------------------------------
STAGE1_LR = 1e-3            # peak learning rate
STAGE1_MIN_LR = 1e-5        # cosine 종료 지점 (cooldown 도 여기에 머무름)
STAGE1_WARMUP_LR = 1e-6     # warmup 시작 LR (epoch 0)
STAGE1_WARMUP_EP = 5
STAGE1_COOLDOWN_EP = 10
STAGE1_EPOCHS = 250

def stage1_lr_lambda(epoch: int) -> float:
    if epoch < STAGE1_WARMUP_EP:
        # WARMUP_LR 에서 LR 까지 선형
        lr = STAGE1_WARMUP_LR + (STAGE1_LR - STAGE1_WARMUP_LR) * epoch / STAGE1_WARMUP_EP
        
        return lr / STAGE1_LR

    if epoch >= STAGE1_EPOCHS - STAGE1_COOLDOWN_EP:
        return STAGE1_MIN_LR / STAGE1_LR

    progress = (epoch - STAGE1_WARMUP_EP) / (STAGE1_EPOCHS - STAGE1_WARMUP_EP - STAGE1_COOLDOWN_EP)
    lr = STAGE1_MIN_LR + 0.5 * (STAGE1_LR - STAGE1_MIN_LR) * (1 + math.cos(math.pi * progress))
    
    return lr / STAGE1_LR

def stage1_loss_weights(epoch: int) -> tuple[float, float]:
    t = epoch / max(STAGE1_EPOCHS - 1, 1)
    fine_w = 0.9 + (0.1 - 0.9) * t
    sc_w = 0.1 + (0.9 - 0.1) * t
    
    return fine_w, sc_w


# ---------------------------------------------------------------------------
# Stage 2 LR — cosine + warmup (cooldown 없음)
# ---------------------------------------------------------------------------
STAGE2_LR = 1e-4            # Stage 1 대비 10배 낮음 (fine-tune)
STAGE2_MIN_LR = 1e-6
STAGE2_WARMUP_LR = 1e-6
STAGE2_WARMUP_EP = 3
STAGE2_EPOCHS = 100
STAGE2_MIXUP_OFF_EP = 80    # 이 epoch 부터 Mixup/CutMix off (마지막 20ep 클린)

def stage2_lr_lambda(epoch: int) -> float:
    if epoch < STAGE2_WARMUP_EP:
        lr = STAGE2_WARMUP_LR + (STAGE2_LR - STAGE2_WARMUP_LR) * epoch / STAGE2_WARMUP_EP
        
        return lr / STAGE2_LR
    
    progress = (epoch - STAGE2_WARMUP_EP) / (STAGE2_EPOCHS - STAGE2_WARMUP_EP)
    lr = STAGE2_MIN_LR + 0.5 * (STAGE2_LR - STAGE2_MIN_LR) * (1 + math.cos(math.pi * progress))
    
    return lr / STAGE2_LR


def stage2_loss_weights(_epoch: int) -> tuple[float, float]:
    """flat: fine=0.9, sc=0.1 — Stage 2 에서는 schedule 없음"""
    return 0.9, 0.1

def stage2_mixup_enabled(epoch: int) -> bool:
    """epoch < 80 이면 ON, 그 이후 OFF (DeiT처럼 마지막 20ep clean 학습)"""
    return epoch < STAGE2_MIXUP_OFF_EP