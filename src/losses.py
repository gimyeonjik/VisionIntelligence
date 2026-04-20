from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Soft70 확률 분포
# ---------------------------------------------------------------------------
SC_CORRECT_PROB = 0.70    # GT fine class 의 정답 확률
SC_SAME_TOTAL   = 0.205   # GT 와 같은 super-class에 속한 4개 class가 합계로 받는 확률
SC_OTHER_EACH   = 0.001   # 그 외 95개 class 각각이 받는 확률

def build_sc_soft_labels(fine_to_coarse: Sequence[int], num_classes: int = 100, device: torch.device | str = "cuda") -> torch.Tensor:
    n = num_classes
    soft = torch.full((n, n), SC_OTHER_EACH, device=device)
    for gt in range(n):
        gt_sc = fine_to_coarse[gt]
        same = [c for c in range(n) if fine_to_coarse[c] == gt_sc and c != gt]
        assert len(same) == 4, "CIFAR-100 super-class 는 각 5개 fine class 여야 함"
        per = SC_SAME_TOTAL / len(same)                          # 0.205 / 4 = 0.05125
        for c in same:
            soft[gt, c] = per
        soft[gt, gt] = SC_CORRECT_PROB                            # 정답 확률
        soft[gt] = soft[gt] / soft[gt].sum()
        
    return soft

def sc_aware_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, soft_matrix: torch.Tensor) -> torch.Tensor:
    soft_targets = soft_matrix[targets]
    log_probs = F.log_softmax(logits, dim=1)
    
    return -(soft_targets * log_probs).sum(dim=1).mean()

# ---------------------------------------------------------------------------
# Super-class logits (logsumexp aggregation)
# ---------------------------------------------------------------------------
def build_superclass_indices(fine_to_coarse: Sequence[int], num_superclasses: int = 20) -> List[List[int]]:
    groups: List[List[int]] = [[] for _ in range(num_superclasses)]
    for fine, coarse in enumerate(fine_to_coarse):
        groups[coarse].append(fine)
    assert all(len(g) == 5 for g in groups)
    
    return groups


def compute_superclass_logits(logits: torch.Tensor, sc_indices: List[torch.Tensor]) -> torch.Tensor:
    out = logits.new_empty(logits.size(0), len(sc_indices))
    for i, idx in enumerate(sc_indices):
        out[:, i] = torch.logsumexp(logits.index_select(1, idx), dim=1)
        
    return out

# ---------------------------------------------------------------------------
# Mixup 호환 combined loss
# ---------------------------------------------------------------------------
class CombinedLoss:
    def __init__(self, soft_matrix: torch.Tensor,
                 sc_indices: List[torch.Tensor],
                 fine_to_coarse_tensor: torch.Tensor):
        self.soft_matrix = soft_matrix                   # [100, 100]
        self.sc_indices = sc_indices                     # 20 × [5] LongTensor
        self.fine_to_coarse = fine_to_coarse_tensor      # [100] int → super-class
        self.crit_sc = nn.CrossEntropyLoss(label_smoothing=0.1)

    def __call__(self, logits: torch.Tensor,
                 targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float):
        # fine CE with Soft70
        fine_a = sc_aware_cross_entropy(logits, targets_a, self.soft_matrix)
        if lam == 1.0:
            fine_loss = fine_a
        else:
            fine_b = sc_aware_cross_entropy(logits, targets_b, self.soft_matrix)
            fine_loss = lam * fine_a + (1 - lam) * fine_b

        # super-class CE
        sc_logits = compute_superclass_logits(logits, self.sc_indices)
        sc_ta = self.fine_to_coarse[targets_a]
        sc_loss_a = self.crit_sc(sc_logits, sc_ta)
        if lam == 1.0:
            sc_loss = sc_loss_a
        else:
            sc_tb = self.fine_to_coarse[targets_b]
            sc_loss_b = self.crit_sc(sc_logits, sc_tb)
            sc_loss = lam * sc_loss_a + (1 - lam) * sc_loss_b

        return fine_loss, sc_loss