from __future__ import annotations
import math
import random
from typing import Tuple
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Sampler
from constants import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_transforms(input_size: int = 32) -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose([
        T.RandomCrop(input_size, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9, fill=(128, 128, 128)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, value="random"),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, test_transform

# ---------------------------------------------------------------------------
# Repeated Augmentation Sampler
# ---------------------------------------------------------------------------
class RASampler(Sampler):
    def __init__(self, dataset, num_repeats: int = 3, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset) * num_repeats   # 50000 * 3 = 150000

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices = indices * self.num_repeats
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

# ---------------------------------------------------------------------------
# Mixup + CutMix (batch level, 50/50)
# ---------------------------------------------------------------------------
def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    H, W = size[-2], size[-1]
    cut_rat = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(cy - cut_h // 2, 0)
    x1 = max(cx - cut_w // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    x2 = min(cx + cut_w // 2, W)
    
    return y1, x1, y2, x2

class MixupCutmix:
    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0, switch_prob: float = 0.5, enabled: bool = True):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.enabled = enabled

    def __call__(self, images: torch.Tensor, labels: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if not self.enabled:
            return images, labels, labels, 1.0

        use_cutmix = random.random() < self.switch_prob
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = float(np.random.beta(alpha, alpha))

        index = torch.randperm(images.size(0), device=images.device)
        targets_a, targets_b = labels, labels[index]

        if use_cutmix:
            y1, x1, y2, x2 = _rand_bbox(images.size(), lam)
            images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (images.size(-2) * images.size(-1))
        else:
            images = lam * images + (1 - lam) * images[index]

        return images, targets_a, targets_b, lam

# ---------------------------------------------------------------------------
# Dataloader 조립
# ---------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """DataLoader worker 각각에 결정론적 seed 주입 (재현성)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloaders(data_root: str = "./data", batch_size: int = 256,
                      num_workers: int = 8, seed: int = 0,
                      ra_repeats: int = 3, download: bool = True
                      ) -> Tuple[DataLoader, DataLoader, RASampler]:
    train_transform, test_transform = build_transforms(input_size=32)
    trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=download, transform=test_transform)
    train_sampler = RASampler(trainset, num_repeats=ra_repeats, shuffle=True, seed=seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker, generator=g,
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
    )
    
    return train_loader, test_loader, train_sampler