from __future__ import annotations
import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 가중치 초기화
# ---------------------------------------------------------------------------
def _trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> torch.Tensor:
    # 평균이 [a, b] 밖이면 경고 (사용상 거의 없음)
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("trunc_normal_: mean outside [a, b] ± 2 * std — 분포가 왜곡됨")

    with torch.no_grad():
        def _norm_cdf(x: float) -> float:
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        lo = _norm_cdf((a - mean) / std)
        hi = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # shape = (B, 1, 1, ...) — batch dim 만 살리고 나머지는 broadcast
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"

# ---------------------------------------------------------------------------
# Smooth Overlapping Patch Embedding (SOPE)
# ---------------------------------------------------------------------------
class Affine(nn.Module):
    """
    채널별 학습 가능 스케일/바이어스 (shape 은 conv 특징 맵 [B, C, H, W] 기준)
    """
    def __init__(self, dim: int):
        super().__init__()
        # 초기값: scale=1, bias=0 → 처음엔 identity
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta

def _conv3x3_bn(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
    )

class ConvPatchEmbed(nn.Module):
    """
    img[B, 3, 32, 32] -> pre_affine -> conv3x3+BN -> GELU -> conv2x2 -> [B, 192, 16, 16]
    num_patches = 16x16 = 256, 출력은 [B, 256, 192]
    """
    def __init__(self, img_size: int = 32, patch_size: int = 2, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        if patch_size != 2:
            raise ValueError(f"이 repo 는 DHVT-T CIFAR-100 (patch=2) 전용. patch_size={patch_size} 받음.")
        self.img_size = img_size
        self.patch_size = patch_size
        # (32/2, 32/2) = (16, 16) = 256 patches
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # pre-affine on raw RGB input
        self.pre_affine = Affine(in_chans)

        # Conv stack: patch_size=2 는 `conv3x3(stride=2)+GELU` 한 블록으로 충분.
        self.proj = nn.Sequential(
            _conv3x3_bn(in_chans, embed_dim, stride=2),  # [B, 3, 32, 32] → [B, 192, 16, 16]
            nn.GELU(),
        )

        # post-affine on embedded feature map
        self.post_affine = Affine(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        x = self.pre_affine(x)                 # [B, 3, 32, 32]
        x = self.proj(x)                       # [B, 192, 16, 16]
        x = self.post_affine(x)                # [B, 192, 16, 16]
        x = x.flatten(2).transpose(1, 2)       # [B, 256, 192]
        return x

# ---------------------------------------------------------------------------
# Head-Injected Multi-Head Self-Attention (HI-MHSA)
# ---------------------------------------------------------------------------
class HI_Attention(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 4, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # 표준 QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Head-token 생성용 서브모듈
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(head_dim, dim, bias=True)
        self.ht_norm = nn.LayerNorm(head_dim)
        # head-token 4개 각각에 대한 learnable Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_heads, dim))
        _trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape     # N = 257 (= 1 CLS + 256 patches)

        # ── Head-token 생성 ──
        head_in = x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # sequence dim 을 평균 → [B, h, d]  (head 당 단일 프로토타입 벡터)
        head_in = head_in.mean(dim=2)
        # 48 → 192 projection, 그리고 [B, h, h, d] 로 reshape 후 flatten 하여 [B, h, C] 형태의 4개 head-token
        head_tok = self.ht_proj(head_in)                                      # [B, h, C]
        head_tok = head_tok.reshape(B, self.num_heads, self.num_heads, self.head_dim)
        head_tok = self.act(self.ht_norm(head_tok)).flatten(2)                # [B, h, C]
        head_tok = head_tok + self.pos_embed                                  # + PE

        # 시퀀스에 head-token 4개 concat → 길이 257 + 4 = 261
        x = torch.cat([x, head_tok], dim=1)                                   # [B, 261, C]

        # ── standard MHSA on concat sequence ──
        Nh = N + self.num_heads                                               # 261
        qkv = self.qkv(x)                                                     # [B, 261, 3C]
        # [B, 261, 3, h, d] → permute → [3, B, h, 261, d]
        qkv = qkv.reshape(B, Nh, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                      # each [B, h, 261, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale                         # [B, h, 261, 261]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Nh, C)                      # [B, 261, C]
        x = self.proj(x)

        # ── Head-token 을 CLS 에 흡수 ──
        cls, patches, ht = torch.split(x, [1, N - 1, self.num_heads], dim=1)
        cls = cls + ht.mean(dim=1, keepdim=True)                              # [B, 1, C]
        x = torch.cat([cls, patches], dim=1)                                  # [B, 257, C]
        x = self.proj_drop(x)
        return x

# ---------------------------------------------------------------------------
# Dual Attention Feed-Forward (DAFF)
# ---------------------------------------------------------------------------
class DAFF(nn.Module):
    def __init__(self, in_features: int = 192, hidden_features: int = 768, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) // 2

        # pointwise 확장
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        # depthwise 3x3 (groups=hidden_features -> 채널별 독립 conv)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size,
                               padding=pad, groups=hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        # pointwise 축소 (원래 채널로 되돌림)
        self.conv3 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(in_features)

        self.act = nn.GELU()

        # Squeeze-Excitation: 채널 reduce ratio=4 (in_features // 4)
        # compress 는 in_features(=192) 입력 - in_fea>tures // 4 (=48) 출력
        # excitation 은 다시 in_features(=192) 복원
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N=257, C=192]
        B, N, C = x.shape
        # CLS 와 patches 분리
        cls_token, patches = torch.split(x, [1, N - 1], dim=1)      # [B,1,C], [B,256,C]

        # patches 를 공간 텐서로 되돌림: 256 tokens = 16x16 grid
        hw = int(math.sqrt(N - 1))                                  # 16
        x_sp = patches.reshape(B, hw, hw, C).permute(0, 3, 1, 2)    # [B, C, 16, 16]

        # Conv 스택
        x_sp = self.act(self.bn1(self.conv1(x_sp)))                 # [B, 768, 16, 16]
        shortcut = x_sp
        x_sp = self.act(self.bn2(self.conv2(x_sp))) + shortcut       # DW-Conv + residual
        x_sp = self.bn3(self.conv3(x_sp))                           # [B, 192, 16, 16]

        # SE 채널 가중치 (sigmoid 없음 — 오리지널 그대로, excitation linear 결과를 직접 곱)
        w = self.squeeze(x_sp).flatten(1).reshape(B, 1, C)          # [B, 1, 192]
        w = self.excitation(self.act(self.compress(w)))             # [B, 1, 192]
        # CLS 만 reweight (patches 는 그대로)
        cls_token = cls_token * w

        # patches 를 다시 시퀀스로
        patches = x_sp.flatten(2).permute(0, 2, 1)                  # [B, 256, 192]
        return torch.cat([cls_token, patches], dim=1)                # [B, 257, 192]

# ---------------------------------------------------------------------------
# DHVT Transformer Block (HI-MHSA + DAFF)
# ---------------------------------------------------------------------------
class DHVT_Block(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 4, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = HI_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = DAFF(in_features=dim, hidden_features=hidden, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---------------------------------------------------------------------------
# DHVT-T Main Model
# ---------------------------------------------------------------------------
class DHVT(nn.Module):
    def __init__(self, num_classes: int = 100, img_size: int = 32, patch_size: int = 2,
                 in_chans: int = 3, embed_dim: int = 192, depth: int = 12,
                 num_heads: int = 4, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding (SOPE)
        self.patch_embed = ConvPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 256

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 12 개 DHVT block
        self.blocks = nn.ModuleList([
            DHVT_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path_rate=dpr[i])
            for i in range(depth)
        ])

        # 최종 LN + classifier head
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # 가중치 초기화
        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set:
        return {"cls_token"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """patch embed → CLS prepend → 12 blocks → LN → CLS 반환."""
        x = self.patch_embed(x)                              # [B, 256, 192]
        cls = self.cls_token.expand(x.shape[0], -1, -1)      # [B, 1, 192]
        x = torch.cat([cls, x], dim=1)                       # [B, 257, 192]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                       # CLS 만 반환 [B, 192]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))           # [B, num_classes]


def build_dhvt_t(num_classes: int = 100, drop_path_rate: float = 0.1) -> DHVT:
    return DHVT(
        num_classes=num_classes,
        img_size=32, patch_size=2, in_chans=3,
        embed_dim=192, depth=12, num_heads=4, mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )