"""
Restormer-style building blocks used as the backbone of ClearAIR.

Reference:
- Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image
  Restoration", CVPR 2022.
- ClearAIR's Prompt Transformer Block (PTB) reuses the same MDTA + GDFN
  structure; conditioning is injected by external modules (QGM/SCA/DAM)
  defined in `modules.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """BiasFree LayerNorm over channel dim for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


# ---------------------------------------------------------------------------
# Multi-Dconv Head Transposed Attention (MDTA)
# ---------------------------------------------------------------------------
class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # rearrange to (B, heads, head_dim, H*W)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.reshape(b, c, h, w)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Gated-Dconv Feed-Forward Network (GDFN)
# ---------------------------------------------------------------------------
class GDFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Prompt Transformer Block (PTB) — pre-norm Restormer block
# ---------------------------------------------------------------------------
class PromptTransformerBlock(nn.Module):
    """A single PTB. Conditioning is applied externally via QGM/SCA/DAM."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch (down/up)-sampling for the U-shaped backbone
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Initial / final shallow convs (Extraction & Reconstruction in Fig. 2)
# ---------------------------------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """Initial 3x3 conv to extract shallow features (Fig. 2: Extraction)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
