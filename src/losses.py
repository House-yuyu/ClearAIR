"""
Losses for ClearAIR.

Total loss (Eq. 1):
    L_total = L1 + alpha * L_inter,        alpha = 0.25

Internal Clue Reuse Mechanism — ICRM (Fig. 3, Eq. 15-17):
    I_r       : restored output
    I_r^w     : weak augmentation of I_r           (random crop)
    I_r^s     : strong augmentation of I_r^w       (color jitter + Gaussian blur)
    L_inter   = gamma * ||I_r^w - I_r^s||_2^2,     gamma = 0.05
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
def _random_crop_resize(images: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
    """Random crop a (scale * H, scale * W) patch and resize back."""
    b, _, h, w = images.shape
    ch, cw = int(h * scale), int(w * scale)
    out = torch.empty_like(images)
    for i in range(b):
        top = torch.randint(0, h - ch + 1, (1,)).item()
        left = torch.randint(0, w - cw + 1, (1,)).item()
        crop = images[i:i + 1, :, top:top + ch, left:left + cw]
        out[i] = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
    return out


def _color_jitter(images: torch.Tensor, strength: float = 0.4) -> torch.Tensor:
    """Brightness / contrast / saturation jitter on a (B, 3, H, W) tensor."""
    b = images.shape[0]
    out = images.clone()
    for i in range(b):
        bf = 1.0 + (torch.rand(1).item() - 0.5) * 2 * strength
        cf = 1.0 + (torch.rand(1).item() - 0.5) * 2 * strength
        sf = 1.0 + (torch.rand(1).item() - 0.5) * 2 * strength
        out[i] = TF.adjust_brightness(out[i], max(0.0, bf))
        out[i] = TF.adjust_contrast(out[i], max(0.0, cf))
        out[i] = TF.adjust_saturation(out[i], max(0.0, sf))
    return out


def _gaussian_blur(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


# ---------------------------------------------------------------------------
# ICRM
# ---------------------------------------------------------------------------
@dataclass
class ICRMConfig:
    weak_scale: float = 0.7              # random-crop ratio for weak aug
    color_jitter_strength: float = 0.4
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    gamma: float = 0.05                  # initial weight, Eq. 17


class ICRMLoss(nn.Module):
    """Internal Clue Reuse Mechanism loss (Eq. 15-17)."""

    def __init__(self, cfg: ICRMConfig | None = None):
        super().__init__()
        self.cfg = cfg or ICRMConfig()

    @torch.no_grad()
    def _augment(self, restored: torch.Tensor):
        # Eq. 15: weak augmentation
        weak = _random_crop_resize(restored, scale=self.cfg.weak_scale)
        # Eq. 16: strong augmentation on weak
        strong = _color_jitter(weak, strength=self.cfg.color_jitter_strength)
        strong = _gaussian_blur(
            strong,
            kernel_size=self.cfg.blur_kernel_size,
            sigma=self.cfg.blur_sigma,
        )
        return weak, strong

    def forward(self, restored: torch.Tensor) -> torch.Tensor:
        weak, strong = self._augment(restored)
        # Eq. 17: L2 distance, weighted by gamma
        diff = (weak - strong).pow(2).mean()
        return self.cfg.gamma * diff


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------
@dataclass
class ClearAIRLossConfig:
    alpha: float = 0.25                   # Eq. 1
    icrm: ICRMConfig = None  # type: ignore[assignment]


class ClearAIRLoss(nn.Module):
    def __init__(self, cfg: ClearAIRLossConfig | None = None):
        super().__init__()
        cfg = cfg or ClearAIRLossConfig()
        if cfg.icrm is None:
            cfg.icrm = ICRMConfig()
        self.cfg = cfg
        self.l1 = nn.L1Loss()
        self.icrm = ICRMLoss(cfg.icrm)

    def forward(
        self,
        restored: torch.Tensor,
        target: torch.Tensor,
    ):
        l1 = self.l1(restored, target)
        l_inter = self.icrm(restored)
        total = l1 + self.cfg.alpha * l_inter
        return total, {"l1": l1.detach(), "l_inter": l_inter.detach()}
