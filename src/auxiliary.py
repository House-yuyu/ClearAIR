from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) MLLM-based IQA  (Section "Overall Assessment", Eq. 2)
# ---------------------------------------------------------------------------
class MLLMIQA(nn.Module):
    """
    Wraps a frozen MLLM-IQA model (DeQA in the paper).

    The paper extracts the hidden state Q from the layer *preceding* the
    'quality level' token. Implementations differ across MLLMs, so we expose a
    single hook `_extract_quality_state(images, prompt_text)` that the user
    overrides to plug in their MLLM of choice.

    The default `dummy=True` mode returns a lightweight frozen proxy embedding
    so that the rest of the pipeline can be shape-tested without the heavy
    MLLM. It is not a replacement for DeQA in paper-reproduction experiments.
    """

    DEFAULT_PROMPT = (
        "USER: How would you rate the quality of this image?\n"
        "<|image|>\nASSISTANT: The quality of the image is"
    )

    def __init__(
        self,
        hidden_dim: int = 4096,       # DeQA / Qwen2.5-VL hidden size
        out_dim: int = 4096,
        dummy: bool = True,
        mllm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dummy = dummy

        if dummy:
            # very small CNN -> vector. only used until the real MLLM is loaded.
            self.dummy_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
            )
        else:
            assert mllm is not None, "Provide an MLLM module when dummy=False."
            self.mllm = mllm
            for p in self.mllm.parameters():
                p.requires_grad = False

        self.proj = nn.Identity() if hidden_dim == out_dim else nn.Linear(hidden_dim, out_dim)
        if dummy:
            # Dummy mode is for pipeline validation only. Its proxy must not be
            # counted as a trainable substitute for the frozen DeQA model.
            for p in self.parameters():
                p.requires_grad = False

    def _extract_quality_state(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        """Return Q with shape (B, hidden_dim).

        Implement by:
          1. tokenizing `prompt` and the (vision-encoded) image,
          2. running the MLLM forward pass with `output_hidden_states=True`,
          3. selecting the hidden state at the position of the token preceding
             the predicted 'quality level' token,
          4. returning that hidden state.
        """
        return self.mllm(images, prompt)

    # ----------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.dummy:
                q = self.dummy_encoder(images)
            else:
                q = self._extract_quality_state(images, self.DEFAULT_PROMPT)
        # External backends use inference_mode; clone after leaving that
        # context so trainable adapters may safely save this tensor backward.
        q = q.clone()
        projection_parameter = next(self.proj.parameters(), None)
        target_device = projection_parameter.device if projection_parameter is not None else images.device
        target_dtype = projection_parameter.dtype if projection_parameter is not None else q.dtype
        q = q.to(device=target_device, dtype=target_dtype)
        return self.proj(q)


# ---------------------------------------------------------------------------
# 2) Semantic Guidance Unit  (Section "Region Awareness", Eq. 5)
# ---------------------------------------------------------------------------
class SemanticGuidanceUnit(nn.Module):
    """
    Wraps a frozen SAM2 to produce N_m binary masks of shape
    (B, N_m, H, W). For training we follow the paper's *mask dropout*: during
    training a random subset of masks is removed and merged into background.

    `dummy=True` produces grid-based fake masks so that downstream code can be
    debugged without SAM2 installed.
    """

    def __init__(
        self,
        num_masks: int = 16,
        mask_dropout: float = 0.2,
        dummy: bool = True,
        sam2: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_masks = num_masks
        self.mask_dropout = mask_dropout
        self.dummy = dummy

        if not dummy:
            assert sam2 is not None, "Provide a SAM2 module when dummy=False."
            self.sam2 = sam2
            for p in self.sam2.parameters():
                p.requires_grad = False

    def _segment(self, images: torch.Tensor) -> torch.Tensor:
        """Return masks (B, N_m, H, W) in {0, 1}."""
        return self.sam2(images, num_masks=self.num_masks)

    # ---------------------------------------------------------------------
    def _dummy_masks(self, images: torch.Tensor) -> torch.Tensor:
        """Grid partition with N_m cells. Useful when SAM2 is not available."""
        b, _, h, w = images.shape
        nm = self.num_masks
        grid = int(round(nm ** 0.5))
        nm = grid * grid
        masks = torch.zeros(b, nm, h, w, device=images.device, dtype=images.dtype)
        ph, pw = h // grid, w // grid
        idx = 0
        for i in range(grid):
            for j in range(grid):
                masks[:, idx, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 1.0
                idx += 1
        return masks

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        masks = self._dummy_masks(images) if self.dummy else self._segment(images)
        masks = masks.clone()

        # mask dropout: merge dropped masks into a single background channel
        if self.training and self.mask_dropout > 0:
            b, nm, h, w = masks.shape
            keep = (torch.rand(b, nm, device=masks.device) > self.mask_dropout).float()
            keep = keep.unsqueeze(-1).unsqueeze(-1)
            masks = masks * keep
        return masks


# ---------------------------------------------------------------------------
# 3) Task Identifier  (Section "Task Recognition")
# ---------------------------------------------------------------------------
class TaskIdentifier(nn.Module):
    """
    Wraps DA-CLIP to produce a 512-d content embedding F_c and a 512-d
    degradation embedding F_d for each image (paper Eq. 10 surroundings).

    Returns
    -------
    fc, fd : (B, 512), (B, 512)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        dummy: bool = True,
        da_clip: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = dummy

        if dummy:
            self.dummy_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.fc_head = nn.Linear(64, embed_dim)
            self.fd_head = nn.Linear(64, embed_dim)
        else:
            assert da_clip is not None, "Provide a DA-CLIP module when dummy=False."
            self.da_clip = da_clip
            for p in self.da_clip.parameters():
                p.requires_grad = False

        if dummy:
            # Keep the test-only proxy consistent with the frozen paper prior.
            for p in self.parameters():
                p.requires_grad = False

    def _encode(self, images: torch.Tensor):
        return self.da_clip(images)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        if self.dummy:
            feat = self.dummy_encoder(images)
            return self.fc_head(feat), self.fd_head(feat)
        fc, fd = self._encode(images)
        return (
            fc.clone().to(images.device, dtype=images.dtype),
            fd.clone().to(images.device, dtype=images.dtype),
        )
