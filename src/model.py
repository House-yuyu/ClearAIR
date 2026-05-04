"""
Main ClearAIR model.

Architecture (Fig. 2 of the paper):

  Input ─► Extraction (3x3 conv) ─►  L1 ─► L2 ─► L3 ─► Bottleneck (L4)
                                      │     │     │
                                      ▼     ▼     ▼
                                     skip  skip  skip
                                      │     │     │
                                            ◄─────────────────────────────
            Output ◄─ Reconstruction ◄─ L1' ◄─ L2' ◄─ L3' ◄─ Bottleneck

Each level (encoder & decoder) consists of:
    PTB × n_blocks_first_half
    │
    ▼  ── conditioning block ──
    ├─► QGM(score_emb)
    ├─► SCA(semantic_features)
    └─► DAM(content, deg_prompt)
    │
    ▼
    PTB × n_blocks_second_half

Block counts per level (paper, level-1..level-4): [3, 5, 6, 8].
We split each level's PTBs in half around a single conditioning unit so the
modules' influence reaches every resolution of the U-Net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import (
    Downsample,
    OverlapPatchEmbed,
    PromptTransformerBlock,
    Upsample,
)
from .modules import (
    DegradationAwareModule,
    DegradationPromptGenerator,
    IQAAdapter,
    QualityGuidanceModule,
    SemanticCrossAttention,
    mask_average_pool,
)
from .auxiliary import MLLMIQA, SemanticGuidanceUnit, TaskIdentifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ClearAIRConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_dim: int = 48
    num_blocks: List[int] = field(default_factory=lambda: [3, 5, 6, 8])
    num_heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ffn_expansion: float = 2.66
    bias: bool = False

    # auxiliary networks
    iqa_hidden_dim: int = 4096
    iqa_out_dim: int = 256
    fc_dim: int = 512
    fd_dim: int = 512
    num_deg_prompts: int = 5

    # SGU
    num_masks: int = 16
    mask_dropout: float = 0.2

    # whether auxiliaries are wired to real frozen models or use dummy stand-ins
    dummy_auxiliaries: bool = True


# ---------------------------------------------------------------------------
# A single U-Net level with embedded conditioning.
# ---------------------------------------------------------------------------
class ConditionedLevel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int,
        prompt_dim: int,
        fq_dim: int,
        fc_dim: int,
        ffn_expansion: float = 2.66,
        bias: bool = False,
        sca_heads: int = 4,
        dam_heads: int = 4,
    ):
        super().__init__()
        # split PTBs in half around the single conditioning unit
        first = num_blocks // 2
        second = num_blocks - first

        self.pre_blocks = nn.ModuleList(
            [PromptTransformerBlock(dim, num_heads, ffn_expansion, bias) for _ in range(first)]
        )
        # conditioning units
        self.qgm = QualityGuidanceModule(dim, fq_dim)
        self.sca = SemanticCrossAttention(dim, num_heads=sca_heads, bias=bias)
        self.dam = DegradationAwareModule(
            dim, prompt_dim=prompt_dim, fc_dim=fc_dim,
            num_heads=dam_heads, bias=bias,
        )

        self.post_blocks = nn.ModuleList(
            [PromptTransformerBlock(dim, num_heads, ffn_expansion, bias) for _ in range(second)]
        )

        # adapter that matches the semantic feature channel count to `dim`
        # (SGU's MAP returns features in the level's own dim, so this is identity
        #  if shallow features were lifted to `dim` already)
        self.sem_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        f_q: torch.Tensor,
        f_sem: torch.Tensor,
        f_c: torch.Tensor,
        f_p: torch.Tensor,
    ) -> torch.Tensor:
        for blk in self.pre_blocks:
            x = blk(x)

        x = self.qgm(x, f_q)
        x = self.sca(x, self.sem_proj(f_sem))
        x = self.dam(x, f_c, f_p)

        for blk in self.post_blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# ClearAIR
# ---------------------------------------------------------------------------
class ClearAIR(nn.Module):
    def __init__(self, cfg: ClearAIRConfig | None = None):
        super().__init__()
        cfg = cfg or ClearAIRConfig()
        self.cfg = cfg

        dim = cfg.base_dim
        num_blocks = cfg.num_blocks
        num_heads = cfg.num_heads
        bias = cfg.bias
        ffn = cfg.ffn_expansion

        # ---- frozen auxiliary networks ---------------------------------
        self.iqa = MLLMIQA(
            hidden_dim=cfg.iqa_hidden_dim,
            out_dim=cfg.iqa_out_dim,
            dummy=cfg.dummy_auxiliaries,
        )
        self.sgu = SemanticGuidanceUnit(
            num_masks=cfg.num_masks,
            mask_dropout=cfg.mask_dropout,
            dummy=cfg.dummy_auxiliaries,
        )
        self.task_id = TaskIdentifier(
            embed_dim=cfg.fc_dim,
            dummy=cfg.dummy_auxiliaries,
        )

        # ---- IQA adapter (Eq. 3) and degradation prompt (Eq. 10) -------
        self.iqa_adapter = IQAAdapter(cfg.iqa_out_dim, dim * 4)
        # F_q is fed into every level; we project it to per-level dims:
        self.fq_per_level = nn.ModuleList(
            [nn.Linear(dim * 4, dim * (2 ** i)) for i in range(4)]
        )

        prompt_dim = dim * 4  # shared prompt dimension
        self.deg_prompt_gen = DegradationPromptGenerator(
            fd_dim=cfg.fd_dim,
            num_prompts=cfg.num_deg_prompts,
            prompt_dim=prompt_dim,
        )

        # ---- shallow extraction (Fig. 2: Extraction) -------------------
        self.patch_embed = OverlapPatchEmbed(cfg.in_channels, dim, bias=bias)

        # ---- encoder levels --------------------------------------------
        self.enc1 = ConditionedLevel(
            dim, num_blocks[0], num_heads[0],
            prompt_dim=prompt_dim, fq_dim=dim, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )
        self.down1 = Downsample(dim)

        self.enc2 = ConditionedLevel(
            dim * 2, num_blocks[1], num_heads[1],
            prompt_dim=prompt_dim, fq_dim=dim * 2, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )
        self.down2 = Downsample(dim * 2)

        self.enc3 = ConditionedLevel(
            dim * 4, num_blocks[2], num_heads[2],
            prompt_dim=prompt_dim, fq_dim=dim * 4, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )
        self.down3 = Downsample(dim * 4)

        self.bottleneck = ConditionedLevel(
            dim * 8, num_blocks[3], num_heads[3],
            prompt_dim=prompt_dim, fq_dim=dim * 8, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )

        # ---- decoder levels --------------------------------------------
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dec3 = ConditionedLevel(
            dim * 4, num_blocks[2], num_heads[2],
            prompt_dim=prompt_dim, fq_dim=dim * 4, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dec2 = ConditionedLevel(
            dim * 2, num_blocks[1], num_heads[1],
            prompt_dim=prompt_dim, fq_dim=dim * 2, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )

        self.up1 = Upsample(dim * 2)
        self.dec1 = ConditionedLevel(
            dim * 2, num_blocks[0], num_heads[0],   # paper keeps 2*dim at level-1 of decoder
            prompt_dim=prompt_dim, fq_dim=dim * 2, fc_dim=cfg.fc_dim,
            ffn_expansion=ffn, bias=bias,
        )

        # final refinement
        self.refine = nn.Sequential(
            *[PromptTransformerBlock(dim * 2, num_heads[0], ffn, bias)
              for _ in range(num_blocks[0])]
        )

        # ---- reconstruction (Fig. 2: Reconstruction) -------------------
        self.output = nn.Conv2d(
            dim * 2, cfg.out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _resize_masks(self, masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Nearest-neighbour resize to (h, w), keeping binary semantics."""
        return torch.nn.functional.interpolate(masks, size=(h, w), mode="nearest")

    def _semantic_features(
        self,
        feats: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = feats.shape
        masks_resized = self._resize_masks(masks, h, w)
        return mask_average_pool(feats, masks_resized)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) frozen priors -------------------------------------------------
        q = self.iqa(x)                                       # (B, iqa_out_dim)
        masks = self.sgu(x)                                   # (B, Nm, H, W)
        f_c, f_d = self.task_id(x)                            # (B, 512), (B, 512)

        # 2) project priors to per-level conditioning ---------------------
        f_q_root = self.iqa_adapter(q)                        # (B, dim*4)
        f_q_levels = [proj(f_q_root) for proj in self.fq_per_level]  # 4 tensors
        f_p = self.deg_prompt_gen(f_d)                        # (B, prompt_dim)

        # 3) shallow extraction -------------------------------------------
        x0 = self.patch_embed(x)                              # (B, dim, H, W)

        # 4) encoder -------------------------------------------------------
        e1 = self.enc1(
            x0, f_q_levels[0],
            self._semantic_features(x0, masks),
            f_c, f_p,
        )
        e2_in = self.down1(e1)
        e2 = self.enc2(
            e2_in, f_q_levels[1],
            self._semantic_features(e2_in, masks),
            f_c, f_p,
        )
        e3_in = self.down2(e2)
        e3 = self.enc3(
            e3_in, f_q_levels[2],
            self._semantic_features(e3_in, masks),
            f_c, f_p,
        )
        b_in = self.down3(e3)
        b_out = self.bottleneck(
            b_in, f_q_levels[3],
            self._semantic_features(b_in, masks),
            f_c, f_p,
        )

        # 5) decoder + skip connections -----------------------------------
        d3 = self.up3(b_out)
        d3 = self.reduce3(torch.cat([d3, e3], dim=1))
        d3 = self.dec3(
            d3, f_q_levels[2],
            self._semantic_features(d3, masks),
            f_c, f_p,
        )

        d2 = self.up2(d3)
        d2 = self.reduce2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2(
            d2, f_q_levels[1],
            self._semantic_features(d2, masks),
            f_c, f_p,
        )

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)                       # 2*dim channels
        d1 = self.dec1(
            d1, f_q_levels[1],
            self._semantic_features(d1, masks),
            f_c, f_p,
        )

        # 6) refinement and reconstruction --------------------------------
        out = self.refine(d1)
        residual = self.output(out)
        return x + residual                                   # global skip
