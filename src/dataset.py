"""
Multi-degradation dataset for All-in-One Image Restoration (AiOIR).

Expected directory layout (one sub-directory per degradation type):

    <root>/
        denoise/
            input/
            target/
        dehaze/
            input/
            target/
        derain/
            input/
            target/
        ...

Where `target/` is the ground-truth clean image and `input/` is the degraded
counterpart with matching filename. For datasets that only provide clean
images (e.g. BSD400 for noise), the user supplies a pre-generated noisy copy
under `input/` so the dataset stays format-agnostic.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.suffix.lower() in _IMG_EXT])


class AiOIRDataset(Dataset):
    """A generic AiOIR dataset that mixes several degradation sub-folders."""

    def __init__(
        self,
        root: str,
        degradations: List[str],
        patch_size: int = 256,
        train: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.patch_size = patch_size

        self.samples: List[Tuple[Path, Path, str]] = []
        for deg in degradations:
            in_dir = self.root / deg / "input"
            gt_dir = self.root / deg / "target"
            if not in_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(f"Missing folders for '{deg}' under {self.root}.")

            ins = _list_images(in_dir)
            gts = {p.stem: p for p in _list_images(gt_dir)}
            for ip in ins:
                if ip.stem in gts:
                    self.samples.append((ip, gts[ip.stem], deg))

        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root}.")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    # ----------------------------------------------------------------------
    # Augmentation: random crop + horizontal/vertical flip (training only)
    # ----------------------------------------------------------------------
    def _augment(self, lq: Image.Image, gt: Image.Image):
        if not self.train:
            return self.to_tensor(lq), self.to_tensor(gt)

        w, h = lq.size
        ps = self.patch_size
        # if the image is smaller than the patch, resize up first
        if w < ps or h < ps:
            scale = ps / min(w, h)
            lq = lq.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            gt = gt.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = lq.size

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        lq = lq.crop((x, y, x + ps, y + ps))
        gt = gt.crop((x, y, x + ps, y + ps))

        if random.random() < 0.5:
            lq = lq.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lq = lq.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)

        return self.to_tensor(lq), self.to_tensor(gt)

    def __getitem__(self, idx: int):
        in_path, gt_path, deg = self.samples[idx]
        lq = Image.open(in_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        lq, gt = self._augment(lq, gt)
        return {"lq": lq, "gt": gt, "deg": deg}
