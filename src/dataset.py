
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def adair_center_crop_to_multiple(image: np.ndarray, base: int = 16) -> np.ndarray:
    """Match AdaIR's centered test-time crop to a spatial multiple of ``base``."""
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {image.shape}.")
    height, width = image.shape[:2]
    crop_h = height % base
    crop_w = width % base
    top, left = crop_h // 2, crop_w // 2
    return image[top:height - crop_h + top, left:width - crop_w + left, :]


def _list_images(root: Path) -> List[Path]:
    """List image files recursively in a stable order."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXT)


@dataclass(frozen=True)
class _Sample:
    input_path: Optional[Path]
    target_path: Path
    degradation: str
    dataset: str
    noise_sigma: Optional[int] = None


_ADAIR_NOISE_SIGMAS = (15, 25, 50)


def _adair_repeat_count(degradation: str) -> int:
    """Return AdaIR's explicit repetition factor for paired training sets."""
    return {
        "derain": 120,
        "deblur": 5,
        "lowlight": 20,
        "enhance": 20,
    }.get(degradation, 1)


def _source_is_selected(source_index: int, subset_modulo: int, subset_index: int) -> bool:
    """Select a deterministic source-image subset before task repetition."""
    if subset_modulo < 1:
        raise ValueError("source subset modulo must be at least one")
    if not 0 <= subset_index < subset_modulo:
        raise ValueError("source subset index must be in [0, source subset modulo)")
    return source_index % subset_modulo == subset_index


def _stem_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _list_images(root):
        if path.stem in result:
            raise RuntimeError(f"Duplicate target stem '{path.stem}' under {root}.")
        result[path.stem] = path
    return result


def _paper_root(root: Path) -> Path:
    candidate = root / "AiO"
    return candidate if candidate.is_dir() else root


def _paper_train_samples(
    root: Path,
    degradations: Sequence[str],
    sampling_profile: str = "natural",
    source_subset_modulo: int = 1,
    source_subset_index: int = 0,
) -> list[_Sample]:
    """Build paper-layout samples, optionally matching AdaIR's task balance.

    AdaIR does not sample each source image once.  Its official dataset code
    instantiates every denoising image three times at each sigma (nine entries
    total), repeats Rain100L 120 times, GoPro 5 times, and LOL 20 times.  This
    keeps the tiny paired datasets from being drowned out by the 72K OTS haze
    pairs.  ``natural`` retains the original one-file-one-entry behavior.
    """
    if sampling_profile not in {"natural", "adair"}:
        raise ValueError(f"Unsupported sampling profile: {sampling_profile!r}")
    _source_is_selected(0, source_subset_modulo, source_subset_index)
    root = _paper_root(root) / "train"
    mapping = {
        "denoise": ("Denosie", "gt", None),
        "dehaze": ("Dehaze", "gt", "input"),
        "derain": ("Derain", "gt", "input"),
        "deblur": ("Deblur", "gt", "input"),
        "lowlight": ("Enhance", "gt", "input"),
        "enhance": ("Enhance", "gt", "input"),
    }
    samples: list[_Sample] = []
    for degradation in degradations:
        key = degradation.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported paper degradation '{degradation}'.")
        folder, target_name, input_name = mapping[key]
        base = root / folder
        targets = _stem_index(base / target_name)
        if not targets:
            raise FileNotFoundError(f"No target images found under {base / target_name}.")
        if input_name is None:
            if sampling_profile == "adair":
                for source_index, path in enumerate(targets.values()):
                    if not _source_is_selected(source_index, source_subset_modulo, source_subset_index):
                        continue
                    for sigma in _ADAIR_NOISE_SIGMAS:
                        samples.extend(
                            _Sample(None, path, key, "BSD400+WED", noise_sigma=sigma)
                            for _ in range(3)
                        )
            else:
                samples.extend(
                    _Sample(None, path, key, "BSD400+WED")
                    for source_index, path in enumerate(targets.values())
                    if _source_is_selected(source_index, source_subset_modulo, source_subset_index)
                )
            continue

        input_root = base / input_name
        before = len(samples)
        repeats = _adair_repeat_count(key) if sampling_profile == "adair" else 1
        for source_index, input_path in enumerate(_list_images(input_root)):
            if not _source_is_selected(source_index, source_subset_modulo, source_subset_index):
                continue
            stem = input_path.stem
            if key == "dehaze":
                target_key = stem.split("_", 1)[0]
            elif key == "derain":
                target_key = "norain-" + stem.removeprefix("rain-")
            else:
                target_key = stem
            target_path = targets.get(target_key)
            if target_path is not None:
                samples.extend(
                    _Sample(input_path, target_path, key, folder) for _ in range(repeats)
                )
        if len(samples) == before:
            raise RuntimeError(f"No paired samples found for paper degradation '{degradation}'.")
    return samples


def _paper_test_samples(root: Path, degradations: Sequence[str]) -> list[_Sample]:
    root = _paper_root(root) / "test"
    samples: list[_Sample] = []
    for degradation in degradations:
        key = degradation.lower()
        if key == "denoise":
            for dataset in ("bsd68", "urban100", "kodak24"):
                for target_path in _list_images(root / "denoise" / dataset / "target"):
                    samples.append(_Sample(None, target_path, key, dataset))
        elif key == "dehaze":
            target_index = _stem_index(root / "dehaze" / "target")
            for input_path in _list_images(root / "dehaze" / "input"):
                target_key = input_path.stem.split("_", 1)[0]
                target_path = target_index.get(target_key)
                if target_path is not None:
                    samples.append(_Sample(input_path, target_path, key, "SOTS-Outdoor"))
        elif key == "derain":
            target_index = _stem_index(root / "derain" / "Rain100L" / "target")
            for input_path in _list_images(root / "derain" / "Rain100L" / "input"):
                target_path = target_index.get(input_path.stem)
                if target_path is not None:
                    samples.append(_Sample(input_path, target_path, key, "Rain100L"))
        elif key == "deblur":
            target_index = _stem_index(root / "deblur" / "gopro" / "target")
            for input_path in _list_images(root / "deblur" / "gopro" / "input"):
                target_path = target_index.get(input_path.stem)
                if target_path is not None:
                    samples.append(_Sample(input_path, target_path, key, "GoPro"))
        elif key in {"lowlight", "enhance"}:
            target_index = _stem_index(root / "enhance" / "lol" / "target")
            for input_path in _list_images(root / "enhance" / "lol" / "input"):
                target_path = target_index.get(input_path.stem)
                if target_path is not None:
                    samples.append(_Sample(input_path, target_path, "lowlight", "LOLv1"))
        else:
            raise ValueError(f"Unsupported paper degradation '{degradation}'.")
    if not samples:
        raise RuntimeError(f"No test samples found under {root}.")
    return samples


class AiOIRDataset(Dataset):
    """A generic or paper-layout All-in-One image restoration dataset."""

    def __init__(
        self,
        root: str,
        degradations: List[str],
        patch_size: int = 256,
        train: bool = True,
        paper_layout: bool = False,
        noise_sigmas: Sequence[int] = (15, 25, 50),
        noise_seed: Optional[int] = None,
        adair_test_protocol: bool = False,
        sampling_profile: str = "natural",
        source_subset_modulo: int = 1,
        source_subset_index: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.patch_size = patch_size
        self.paper_layout = paper_layout
        self.noise_seed = noise_seed
        self.adair_test_protocol = adair_test_protocol
        self.sampling_profile = sampling_profile
        self.source_subset_modulo = source_subset_modulo
        self.source_subset_index = source_subset_index
        self.noise_sigmas = tuple(int(sigma) for sigma in noise_sigmas)
        if not self.noise_sigmas or any(sigma <= 0 for sigma in self.noise_sigmas):
            raise ValueError("noise_sigmas must contain positive values.")
        if paper_layout:
            self.samples = (
                _paper_train_samples(
                    self.root,
                    degradations,
                    sampling_profile,
                    source_subset_modulo,
                    source_subset_index,
                )
                if train else _paper_test_samples(self.root, degradations)
            )
        else:
            self.samples = self._generic_samples(degradations)
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root}.")
        self.to_tensor = transforms.ToTensor()

    def _generic_samples(self, degradations: Sequence[str]) -> list[_Sample]:
        samples: list[_Sample] = []
        for degradation in degradations:
            in_dir = self.root / degradation / "input"
            gt_dir = self.root / degradation / "target"
            if not in_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(f"Missing folders for '{degradation}' under {self.root}.")
            targets = _stem_index(gt_dir)
            for input_path in _list_images(in_dir):
                target_path = targets.get(input_path.stem)
                if target_path is not None:
                    samples.append(_Sample(input_path, target_path, degradation, degradation))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, lq: Image.Image, gt: Image.Image):
        if not self.train:
            return self.to_tensor(lq), self.to_tensor(gt)
        w, h = lq.size
        ps = self.patch_size
        if w < ps or h < ps:
            scale = ps / min(w, h)
            size = (max(ps, int(round(w * scale))), max(ps, int(round(h * scale))))
            lq = lq.resize(size, Image.BICUBIC)
            gt = gt.resize(size, Image.BICUBIC)
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

    def _add_gaussian_noise(self, clean: torch.Tensor, sigma: int, idx: int) -> torch.Tensor:
        if self.noise_seed is None:
            noise = torch.randn_like(clean)
        else:
            generator = torch.Generator(device=clean.device).manual_seed(self.noise_seed + idx)
            noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
        noisy = (clean * 255.0 + noise * float(sigma)).clamp(0, 255)
        return noisy.to(torch.uint8).float() / 255.0

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.adair_test_protocol and not self.train:
            gt_array = adair_center_crop_to_multiple(
                np.array(Image.open(sample.target_path).convert("RGB"))
            )
            if sample.input_path is None:
                sigma = random.choice(self.noise_sigmas)
                noise = np.random.randn(*gt_array.shape)
                lq_array = np.clip(gt_array + noise * float(sigma), 0, 255).astype(np.uint8)
            else:
                lq_array = adair_center_crop_to_multiple(
                    np.array(Image.open(sample.input_path).convert("RGB"))
                )
            return {
                "lq": self.to_tensor(lq_array),
                "gt": self.to_tensor(gt_array),
                "deg": sample.degradation,
                "dataset": sample.dataset,
            }

        gt = Image.open(sample.target_path).convert("RGB")
        if sample.input_path is None:
            # Crop/flip first, then synthesize noise in the uint8 range.
            lq, gt = self._augment(gt, gt.copy())
            sigma = sample.noise_sigma or random.choice(self.noise_sigmas)
            lq = self._add_gaussian_noise(gt, sigma, idx)
        else:
            lq_image = Image.open(sample.input_path).convert("RGB")
            lq, gt = self._augment(lq_image, gt)
        return {"lq": lq, "gt": gt, "deg": sample.degradation, "dataset": sample.dataset}
