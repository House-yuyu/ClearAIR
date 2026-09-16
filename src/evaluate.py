"""Evaluate a ClearAIR checkpoint on the paper's AiOIR test layout.

The script reports the same reference metrics used in Tables 2 and 3:
RGB PSNR and SSIM, grouped by benchmark.  Denoising inputs are synthesized
with Gaussian noise at sigma 15, 25, and 50, matching the paper protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from clearair import AiOIRDataset, ClearAIR, ClearAIRConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--degradations",
        nargs="+",
        default=["denoise", "dehaze", "derain"],
        help="Three-task default; add deblur lowlight for the five-task protocol.",
    )
    parser.add_argument("--paper-layout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aux-device", default=None)
    parser.add_argument(
        "--dummy-aux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the auxiliary mode saved in the checkpoint.",
    )
    parser.add_argument("--deqa-model", default=None)
    parser.add_argument("--deqa-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sam2-model", default=None)
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--daclip-checkpoint", default=None)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--noise-sigmas", nargs="+", type=int, default=[15, 25, 50])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--adair-protocol",
        action="store_true",
        help=(
            "Match official AdaIR three-task testing: full BSD68 at sigma 15/25/50, Rain100L, "
            "and SOTS-Outdoor; centered 16-multiple crop and direct full-image inference."
        ),
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="Evaluate only these benchmark names (case-insensitive), e.g. bsd68 SOTS-Outdoor Rain100L.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Deterministically sample this fraction from each selected benchmark (0 < ratio <= 1).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument("--y-channel", action="store_true", help="Use luminance instead of RGB metrics.")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _load_model(args: argparse.Namespace, device: torch.device) -> ClearAIR:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    valid_fields = {field.name for field in fields(ClearAIRConfig)}
    cfg_kwargs = {key: value for key, value in saved_cfg.items() if key in valid_fields}

    if args.dummy_aux is not None:
        cfg_kwargs["dummy_auxiliaries"] = args.dummy_aux
    else:
        cfg_kwargs.setdefault("dummy_auxiliaries", True)
    if args.aux_device is not None:
        cfg_kwargs["auxiliary_device"] = args.aux_device
    elif not cfg_kwargs["dummy_auxiliaries"]:
        cfg_kwargs["auxiliary_device"] = str(device)
    for name in ("deqa_model_path", "sam2_model_path", "sam2_config", "daclip_checkpoint_path"):
        arg_name = name.removesuffix("_path")
        value = getattr(args, arg_name, None)
        if value is not None:
            cfg_kwargs[name] = value
    if args.deqa_4bit is not None:
        cfg_kwargs["deqa_load_in_4bit"] = args.deqa_4bit

    model = ClearAIR(ClearAIRConfig(**cfg_kwargs)).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.inference_mode()
def infer_tiled(model: ClearAIR, image: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    """Run tiled inference and average overlapping predictions."""
    if tile_size == 0:
        return model(image)
    if tile_size <= 0:
        raise ValueError("tile-size must be non-negative")
    if not 0 <= overlap < tile_size:
        raise ValueError("tile-overlap must satisfy 0 <= overlap < tile-size")
    _, _, height, width = image.shape
    stride = tile_size - overlap
    output = torch.zeros_like(image)
    weights = torch.zeros_like(image)
    for top in _positions(height, tile_size, stride):
        for left in _positions(width, tile_size, stride):
            tile = image[..., top:min(top + tile_size, height), left:min(left + tile_size, width)]
            tile_h, tile_w = tile.shape[-2:]
            if tile_h != tile_size or tile_w != tile_size:
                tile = F.pad(tile, (0, tile_size - tile_w, 0, tile_size - tile_h), mode="replicate")
            restored = model(tile)[..., :tile_h, :tile_w]
            output[..., top:top + tile_h, left:left + tile_w] += restored
            weights[..., top:top + tile_h, left:left + tile_w] += 1
    return output / weights.clamp_min(1)


def _metric_arrays(pred: torch.Tensor, target: torch.Tensor, crop_border: int, y_channel: bool):
    if crop_border:
        if pred.shape[-2] <= 2 * crop_border or pred.shape[-1] <= 2 * crop_border:
            raise ValueError("crop-border is larger than the evaluated image")
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    if y_channel:
        # ITU-R BT.601, the convention used by common restoration metrics.
        weights = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        pred = (pred * weights).sum(dim=1, keepdim=True)
        target = (target * weights).sum(dim=1, keepdim=True)
    return pred.clamp(0, 1), target.clamp(0, 1)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - depends on optional eval extra
        raise RuntimeError("SSIM requires scikit-image; install `pip install -e '.[evaluation]'`.") from exc
    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if pred_np.shape[2] == 1:
        pred_np, target_np = pred_np[..., 0], target_np[..., 0]
    return float(structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=-1 if pred_np.ndim == 3 else None))


def _evaluate_group(
    model: ClearAIR,
    dataset: AiOIRDataset,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    crop_border: int,
    y_channel: bool,
    sample_ratio: float,
    max_samples: int | None,
) -> list[dict]:
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample-ratio must satisfy 0 < ratio <= 1")
    sample_count = max(1, math.ceil(len(dataset) * sample_ratio))
    if max_samples is not None:
        sample_count = min(sample_count, max_samples)
    if sample_count <= 0:
        return []
    if sample_count == len(dataset):
        indices = range(len(dataset))
    else:
        # Evenly-spaced indices make the lightweight validation deterministic
        # and cover the benchmark rather than taking a filename-ordered prefix.
        indices = [index * len(dataset) // sample_count for index in range(sample_count)]

    rows: list[dict] = []
    for position, index in enumerate(indices, start=1):
        sample = dataset[index]
        lq = sample["lq"].unsqueeze(0).to(device)
        gt = sample["gt"].unsqueeze(0).to(device)
        pred = infer_tiled(model, lq, tile_size, tile_overlap)
        pred, gt = _metric_arrays(pred, gt, crop_border, y_channel)
        rows.append(
            {
                "dataset": sample["dataset"],
                "degradation": sample["deg"],
                "psnr": psnr(pred, gt),
                "ssim": ssim(pred, gt),
            }
        )
        if position % 25 == 0 or position == sample_count:
            print(f"[eval] {sample['dataset']}: {position}/{sample_count} (from {len(dataset)})")
    return rows


def _summarize(rows: Iterable[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    result = {}
    for name, values in sorted(grouped.items()):
        result[name] = {
            "count": len(values),
            "psnr": sum(row["psnr"] for row in values) / len(values),
            "ssim": sum(row["ssim"] for row in values) / len(values),
        }
    if rows := [row for values in grouped.values() for row in values]:
        result["Average"] = {
            "count": len(rows),
            "psnr": sum(row["psnr"] for row in rows) / len(rows),
            "ssim": sum(row["ssim"] for row in rows) / len(rows),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.adair_protocol:
        if args.sample_ratio != 1.0 or args.max_samples is not None:
            raise ValueError("AdaIR protocol evaluates every image; omit sample-ratio and max-samples.")
        if args.crop_border or args.y_channel:
            raise ValueError("AdaIR protocol uses RGB metrics without border cropping.")
        if set(args.noise_sigmas) != {15, 25, 50}:
            raise ValueError("AdaIR protocol requires noise sigmas 15, 25, and 50.")
        args.dataset_names = ["bsd68", "SOTS-Outdoor", "Rain100L"]
        args.seed = 0
        args.tile_size = 0
        args.tile_overlap = 0
        np.random.seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _load_model(args, device)
    all_rows: list[dict] = []
    requested_benchmarks = (
        {name.casefold() for name in args.dataset_names}
        if args.dataset_names is not None
        else None
    )

    for degradation in args.degradations:
        sigmas = args.noise_sigmas if degradation.lower() == "denoise" else [None]
        for sigma in sigmas:
            dataset = AiOIRDataset(
                root=args.data_root,
                degradations=[degradation],
                train=False,
                paper_layout=args.paper_layout,
                noise_sigmas=args.noise_sigmas if sigma is None else [sigma],
                noise_seed=args.seed,
                adair_test_protocol=args.adair_protocol,
            )
            if requested_benchmarks is not None:
                dataset.samples = [
                    sample for sample in dataset.samples
                    if sample.dataset.casefold() in requested_benchmarks
                ]
                if not dataset.samples:
                    continue
            rows = _evaluate_group(
                model, dataset, device, args.tile_size, args.tile_overlap,
                args.crop_border, args.y_channel, args.sample_ratio, args.max_samples,
            )
            if sigma is not None:
                for row in rows:
                    row["sigma"] = sigma
                    row["dataset"] = f"{row['dataset']}_sigma{sigma}"
            all_rows.extend(rows)

    summary = _summarize(all_rows)
    for name, values in summary.items():
        print(f"{name:>20s}  n={values['count']:>4d}  PSNR={values['psnr']:.3f}  SSIM={values['ssim']:.4f}")
    payload = {"summary": summary, "samples": all_rows, "args": vars(args)}
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[save] {path}")


if __name__ == "__main__":
    main()
