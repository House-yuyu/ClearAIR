"""Minimal image inference entry point for ClearAIR checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from clearair import ClearAIR, ClearAIRConfig


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore an image or image directory with ClearAIR.")
    parser.add_argument("--checkpoint", required=True, help="A checkpoint saved by clearair-train.")
    parser.add_argument("--input", required=True, help="Input image or directory of input images.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Torch device; falls back to CPU when CUDA is unavailable.")
    parser.add_argument(
        "--dummy-aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the lightweight auxiliary stand-ins included in this repository (default: true).",
    )
    parser.add_argument("--aux-device", default=None)
    parser.add_argument("--deqa-model", default="pretrained/DeQA-Score-Mix3")
    parser.add_argument("--deqa-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam2-model", default="pretrained/sam2/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-points-per-batch", type=int, default=16)
    parser.add_argument("--sam2-points-per-crop", type=int, default=4)
    parser.add_argument("--sam2-pred-iou-thresh", type=float, default=0.70)
    parser.add_argument("--daclip-checkpoint", default="pretrained/daclip/daclip_ViT-B-32.pt")
    return parser.parse_args()


def input_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad BCHW image tensors for the three encoder down-sampling stages."""
    _, _, height, width = image.shape
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate"), (height, width)


def load_model(args: argparse.Namespace, device: torch.device) -> ClearAIR:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = ClearAIR(
        ClearAIRConfig(
            dummy_auxiliaries=args.dummy_aux,
            auxiliary_device=args.aux_device or str(device),
            deqa_model_path=args.deqa_model,
            deqa_load_in_4bit=args.deqa_4bit,
            sam2_model_path=args.sam2_model,
            sam2_config=args.sam2_config,
            sam2_points_per_batch=args.sam2_points_per_batch,
            sam2_points_per_crop=args.sam2_points_per_crop,
            sam2_pred_iou_thresh=args.sam2_pred_iou_thresh,
            daclip_checkpoint_path=args.daclip_checkpoint,
        )
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    paths = input_images(Path(args.input))
    if not paths:
        raise RuntimeError("No supported images were found under --input.")

    for path in paths:
        image = TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        image, (height, width) = pad_to_multiple(image)
        with torch.inference_mode():
            restored = model(image).clamp(0, 1)[..., :height, :width]

        target = output_dir / f"{path.stem}.png"
        TF.to_pil_image(restored.squeeze(0).cpu()).save(target)
        print(f"[saved] {target}")


if __name__ == "__main__":
    main()
