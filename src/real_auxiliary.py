"""Frozen real auxiliary backends used by ClearAIR.

The heavy third-party models are intentionally stored outside the PyTorch
module tree. This prevents ``ClearAIR.to(device)`` and checkpoint serialization
from moving or duplicating DeQA/SAM2/DA-CLIP weights. Only the restoration
network and its small trainable adapters are saved in ClearAIR checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF


IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN = "<|image|>"


def _require_path(path: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    return str(resolved)


def _device(device: str) -> torch.device:
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA auxiliary device requested but CUDA is unavailable: {device}")
    return parsed


def _to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
    return [TF.to_pil_image(image.detach().float().clamp(0, 1).cpu()) for image in images]


def _tokenizer_image_token(prompt: str, tokenizer: Any) -> torch.Tensor:
    chunks = [tokenizer(chunk).input_ids if chunk else [] for chunk in prompt.split(IMAGE_TOKEN)]
    input_ids: list[int] = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    interleaved: list[list[int]] = []
    for index, chunk in enumerate(chunks):
        if index > 0:
            interleaved.append([IMAGE_TOKEN_INDEX] * (offset + 1))
        interleaved.append(chunk)
    for piece in interleaved:
        input_ids.extend(piece[offset:])
    return torch.tensor(input_ids, dtype=torch.long)


def _expand_to_square(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), background)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


class FrozenExternalBackend(nn.Module):
    """A parameter-free facade around a large model owned outside nn.Module."""

    def _set_external(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class DeQAFeatureExtractor(FrozenExternalBackend):
    """Extract DeQA's final prompt state immediately before quality prediction."""

    def __init__(self, model_path: str, device: str, load_in_4bit: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

        model_path = _require_path(model_path, "DeQA model")
        aux_device = _device(device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(config, "mlp_bias"):
            config.mlp_bias = False

        kwargs: dict[str, Any] = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "device_map": {"": str(aux_device)},
            "torch_dtype": torch.float16 if aux_device.type == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            if aux_device.type != "cuda":
                raise RuntimeError("DeQA 4-bit loading requires a CUDA auxiliary device.")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        self._set_external("_model", model)
        self.device = aux_device

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        model = self._model
        processor = model.image_processor
        background = tuple(int(channel * 255) for channel in processor.image_mean)
        pil_images = [_expand_to_square(image, background) for image in _to_pil_batch(images)]
        input_ids = _tokenizer_image_token(prompt, model.tokenizer).unsqueeze(0).to(self.device)
        pixel_values = processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]
        pixel_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pixel_values = pixel_values.to(device=self.device, dtype=pixel_dtype)

        outputs = model(
            input_ids=input_ids.repeat(pixel_values.shape[0], 1),
            images=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # The final prompt position is the state used to predict the quality
        # adjective (excellent/good/fair/poor/bad), matching DeQA.score().
        return outputs.hidden_states[-1][:, -1, :].float()


class SAM2MaskExtractor(FrozenExternalBackend):
    """Generate ranked semantic masks with the official SAM2 HF pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        model_config: str,
        device: str,
        points_per_batch: int = 16,
        points_per_crop: int = 4,
        pred_iou_thresh: float = 0.70,
    ):
        super().__init__()
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint_path = _require_path(checkpoint_path, "SAM2 checkpoint")
        aux_device = _device(device)
        model = build_sam2(
            config_file=model_config,
            ckpt_path=checkpoint_path,
            device=str(aux_device),
            mode="eval",
        )
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_crop,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            output_mode="binary_mask",
        )
        self._set_external("_generator", generator)
        self.device = aux_device
        self.points_per_batch = points_per_batch
        self.points_per_crop = points_per_crop
        self.pred_iou_thresh = pred_iou_thresh

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, num_masks: int) -> torch.Tensor:
        batch, _, height, width = images.shape
        output = torch.zeros(batch, num_masks, height, width, dtype=images.dtype, device=images.device)
        for batch_index, image in enumerate(_to_pil_batch(images)):
            records = self._generator.generate(np.array(image, copy=True))
            records = sorted(records, key=lambda record: record.get("predicted_iou", 0.0), reverse=True)
            if not records:
                output[batch_index, 0] = 1
                continue
            for mask_index, record in enumerate(records[:num_masks]):
                mask = record["segmentation"]
                mask_tensor = torch.as_tensor(mask, dtype=images.dtype, device=images.device)
                if mask_tensor.ndim > 2:
                    mask_tensor = mask_tensor.squeeze()
                if mask_tensor.shape != (height, width):
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor[None, None], size=(height, width), mode="nearest"
                    )[0, 0]
                output[batch_index, mask_index] = mask_tensor
        return output


class DAClipFeatureExtractor(FrozenExternalBackend):
    """Return frozen DA-CLIP content and degradation embeddings."""

    def __init__(self, checkpoint_path: str, device: str):
        super().__init__()
        import open_clip

        if "daclip_ViT-B-32" not in open_clip.list_models():
            raise RuntimeError(
                "The installed open_clip is not the official DA-CLIP fork. "
                "Run scripts/setup_real_aux.sh in the clearair environment."
            )
        checkpoint_path = _require_path(checkpoint_path, "DA-CLIP checkpoint")
        aux_device = _device(device)
        precision = "fp16" if aux_device.type == "cuda" else "fp32"
        model, preprocess = open_clip.create_model_from_pretrained(
            "daclip_ViT-B-32",
            pretrained=checkpoint_path,
            precision=precision,
            device=aux_device,
        )
        model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self._set_external("_model", model)
        self._set_external("_preprocess", preprocess)
        self.device = aux_device

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.stack([self._preprocess(image) for image in _to_pil_batch(images)]).to(self.device)
        model_dtype = self._model.visual.conv1.weight.dtype
        content, degradation = self._model.encode_image(
            inputs.to(dtype=model_dtype), control=True, normalize=True
        )
        return content.float(), degradation.float()


@dataclass
class RealAuxiliaryBundle:
    deqa: DeQAFeatureExtractor
    sam2: SAM2MaskExtractor
    daclip: DAClipFeatureExtractor


def build_real_auxiliaries(config: Any) -> RealAuxiliaryBundle:
    """Load all three frozen paper auxiliaries from local paths."""
    return RealAuxiliaryBundle(
        deqa=DeQAFeatureExtractor(
            config.deqa_model_path,
            config.auxiliary_device,
            load_in_4bit=config.deqa_load_in_4bit,
        ),
        sam2=SAM2MaskExtractor(
            config.sam2_model_path,
            config.sam2_config,
            config.auxiliary_device,
            points_per_batch=config.sam2_points_per_batch,
            points_per_crop=config.sam2_points_per_crop,
            pred_iou_thresh=config.sam2_pred_iou_thresh,
        ),
        daclip=DAClipFeatureExtractor(config.daclip_checkpoint_path, config.auxiliary_device),
    )
