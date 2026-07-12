"""
Training entry point for ClearAIR.

Defaults follow the paper:
    optimizer       AdamW (beta1=0.9, beta2=0.999)
    learning rate   2e-4
    batch size      4
    iterations      300_000
    patch size      256x256
    horizontal+vertical flips
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from clearair import (
    AiOIRDataset,
    ClearAIR,
    ClearAIRConfig,
    ClearAIRLoss,
    ClearAIRLossConfig,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument(
        "--degradations",
        nargs="+",
        default=["denoise", "dehaze", "derain"],
        help="Sub-directories under data-root to mix for training.",
    )
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-iters", type=int, default=300_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--dummy-aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use lightweight stand-ins for DeQA/SAM2/DA-CLIP (default: true).",
    )
    p.add_argument("--aux-device", default=None, help="Device for frozen auxiliaries; defaults to --device.")
    p.add_argument("--deqa-model", default="pretrained/DeQA-Score-Mix3")
    p.add_argument("--deqa-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sam2-model", default="pretrained/sam2/sam2.1_hiera_tiny.pt")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam2-points-per-batch", type=int, default=16)
    p.add_argument("--sam2-points-per-crop", type=int, default=4)
    p.add_argument("--sam2-pred-iou-thresh", type=float, default=0.70)
    p.add_argument("--daclip-checkpoint", default="pretrained/daclip/daclip_ViT-B-32.pt")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- data --------------------------------------------------------
    train_set = AiOIRDataset(
        root=args.data_root,
        degradations=args.degradations,
        patch_size=args.patch_size,
        train=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_iter = cycle(train_loader)

    # ---- model -------------------------------------------------------
    cfg = ClearAIRConfig(
        dummy_auxiliaries=args.dummy_aux,
        auxiliary_device=args.aux_device or args.device,
        deqa_model_path=args.deqa_model,
        deqa_load_in_4bit=args.deqa_4bit,
        sam2_model_path=args.sam2_model,
        sam2_config=args.sam2_config,
        sam2_points_per_batch=args.sam2_points_per_batch,
        sam2_points_per_crop=args.sam2_points_per_crop,
        sam2_pred_iou_thresh=args.sam2_pred_iou_thresh,
        daclip_checkpoint_path=args.daclip_checkpoint,
    )
    model = ClearAIR(cfg).to(device)

    loss_fn = ClearAIRLoss(ClearAIRLossConfig()).to(device)

    # ---- optim -------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optim, T_max=args.total_iters, eta_min=1e-6)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_iter = ckpt["iter"]
        print(f"[resume] loaded {args.resume} at iter {start_iter}")

    # ---- training loop ----------------------------------------------
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "l1": 0.0, "l_inter": 0.0, "psnr": 0.0, "n": 0}

    for it in range(start_iter, args.total_iters):
        batch = next(train_iter)
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = model(lq)
        loss, parts = loss_fn(pred, gt)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        scheduler.step()

        running["loss"] += loss.item()
        running["l1"] += parts["l1"].item()
        running["l_inter"] += parts["l_inter"].item()
        running["psnr"] += psnr(pred.detach().clamp(0, 1), gt)
        running["n"] += 1

        if (it + 1) % args.log_every == 0:
            n = running["n"]
            elapsed = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]
            print(
                f"[iter {it + 1:>7d}/{args.total_iters}] "
                f"loss={running['loss'] / n:.4f}  "
                f"l1={running['l1'] / n:.4f}  "
                f"l_inter={running['l_inter'] / n:.4f}  "
                f"psnr={running['psnr'] / n:.2f}  "
                f"lr={lr_now:.2e}  "
                f"({n / elapsed:.2f} it/s)"
            )
            running = {"loss": 0.0, "l1": 0.0, "l_inter": 0.0, "psnr": 0.0, "n": 0}
            t0 = time.time()

        if (it + 1) % args.save_every == 0 or (it + 1) == args.total_iters:
            ckpt_path = save_dir / f"clearair_iter{it + 1}.pth"
            torch.save(
                {
                    "iter": it + 1,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": scheduler.state_dict(),
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")


if __name__ == "__main__":
    main()
