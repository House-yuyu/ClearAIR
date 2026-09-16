

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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
    p.add_argument(
        "--paper-layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read the paper's AiOIR/{train,test} directory layout in-place.",
    )
    p.add_argument(
        "--noise-sigmas",
        nargs="+",
        type=int,
        default=[15, 25, 50],
        help="Gaussian noise levels used for clean-only denoising images.",
    )
    p.add_argument(
        "--sampling-profile",
        choices=("natural", "adair"),
        default="natural",
        help="Task sampling profile for paper-layout training; 'adair' repeats small datasets as AdaIR does.",
    )
    p.add_argument(
        "--source-subset-modulo",
        type=int,
        default=1,
        help="Keep source images whose stable sorted index modulo this value equals --source-subset-index.",
    )
    p.add_argument(
        "--source-subset-index",
        type=int,
        default=0,
        help="Selected residue class for --source-subset-modulo.",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-iters", type=int, default=300_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Autocast precision for the restoration path; fp32 preserves the original behavior.",
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer update (effective batch = batch-size * this value).",
    )
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
def cycle(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def load_optimizer_state_compatible(
    optimizer: torch.optim.Optimizer,
    saved_state: dict,
    model: nn.Module,
    checkpoint_model: dict,
) -> list[str]:
    """Restore Adam state by parameter name after an additive model change.

    A positional optimizer state cannot be loaded when a compatibility fix
    adds a new trainable layer.  State is copied only for parameter names that
    existed in the checkpoint; newly introduced parameters deliberately start
    with fresh optimizer moments.
    """
    if len(saved_state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer has a different number of parameter groups")

    old_names = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad and name in checkpoint_model]
    current_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    saved_ids = [parameter_id for group in saved_state["param_groups"] for parameter_id in group["params"]]
    current = optimizer.state_dict()
    current_ids = [parameter_id for group in current["param_groups"] for parameter_id in group["params"]]
    if len(old_names) != len(saved_ids):
        raise ValueError("could not align checkpoint optimizer parameters with checkpoint model")
    if len(current_names) != len(current_ids):
        raise ValueError("could not align current optimizer parameters with current model")

    old_state_by_name = dict(zip(old_names, saved_ids))
    restored = {"state": {}, "param_groups": current["param_groups"]}
    for name, parameter_id in zip(current_names, current_ids):
        old_id = old_state_by_name.get(name)
        if old_id is not None and old_id in saved_state["state"]:
            restored["state"][parameter_id] = saved_state["state"][old_id]
    for new_group, old_group in zip(restored["param_groups"], saved_state["param_groups"]):
        for key, value in old_group.items():
            if key != "params":
                new_group[key] = value
    optimizer.load_state_dict(restored)
    return [name for name in current_names if name not in old_state_by_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if local_rank < 0:
        local_rank = 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    is_main = rank == 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.precision != "fp32":
        raise ValueError("--precision bf16/fp16 requires a CUDA device.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1.")
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    # ---- data --------------------------------------------------------
    train_set = AiOIRDataset(
        root=args.data_root,
        degradations=args.degradations,
        patch_size=args.patch_size,
        train=True,
        paper_layout=args.paper_layout,
        noise_sigmas=args.noise_sigmas,
        sampling_profile=args.sampling_profile,
        source_subset_modulo=args.source_subset_modulo,
        source_subset_index=args.source_subset_index,
    )
    if is_main:
        counts = dict(sorted(Counter(sample.degradation for sample in train_set.samples).items()))
        print(
            f"[data] {len(train_set)} training pairs across {args.degradations}; "
            f"sampling={args.sampling_profile}; source-subset="
            f"{args.source_subset_index}/{args.source_subset_modulo}; counts={counts}"
        )
    sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_iter = cycle(train_loader, sampler=sampler)

    # ---- model -------------------------------------------------------
    if distributed:
        # Each DDP rank owns one complete copy of the frozen auxiliaries.
        # Omitting --aux-device is intentional: map them to the rank-local GPU.
        auxiliary_device = f"cuda:{local_rank}"
    else:
        auxiliary_device = args.aux_device or args.device
    cfg = ClearAIRConfig(
        dummy_auxiliaries=args.dummy_aux,
        auxiliary_device=auxiliary_device,
        deqa_model_path=args.deqa_model,
        deqa_load_in_4bit=args.deqa_4bit,
        sam2_model_path=args.sam2_model,
        sam2_config=args.sam2_config,
        sam2_points_per_batch=args.sam2_points_per_batch,
        sam2_points_per_crop=args.sam2_points_per_crop,
        sam2_pred_iou_thresh=args.sam2_pred_iou_thresh,
        daclip_checkpoint_path=args.daclip_checkpoint,
    )
    raw_model = ClearAIR(cfg).to(device)

    loss_fn = ClearAIRLoss(ClearAIRLossConfig()).to(device)

    # ---- optim -------------------------------------------------------
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optim, T_max=args.total_iters, eta_min=1e-6)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        try:
            optim.load_state_dict(ckpt["optim"])
        except ValueError:
            new_parameters = load_optimizer_state_compatible(
                optim, ckpt["optim"], raw_model, ckpt["model"]
            )
            if is_main:
                print(
                    "[resume] restored optimizer state by parameter name; "
                    f"fresh state for {len(new_parameters)} new parameters"
                )
        scheduler.load_state_dict(ckpt["sched"])
        start_iter = ckpt["iter"]
        if is_main:
            print(f"[resume] loaded {args.resume} at iter {start_iter}")

    model = (
        DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if distributed
        else raw_model
    )


    if distributed:
        rank_seed = args.seed + rank
        random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)

    # ---- training loop ----------------------------------------------
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "l1": 0.0, "l_inter": 0.0, "psnr": 0.0, "n": 0}

    def save_checkpoint(path: Path, iteration: int) -> None:
        """Persist only the trainable model and optimizer state."""
        torch.save(
            {
                "iter": iteration,
                "model": raw_model.state_dict(),
                "optim": optim.state_dict(),
                "sched": scheduler.state_dict(),
                "cfg": cfg.__dict__,
            },
            path,
        )

    if is_main:
        effective_batch = args.batch_size * args.grad_accum_steps * world_size
        print(
            f"[optim] micro-batch/rank={args.batch_size}, accumulation={args.grad_accum_steps}, "
            f"world-size={world_size}, effective batch={effective_batch}"
        )

    optim.zero_grad(set_to_none=True)
    for it in range(start_iter, args.total_iters):
        for _ in range(args.grad_accum_steps):
            batch = next(train_iter)
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            autocast_enabled = autocast_dtype is not None
            sync_context = (
                model.no_sync()
                if distributed and args.grad_accum_steps > 1 and _ < args.grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    pred = model(lq)
                    loss, parts = loss_fn(pred, gt)
                # Keep the optimizer update equivalent to a mean loss over
                # the effective batch while retaining one micro-batch on GPU.
                (loss / args.grad_accum_steps).backward()

            running["loss"] += loss.item()
            running["l1"] += parts["l1"].item()
            running["l_inter"] += parts["l_inter"].item()
            running["psnr"] += psnr(pred.detach().clamp(0, 1), gt)
            running["n"] += 1

        grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
        finite_flag = torch.tensor(
            int(torch.isfinite(grad_norm).item()), device=device, dtype=torch.int32
        )
        if distributed:
            # All ranks must make the same stop/continue decision; otherwise
            # the next DDP collective would hang after a rank detects NaN.
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        if finite_flag.item() == 0:
            if is_main:
                # ``optim.step`` has not run, so this checkpoint captures the
                # last valid optimization state and can be used to reproduce
                # the failing batch without ever persisting NaN parameters.
                diagnostic_path = save_dir / f"clearair_pre_nonfinite_iter{it + 1}.pth"
                save_checkpoint(diagnostic_path, it)
                print(f"[diagnostic-save] {diagnostic_path}", flush=True)
            print(
                f"[nonfinite][rank {rank}] iter={it + 1} "
                f"loss={loss.detach().item()} grad_norm={grad_norm.item()}",
                flush=True,
            )
            if distributed:
                dist.destroy_process_group()
            raise FloatingPointError("non-finite gradient detected; optimizer step was skipped")
        optim.step()
        scheduler.step()

        if (it + 1) % args.log_every == 0:
            if distributed:
                stats = torch.tensor(
                    [
                        running["loss"],
                        running["l1"],
                        running["l_inter"],
                        running["psnr"],
                        float(running["n"]),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats /= world_size
                n = stats[4].item()
                loss_avg = stats[0].item() / n
                l1_avg = stats[1].item() / n
                inter_avg = stats[2].item() / n
                psnr_avg = stats[3].item() / n
            else:
                n = running["n"]
                loss_avg = running["loss"] / n
                l1_avg = running["l1"] / n
                inter_avg = running["l_inter"] / n
                psnr_avg = running["psnr"] / n
            elapsed = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]
            if is_main:
                print(
                    f"[iter {it + 1:>7d}/{args.total_iters}] "
                    f"loss={loss_avg:.4f}  "
                    f"l1={l1_avg:.4f}  "
                    f"l_inter={inter_avg:.4f}  "
                    f"psnr={psnr_avg:.2f}  "
                    f"lr={lr_now:.2e}  "
                    f"({n / elapsed:.2f} rank-steps/s)"
                )
            running = {"loss": 0.0, "l1": 0.0, "l_inter": 0.0, "psnr": 0.0, "n": 0}
            t0 = time.time()

        if (it + 1) % args.save_every == 0 or (it + 1) == args.total_iters:
            if is_main:
                ckpt_path = save_dir / f"clearair_iter{it + 1}.pth"
                save_checkpoint(ckpt_path, it + 1)
                print(f"[save] {ckpt_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
