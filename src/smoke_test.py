"""Quick forward / loss / backward sanity check for ClearAIR."""

from __future__ import annotations

import argparse

import torch

from clearair import ClearAIR, ClearAIRConfig, ClearAIRLoss, ClearAIRLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ClearAIR forward/backward sanity check.")
    parser.add_argument("--quick", action="store_true", help="Use a compact 32x32 model suitable for CPU checks.")
    parser.add_argument("--device", default="cuda", help="Torch device; falls back to CPU when CUDA is unavailable.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    cfg = ClearAIRConfig(
        base_dim=8 if args.quick else 48,
        num_blocks=[1, 1, 1, 1] if args.quick else [3, 5, 6, 8],
        num_heads=[1, 1, 1, 1] if args.quick else [1, 2, 4, 8],
        dummy_auxiliaries=True,
    )
    model = ClearAIR(cfg).to(device)
    loss_fn = ClearAIRLoss(ClearAIRLossConfig()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f} M")

    # Inputs must divide cleanly through three down-sampling stages.
    batch_size, patch_size = (1, 32) if args.quick else (2, 256)
    x = torch.rand(batch_size, 3, patch_size, patch_size, device=device)
    y = torch.rand(batch_size, 3, patch_size, patch_size, device=device)

    out = model(x)
    assert out.shape == x.shape, f"unexpected output shape: {out.shape}"
    print(f"forward ok | out shape: {tuple(out.shape)} | range: [{out.min():.3f}, {out.max():.3f}]")

    loss, parts = loss_fn(out, y)
    print(f"loss = {loss.item():.4f}  (l1={parts['l1'].item():.4f}, l_inter={parts['l_inter'].item():.4f})")

    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable)
    print(f"backward ok: {grad_ok}")


if __name__ == "__main__":
    main()
