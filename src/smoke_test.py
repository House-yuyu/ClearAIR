"""Quick forward / loss / backward sanity check for ClearAIR."""

import torch

from clearair import ClearAIR, ClearAIRConfig, ClearAIRLoss, ClearAIRLossConfig


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ClearAIRConfig(dummy_auxiliaries=True)
    model = ClearAIR(cfg).to(device)
    loss_fn = ClearAIRLoss(ClearAIRLossConfig()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f} M")

    # 256x256 must divide cleanly through 3 down-samples (-> 32x32 at bottleneck)
    x = torch.rand(2, 3, 256, 256, device=device)
    y = torch.rand(2, 3, 256, 256, device=device)

    out = model(x)
    assert out.shape == x.shape, f"unexpected output shape: {out.shape}"
    print(f"forward ok | out shape: {tuple(out.shape)} | range: [{out.min():.3f}, {out.max():.3f}]")

    loss, parts = loss_fn(out, y)
    print(f"loss = {loss.item():.4f}  (l1={parts['l1'].item():.4f}, l_inter={parts['l_inter'].item():.4f})")

    loss.backward()
    grad_ok = all(
        (p.grad is not None and torch.isfinite(p.grad).all().item())
        for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    print(f"backward ok: {grad_ok}")


if __name__ == "__main__":
    main()
