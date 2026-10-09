"""The optimizer: AdamW, with weight decay on matrices only.

AdamW keeps two running averages per parameter (the gradient's mean and its squared size) and
scales each update by them, so every parameter gets a step size suited to its own gradients.
The "W" is decoupled weight decay: each step shrinks weights slightly towards zero, which
discourages the model from relying on a few very large weights.

Decay applies to 2-D parameters (every weight matrix, including the tied embedding) and not to
1-D ones (the RMSNorm gains). Shrinking a norm's gain towards zero would fight its purpose of
setting the scale, and those vectors hold a negligible share of the parameters. Same split as
nanoGPT and most LLM training code.
"""

from __future__ import annotations

import torch
from torch import nn

from slmkit.config.schema import TrainConfig


def build_optimizer(
    model: nn.Module, cfg: TrainConfig, device: torch.device
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw8bit":
        # Optional and unverified until M3's A/B test against fp32 AdamW (CONTRIBUTING.md).
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("optimizer adamw8bit needs `uv sync --extra optim8bit`") from exc
        opt: torch.optim.Optimizer = bnb.optim.AdamW8bit(  # type: ignore[attr-defined,no-untyped-call,unused-ignore]
            groups, lr=cfg.lr, betas=cfg.betas
        )
        return opt
    # fused=True runs the whole update as one kernel per parameter group on the GPU.
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=device.type == "cuda")
