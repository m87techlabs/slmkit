"""Weight initialisation: where training starts from.

A network starts as random numbers, and *which* random numbers matters more than it looks.
Too large and activations grow layer by layer until they overflow; too small and gradients
shrink to nothing before they reach the early layers. Two rules, both from GPT-2 (and used by
nanoGPT, the reference M1 must match):

1. Every weight matrix and the embedding start as Normal(0, 0.02). Small enough that the first
   predictions are nearly uniform (initial loss ~ ln(vocab_size), which the tests check), large
   enough that every unit starts out different from its neighbours.

2. The two projections that write *into* the residual stream (attention `o_proj`, MLP
   `down_proj`) use std 0.02 / sqrt(2 * n_layers). The residual stream is a running sum of
   2 * n_layers contributions; scaling each one down keeps the sum's variance roughly
   constant with depth instead of growing with it.

RMSNorm gains start at 1 (set in the module itself), so normalisation begins as a pure rescale.

Rejected: PyTorch's default `nn.Linear` init (Kaiming-uniform), which is tuned for ReLU
networks and gives a noticeably worse start for transformers; and HF Llama's init, which is
0.02 everywhere with no depth scaling.
"""

from __future__ import annotations

import math

from torch import nn

BASE_STD = 0.02
RESIDUAL_PROJECTIONS = ("o_proj.weight", "down_proj.weight")


def init_weights(model: nn.Module, n_layers: int) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=BASE_STD)
    residual_std = BASE_STD / math.sqrt(2 * n_layers)
    for name, param in model.named_parameters():
        if name.endswith(RESIDUAL_PROJECTIONS):
            nn.init.normal_(param, mean=0.0, std=residual_std)
