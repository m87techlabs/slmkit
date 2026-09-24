"""Seeding and random-number-generator state.

Training draws random numbers from four independent generators: Python's `random`, NumPy's
global generator, PyTorch's CPU generator, and PyTorch's CUDA generator (dropout on the GPU).
`seed_everything` sets all four at the start of a fresh run. A checkpoint saves all four, so
a resumed run continues the *same* random sequence rather than a new one; together with the
sampler's own state that is what makes resume equivalent to never having stopped.

The goal is statistical reproducibility, not bit-exact determinism: GPU kernels may sum in a
different order run to run, so two GPU runs with the same seed agree closely but not exactly.
The resume-equivalence test runs on the CPU, where they do agree exactly.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
