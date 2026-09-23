"""Random-window batch sampler over a packed split, with resumable state.

Each batch is `batch_size` windows of `block_size + 1` tokens starting at random offsets:

    window  = t0 t1 t2 ... tN
    x       = t0 t1 ... t(N-1)     what the model sees
    y       = t1 t2 ...  tN        what it must predict at each position (x shifted by one)

The shift is the whole of language-model training in one line, and an off-by-one here is the
classic silent bug: the model learns to copy its input and loss looks great. `test_sampler.py`
pins it down.

Why random offsets rather than epochs over fixed chunks: with a tiny corpus (Shakespeare is
~1M tokens) random windows give far more distinct training examples than fixed boundaries.
The RNG state is part of every checkpoint, so a resumed run draws exactly the batches an
uninterrupted run would have (DESIGN 7, resume equivalence).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomWindowSampler:
    def __init__(self, data: np.ndarray, block_size: int, batch_size: int, seed: int) -> None:
        if len(data) < block_size + 1:
            raise ValueError(
                f"split has {len(data)} tokens, fewer than block_size + 1 = {block_size + 1}"
            )
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y), each int64 of shape (batch_size, block_size)."""
        starts = self.rng.integers(0, len(self.data) - self.block_size, size=self.batch_size)
        windows = np.stack(
            [np.asarray(self.data[s : s + self.block_size + 1], dtype=np.int64) for s in starts]
        )
        return windows[:, :-1], windows[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
