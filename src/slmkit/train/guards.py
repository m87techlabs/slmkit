"""Numerical guards: skip a step whose loss or gradients are not finite, abort if it keeps happening.

One NaN written into the weights is permanent: every later forward pass produces NaN and the
run is dead. Checking before `optimizer.step()` costs almost nothing and turns a silent
corruption into a logged, skipped step. A long streak of skips means something is actually
broken (learning rate too high, bad data, a numerics bug), so the run stops with a clear error
instead of burning GPU-hours on nothing. Debug order when it fires: CONTRIBUTING.md, "When loss
misbehaves".
"""

from __future__ import annotations

import math


class TrainingDiverged(RuntimeError):
    pass


class NonFiniteGuard:
    def __init__(self, max_consecutive: int) -> None:
        self.max_consecutive = max_consecutive
        self.consecutive = 0
        self.total_skipped = 0

    def check(self, loss: float, grad_norm: float) -> bool:
        """True if the step may be applied. Raises after too many bad steps in a row."""
        if math.isfinite(loss) and math.isfinite(grad_norm):
            self.consecutive = 0
            return True
        self.consecutive += 1
        self.total_skipped += 1
        if self.consecutive >= self.max_consecutive:
            raise TrainingDiverged(
                f"{self.consecutive} consecutive non-finite steps (loss={loss}, "
                f"grad_norm={grad_norm}). Try a lower lr or longer warmup; see CONTRIBUTING.md, "
                "'When loss misbehaves'."
            )
        return False
