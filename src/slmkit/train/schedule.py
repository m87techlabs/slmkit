"""Learning-rate schedule: linear warmup, then cosine decay, both measured in tokens.

    lr
    │      ╭──╮
    │     ╱    ╲___
    │    ╱         ╲___
    │   ╱              ╲____
    │  ╱                    ╲______  ← floor: lr × min_lr_ratio
    └─┴────────────────────────────── tokens
      warmup                 max_tokens

Why warmup: at step 0 the weights are random and AdamW's running estimates of gradient size
are empty, so its first updates are unreliable. A full-size learning rate then can throw the
model somewhere it never recovers from. Ramping up over the first ~1-2% of training avoids that.

Why cosine decay: large steps early make fast progress; small steps late let the model settle
into a good minimum instead of bouncing around it. Cosine is the smooth, standard way to go
from one to the other. The floor (10% of peak here) keeps learning from stopping entirely.

Why tokens rather than steps: changing the batch size changes how many steps a run has, but
not how much data it sees. A schedule in tokens means the same experiment with a different
batch size still warms up over the same amount of data. (DESIGN 6.6.)

The schedule is a pure function of tokens seen, so it needs no state of its own: restoring
`tokens_seen` from a checkpoint restores the schedule exactly. The trainer passes the token
count *after* the step being taken, so the first step already has a small non-zero rate
(lr x tokens_per_step / warmup_tokens), as in nanoGPT.
"""

from __future__ import annotations

import math


def lr_at(
    tokens_seen: int, *, lr: float, warmup_tokens: int, max_tokens: int, min_lr_ratio: float
) -> float:
    min_lr = lr * min_lr_ratio
    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        return lr * tokens_seen / warmup_tokens
    if tokens_seen >= max_tokens:
        return min_lr
    progress = (tokens_seen - warmup_tokens) / max(1, max_tokens - warmup_tokens)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
