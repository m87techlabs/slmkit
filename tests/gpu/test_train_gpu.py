"""GPU smoke test for the trainer (DESIGN 7): 50 steps on `nano` with compile and bf16."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from slmkit.config import load_experiment
from slmkit.train import checkpoint
from slmkit.train.trainer import Trainer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def test_fifty_compiled_bf16_steps(toy_repo: Path, slm_home: Path) -> None:
    exp = load_experiment(
        "toy/base",
        [
            "run.tracker=none",
            "train.compile=true",
            "train.max_tokens=1600",  # 50 steps of 4 x 8
            "train.warmup_tokens=160",
            "train.log_every_steps=10",
            "train.eval_every_steps=50",
            "train.eval_iters=2",
            "train.eval_samples=1",
            "train.sample_tokens=8",
        ],
    )
    log: list[str] = []
    trainer = Trainer(exp, device="cuda", log=log.append)
    assert trainer.run() == "complete"
    print("\n".join(line for line in log if "tok/s" in line or "measured" in line))

    records = [json.loads(x) for x in (trainer.run_dir / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in records if r["kind"] == "train"]
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0] - 0.5
    assert any("measured" in line and "tokens/s" in line for line in log)
    assert checkpoint.latest(trainer.run_dir) is not None
