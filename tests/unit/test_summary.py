"""`slm run` evaluating what it trained, and `slm runs summary` averaging over training seeds."""

from __future__ import annotations

import statistics
from pathlib import Path

from typer.testing import CliRunner

from slmkit.cli import app
from slmkit.eval.summary import summarize

# Small enough for the CPU: 2 steps of eval sampling instead of 3 seeds x 200 samples x 600 tokens.
FAST = ["run.tracker=none", "train.compile=false", "train.max_tokens=640",
        "train.warmup_tokens=64", "train.eval_every_steps=10", "train.eval_iters=2",
        "train.eval_samples=1", "train.sample_tokens=4",
        "eval.num_samples=4", "eval.max_new_tokens=8", "eval.seeds=[0,1]"]  # fmt: skip


def _run(seed: int) -> str:
    args = ["run", "toy/base", "--device", "cpu"]
    for s in [*FAST, f"run.seed={seed}"]:
        args += ["--set", s]
    out = CliRunner().invoke(app, args)
    assert out.exit_code == 0, out.output
    return out.output


def test_run_trains_then_evaluates_each_stage(toy_repo: Path) -> None:
    output = _run(1)
    # toy/base has no `sft:` section, so one stage: trained, then evaluated.
    assert output.count("report:") == 1 and "novelty" in output
    assert "already exists" in _run(1)  # re-running reuses the run and its report


def test_summary_averages_over_training_seeds(toy_repo: Path) -> None:
    _run(1)
    _run(2)
    (group,) = summarize(project="toy")
    assert (group.experiment, group.stage) == ("toy/base", "pretrain")
    assert sorted(group.seeds) == [1, 2] and group.evaluated == 2
    loss = [r["best val loss"] for r in group.per_run]
    assert abs(group.stats["best val loss"]["mean"] - statistics.fmean(loss)) < 1e-12
    assert abs(group.stats["best val loss"]["std"] - statistics.stdev(loss)) < 1e-12
    assert {"best val bpc", "novelty", "ended"} <= set(group.stats)


def test_summary_cli(toy_repo: Path) -> None:
    _run(1)
    out = CliRunner().invoke(app, ["runs", "summary"])
    assert out.exit_code == 0, out.output
    assert "toy/base" in out.output and "(1 run)" in out.output
