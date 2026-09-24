"""The trainer end to end on the CPU, with the synthetic toy project.

The most important test in the suite is resume equivalence (DESIGN 7): on a machine that is
switched off most nights every real run is a resumed run, so stopping and resuming must be
indistinguishable from never having stopped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slmkit.config import load_experiment
from slmkit.train import checkpoint
from slmkit.train.run import list_runs, run_id
from slmkit.train.trainer import Trainer

STEPS = 20
TOKENS_PER_STEP = 4 * 8  # batch_size x block_size in the toy experiment
FAST = [
    "run.tracker=none",
    "model.dropout=0.1",  # dropout draws random numbers every step, so RNG restore matters
    "train.compile=false",
    f"train.max_tokens={STEPS * TOKENS_PER_STEP}",
    "train.warmup_tokens=64",
    "train.log_every_steps=1",
    "train.eval_every_steps=10",
    "train.eval_iters=2",
    "train.eval_samples=1",
    "train.sample_tokens=5",
]


def _trainer(extra: list[str] | None = None, log: list[str] | None = None) -> Trainer:
    exp = load_experiment("toy/base", FAST + (extra or []))
    return Trainer(exp, device="cpu", log=(log.append if log is not None else lambda _: None))


def _train_losses(run_dir: Path) -> dict[int, float]:
    records = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in records if r["kind"] == "train"}


def _straight_and_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[int, float], dict[int, float]]:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "straight"))
    straight = _trainer()
    assert straight.run() == "complete"

    monkeypatch.setenv("SLM_HOME", str(tmp_path / "split"))
    first = _trainer()
    assert first.run(max_steps=10) == "stopped"  # the evening ends here
    second = _trainer()  # the next evening: same command, fresh process state
    assert second.resumed_from is not None and second.step == 10
    assert second.run() == "complete"
    return _train_losses(straight.run_dir), _train_losses(second.run_dir)


def test_resume_is_equivalent_to_never_stopping(
    toy_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight, split = _straight_and_split(tmp_path, monkeypatch)
    assert sorted(straight) == sorted(split) == list(range(1, STEPS + 1))
    for step in straight:
        assert split[step] == pytest.approx(straight[step], abs=1e-4), f"step {step}"


def test_resume_test_is_sensitive(
    toy_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break one piece of resume (the RNG restore) and the losses must diverge."""
    monkeypatch.setattr("slmkit.train.trainer.set_rng_state", lambda _state: None)
    straight, split = _straight_and_split(tmp_path, monkeypatch)
    assert any(abs(split[s] - straight[s]) > 1e-4 for s in range(11, STEPS + 1))


def test_loss_falls_and_samples_are_printed(toy_repo: Path, slm_home: Path) -> None:
    log: list[str] = []
    trainer = _trainer(["train.max_tokens=3200"], log)  # 100 steps
    trainer.run()
    losses = _train_losses(trainer.run_dir)
    assert losses[100] < losses[1] - 1.0
    assert any("sample @ step 0" in line for line in log)  # the untrained baseline
    assert any("sample @ step 100" in line for line in log)
    for name in ("config.resolved.yaml", "manifest.json", "status.json", "train.log"):
        assert (trainer.run_dir / name).is_file()


def test_completed_run_is_not_retrained(toy_repo: Path, slm_home: Path) -> None:
    _trainer().run()
    again = _trainer()
    assert again.run() == "complete"
    assert again.step == STEPS


def test_checkpoints_are_pruned_and_best_is_kept(toy_repo: Path, slm_home: Path) -> None:
    trainer = _trainer(["train.keep_last_n_ckpts=2"])
    for _ in range(4):
        _trainer(["train.keep_last_n_ckpts=2"]).run(max_steps=5)
    steps = [c.step for c in checkpoint.list_checkpoints(trainer.run_dir)]
    assert steps == [15, 20]
    assert (trainer.run_dir / checkpoint.CKPT_DIR / checkpoint.BEST / "model.pt").is_file()


def test_run_identity(toy_repo: Path, slm_home: Path) -> None:
    base = load_experiment("toy/base", FAST)
    ids = ("pk-x", "tk-y")
    # Changing how a run is watched keeps the same run...
    watched = load_experiment("toy/base", FAST + ["train.eval_every_steps=3", "run.name=other"])
    assert run_id(base, *ids) == run_id(watched, *ids)
    # ...changing what it learns starts a new one.
    assert run_id(base, *ids) != run_id(load_experiment("toy/base", FAST + ["train.lr=5e-4"]), *ids)
    assert run_id(base, *ids) != run_id(base, "pk-other", ids[1])


def test_status_feeds_runs_list(toy_repo: Path, slm_home: Path) -> None:
    _trainer().run(max_steps=7)
    [(_, status)] = list_runs()
    assert status["step"] == 7 and status["tokens_seen"] == 7 * TOKENS_PER_STEP
    assert status["complete"] is False
    assert status["last_checkpoint"]["step"] == 7
