"""Schedule, guards, checkpoints and sampling, each on its own."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch

from slmkit.model import CausalLM, ModelArgs
from slmkit.sampling import filter_logits, generate
from slmkit.train import checkpoint
from slmkit.train.guards import NonFiniteGuard, TrainingDiverged
from slmkit.train.schedule import lr_at

SCHED = {"lr": 1e-3, "warmup_tokens": 1000, "max_tokens": 11_000, "min_lr_ratio": 0.1}


# --------------------------------------------------------------------------- schedule


def test_warmup_is_linear() -> None:
    assert lr_at(100, **SCHED) == pytest.approx(1e-4)
    assert lr_at(500, **SCHED) == pytest.approx(5e-4)
    assert lr_at(1000, **SCHED) == pytest.approx(1e-3)


def test_cosine_peak_midpoint_and_floor() -> None:
    assert lr_at(1000, **SCHED) == pytest.approx(1e-3)
    assert lr_at(6000, **SCHED) == pytest.approx(0.55e-3)  # halfway: midway between peak and floor
    assert lr_at(11_000, **SCHED) == pytest.approx(1e-4)
    assert lr_at(50_000, **SCHED) == pytest.approx(1e-4)  # never below the floor


def test_decay_is_monotonic() -> None:
    lrs = [lr_at(t, **SCHED) for t in range(1000, 11_001, 250)]
    assert all(a >= b for a, b in itertools.pairwise(lrs))


# --------------------------------------------------------------------------- guards


def test_guard_skips_then_aborts() -> None:
    guard = NonFiniteGuard(max_consecutive=3)
    assert guard.check(2.0, 0.5)
    assert not guard.check(math.nan, 0.5)
    assert not guard.check(2.0, math.inf)
    assert guard.check(2.0, 0.5)  # a good step resets the streak
    assert guard.total_skipped == 2
    guard.check(math.nan, 1.0)
    guard.check(math.nan, 1.0)
    with pytest.raises(TrainingDiverged, match="3 consecutive"):
        guard.check(math.nan, 1.0)


# --------------------------------------------------------------------------- checkpoints


def _save(run_dir: Path, step: int, keep: int = 3) -> None:
    checkpoint.save(run_dir, step, {"w": torch.tensor([float(step)])}, {}, {"step": step},
                    keep_last_n=keep)  # fmt: skip


def test_checkpoint_round_trip_and_pruning(tmp_path: Path) -> None:
    for step in (100, 200, 300, 400):
        _save(tmp_path, step)
    assert [c.step for c in checkpoint.list_checkpoints(tmp_path)] == [200, 300, 400]
    model, _, state = checkpoint.load(checkpoint.latest(tmp_path).path, torch.device("cpu"))  # type: ignore[union-attr]
    assert model["w"].item() == 400.0 and state["step"] == 400


def test_half_written_checkpoint_is_ignored(tmp_path: Path) -> None:
    _save(tmp_path, 100)
    (tmp_path / "ckpt" / "step_0000200.tmp").mkdir()  # a power cut mid-save
    assert checkpoint.latest(tmp_path).step == 100  # type: ignore[union-attr]
    _save(tmp_path, 200)  # the next save of that step clears the leftover
    assert not (tmp_path / "ckpt" / "step_0000200.tmp").exists()


def test_best_is_replaced_atomically(tmp_path: Path) -> None:
    for value in (1.0, 2.0):
        checkpoint.save_best(tmp_path, {"w": torch.tensor([value])}, {}, {"step": 0})
    best = tmp_path / "ckpt" / "best"
    assert torch.load(best / "model.pt")["w"].item() == 2.0
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["best"]


# --------------------------------------------------------------------------- sampling


def test_filter_logits_worked_example() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.1]])

    def probs(**kw: object) -> list[float]:
        p = filter_logits(logits, **kw).softmax(-1)  # type: ignore[arg-type]
        return [round(x, 2) for x in p[0].tolist()]

    assert probs(temperature=1.0, top_k=None, top_p=None) == [0.66, 0.24, 0.10]
    assert probs(temperature=0.5, top_k=None, top_p=None) == [0.86, 0.12, 0.02]
    assert probs(temperature=1.0, top_k=2, top_p=None) == [0.73, 0.27, 0.0]
    assert probs(temperature=1.0, top_k=None, top_p=0.5) == [1.0, 0.0, 0.0]


def _tiny_model() -> CausalLM:
    torch.manual_seed(0)
    return CausalLM(ModelArgs(vocab_size=20, block_size=8, n_layers=1, d_model=32,
                              n_heads=2, n_kv_heads=2, ffn_hidden=64))  # fmt: skip


def test_generate_is_reproducible_with_a_generator() -> None:
    model = _tiny_model()
    idx = torch.tensor([[1, 2, 3]])
    a = generate(model, idx, 12, temperature=1.0, generator=torch.Generator().manual_seed(7))
    b = generate(model, idx, 12, temperature=1.0, generator=torch.Generator().manual_seed(7))
    assert a.shape == (1, 15)  # generates past block_size by cropping the context
    assert torch.equal(a, b)
    assert torch.equal(a[:, :3], idx)


def test_greedy_and_stop_token() -> None:
    model = _tiny_model()
    idx = torch.tensor([[1]])
    greedy = generate(model, idx, 5, temperature=0.0)
    assert torch.equal(greedy, generate(model, idx, 5, temperature=0.0))
    first = int(greedy[0, 1])
    stopped = generate(model, idx, 5, temperature=0.0, stop_token=first)
    assert stopped.shape == (1, 2)  # stopped right after producing the stop token
