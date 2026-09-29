"""Generic graders, the eval runner and `slm runs compare`, on the toy project (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from slmkit.cli import app
from slmkit.config import load_experiment
from slmkit.eval.runner import EvalSettings, _split_counts, aggregate, evaluate
from slmkit.graders import completion, ngram_novelty, parse_rate
from slmkit.inference import load_run
from slmkit.project_api import EvalPrompt
from slmkit.registry import load_project
from slmkit.train.trainer import Trainer

PROMPT = EvalPrompt(id="p", prompt="R:reel\n")
TRAIN = ["|:ABcd efga|bagf edcB|ABcd efga|bagf e4:|", "|:GABc dBGB|cBAG FGAB|"]


# ------------------------------------------------------------------------------ graders


def test_completion_strips_the_prompt() -> None:
    assert completion(PROMPT, "R:reel\nABC") == "ABC"
    assert completion(PROMPT, "no prompt here") == "no prompt here"


def test_novelty_is_zero_for_a_copy_and_one_for_new_text() -> None:
    novelty = ngram_novelty(TRAIN, n=8)
    assert novelty(PROMPT, "R:reel\n" + TRAIN[0])["novelty"] == 0.0
    assert novelty(PROMPT, "R:reel\nzzzzzzzzzzzzzzzzzzzz")["novelty"] == 1.0
    half = novelty(PROMPT, "R:reel\n" + TRAIN[0][:20] + "zzzzzzzzzzzzzzzzzzzz")["novelty"]
    assert 0.2 < half < 0.9


def test_novelty_ignores_the_prompt_itself() -> None:
    # The prompt is copied from training data by construction; only the completion counts.
    novelty = ngram_novelty(["R:reel\nR:reel\nR:reel\n"], n=4)
    prompt = EvalPrompt(id="p", prompt="R:reel\nR:reel\n")
    assert novelty(prompt, "R:reel\nR:reel\nzzzz")["novelty"] == 1.0


def test_parse_rate_wraps_any_parser() -> None:
    ok = parse_rate("parses", int)
    assert ok(EvalPrompt(id="p", prompt=""), "42") == {"parses": 1.0}
    assert ok(EvalPrompt(id="p", prompt=""), "forty-two") == {"parses": 0.0}


def test_split_counts_and_aggregate() -> None:
    assert _split_counts(10, 4) == [3, 3, 2, 2]
    agg = aggregate({0: {"m": 0.5}, 1: {"m": 0.7}, 2: {"m": 0.6}})
    assert abs(agg["m"]["mean"] - 0.6) < 1e-9 and abs(agg["m"]["std"] - 0.1) < 1e-9


# ------------------------------------------------------------------------------ runner


def _trained(seed: int = 1337) -> str:
    exp = load_experiment(
        "toy/base",
        ["run.tracker=none", "train.compile=false", "train.max_tokens=640",
         "train.warmup_tokens=64", "train.eval_every_steps=10", "train.eval_iters=2",
         "train.eval_samples=1", "train.sample_tokens=4", f"run.seed={seed}"],
    )  # fmt: skip
    trainer = Trainer(exp, device="cpu", log=lambda _: None)
    trainer.run()
    return trainer.run_id


SETTINGS = EvalSettings(seeds=(0, 1, 2), num_samples=6, temperature=1.0, top_k=None, top_p=None,
                        max_new_tokens=12)  # fmt: skip


def test_eval_writes_a_report_with_seeds_spread_and_baseline(
    toy_repo: Path, slm_home: Path
) -> None:
    run = load_run(_trained(), "best", torch.device("cpu"))
    report, path = evaluate(run, SETTINGS, torch.device("cpu"), log=lambda _: None)
    assert path.is_file() and json.loads(path.read_text())["eval_id"] == report["eval_id"]
    assert set(report["model"]["per_seed"]) == {"0", "1", "2"}
    for side in ("model", "baseline"):
        assert {"ended", "length", "novelty"} <= set(report[side]["aggregate"])
    assert report["examples"]


def test_eval_is_not_redone(toy_repo: Path, slm_home: Path) -> None:
    run = load_run(_trained(), "best", torch.device("cpu"))
    first, _ = evaluate(run, SETTINGS, torch.device("cpu"), log=lambda _: None)
    log: list[str] = []
    again, _ = evaluate(run, SETTINGS, torch.device("cpu"), log=log.append)
    assert again == first and "already exists" in log[0]


def test_runs_compare_shows_both_runs(toy_repo: Path, slm_home: Path) -> None:
    a, b = _trained(1), _trained(2)
    evaluate(load_run(a, "best", torch.device("cpu")), SETTINGS, torch.device("cpu"),
             log=lambda _: None)  # fmt: skip
    out = CliRunner().invoke(app, ["runs", "compare", a[:12], b[:12]])
    assert out.exit_code == 0, out.output
    assert a[:12] in out.output and b[:12] in out.output
    assert "novelty" in out.output  # from a's eval report; b shows "-"


def test_changing_a_prompt_changes_the_eval_id(
    toy_repo: Path, slm_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = load_run(_trained(), "best", torch.device("cpu"))
    first, _ = evaluate(run, SETTINGS, torch.device("cpu"), log=lambda _: None)
    project_cls = type(load_project("toy", {}, slm_home))
    monkeypatch.setattr(project_cls, "eval_prompts",
                        lambda self, split: iter([EvalPrompt(id="p", prompt="abd")]))  # fmt: skip
    again, _ = evaluate(run, SETTINGS, torch.device("cpu"), log=lambda _: None)
    assert again["eval_id"] != first["eval_id"]
