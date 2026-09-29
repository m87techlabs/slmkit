"""`slm runs summary`: every experiment's result, averaged over its training seeds.

One run is one draw. Retraining with a different seed changes the initial weights and the order
the data is read in, and the result moves; DESIGN §7 asks for ≥ 3 seeds before believing a
comparison. This groups complete runs by (experiment, stage) and reports, per metric, the mean
and standard deviation *across runs*. Each run contributes one number per metric: its best
validation loss, and the mean of its latest matching eval report (itself averaged over sampling
seeds, see eval/runner.py).

So there are two spreads in play, and they answer different questions (evaluation.md §4):
`slm eval`'s ± is sampling noise for one model; this ± is training noise across models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slmkit import artifacts
from slmkit.data.pack import count_chars, open_packed
from slmkit.data.split import read_docs
from slmkit.eval.runner import aggregate, latest_report
from slmkit.train.run import list_runs, load_run_config


@dataclass
class Group:
    experiment: str
    stage: str
    seeds: list[int] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    per_run: list[dict[str, float]] = field(default_factory=list)
    gpu_hours: float = 0.0
    eval_kind: str | None = None
    evaluated: int = 0  # runs that have an eval report of the chosen kind

    @property
    def stats(self) -> dict[str, dict[str, float]]:
        """Mean and standard deviation across runs, for metrics every run has."""
        if not self.per_run:
            return {}
        shared = [m for m in self.per_run[0] if all(m in r for r in self.per_run)]
        table = {i: {m: r[m] for m in shared} for i, r in enumerate(self.per_run)}
        return aggregate(table)


def _val_chars_per_token(run_dir: Path, cfg: dict[str, Any]) -> float | None:
    """Validation characters per token, from the run's packed data (pretraining runs only)."""
    inputs = artifacts.read_manifest(run_dir)["inputs"]
    if "packed" not in inputs:
        return None
    packed = artifacts.find_artifact(inputs["packed"])
    dataset = artifacts.find_artifact(artifacts.read_manifest(packed)["inputs"]["dataset"])
    chars = count_chars(read_docs(dataset / "val.jsonl"), append_eos=cfg["data"]["append_eos"])
    return chars / max(1, len(open_packed(packed / "val.bin")))


def run_result(run_dir: Path, stage: str, prompts: str) -> tuple[dict[str, float], str | None]:
    """One run's numbers: best validation loss (and bpc), plus its eval report's means."""
    records = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["kind"] == "eval"]
    out: dict[str, float] = {}
    if evals:
        best = min(evals, key=lambda r: r["val_loss"])
        out["best val loss"] = best["val_loss"]
        bpc = best.get("val_bpc")
        if bpc is None and stage == "pretrain":
            # Runs trained before the trainer logged bpc: the same formula, computed here.
            cpt = _val_chars_per_token(run_dir, load_run_config(run_dir))
            bpc = best["val_loss"] / cpt / math.log(2) if cpt else None
        if bpc is not None:
            out["best val bpc"] = bpc
    kind = prompts if prompts != "auto" else ("sft" if stage == "sft" else "headers")
    report = latest_report(run_dir, kind)
    if report:
        out.update({m: s["mean"] for m, s in report["model"]["aggregate"].items()})
    return out, kind if report else None


def summarize(prompts: str = "auto", project: str | None = None) -> list[Group]:
    groups: dict[tuple[str, str], Group] = {}
    for run_dir, st in sorted(list_runs(), key=lambda r: r[0].name):
        if not st.get("complete"):
            continue
        experiment = st.get("experiment", "?")
        if project and not experiment.startswith(project + "/"):
            continue
        stage = st.get("stage", "pretrain")
        g = groups.setdefault((experiment, stage), Group(experiment, stage))
        result, kind = run_result(run_dir, stage, prompts)
        g.seeds.append(int(load_run_config(run_dir)["run"]["seed"]))
        g.run_ids.append(run_dir.name)
        g.per_run.append(result)
        g.gpu_hours += st.get("gpu_seconds", 0) / 3600
        g.eval_kind = g.eval_kind or kind
        g.evaluated += kind is not None
    return sorted(groups.values(), key=lambda g: (g.experiment, g.stage != "pretrain"))
