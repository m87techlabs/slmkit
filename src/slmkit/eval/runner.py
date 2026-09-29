"""`slm eval`: generate samples, grade every one, report mean and spread across seeds.

For each sampling seed, `num_samples` outputs are generated, spread evenly over the project's
eval prompts, and every grader scores every output. Each metric is averaged per seed, then
reported as mean ± standard deviation across seeds. The spread is how much of a result is luck of
the sample draw; a difference between two runs smaller than it is not a difference.

Every report also scores a **baseline** that needs no model: tokens sampled independently in
proportion to how often they occur in the training data. A trained model has to beat it on every
grader, or the graders (or the model) are not measuring anything (DESIGN 6.7).

Reports are written to `runs/<run>/eval/<eval_id>.json`; the ID hashes the settings and the
checkpoint, so re-running the same evaluation finds the existing report instead of redoing it.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slmkit import artifacts
from slmkit.config.load import stable_hash
from slmkit.data.pack import open_packed
from slmkit.data.split import read_docs
from slmkit.graders import completion, ngram_novelty
from slmkit.inference import LoadedRun
from slmkit.project_api import EvalPrompt, Grader
from slmkit.registry import load_project
from slmkit.sampling import generate
from slmkit.tokenizers import EOS_ID, Tokenizer

EVAL_DIR = "eval"


@dataclass(frozen=True)
class EvalSettings:
    seeds: tuple[int, ...]
    num_samples: int  # per seed
    temperature: float
    top_k: int | None
    top_p: float | None
    max_new_tokens: int


@dataclass(frozen=True)
class Sample:
    prompt: EvalPrompt
    text: str  # prompt + completion, cut at <eos>
    ended: bool  # the model produced <eos> itself (finished the piece) before max_new_tokens


Sampler = Callable[[EvalPrompt, int, int], list[tuple[list[int], bool]]]


def _split_counts(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def _cut(ids: list[int]) -> tuple[list[int], bool]:
    return (ids[: ids.index(EOS_ID)], True) if EOS_ID in ids else (ids, False)


def model_sampler(run: LoadedRun, s: EvalSettings, device: torch.device) -> Sampler:
    def sample(prompt: EvalPrompt, count: int, seed: int) -> list[tuple[list[int], bool]]:
        ids = run.tokenizer.encode(prompt.prompt)
        idx = torch.tensor([ids] * count, device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = generate(run.model, idx, s.max_new_tokens, temperature=s.temperature,
                           top_k=s.top_k, top_p=s.top_p, stop_token=EOS_ID, generator=gen)  # fmt: skip
        return [_cut(row[len(ids) :]) for row in out.tolist()]

    return sample


def frequency_sampler(train_tokens: np.ndarray, vocab: int, s: EvalSettings) -> Sampler:
    """The no-model baseline: each token drawn independently, by its training frequency."""
    probs = np.bincount(train_tokens, minlength=vocab).astype(np.float64)
    probs /= probs.sum()

    def sample(prompt: EvalPrompt, count: int, seed: int) -> list[tuple[list[int], bool]]:
        rng = np.random.default_rng(seed)
        draws = rng.choice(vocab, size=(count, s.max_new_tokens), p=probs)
        return [_cut(row.tolist()) for row in draws]

    return sample


def draw_samples(
    sampler: Sampler, tokenizer: Tokenizer, prompts: list[EvalPrompt], n: int, seed: int
) -> list[Sample]:
    samples = []
    for k, (prompt, count) in enumerate(zip(prompts, _split_counts(n, len(prompts)), strict=True)):
        if count == 0:
            continue
        for ids, ended in sampler(prompt, count, seed * 1000 + k):
            samples.append(Sample(prompt, prompt.prompt + tokenizer.decode(ids), ended))
    return samples


def grade(samples: list[Sample], graders: list[Grader]) -> dict[str, float]:
    """Average every metric over the samples of one seed."""
    totals: dict[str, list[float]] = {}
    for s in samples:
        scores = {"ended": float(s.ended), "length": float(len(completion(s.prompt, s.text)))}
        for g in graders:
            scores.update(g(s.prompt, s.text))
        for key, value in scores.items():
            totals.setdefault(key, []).append(value)
    return {k: float(np.mean(v)) for k, v in totals.items()}


def aggregate(per_seed: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics = next(iter(per_seed.values())).keys()
    out = {}
    for m in metrics:
        values = [per_seed[s][m] for s in per_seed]
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        out[m] = {"mean": statistics.fmean(values), "std": spread}
    return out


def evaluate(
    run: LoadedRun,
    s: EvalSettings,
    device: torch.device,
    *,
    baseline: bool = True,
    prompts_kind: str = "auto",
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], Path]:
    """`prompts_kind`: "headers" (the project's eval_prompts), "sft" (plain-language requests),
    or "auto" (sft for a fine-tuned run, headers otherwise). Asking a *base* model in words
    shows what SFT added."""
    cfg = run.config
    manifest = artifacts.read_manifest(run.run_dir)
    while "packed" not in manifest["inputs"]:  # an SFT run: its data is its parent's
        manifest = artifacts.read_manifest(run.run_dir.parent / manifest["inputs"]["parent"])
    packed = artifacts.find_artifact(manifest["inputs"]["packed"])
    dataset = artifacts.find_artifact(artifacts.read_manifest(packed)["inputs"]["dataset"])
    project = load_project(cfg["project"]["name"], cfg["project"]["args"], artifacts.slm_home())
    # A fine-tuned run is asked in words; a base run is given header prompts. Both carry the
    # same `meta`, so the same graders score whether each did what was asked.
    kind = prompts_kind if prompts_kind != "auto" else ("sft" if run.stage == "sft" else "headers")
    sft_prompts = project.sft_eval_prompts("val") if kind == "sft" else None
    if kind == "sft" and sft_prompts is None:
        raise ValueError(f"project {cfg['project']['name']!r} has no SFT prompts")
    prompts = list(sft_prompts if sft_prompts is not None else project.eval_prompts("val"))
    graders = [
        *project.graders(),
        ngram_novelty(d.text for d in read_docs(dataset / "train.jsonl")),
    ]

    # The ID covers the run, its checkpoint, the settings, the prompts themselves and the
    # graders' own source code, so changing a prompt or a grader invalidates old reports instead
    # of silently reusing numbers they no longer produce.
    eval_id = (
        "ev-"
        + stable_hash(
            {
                "run": run.run_id,
                "settings": asdict(s),
                "checkpoint": run.checkpoint,
                "step": run.state["step"],
                "baseline": baseline,
                "prompts": kind,
                "prompt_items": [[p.id, p.prompt, p.meta] for p in prompts],
                "graders": [inspect.getsource(g) for g in graders],
            }
        )[:10]
    )
    path = run.run_dir / EVAL_DIR / f"{eval_id}.json"
    if path.is_file():
        log(f"{eval_id} already exists: {path}")
        report: dict[str, Any] = json.loads(path.read_text())
        return report, path

    contenders: dict[str, Sampler] = {"model": model_sampler(run, s, device)}
    if baseline:
        train = np.asarray(open_packed(packed / "train.bin"), dtype=np.int64)
        contenders["baseline"] = frequency_sampler(train, run.tokenizer.vocab_size, s)

    results: dict[str, Any] = {}
    examples: list[dict[str, str]] = []
    for name, sampler in contenders.items():
        per_seed = {}
        for seed in s.seeds:
            samples = draw_samples(sampler, run.tokenizer, prompts, s.num_samples, seed)
            per_seed[seed] = grade(samples, graders)
            log(f"  {name:<8} seed {seed}: " + "  ".join(
                f"{k} {v:.3f}" for k, v in per_seed[seed].items()))  # fmt: skip
            if name == "model" and seed == s.seeds[0]:
                examples = [{"prompt": x.prompt.id, "text": x.text} for x in samples[:: max(1, len(samples) // 4)][:4]]  # fmt: skip
        results[name] = {"per_seed": {str(k): v for k, v in per_seed.items()},
                         "aggregate": aggregate(per_seed)}  # fmt: skip

    report = {
        "eval_id": eval_id,
        "run_id": run.run_id,
        "name": cfg["run"]["name"],
        "stage": run.stage,
        "prompts_kind": kind,
        "checkpoint": {
            "which": run.checkpoint,
            "step": run.state["step"],
            "val_loss": run.state.get("last_val"),
        },
        "settings": asdict(s),
        "prompts": [p.id for p in prompts],
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **results,
        "examples": examples,
    }
    path.parent.mkdir(exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    # Return exactly what a later run reads back from disk (JSON turns tuples into lists), so a
    # fresh report and a re-used one are identical to the caller.
    saved: dict[str, Any] = json.loads(text)
    return saved, path


def latest_report(run_dir: Path, prompts_kind: str | None = None) -> dict[str, Any] | None:
    """The most recent eval report, optionally only among those using `prompts_kind` prompts
    ("headers" or "sft").

    "Recent" is the report's own `created` time, not the file's mtime: copying a run directory
    (a backup, `cp -r`, rsync without -t) resets mtimes and would pick an arbitrary report."""
    reports: list[dict[str, Any]] = [
        json.loads(p.read_text()) for p in (run_dir / EVAL_DIR).glob("ev-*.json")
    ]
    reports.sort(key=lambda r: _dt.datetime.fromisoformat(r["created"]))
    for report in reversed(reports):
        if prompts_kind is None or report.get("prompts_kind", "headers") == prompts_kind:
            return report
    return None


def fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}" if not math.isnan(stat["mean"]) else "-"
