"""The checks behind the Phase F sweep's conclusions (docs/concepts/experiments.md).

    uv run python projects/abc_music/check_sweep.py keys          # §5: original vs transposed keys
    uv run python projects/abc_music/check_sweep.py overlap       # §5: is it leakage?
    uv run python projects/abc_music/check_sweep.py parse         # §8: what "parse rate" means
    uv run python projects/abc_music/check_sweep.py temperature   # §8: plays vs temperature

`keys` scores every pretraining run's best checkpoint, in bits per character, on the validation
tunes in their original keys and on transposed copies of the same tunes. `overlap` counts how
much of each validation set appears verbatim in the training text. `parse` measures three
definitions of "parses" on the best models and on the random-character baseline. `temperature`
samples one model at several temperatures. All need a GPU for speed, and take under two minutes.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slmkit import artifacts
from slmkit.data.pack import open_packed
from slmkit.data.split import read_docs
from slmkit.eval.runner import EvalSettings, draw_samples, frequency_sampler, grade, model_sampler
from slmkit.graders import ngram_novelty
from slmkit.inference import LoadedRun, load_run
from slmkit.registry import load_project
from slmkit.tokenizers import EOS_ID
from slmkit.train.run import list_runs

AUGMENTED = {"transpose_semitones": [-2, -1, 1, 2], "max_key_accidentals": 3, "header_dropout": 0.4}
EXPERIMENTS = ("baseline", "noaug", "micro", "micro_noaug")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _project() -> Any:
    return load_project("abc_music", AUGMENTED, artifacts.slm_home())


def _graders() -> Any:
    _project()  # registers the package
    return importlib.import_module("slmkit_projects.abc_music.graders")


def _runs(experiments: tuple[str, ...]) -> list[tuple[str, str]]:
    """(experiment, run_id) for complete pretraining runs, ordered by experiment then seed."""
    rows = []
    for run_dir, st in list_runs():
        exp = st.get("experiment", "").removeprefix("abc_music/")
        if exp in experiments and st.get("stage", "pretrain") == "pretrain" and st.get("complete"):
            rows.append((exp, run_dir.name))
    return sorted(rows)


def _dataset(run: LoadedRun) -> Path:
    packed = artifacts.find_artifact(artifacts.read_manifest(run.run_dir)["inputs"]["packed"])
    return artifacts.find_artifact(artifacts.read_manifest(packed)["inputs"]["dataset"])


def _validation_sets() -> tuple[list[str], list[str]]:
    """Validation tunes that have transposed copies, and those copies. The validation split is
    the same for every experiment (augmentation happens after splitting, train only)."""
    project = _project()
    dataset = _dataset(load_run(_runs(("baseline",))[0][1], "best"))
    pairs = [(d.text, [v.text for v in list(project.augment(d))[1:]])
             for d in read_docs(dataset / "val.jsonl")]  # fmt: skip
    return [o for o, moved in pairs if moved], [t for _, moved in pairs for t in moved]


@torch.no_grad()
def _bpc(run: LoadedRun, texts: list[str]) -> float:
    """Every character scored once: consecutive, non-overlapping windows over the texts."""
    ids: list[int] = []
    chars = 0
    for t in texts:
        ids += run.tokenizer.encode(t) + [EOS_ID]
        chars += len(t) + 1
    x = torch.tensor(ids, device=DEV)
    block, nats = run.model.args.block_size, 0.0
    for i in range(0, len(ids) - 1, block):
        target = x[i + 1 : i + 1 + block][None]
        _, loss = run.model(x[i : i + target.shape[1]][None], target)
        assert loss is not None
        nats += loss.item() * target.shape[1]
    return nats / chars / math.log(2)


def keys() -> None:
    original, moved = _validation_sets()
    print(f"{len(original)} validation tunes with transposed copies, {len(moved)} copies\n")
    results: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for exp, rid in _runs(EXPERIMENTS):
        run = load_run(rid, "best", DEV)
        results[exp].append((_bpc(run, original), _bpc(run, moved)))
        o, m = results[exp][-1]
        print(f"  {exp:<12} {rid}  original keys {o:.3f}  transposed {m:.3f}")
    print(f"\n{'bpc':<12} {'original keys':>16} {'transposed':>16} {'difference':>11}")
    for exp in EXPERIMENTS:
        o = [a for a, _ in results[exp]]
        m = [b for _, b in results[exp]]
        print(f"{exp:<12} {statistics.fmean(o):>8.3f} ± {statistics.stdev(o):.3f} "
              f"{statistics.fmean(m):>8.3f} ± {statistics.stdev(m):.3f} "
              f"{statistics.fmean(m) - statistics.fmean(o):>+11.3f}")  # fmt: skip


def overlap(n: int = 32) -> None:
    original, moved = _validation_sets()
    runs = dict(_runs(("baseline", "noaug")))
    train = {name: {t[i : i + n] for d in read_docs(_dataset(load_run(runs[name], "best")) /
                                                   "train.jsonl")
                    for t in [d.text] for i in range(len(t) - n + 1)}
             for name in ("baseline", "noaug")}  # fmt: skip
    for label, texts in (
        ("validation, original keys", original),
        ("validation, transposed", moved),
    ):
        for name, table in train.items():
            shares = [sum(t[i : i + n] in table for i in range(len(t) - n + 1))
                      / (len(t) - n + 1) for t in texts if len(t) >= n]  # fmt: skip
            print(f"{label:<27} vs {name + ' train':<15} {statistics.fmean(shares):5.1%} of "
                  f"{n}-character windows seen; {sum(s > 0.5 for s in shares)} of {len(texts)} "
                  "tunes more than half seen")  # fmt: skip


def _parse_rates(samples: list[Any]) -> tuple[float, float, float]:
    g = _graders()
    parsed = midi = clean = 0
    for x in samples:
        tune = g._parse(x.text)
        ok = tune is not None and g.note_count(tune) >= g.MIN_NOTES
        out, errors = g.abc2midi("X:1\n" + x.text)
        parsed += ok
        midi += ok and out is not None
        clean += ok and out is not None and not errors
    n = len(samples)
    return parsed / n, midi / n, clean / n


def parse() -> None:
    settings = EvalSettings(seeds=(0,), num_samples=200, temperature=0.8, top_k=None, top_p=None,
                            max_new_tokens=600)  # fmt: skip
    prompts = list(_project().eval_prompts("val"))
    print(f"{'':<34} {'parses':>7} {'makes MIDI':>11} {'plays, no error':>16}")
    first = None
    for exp, rid in _runs(("micro_noaug", "micro", "baseline")):
        run = load_run(rid, "best", DEV)
        first = first or run
        samples = draw_samples(model_sampler(run, settings, DEV), run.tokenizer, prompts, 200, 0)
        p, m, c = _parse_rates(samples)
        print(f"{exp + ' ' + rid:<34} {p:>7.3f} {m:>11.3f} {c:>16.3f}")
    assert first is not None
    packed = artifacts.find_artifact(artifacts.read_manifest(first.run_dir)["inputs"]["packed"])
    train = np.asarray(open_packed(packed / "train.bin"), dtype=np.int64)
    sampler = frequency_sampler(train, first.tokenizer.vocab_size, settings)
    p, m, c = _parse_rates(draw_samples(sampler, first.tokenizer, prompts, 200, 0))
    print(f"{'random characters (baseline)':<34} {p:>7.3f} {m:>11.3f} {c:>16.3f}")


def temperature(run_id: str | None = None) -> None:
    rid = run_id or max(_runs(("micro_noaug",)), key=lambda r: r[1])[1]
    run = load_run(rid, "best", DEV)
    project = _project()
    graders = [*project.graders(), ngram_novelty(d.text for d in read_docs(_dataset(run) /
                                                                           "train.jsonl"))]  # fmt: skip
    print(f"{rid}, 200 samples per temperature, sampling seed 0")
    for t in (1.0, 0.8, 0.6, 0.4, 0.2):
        s = EvalSettings(seeds=(0,), num_samples=200, temperature=t, top_k=None, top_p=None,
                         max_new_tokens=600)  # fmt: skip
        scores = grade(draw_samples(model_sampler(run, s, DEV), run.tokenizer,
                                    list(project.eval_prompts("val")), 200, 0), graders)  # fmt: skip
        print(f"T={t:.1f}  " + "  ".join(f"{k} {scores[k]:.3f}" for k in
                                        ("plays", "bar_accuracy", "ended", "novelty")))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["keys", "overlap", "parse", "temperature"])
    parser.add_argument("--run", help="temperature: the run to sample (default: a micro_noaug run)")
    args = parser.parse_args()
    if args.check == "temperature":
        temperature(args.run)
    else:
        {"keys": keys, "overlap": overlap, "parse": parse}[args.check]()


if __name__ == "__main__":
    main()
