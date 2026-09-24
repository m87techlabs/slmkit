"""Run identity and bookkeeping: which directory a run lives in, and how it is doing.

A run is `$SLM_HOME/runs/<run_id>/`. The ID hashes the configuration that determines *what is
learned* (model, optimisation, seed, data), plus the packed-data and tokenizer artifact IDs.
So re-running the same command finds the same directory and resumes, which is how a run spans
many evenings, while `--set train.lr=6e-4` gets a new directory and a fresh run.

Settings that only change *how the run is watched or saved* are left out of the hash (eval
cadence, checkpoint interval, logging, compile, tracker, run name). You can change them between
sessions and still resume the same run. That list is `OPERATIONAL`.

Unlike data artifacts, a run directory is mutable while training: checkpoints come and go and
`status.json` is rewritten. It becomes immutable once complete.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import yaml

from slmkit import artifacts
from slmkit.config.load import Experiment, dump_resolved

RUNS_DIR = "runs"
RUN_CODE_VERSION = 1

OPERATIONAL: dict[str, set[str]] = {
    "run": {"name", "tracker"},
    "train": {
        "eval_every_steps",
        "eval_iters",
        "eval_samples",
        "sample_tokens",
        "log_every_steps",
        "ckpt_every_minutes",
        "keep_last_n_ckpts",
        "max_consecutive_skips",
        "compile",
        "compile_mode",
    },
}
IGNORED_SECTIONS = {"eval", "sft"}  # used by later stages, not by pretraining


def training_identity(exp: Experiment) -> dict[str, Any]:
    cfg = exp.config.model_dump(mode="json")
    out = {}
    for section, values in cfg.items():
        if section in IGNORED_SECTIONS:
            continue
        drop = OPERATIONAL.get(section, set())
        out[section] = {k: v for k, v in values.items() if k not in drop}
    return out


def run_id(exp: Experiment, packed_id: str, tokenizer_id: str) -> str:
    return artifacts.artifact_id(
        "run",
        code_version=RUN_CODE_VERSION,
        config=training_identity(exp),
        inputs={"packed": packed_id, "tokenizer": tokenizer_id},
    )


def runs_root() -> Path:
    return artifacts.slm_home() / RUNS_DIR


def create_run_dir(exp: Experiment, rid: str, packed_id: str, tokenizer_id: str) -> Path:
    """Create the run directory on first use. Idempotent: an existing run is left alone."""
    run_dir = runs_root() / rid
    if (run_dir / artifacts.MANIFEST).is_file():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = artifacts.build_manifest(
        kind="run",
        artifact=rid,
        project=exp.project,
        code_version=RUN_CODE_VERSION,
        config=exp.config.model_dump(mode="json"),
        inputs={"packed": packed_id, "tokenizer": tokenizer_id},
        stats={"experiment": exp.address, "name": exp.config.run.name},
    )
    (run_dir / "config.resolved.yaml").write_text(dump_resolved(exp.config))
    (run_dir / artifacts.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def write_status(run_dir: Path, status: dict[str, Any]) -> None:
    status = {**status, "updated": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, run_dir / "status.json")  # atomic: readers never see half a file


def read_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "status.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def list_runs() -> list[tuple[Path, dict[str, Any]]]:
    root = runs_root()
    if not root.is_dir():
        return []
    rows = []
    for run_dir in sorted(root.iterdir()):
        status = read_status(run_dir)
        if status is not None:
            rows.append((run_dir, status))
    return sorted(rows, key=lambda r: str(r[1].get("updated", "")), reverse=True)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    return data
