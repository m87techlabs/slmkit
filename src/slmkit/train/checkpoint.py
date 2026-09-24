"""Full-state, atomic checkpoints (CONTRIBUTING.md rule 8).

On a machine that is switched off most nights, every real run is a resumed run, so a
checkpoint must hold *everything* needed to continue as if nothing happened:

    model.pt       weights (fp32)
    optimizer.pt   AdamW's two running averages per parameter; without them the first steps
                   after a resume behave like a cold start and the loss curve jumps
    state.pt       step, tokens seen (which also fixes the LR schedule), GPU-seconds used,
                   best validation loss, the train sampler's position, and all four RNG states

Layout inside a run directory:

    ckpt/step_0000500/   ckpt/step_0001000/   ...   the last `keep_last_n`, oldest pruned
    ckpt/best/                                      lowest validation loss so far

Each checkpoint is written to `<name>.tmp`, fsynced, then renamed: a power cut mid-save leaves
the previous checkpoint intact and a `.tmp` directory that is ignored and later cleared.

Format: `torch.save`, i.e. pickle. Acceptable because only slmkit writes and reads these files,
and they carry Python objects (RNG states) that safetensors cannot hold. Never load a `.pt`
file from someone else: unpickling can run code. Exports use safetensors (docs/MODEL.md 6).
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slmkit.artifacts import fsync_tree, guard_path

CKPT_DIR = "ckpt"
BEST = "best"
_STEP_DIR = re.compile(r"^step_(\d{7})$")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int


def _write(
    target: Path, model: dict[str, Any], optimizer: dict[str, Any], state: dict[str, Any]
) -> None:
    target = guard_path(target)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(model, tmp / "model.pt")
    torch.save(optimizer, tmp / "optimizer.pt")
    torch.save(state, tmp / "state.pt")
    fsync_tree(tmp)
    if target.exists():  # only `best/` is ever replaced; step dirs are unique
        old = target.with_name(target.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        os.rename(target, old)
        os.rename(tmp, target)
        shutil.rmtree(old)
    else:
        os.rename(tmp, target)
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def list_checkpoints(run_dir: Path) -> list[Checkpoint]:
    root = run_dir / CKPT_DIR
    if not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        match = _STEP_DIR.match(child.name)
        if match and (child / "state.pt").is_file():
            found.append(Checkpoint(child, int(match.group(1))))
    return sorted(found, key=lambda c: c.step)


def latest(run_dir: Path) -> Checkpoint | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def save(
    run_dir: Path,
    step: int,
    model: dict[str, Any],
    optimizer: dict[str, Any],
    state: dict[str, Any],
    *,
    keep_last_n: int,
) -> Path:
    target = run_dir / CKPT_DIR / f"step_{step:07d}"
    if not target.exists():
        _write(target, model, optimizer, state)
    for old in list_checkpoints(run_dir)[:-keep_last_n]:
        shutil.rmtree(old.path)
    return target


def save_best(
    run_dir: Path, model: dict[str, Any], optimizer: dict[str, Any], state: dict[str, Any]
) -> Path:
    target = run_dir / CKPT_DIR / BEST
    _write(target, model, optimizer, state)
    return target


def load(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (model state, optimizer state, trainer state)."""
    model = torch.load(path / "model.pt", map_location=device, weights_only=True)
    optimizer = torch.load(path / "optimizer.pt", map_location=device, weights_only=True)
    # state.pt holds RNG states (Python and NumPy objects), which weights_only mode refuses.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    return model, optimizer, state
