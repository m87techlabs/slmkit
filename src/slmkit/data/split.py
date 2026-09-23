"""Train/validation split by `Doc.group` (CONTRIBUTING.md rule 4).

Why groups: random per-document splits leak. If two settings of the same folk tune land on
opposite sides, validation loss rewards memorising the tune rather than learning music, and a
"surprisingly good" result is actually a bug. Splitting whole groups prevents that.

Why hash-sorting rather than `random.random() < val_fraction` per group: with ~100 groups a
coin flip per group gives anywhere from ~5% to ~15% validation. Sorting groups by a salted hash
and taking the first `ceil(n * val_fraction)` gives the exact fraction and is deterministic. A
group's rank never changes, so adding new groups moves at most the few groups nearest the
boundary rather than reshuffling everything.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from slmkit.project_api import Doc

SPLITS = ("train", "val")


def group_rank(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def assign_groups(groups: Iterable[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Map each distinct group to "train" or "val"."""
    unique = sorted(set(groups), key=lambda g: group_rank(g, seed))
    if len(unique) < 2:
        raise ValueError(f"need at least 2 groups to split, got {len(unique)}")
    n_val = min(len(unique) - 1, max(1, math.ceil(len(unique) * val_fraction)))
    return {g: ("val" if i < n_val else "train") for i, g in enumerate(unique)}


def split_documents(
    docs: Iterable[Doc], val_fraction: float, seed: int, exclusions: set[str] | None = None
) -> tuple[dict[str, list[Doc]], dict[str, int]]:
    """Split docs by group. Docs whose id or group is in `exclusions` are dropped entirely.

    Returns (docs per split, stats).
    """
    exclusions = exclusions or set()
    kept: list[Doc] = []
    excluded = 0
    for doc in docs:
        if doc.id in exclusions or doc.group in exclusions:
            excluded += 1
        else:
            kept.append(doc)

    assignment = assign_groups((d.group for d in kept), val_fraction, seed)
    out: dict[str, list[Doc]] = {s: [] for s in SPLITS}
    for doc in kept:
        out[assignment[doc.group]].append(doc)

    stats = {"excluded_docs": excluded}
    for split in SPLITS:
        stats[f"{split}_docs"] = len(out[split])
        stats[f"{split}_groups"] = sum(1 for v in assignment.values() if v == split)
        stats[f"{split}_chars"] = sum(len(d.text) for d in out[split])
    return out, stats


def write_docs(path: Path, docs: Iterable[Doc]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def read_docs(path: Path) -> Iterator[Doc]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield Doc(**json.loads(line))
