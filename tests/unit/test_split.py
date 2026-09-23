"""Group split: no leakage, exact fraction, determinism, exclusions (CONTRIBUTING.md rule 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slmkit.data.split import assign_groups, read_docs, split_documents, write_docs
from slmkit.project_api import Doc


def _docs(n_groups: int = 50, per_group: int = 3) -> list[Doc]:
    return [
        Doc(id=f"g{g}/d{d}", group=f"g{g}", text=f"text {g} {d}")
        for g in range(n_groups)
        for d in range(per_group)
    ]


def test_no_group_in_both_splits() -> None:
    splits, _ = split_documents(_docs(), val_fraction=0.2, seed=0)
    train = {d.group for d in splits["train"]}
    val = {d.group for d in splits["val"]}
    assert train and val
    assert not train & val


def test_documents_of_one_group_stay_together() -> None:
    splits, _ = split_documents(_docs(per_group=5), val_fraction=0.3, seed=0)
    for part in splits.values():
        for group in {d.group for d in part}:
            assert sum(d.group == group for d in part) == 5


def test_fraction_is_exact_by_group_count() -> None:
    assignment = assign_groups([f"g{i}" for i in range(110)], val_fraction=0.1, seed=0)
    assert sum(v == "val" for v in assignment.values()) == 11


def test_deterministic_and_seed_sensitive() -> None:
    groups = [f"g{i}" for i in range(100)]
    assert assign_groups(groups, 0.2, seed=0) == assign_groups(groups, 0.2, seed=0)
    assert assign_groups(groups, 0.2, seed=0) != assign_groups(groups, 0.2, seed=1)


def test_exclusions_drop_docs_by_id_or_group() -> None:
    splits, stats = split_documents(
        _docs(n_groups=10), val_fraction=0.2, seed=0, exclusions={"g3", "g5/d0"}
    )
    ids = {d.id for part in splits.values() for d in part}
    assert not any(i.startswith("g3/") for i in ids)
    assert "g5/d0" not in ids and "g5/d1" in ids
    assert stats["excluded_docs"] == 4


def test_needs_two_groups() -> None:
    with pytest.raises(ValueError):
        assign_groups(["only"], 0.1, seed=0)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    docs = [Doc(id="a", group="g", text="é\nline", meta={"k": 1})]
    write_docs(tmp_path / "d.jsonl", docs)
    assert list(read_docs(tmp_path / "d.jsonl")) == docs
