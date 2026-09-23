"""The data stages end to end on a synthetic project: build, skip, rebuild, trace."""

from __future__ import annotations

from pathlib import Path

from slmkit import artifacts, pipeline
from slmkit.config import load_experiment
from slmkit.data.split import read_docs


def test_builds_every_stage_then_skips(toy_repo: Path, slm_home: Path) -> None:
    exp = load_experiment("toy/base")
    log: list[str] = []
    first = pipeline.ensure_packed(exp, log.append)
    assert [line.split()[:2] for line in log] == [
        ["raw", "built"],
        ["dataset", "built"],
        ["tokenizer", "built"],
        ["packed", "built"],
    ]
    assert not first.skipped
    for sub in ("train.bin", "val.bin", "meta.json", "manifest.json"):
        assert (first.path / sub).is_file()

    log.clear()
    again = pipeline.ensure_packed(exp, log.append)
    assert again.id == first.id and again.skipped
    assert all(" exists " in line for line in log)


def test_changing_config_changes_downstream_ids_only(toy_repo: Path, slm_home: Path) -> None:
    base = load_experiment("toy/base")
    changed = load_experiment("toy/base", ["data.val_fraction=0.4"])
    assert pipeline.ensure_raw(base).id == pipeline.ensure_raw(changed).id
    assert pipeline.ensure_dataset(base).id != pipeline.ensure_dataset(changed).id
    assert pipeline.ensure_packed(base).id != pipeline.ensure_packed(changed).id

    eos_off = load_experiment("toy/base", ["data.append_eos=false"])
    assert pipeline.ensure_dataset(eos_off).id == pipeline.ensure_dataset(base).id
    assert pipeline.ensure_packed(eos_off).id != pipeline.ensure_packed(base).id


def test_split_has_no_leakage_and_honours_exclusions(toy_repo: Path, slm_home: Path) -> None:
    ds = pipeline.ensure_dataset(load_experiment("toy/base"))
    train = list(read_docs(ds.path / "train.jsonl"))
    val = list(read_docs(ds.path / "val.jsonl"))
    assert not {d.group for d in train} & {d.group for d in val}
    assert "tune19" not in {d.group for d in train + val}


def test_augmentation_touches_train_only(toy_repo: Path, slm_home: Path) -> None:
    ds = pipeline.ensure_dataset(load_experiment("toy/base", ["project.args.augment=true"]))
    train = list(read_docs(ds.path / "train.jsonl"))
    val = list(read_docs(ds.path / "val.jsonl"))
    assert any(d.text.isupper() for d in train)
    assert not any(d.text.isupper() for d in val)


def test_lineage_reaches_raw(toy_repo: Path, slm_home: Path) -> None:
    packed = pipeline.ensure_packed(load_experiment("toy/base"))
    kinds = [(m["kind"], seen) for _, _, m, seen in artifacts.lineage(packed.id)]
    assert kinds == [
        ("packed", False),
        ("dataset", False),
        ("raw", False),
        ("tokenizer", False),
        ("dataset", True),  # reached twice: shown once, then referenced
    ]
