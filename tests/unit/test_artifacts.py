"""Artifact IDs, atomic commit and the /mnt guard (CONTRIBUTING.md rules 1 and 3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from slmkit import artifacts
from slmkit.artifacts import ArtifactPathError, artifact_id, commit_dir, guard_path


def _id(**kw: object) -> str:
    args: dict = {"code_version": 1, "config": {"a": 1, "b": 2}, "inputs": {"raw": "raw-x"}}
    args.update(kw)
    return artifact_id("dataset", **args)


def test_id_is_deterministic_and_order_independent() -> None:
    assert _id() == _id()
    assert _id() == _id(config={"b": 2, "a": 1})
    assert _id().startswith("ds-") and len(_id()) == len("ds-") + 12


@pytest.mark.parametrize(
    "change",
    [{"config": {"a": 1, "b": 3}}, {"inputs": {"raw": "raw-y"}}, {"code_version": 2}],
)
def test_id_changes_with_any_input(change: dict) -> None:
    assert _id(**change) != _id()


def test_guard_rejects_windows_drives() -> None:
    with pytest.raises(ArtifactPathError, match="9P"):
        guard_path(Path("/mnt/c/Users/someone/slm"))
    assert guard_path(Path("/mnt/c/export"), allow_windows=True) == Path("/mnt/c/export")


def test_slm_home_refuses_mnt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_HOME", "/mnt/c/slm")
    with pytest.raises(ArtifactPathError):
        artifacts.slm_home()


def test_commit_publishes_complete_directory(tmp_path: Path) -> None:
    final = tmp_path / "packed" / "pk-abc"
    with commit_dir(final, {"id": "pk-abc"}) as tmp:
        assert tmp != final and not final.exists()
        (tmp / "train.bin").write_bytes(b"\x00\x01")
    assert (final / "train.bin").read_bytes() == b"\x00\x01"
    assert artifacts.read_manifest(final)["id"] == "pk-abc"
    assert not final.with_name("pk-abc.tmp").exists()


def test_failed_build_leaves_nothing(tmp_path: Path) -> None:
    final = tmp_path / "pk-fail"
    with pytest.raises(RuntimeError), commit_dir(final, {}) as tmp:
        (tmp / "half.bin").write_bytes(b"x")
        raise RuntimeError("power cut")
    assert not final.exists()
    assert not final.with_name("pk-fail.tmp").exists()


def test_never_overwrites_and_cleans_stale_tmp(tmp_path: Path) -> None:
    final = tmp_path / "pk-x"
    stale = final.with_name("pk-x.tmp")
    stale.mkdir()
    (stale / "junk").write_text("left by a crash")
    with commit_dir(final, {}) as tmp:
        assert not (tmp / "junk").exists()
    with pytest.raises(FileExistsError), commit_dir(final, {}):
        pass
    assert sorted(os.listdir(final)) == ["manifest.json"]
