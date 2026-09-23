"""shakespeare_char: chunking is lossless, groups do not leak, the experiment resolves.

No network: these tests build a raw directory from synthetic text instead of downloading.
"""

from __future__ import annotations

from pathlib import Path

from slmkit.config import load_experiment
from slmkit.data.split import split_documents
from slmkit.registry import load_project

SPEECHES = "".join(f"SPEAKER {i}:\nline one of {i}\nline two of {i}\n\n" for i in range(400))


def _project(tmp_path: Path, block_chars: int = 500):  # type: ignore[no-untyped-def]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "input.txt").write_text(SPEECHES)
    return load_project("shakespeare_char", {"block_chars": block_chars}, tmp_path), raw


def test_chunks_reassemble_to_the_original_text(tmp_path: Path) -> None:
    project, raw = _project(tmp_path)
    docs = list(project.documents(raw))
    assert "".join(d.text for d in docs) == SPEECHES
    assert len(docs) > 10


def test_chunks_end_at_speech_boundaries(tmp_path: Path) -> None:
    project, raw = _project(tmp_path)
    docs = list(project.documents(raw))
    assert all(d.text.endswith("\n\n") for d in docs)
    assert all(d.text.startswith("SPEAKER") for d in docs)


def test_every_block_is_its_own_group_and_splits_cleanly(tmp_path: Path) -> None:
    project, raw = _project(tmp_path)
    docs = list(project.documents(raw))
    assert len({d.group for d in docs}) == len(docs)
    splits, _ = split_documents(docs, val_fraction=0.1, seed=0)
    assert not {d.group for d in splits["train"]} & {d.group for d in splits["val"]}


def test_eval_prompts_and_reference_experiment() -> None:
    cfg = load_experiment("shakespeare_char/ref").config
    assert cfg.data.append_eos is False
    assert cfg.train.max_tokens == 5000 * 64 * 256  # nanoGPT: 5,000 steps of 64 x 256
    project = load_project("shakespeare_char", cfg.project.args, Path("/tmp"))
    assert all(p.prompt.endswith(":\n") for p in project.eval_prompts("val"))
