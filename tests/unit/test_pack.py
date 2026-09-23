"""Packing: counts, dtype, <eos> placement, and a lossless decode (DESIGN 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slmkit.data.pack import open_packed, pack_split
from slmkit.project_api import Doc
from slmkit.tokenizers import EOS_ID, CharTokenizer

DOCS = [Doc(id=str(i), group=str(i), text=t) for i, t in enumerate(["abc", "ba", "cab!"])]


def test_counts_and_dtype(tmp_path: Path) -> None:
    tok = CharTokenizer.train(d.text for d in DOCS)
    stats = pack_split(DOCS, tok, tmp_path / "t.bin", append_eos=True)
    data = open_packed(tmp_path / "t.bin")
    assert data.dtype == np.uint16
    assert stats == {"tokens": 3 + 2 + 4 + 3, "docs": 3, "unk_tokens": 0}
    assert len(data) == stats["tokens"]
    assert int(data.max()) < tok.vocab_size


def test_eos_after_every_document(tmp_path: Path) -> None:
    tok = CharTokenizer.train(d.text for d in DOCS)
    pack_split(DOCS, tok, tmp_path / "t.bin", append_eos=True)
    data = open_packed(tmp_path / "t.bin").tolist()
    assert [i for i, t in enumerate(data) if t == EOS_ID] == [3, 6, 11]


def test_without_eos_decodes_to_the_concatenated_text(tmp_path: Path) -> None:
    tok = CharTokenizer.train(d.text for d in DOCS)
    pack_split(DOCS, tok, tmp_path / "t.bin", append_eos=False)
    assert tok.decode(open_packed(tmp_path / "t.bin").tolist()) == "abcbacab!"


def test_unknown_characters_are_counted(tmp_path: Path) -> None:
    tok = CharTokenizer.train(["abc"])
    stats = pack_split(DOCS, tok, tmp_path / "t.bin", append_eos=False)
    assert stats["unk_tokens"] == 1  # the "!"


def test_vocab_must_fit_uint16(tmp_path: Path) -> None:
    tok = CharTokenizer([chr(0x4E00 + i) for i in range(70_000)])
    with pytest.raises(ValueError, match="uint16"):
        pack_split(DOCS, tok, tmp_path / "t.bin", append_eos=False)
