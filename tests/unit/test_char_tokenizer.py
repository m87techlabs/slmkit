"""Character tokenizer: lossless round trip, special tokens, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from slmkit.tokenizers import EOS_ID, UNK_ID, CharTokenizer, load_tokenizer

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n"


def test_round_trip_is_lossless() -> None:
    tok = CharTokenizer.train([TEXT])
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_sorted_distinct_chars_after_specials() -> None:
    tok = CharTokenizer.train(["bca", "cab"])
    assert tok.vocab_size == 2 + 3
    assert tok.encode("abc") == [2, 3, 4]  # IDs 0 and 1 are <unk> and <eos>


def test_unseen_character_becomes_unk() -> None:
    tok = CharTokenizer.train(["abc"])
    assert tok.encode("abz") == [2, 3, UNK_ID]
    assert tok.decode([2, UNK_ID]) == "a�"


def test_eos_has_no_text() -> None:
    tok = CharTokenizer.train(["ab"])
    assert tok.decode([2, EOS_ID, 3]) == "ab"


def test_save_and_load(tmp_path: Path) -> None:
    tok = CharTokenizer.train([TEXT])
    tok.save(tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert loaded.vocab_size == tok.vocab_size


def test_rejects_malformed_vocab() -> None:
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])
    with pytest.raises(ValueError):
        CharTokenizer(["ab"])
