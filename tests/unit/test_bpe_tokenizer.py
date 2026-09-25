"""BPE tokenizer: lossless, deterministic, specials fixed, and merges never cross a line."""

from __future__ import annotations

from pathlib import Path

import pytest

from slmkit.config import load_experiment
from slmkit.data.pack import count_chars
from slmkit.project_api import Doc
from slmkit.tokenizers import EOS_ID, UNK_ID, BPETokenizer, load_tokenizer

CORPUS = [
    "R:jig\nM:6/8\nL:1/8\nK:G\n|:ABA AGE|(F>E)F f>dB|ABd ecA|BBG G2B:|\n",
    "R:reel\nM:4/4\nL:1/8\nK:D\n|:AFDF AFDF|AGFE D2EF|GECE GECE|GFED C2AG:|\n",
    "First Citizen:\nBefore we proceed any further, hear me speak.\n",
] * 20


def train(vocab_size: int = 80) -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size)


def test_round_trip_is_lossless() -> None:
    tok = train()
    for text in set(CORPUS):
        assert tok.decode(tok.encode(text)) == text


def test_specials_have_the_same_ids_as_the_char_tokenizer() -> None:
    tok = train()
    assert (tok.hf.token_to_id("<unk>"), tok.hf.token_to_id("<eos>")) == (UNK_ID, EOS_ID)
    assert tok.decode([*tok.encode("AB"), EOS_ID]) == "AB"  # <eos> has no text


def test_merges_compress_and_respect_the_vocab_size() -> None:
    tok = train(80)
    text = CORPUS[0]
    assert tok.vocab_size <= 80
    assert len(tok.encode(text)) < len(text)  # fewer tokens than characters


def test_training_is_deterministic() -> None:
    a, b = train(), train()
    assert a.hf.get_vocab() == b.hf.get_vocab()


def test_no_token_spans_a_newline_or_starts_mid_space() -> None:
    tok = train(200)
    for i in range(2, tok.vocab_size):
        piece = tok.token(i)
        assert piece == "\n" or "\n" not in piece
        assert not piece.startswith("  ")  # a space only ever leads a chunk


def test_unseen_characters_become_unk() -> None:
    tok = train()
    assert UNK_ID in tok.encode("A§B")
    assert tok.decode(tok.encode("A§B")) == "A�B"


def test_save_and_load_via_the_generic_loader(tmp_path: Path) -> None:
    tok = train()
    tok.save(tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(CORPUS[1]) == tok.encode(CORPUS[1])


def test_count_chars_includes_one_eos_per_document() -> None:
    docs = [Doc(id="a", group="a", text="abc"), Doc(id="b", group="b", text="de")]
    assert count_chars(docs, append_eos=True) == 7
    assert count_chars(docs, append_eos=False) == 5


def test_pipeline_builds_a_bpe_tokenizer(toy_repo: Path, slm_home: Path) -> None:
    from slmkit import pipeline

    exp = load_experiment("toy/base", ["tokenizer.type=bpe", "tokenizer.vocab_size=40"])
    packed = pipeline.ensure_packed(exp)
    tok = load_tokenizer(pipeline.ensure_tokenizer(exp).path)
    assert isinstance(tok, BPETokenizer) and tok.vocab_size <= 40
    assert (packed.path / "train.bin").stat().st_size > 0


def test_bpe_needs_a_vocab_size(toy_repo: Path, slm_home: Path) -> None:
    from slmkit import pipeline

    with pytest.raises(ValueError, match="vocab_size"):
        pipeline.ensure_tokenizer(load_experiment("toy/base", ["tokenizer.type=bpe"]))
