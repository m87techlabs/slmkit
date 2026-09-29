"""Tokenizers: `char` (M1), `bpe` (M2), `fixed` (M3). See docs/concepts/tokenization.md."""

from __future__ import annotations

import json
from pathlib import Path

from slmkit.tokenizers.base import EOS_ID, SPECIALS, TOKENIZER_FILE, UNK_ID, Tokenizer
from slmkit.tokenizers.bpe import BPETokenizer
from slmkit.tokenizers.char import CharTokenizer


def load_tokenizer(directory: Path) -> Tokenizer:
    data = json.loads((directory / TOKENIZER_FILE).read_text())
    if data.get("type") == "char":
        return CharTokenizer.load(directory)
    model = data.get("model", {})
    if model.get("type") == "BPE":  # Hugging Face's tokenizer.json format
        return BPETokenizer.load(directory)
    if model.get("type") == "WordLevel":  # an exported char tokenizer (CharTokenizer.to_hf)
        return CharTokenizer.from_hf_vocab(model["vocab"])
    raise NotImplementedError(f"unrecognised tokenizer in {directory}")


__all__ = [
    "EOS_ID",
    "SPECIALS",
    "UNK_ID",
    "BPETokenizer",
    "CharTokenizer",
    "Tokenizer",
    "load_tokenizer",
]
