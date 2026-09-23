"""Tokenizers: `char` (M1), `bpe` (M2), `fixed` (M3). See docs/concepts/tokenization.md."""

from __future__ import annotations

import json
from pathlib import Path

from slmkit.tokenizers.base import EOS_ID, SPECIALS, TOKENIZER_FILE, UNK_ID, Tokenizer
from slmkit.tokenizers.char import CharTokenizer


def load_tokenizer(directory: Path) -> Tokenizer:
    kind = json.loads((directory / TOKENIZER_FILE).read_text()).get("type")
    if kind == "char":
        return CharTokenizer.load(directory)
    raise NotImplementedError(f"tokenizer type {kind!r} arrives in a later milestone")


__all__ = [
    "EOS_ID",
    "SPECIALS",
    "UNK_ID",
    "CharTokenizer",
    "Tokenizer",
    "load_tokenizer",
]
