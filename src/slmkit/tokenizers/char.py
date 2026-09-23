"""Character-level tokenizer: one token per distinct character.

The simplest possible tokenizer, and the right first one: every token is human-readable, the
vocabulary is tiny (65 characters for Shakespeare), and there is nothing to get subtly wrong.
The cost is long sequences: a model sees ~4x fewer words per context window than with BPE.
See docs/concepts/tokenization.md.

The vocabulary is built from the **train split only**. A character that appears only in
validation encodes as <unk> rather than silently getting an ID the model never trained on;
pack statistics report how many there were.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from slmkit.tokenizers.base import SPECIALS, TOKENIZER_FILE, UNK_ID, Tokenizer


class CharTokenizer(Tokenizer):
    type = "char"

    def __init__(self, chars: Iterable[str]) -> None:
        chars = list(chars)
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        if any(len(c) != 1 for c in chars):
            raise ValueError("every vocabulary entry must be a single character")
        self.chars = chars
        self._itos = list(SPECIALS) + chars
        self._stoi = {c: i + len(SPECIALS) for i, c in enumerate(chars)}

    @classmethod
    def train(cls, texts: Iterable[str]) -> CharTokenizer:
        """Vocabulary = every distinct character, sorted by code point so IDs are stable."""
        seen: set[str] = set()
        for text in texts:
            seen.update(text)
        return cls(sorted(seen))

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str) -> list[int]:
        return [self._stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids: list[int]) -> str:
        # Special tokens have no text form; generated <eos> ends a document, <unk> is shown.
        out = []
        for i in ids:
            if i >= len(SPECIALS):
                out.append(self._itos[i])
            elif i == UNK_ID:
                out.append("�")  # the Unicode replacement character
        return "".join(out)

    def save(self, directory: Path) -> None:
        payload = {"type": self.type, "specials": list(SPECIALS), "chars": self.chars}
        (directory / TOKENIZER_FILE).write_text(json.dumps(payload, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> CharTokenizer:
        payload = json.loads((directory / TOKENIZER_FILE).read_text())
        if payload.get("type") != cls.type or payload.get("specials") != list(SPECIALS):
            raise ValueError(f"{directory} does not hold a compatible char tokenizer")
        return cls(payload["chars"])
