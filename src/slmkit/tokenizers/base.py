"""The tokenizer interface every tokenizer type implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

TOKENIZER_FILE = "tokenizer.json"

# Special tokens always occupy the lowest IDs, in this order, for every tokenizer type, so
# engine code can rely on them without asking the tokenizer.
#   <unk>  a character never seen in the training split (only possible at encode time)
#   <eos>  end of document, appended between documents when packing
UNK, EOS = "<unk>", "<eos>"
SPECIALS = (UNK, EOS)
UNK_ID, EOS_ID = 0, 1


class Tokenizer(ABC):
    type: str

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...
