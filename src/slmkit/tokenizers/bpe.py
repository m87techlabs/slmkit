"""Byte-pair encoding (BPE) tokenizer, trained with Hugging Face `tokenizers`.

BPE starts from single characters and repeatedly merges the most frequent adjacent pair into a
new token, until the vocabulary reaches the target size. Frequent chunks become one token, so the
same text becomes fewer, more meaningful tokens: `M:6/8` is one token instead of five, and a model
with a 512-token context sees ~1.9x more ABC than a character model with the same context.

Pre-splitting (what merges may cross): a space attaches to the chunk that follows it, GPT-2 style,
and newlines stand alone. Measured on the ABC corpus at 512 tokens: 1.87 characters per token,
against 1.55 when spaces are their own tokens (a third of all tokens were spaces) and 2.16 with no
splitting at all (tokens like ") | (" straddle bar lines, so they stop meaning anything musical).
See docs/concepts/tokenization.md section 6.

Training the merges is performance-sensitive and well understood, so slmkit uses the library for
that (CONTRIBUTING.md allows HF `tokenizers`); the model, trainer and sampler stay hand-written.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Regex, decoders, models, pre_tokenizers, trainers
from tokenizers import Tokenizer as HFTokenizer

from slmkit.tokenizers.base import EOS_ID, SPECIALS, TOKENIZER_FILE, UNK, UNK_ID, Tokenizer

_PRE_SPLIT = Regex(r" ?[^\s]+|\s")


class BPETokenizer(Tokenizer):
    type = "bpe"

    def __init__(self, hf: HFTokenizer) -> None:
        if [hf.token_to_id(s) for s in SPECIALS] != [UNK_ID, EOS_ID]:
            raise ValueError(f"special tokens must be {SPECIALS} at IDs {UNK_ID}, {EOS_ID}")
        self.hf = hf

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> BPETokenizer:
        """Learn merges from `texts` (the train split). Deterministic for the same input."""
        hf = HFTokenizer(models.BPE(unk_token=UNK))
        hf.pre_tokenizer = pre_tokenizers.Split(_PRE_SPLIT, behavior="isolated")
        hf.decoder = decoders.Fuse()  # tokens are plain substrings: decoding is concatenation
        trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size, special_tokens=list(SPECIALS), show_progress=False
        )
        hf.train_from_iterator(texts, trainer)
        return cls(hf)

    @property
    def vocab_size(self) -> int:
        return int(self.hf.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        return list(self.hf.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        # Match the char tokenizer: <eos> has no text, <unk> shows as the replacement character.
        pieces = []
        for i in ids:
            if i == EOS_ID:
                continue
            pieces.append("�" if i == UNK_ID else self.token(i))
        return "".join(pieces)

    def token(self, i: int) -> str:
        """The text of one token, for inspecting what the merges learned."""
        return str(self.hf.id_to_token(i))

    def save(self, directory: Path) -> None:
        # Hugging Face's own format: the export (M2 Phase E) can ship this file unchanged.
        self.hf.save(str(directory / TOKENIZER_FILE))

    def to_hf(self) -> HFTokenizer:
        return self.hf

    @classmethod
    def load(cls, directory: Path) -> BPETokenizer:
        return cls(HFTokenizer.from_file(str(directory / TOKENIZER_FILE)))
