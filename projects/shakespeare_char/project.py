"""Character-level Shakespeare: slmkit's correctness reference (DESIGN 5.0).

The same ~1.1 MB file nanoGPT's `shakespeare_char` example trains on. Its published
validation loss (~1.47) is the number M1 must approach, which is the only way to tell "the
trainer is broken" apart from "the task is hard".
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slmkit.project_api import Doc, EvalPrompt, Grader, Project, TokenizerSpec
from slmkit.registry import register_project

SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Pinned so a changed upstream file fails loudly instead of silently changing the dataset.
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
RAW_FILE = "input.txt"


class ShakespeareArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The file is one continuous text, so "documents" are contiguous blocks of about this
    # many characters, cut at the next blank line (a speech boundary). Each block is its own
    # split group. ~110 blocks of 10K characters gives a validation set of whole scenes
    # rather than scattered lines.
    block_chars: int = Field(10_000, gt=0)


def chunk_text(text: str, block_chars: int) -> list[str]:
    """Cut `text` into consecutive pieces of at least `block_chars`, ending at blank lines.

    Concatenating the result gives back `text` exactly, so no character is lost or duplicated.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        cut = text.find("\n\n", start + block_chars)
        end = len(text) if cut == -1 else cut + 2
        pieces.append(text[start:end])
        start = end
    return pieces


@register_project("shakespeare_char")
class ShakespeareChar(Project):
    Args = ShakespeareArgs

    def ingest(self, raw_dir: Path) -> None:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
            payload = resp.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != SOURCE_SHA256:
            raise RuntimeError(
                f"{SOURCE_URL} changed upstream (sha256 {digest}, expected {SOURCE_SHA256}). "
                "Check the new file, then update SOURCE_SHA256 deliberately."
            )
        (raw_dir / RAW_FILE).write_bytes(payload)

    def documents(self, raw_dir: Path) -> Iterator[Doc]:
        assert isinstance(self.args, ShakespeareArgs)
        text = (raw_dir / RAW_FILE).read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text, self.args.block_chars)):
            block = f"block-{i:04d}"
            yield Doc(id=block, group=block, text=piece)

    def tokenizer_spec(self) -> TokenizerSpec:
        return TokenizerSpec(type="char")

    def eval_prompts(self, split: str) -> Iterator[EvalPrompt]:
        # Speaker names followed by a newline: the model should continue with a speech.
        for speaker in ("ROMEO", "JULIET", "First Citizen", "KING RICHARD III"):
            yield EvalPrompt(id=speaker, prompt=f"{speaker}:\n")

    def graders(self) -> list[Grader]:
        # Validation loss is this project's grade; there is no domain grader to add.
        return []
