"""SFT batches: one prompt + answer per row, with the loss masked on the prompt.

Pretraining reads random windows from one long stream. Supervised fine-tuning is different: each
row is exactly one example,

    ids     = [prompt tokens][answer tokens][<eos>]
    x       = ids[:-1]                               what the model sees
    y       = ids[1:], with every target that is a prompt token set to IGNORE_INDEX (-100)

so the loss (and therefore the gradient) comes only from predicting the answer and the `<eos>`
that ends it. The model is never trained to *write the request*, only to respond to it. Rows are
padded on the right to `block_size`: pad inputs are `<eos>`, pad targets are IGNORE_INDEX, and
because attention is causal, real tokens never see the padding after them.

Why mask the prompt: without it, a large share of the gradient goes into predicting prompts,
which are short, formulaic and never generated at inference time. With the mask, all of the
learning signal goes into the answer. `test_sft.py` checks that prompt positions get exactly zero
loss and zero gradient.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from slmkit.model import IGNORE_INDEX
from slmkit.project_api import SFTExample
from slmkit.tokenizers import EOS_ID, Tokenizer


@dataclass(frozen=True)
class EncodedExample:
    ids: np.ndarray  # prompt + answer + <eos>
    prompt_len: int


def encode_examples(
    examples: Iterable[SFTExample], tokenizer: Tokenizer, block_size: int
) -> tuple[list[EncodedExample], int]:
    """Tokenize examples; drop any that don't fit in one row. Returns (kept, dropped count).

    Prompt and answer are encoded separately and concatenated, which is exactly how generation
    sees them (the prompt is encoded alone, then the model continues).
    """
    kept: list[EncodedExample] = []
    dropped = 0
    for ex in examples:
        prompt = tokenizer.encode(ex.prompt)
        ids = [*prompt, *tokenizer.encode(ex.completion), EOS_ID]
        if len(ids) > block_size + 1:
            dropped += 1
            continue
        kept.append(EncodedExample(np.asarray(ids, dtype=np.int64), len(prompt)))
    return kept, dropped


def make_rows(batch: list[EncodedExample], block_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((len(batch), block_size), EOS_ID, dtype=np.int64)
    y = np.full((len(batch), block_size), IGNORE_INDEX, dtype=np.int64)
    for row, ex in enumerate(batch):
        n = len(ex.ids) - 1
        x[row, :n] = ex.ids[:-1]
        y[row, :n] = ex.ids[1:]
        # Position j predicts token j + 1. Targets j + 1 < prompt_len are prompt tokens: masked.
        y[row, : max(0, ex.prompt_len - 1)] = IGNORE_INDEX
    return x, y


class SFTSampler:
    """Random examples, one per row, with resumable RNG state (like RandomWindowSampler)."""

    def __init__(
        self, examples: list[EncodedExample], block_size: int, batch_size: int, seed: int
    ) -> None:
        if not examples:
            raise ValueError("no SFT examples fit in block_size")
        self.examples = examples
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        picks = self.rng.integers(0, len(self.examples), size=self.batch_size)
        return make_rows([self.examples[i] for i in picks], self.block_size)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
