"""Pack tokenized documents into one flat `uint16` file per split.

Training reads random windows from a single long array, so documents are encoded and
concatenated end to end (optionally with <eos> between them) and written as raw `uint16`.

Why one big file rather than one file per document: the sampler needs O(1) random access to
any offset, and the OS page cache handles one large file far better than thousands of small
ones. Why `uint16`: token IDs up to 65,535 in 2 bytes, half the size of int32, and every
vocabulary slmkit plans fits. See DESIGN 4 rule 4.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from slmkit.project_api import Doc
from slmkit.tokenizers import EOS_ID, UNK_ID, Tokenizer

DTYPE = np.uint16
MAX_VOCAB = int(np.iinfo(DTYPE).max) + 1


def pack_split(
    docs: Iterable[Doc], tokenizer: Tokenizer, out: Path, *, append_eos: bool
) -> dict[str, int]:
    """Encode `docs` into `out` (raw uint16). Returns token statistics."""
    if tokenizer.vocab_size > MAX_VOCAB:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in uint16")
    n_tokens = n_docs = n_unk = 0
    with out.open("wb") as f:
        for doc in docs:
            ids = tokenizer.encode(doc.text)
            if append_eos:
                ids.append(EOS_ID)
            arr = np.asarray(ids, dtype=DTYPE)
            n_unk += int(np.count_nonzero(arr == UNK_ID))
            arr.tofile(f)
            n_tokens += arr.size
            n_docs += 1
    return {"tokens": n_tokens, "docs": n_docs, "unk_tokens": n_unk}


def open_packed(path: Path) -> np.memmap:
    """Map a packed split read-only. Nothing is loaded until a window is actually read."""
    return np.memmap(path, dtype=DTYPE, mode="r")
