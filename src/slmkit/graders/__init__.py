"""Generic graders reused across projects: n-gram novelty and a parse-rate wrapper.

A grader is a pure function `(EvalPrompt, output) -> {metric: score}`, where `output` is the
prompt followed by what the model generated. Domain graders (bar lengths, legal chess moves) live
in their project; these work on any text. See docs/concepts/evaluation.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from slmkit.project_api import EvalPrompt, Grader


def completion(prompt: EvalPrompt, output: str) -> str:
    """The part the model wrote: `output` minus the prompt it was given."""
    return output.removeprefix(prompt.prompt)


def ngram_novelty(train_texts: Iterable[str], n: int = 32) -> Grader:
    """Share of the generated text's n-character windows that never occur in training data.

    1.0 means nothing longer than n-1 characters was copied; 0.0 means every window was. Short,
    common fragments ("|: A2") repeat everywhere and prove nothing, so n is long enough that a
    match means copying: 32 characters is about three bars of ABC.

    Windows are stored as 64-bit hashes in a sorted array: 2.85M training windows take ~23 MB,
    and each lookup is a binary search.
    """
    hashes = np.unique(
        np.fromiter(
            (hash(t[i : i + n]) for t in train_texts for i in range(len(t) - n + 1)),
            dtype=np.int64,
        )
    )

    def novelty(prompt: EvalPrompt, output: str) -> dict[str, float]:
        text = completion(prompt, output)
        windows = np.fromiter(
            (hash(text[i : i + n]) for i in range(len(text) - n + 1)), dtype=np.int64
        )
        if windows.size == 0:
            return {"novelty": 1.0}
        pos = np.searchsorted(hashes, windows).clip(max=len(hashes) - 1)
        copied = hashes[pos] == windows
        return {"novelty": float(1.0 - copied.mean())}

    return novelty


def parse_rate(name: str, parser: Callable[[str], object]) -> Grader:
    """Wrap any parser: 1.0 if it accepts the generated text (returns truthy, doesn't raise)."""

    def grade(prompt: EvalPrompt, output: str) -> dict[str, float]:
        try:
            ok = bool(parser(completion(prompt, output)))
        except Exception:  # noqa: BLE001 - any parser failure means "doesn't parse"
            ok = False
        return {name: float(ok)}

    return grade
