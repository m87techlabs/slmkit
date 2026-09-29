"""Graders for generated ABC tunes. Each is a pure function (prompt, output) -> scores.

`output` is the full generated text: the prompt's header lines followed by what the model wrote.

| grader        | score          | meaning                                                       |
|---------------|----------------|---------------------------------------------------------------|
| plays         | 0 or 1         | abc2midi, the reference ABC player, plays it without errors   |
| bar_accuracy  | 0..1           | share of bars whose length matches the requested meter        |
| ends_on_tonic | 0 or 1         | the last note is the tonic of the requested key               |

"Requested" means the prompt's meter and key (`EvalPrompt.meta`) when the prompt gives them, so
the same graders measure prompt adherence after SFT; otherwise the tune's own headers.

Calibration on the human-transcribed corpus (check_corpus.py): bar accuracy 0.990 on average,
80.5% of tunes end on the tonic (many real tunes end on the third or fifth), and the handful of
tunes scoring 0 on bars have wrong meter headers in the source files. Those are the ceilings a
model should be compared with, not 1.0.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from slmkit.project_api import EvalPrompt

from .abcnotation import (
    Tune,
    bar_accuracy,
    clean_tune,
    final_pitch_class,
    note_count,
    tonic_pitch_class,
)

MIN_NOTES = 8  # a "tune" of a few notes isn't a tune


@lru_cache(maxsize=4096)
def _parse(output: str) -> Tune | None:
    return clean_tune("X:1\n" + output, "generated")


def abc2midi(abc: str) -> tuple[bytes | None, list[str]]:
    """Play a complete ABC file (with `X:`) through abc2midi: (MIDI bytes or None, errors)."""
    if shutil.which("abc2midi") is None:
        raise RuntimeError("abc2midi is not installed (scripts/setup-ml-distro.sh)")
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "t.abc", Path(tmp) / "t.mid"
        src.write_text(abc)
        # errors="replace": abc2midi quotes the input in its messages and can cut a multi-byte
        # character in half; a broken sample must not crash the grader.
        proc = subprocess.run(
            ["abc2midi", str(src), "-o", str(out)],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        errors = [ln for ln in proc.stdout.splitlines() if ln.startswith("Error")]
        return (out.read_bytes() if out.is_file() else None), errors


@lru_cache(maxsize=4096)
def _abc2midi_ok(output: str) -> bool:
    midi, errors = abc2midi("X:1\n" + output)
    return midi is not None and not errors


def plays(prompt: EvalPrompt, output: str) -> dict[str, float]:
    tune = _parse(output)
    enough = tune is not None and note_count(tune) >= MIN_NOTES
    return {"plays": float(enough and _abc2midi_ok(output))}


def bars(prompt: EvalPrompt, output: str) -> dict[str, float]:
    tune = _parse(output)
    score = bar_accuracy(tune, prompt.meta.get("meter")) if tune else None
    return {"bar_accuracy": score if score is not None else 0.0}


def tonic(prompt: EvalPrompt, output: str) -> dict[str, float]:
    """The last note is the key's home note.

    There is deliberately no "ends in key" companion: in ABC the key signature applies to every
    bare letter, so almost any ending is in key by construction (random characters score 98.7%).
    See docs/concepts/evaluation.md section 5.
    """
    tune = _parse(output)
    if tune is None:
        return {"ends_on_tonic": 0.0}
    wanted = tonic_pitch_class(prompt.meta.get("key") or tune.key)
    return {"ends_on_tonic": float(wanted is not None and final_pitch_class(tune) == wanted)}


GRADERS = [plays, bars, tonic]
