"""abc_music graders: known-good and known-bad fixtures, durations, and corpus calibration."""

from __future__ import annotations

import importlib
import shutil
import statistics
from fractions import Fraction
from pathlib import Path

import pytest

from slmkit.project_api import EvalPrompt
from slmkit.registry import load_project

# Loading the project registers its package; the graders module is only imported when
# `graders()` is first called, so import each module explicitly rather than looking it up.
load_project("abc_music", {}, Path("/tmp"))
abcn = importlib.import_module("slmkit_projects.abc_music.abcnotation")
graders = importlib.import_module("slmkit_projects.abc_music.graders")
proj = importlib.import_module("slmkit_projects.abc_music.project")

JIG_PROMPT = EvalPrompt(id="jig-G", prompt="R:jig\nM:6/8\nL:1/8\nK:G\n",
                        meta={"rhythm": "jig", "meter": "6/8", "key": "G"})  # fmt: skip
GOOD_JIG = JIG_PROMPT.prompt + "D|GAB AGE|FGA d2B|GAB AGE|FDF G2:|\n"
LONG_BARS = JIG_PROMPT.prompt + "D|GAB AGEF|FGA d2Bc|GAB AGE|FDF G2:|\n"
ENDS_ON_D = JIG_PROMPT.prompt + "D|GAB AGE|FGA d2B|GAB AGE|FGA D3:|\n"
GARBAGE = JIG_PROMPT.prompt + "(]]{{A%%/|:q2\n"


def tune(text: str) -> abcn.Tune:
    t = abcn.clean_tune("X:1\n" + text, "t")
    assert t is not None
    return t


def score(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for g in graders.GRADERS:
        out.update(g(JIG_PROMPT, text))
    return out


# ------------------------------------------------------------------------------ durations


@pytest.mark.parametrize(
    ("body", "units"),
    [
        ("ABc", 3),            # three eighths
        ("A2B/c/", 3),         # a quarter and two sixteenths
        ("A>B c", 3),          # broken rhythm: dotted eighth + sixteenth = 2
        ("(3ABc d", 3),        # triplet: three in the time of two
        ("[CEG]2 A", 3),       # a chord lasts as long as its notes
        ("{g}A2 B", 3),        # grace notes take no time
        ('"G"A3', 3),          # chord symbols take no time
        ("z2 A", 3),           # rests count
    ],
)  # fmt: skip
def test_bar_durations(body: str, units: int) -> None:
    t = tune(f"M:3/8\nL:1/8\nK:G\n|{body}|{body}|")
    assert [b.units for b in abcn.bars(t)] == [Fraction(units)] * 2


def test_meter_units() -> None:
    assert abcn.meter_units("6/8", "1/8") == 6
    assert abcn.meter_units("C|", "1/8") == 8  # C| is 2/2
    assert abcn.meter_units("2/4", "1/16") == 8


def test_pickup_bars_are_not_counted() -> None:
    # "D|" is an upbeat, and "G2:|" completes it: neither is judged.
    assert abcn.bar_accuracy(tune(GOOD_JIG)) == 1.0


# ------------------------------------------------------------------------------ graders


def test_known_good_tune_scores_full_marks() -> None:
    if shutil.which("abc2midi") is None:
        pytest.skip("needs abc2midi")
    assert score(GOOD_JIG) == {"plays": 1.0, "bar_accuracy": 1.0, "ends_on_tonic": 1.0}


def test_wrong_bar_lengths_are_caught() -> None:
    assert score(LONG_BARS)["bar_accuracy"] == pytest.approx(1 / 3)


def test_ending_off_the_tonic_is_caught() -> None:
    # D in G major: a note of the key, but not its home note. The typical small-model ending.
    assert score(ENDS_ON_D)["ends_on_tonic"] == 0.0


def test_garbage_does_not_play() -> None:
    if shutil.which("abc2midi") is None:
        pytest.skip("needs abc2midi")
    assert score(GARBAGE)["plays"] == 0.0


def test_requested_meter_beats_the_tunes_own() -> None:
    # A well-formed 4/4 tune is still wrong if the prompt asked for 6/8 (prompt adherence).
    reel = "R:reel\nM:4/4\nL:1/8\nK:G\n|GABc dBAG|FGAB c2BA|GABc dBAG|FDEF G4|\n"
    assert graders.bars(JIG_PROMPT, reel)["bar_accuracy"] == 0.0
    assert graders.bars(EvalPrompt(id="free", prompt=""), reel)["bar_accuracy"] == 1.0


def test_key_accidentals_decide_the_final_pitch() -> None:
    assert abcn.final_pitch_class(tune("M:2/4\nL:1/8\nK:D\n|ABcd|f4|")) == 6  # F#
    assert abcn.final_pitch_class(tune("M:2/4\nL:1/8\nK:D\n|ABc=f|f4|")) == 6  # bar reset
    assert abcn.final_pitch_class(tune("M:2/4\nL:1/8\nK:D\n|AB=f2 f2|")) == 5  # F natural


# ------------------------------------------------------------------------------ calibration


def test_graders_rate_the_human_corpus_highly() -> None:
    """The ceiling: real transcriptions must score near-perfectly on bars, or the grader is wrong."""
    music21 = pytest.importorskip("music21")
    tunes = proj.load_tunes(Path(music21.__file__).parent / "corpus")[::10]
    acc = [x for t in tunes if (x := abcn.bar_accuracy(t)) is not None]
    assert statistics.fmean(acc) > 0.97
    tonic = [abcn.final_pitch_class(t) == abcn.tonic_pitch_class(t.key) for t in tunes]
    assert 0.7 < statistics.fmean(tonic) < 0.9


@pytest.mark.skipif(shutil.which("abc2midi") is None, reason="needs abc2midi")
def test_render_sample_writes_abc_and_midi() -> None:
    """`slm export --to-windows` files: an .abc any ABC app opens, plus MIDI to listen to."""
    project = load_project("abc_music", {}, Path("/tmp"))
    files = project.render_sample(JIG_PROMPT, GOOD_JIG)
    abc = files[".abc"].decode()
    assert abc.startswith("X:1\nT:jig-G (generated)\n") and abc.endswith(GOOD_JIG)
    assert files[".mid"][:4] == b"MThd"  # the MIDI file signature
