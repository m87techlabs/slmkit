"""abc_music: cleaning, keys, transposition, fingerprints, grouping, and augmentation."""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slmkit.data.split import split_documents
from slmkit.registry import load_project

PROJECT = load_project("abc_music", {}, Path("/tmp"))
abcn = sys.modules["slmkit_projects.abc_music.abcnotation"]
proj = sys.modules["slmkit_projects.abc_music.project"]

REEL = """X: 12
T: The Test Reel
Z: 1997 by A Transcriber <someone@example.org>
N: "Collected somewhere"
B: Some Book
R: Reel
M: C|
L: 1/8
K: G Major
|:"G"GABc dBgB|"D"^cdef g2fe|"4"dBAG "^SEGUE"FGAB|[1 dcBA G4:|[2 {g}f2 "Em/B"e2 z4|]
P:B
w: some lyric words
|:gfed "B MINOR"BAGF|GABc d4:|
"""


def tune(text: str = REEL) -> abcn.Tune:
    t = abcn.clean_tune(text, "col/file")
    assert t is not None
    return t


# ------------------------------------------------------------------------------ cleaning


def test_cleaning_drops_people_titles_and_notes() -> None:
    t = tune()
    text = t.render()
    assert "@" not in text and "Transcriber" not in text  # Z: carries names and emails
    assert "Test Reel" not in text and "Collected" not in text and "Some Book" not in text
    assert "lyric" not in text and "P:" not in text


def test_cleaning_normalises_headers_in_a_fixed_order() -> None:
    assert tune().render().splitlines()[:4] == ["R:reel", "M:2/2", "L:1/8", "K:G"]


def test_annotations_go_chords_stay() -> None:
    body = tune().body
    assert '"G"' in body and '"Em/B"' in body and '"D"' in body
    assert '"4"' not in body and "SEGUE" not in body and "MINOR" not in body


@pytest.mark.parametrize(
    ("value", "expected"),
    [("E Minor", "Em"), ("Em", "Em"), ("A Mixolydian", "Amix"), ("AMix", "Amix"),
     ("G major", "G"), ("Bb", "Bb"), ("D clef=treble", "D"), ("F#m", "F#m"), ("HP", None),
     ("Bn", None)],
)  # fmt: skip
def test_parse_key(value: str, expected: str | None) -> None:
    key = abcn.parse_key(value)
    assert (str(key) if key else None) == expected


def test_key_signatures() -> None:
    assert abcn.parse_key("D").signature() == {"F": 1, "C": 1}
    assert abcn.parse_key("Ador").signature() == {"F": 1}  # A dorian = G major
    assert abcn.parse_key("Gm").signature() == {"B": -1, "E": -1}


def test_rhythm_and_meter_aliases() -> None:
    assert abcn.normalise_rhythm("Slipjig") == "slip jig"
    assert abcn.normalise_rhythm("double jig") == "jig"
    assert abcn.normalise_meter("C") == "4/4"
    assert abcn.default_unit("2/4") == "1/16" and abcn.default_unit("6/8") == "1/8"


# ------------------------------------------------------------------------------ transposition


def test_transpose_up_a_tone() -> None:
    moved = abcn.transpose(tune(), 2)
    assert moved.key == "A"
    assert moved.body.startswith('|:"A"ABcd ecac|"E"^defg a2gf|')


def test_transpose_writes_naturals_the_new_key_needs() -> None:
    # C# down a tone is B natural; in F major B is flat, so it needs an explicit natural.
    assert '"C"=Bcde' in abcn.transpose(tune(), -2).body


def test_transpose_moves_chord_bass_notes() -> None:
    assert '"F#m/C#"' in abcn.transpose(tune(), 2).body


def test_refuses_mid_tune_key_change() -> None:
    changing = tune(REEL.replace("P:B", "K:D"))
    assert changing.changes_mid_tune
    with pytest.raises(abcn.NotTransposable):
        abcn.transpose(changing, 2)


def test_refuses_accidental_ambiguous_across_octaves() -> None:
    # {^g} then G: the ABC standard says G natural, abc2midi says G sharp. No right answer.
    ambiguous = tune("X:1\nM:6/8\nL:1/8\nK:Am\nz2{^g}a2 G2A2|\n")
    with pytest.raises(abcn.NotTransposable):
        abcn.transpose(ambiguous, 2)


def test_fingerprint_survives_transposition() -> None:
    t = tune()
    assert abcn.fingerprint(t.body) == abcn.fingerprint(abcn.transpose(t, -1).body)
    other = tune(REEL.replace("GABc dBgB", "GFED CEDC"))
    assert abcn.fingerprint(other.body) != abcn.fingerprint(t.body)


def _played(t: abcn.Tune, tmp: Path) -> list[int]:
    from music21 import midi as m21midi

    src, out = tmp / "t.abc", tmp / "t.mid"
    src.write_text("X:1\n" + t.render())
    out.unlink(missing_ok=True)
    subprocess.run(
        ["abc2midi", str(src), "-o", str(out), "-silent"], capture_output=True, check=False
    )
    mf = m21midi.MidiFile()
    mf.open(str(out))
    mf.read()
    mf.close()
    on = m21midi.ChannelVoiceMessages.NOTE_ON
    return [e.pitch for tr in mf.tracks for e in tr.events if e.type == on and e.velocity]


@pytest.mark.skipif(shutil.which("abc2midi") is None, reason="needs abc2midi (setup script)")
def test_transposed_tunes_play_the_right_notes(tmp_path: Path) -> None:
    """abc2midi is the reference player: every note must sound exactly n semitones higher.

    (music21 is *not* used as the judge: it does not carry an accidental to later notes in
    the bar, which the ABC standard requires, so it misreads correct transpositions.)
    """
    music21 = pytest.importorskip("music21")
    tunes = proj.load_tunes(Path(music21.__file__).parent / "corpus")
    random.seed(0)
    checked = 0
    for t in random.sample(tunes, 30):
        for n in (-2, 1):
            try:
                moved = abcn.transpose(t, n)
            except abcn.NotTransposable:
                continue
            before, after = _played(t, tmp_path), _played(moved, tmp_path)
            assert before and [b - a for a, b in zip(before, after, strict=True)] == [n] * len(
                before
            ), f"{t.ref} {n:+d}"
            checked += 1
    assert checked >= 40


# ------------------------------------------------------------------------------ groups and docs


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for i, collection in enumerate(proj.COLLECTIONS):
        (raw / collection).mkdir(parents=True)
        text = REEL.replace("X: 12", f"X: {i}")  # the same tune in every collection
        text += "\n" + REEL.replace("X: 12", "X: 99").replace("The Test Reel", f"Other {i}").replace(
            "GABc dBgB", "GFED CEDC").replace("gfed", "cdef")  # fmt: skip
        (raw / collection / "book.abc").write_text(text)
    return raw


def test_same_tune_in_different_books_is_one_group(tmp_path: Path) -> None:
    docs = list(PROJECT.documents(_raw(tmp_path)))
    groups = {d.id: d.group for d in docs}
    copies = [g for ref, g in groups.items() if ref.endswith(("#0", "#1", "#2"))]
    assert len(set(copies)) == 1
    splits, _ = split_documents(docs, val_fraction=0.3, seed=0)
    assert not {d.group for d in splits["train"]} & {d.group for d in splits["val"]}


def test_generic_titles_do_not_merge_tunes() -> None:
    a = tune(REEL.replace("The Test Reel", "Reel"))
    tunes = [a] + [
        abcn.Tune(ref=f"x#{i}", title="Reel", rhythm=None, meter="4/4", unit="1/8", key="D",
                  body=f"{'ABcd' * (i + 1)} efga bagf edcB|", changes_mid_tune=False)
        for i in range(4)
    ]  # fmt: skip
    assert len(set(proj.assign_groups(tunes).values())) == 5


def test_no_document_contains_an_email(tmp_path: Path) -> None:
    assert not any("@" in d.text for d in PROJECT.documents(_raw(tmp_path)))


def test_augmentation_adds_common_key_transpositions_deterministically(tmp_path: Path) -> None:
    doc = next(iter(PROJECT.documents(_raw(tmp_path))))
    first = list(PROJECT.augment(doc))
    assert [d.text for d in first] == [d.text for d in PROJECT.augment(doc)]  # reproducible
    keys = [d.meta["key"] for d in first]
    assert keys[0] == "G" and len(first) > 1
    assert all(abs(abcn.parse_key(k).fifths) <= 3 for k in keys)
    assert all(d.group == doc.group for d in first)


def test_header_dropout_rate_is_roughly_as_configured() -> None:
    t = tune()
    project = load_project("abc_music", {"header_dropout": 0.4}, Path("/tmp"))
    dropped = sum(
        project._render(abcn.Tune(**{**t.__dict__, "ref": f"t#{i}"})).count("K:") == 0
        or not project._render(abcn.Tune(**{**t.__dict__, "ref": f"t#{i}"})).startswith("R:")
        for i in range(2000)
    )
    assert 0.3 < dropped / 2000 < 0.5
