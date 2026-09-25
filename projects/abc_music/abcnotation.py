"""ABC notation handling for the abc_music project: split, clean, normalise, fingerprint, transpose.

ABC is plain text: header lines (`M:6/8`, `K:G`) followed by a body of notes (`GAB c2d|`).
Letters A-G are notes, lower case is an octave up, `'` and `,` shift octaves, `^ _ =` are sharp,
flat and natural. The key signature (`K:`) sharpens or flattens letters implicitly, and an
explicit accidental lasts until the next bar line.

Everything here is pure Python on purpose: music21 can *read* ABC but cannot write it, so
transposition has to be done by rewriting the text. music21 is used only in the tests, as an
independent oracle that the rewritten tune has every pitch moved by exactly n semitones.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------------------ keys

LETTERS = "CDEFGAB"
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Position of each major key on the circle of fifths (number of sharps; negative = flats).
MAJOR_FIFTHS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}  # fmt: skip
# A mode's signature relative to the major key on the same tonic (A dorian = G major = 1 sharp).
MODE_OFFSET = {"": 0, "mix": -1, "dor": -2, "m": -3, "phr": -4, "loc": -5, "lyd": 1}
MODE_ALIASES = {
    "": "", "maj": "", "major": "", "ion": "", "ionian": "",
    "m": "m", "min": "m", "minor": "m", "aeo": "m", "aeolian": "m",
    "mix": "mix", "mixolydian": "mix", "dor": "dor", "dorian": "dor",
    "phr": "phr", "phrygian": "phr", "lyd": "lyd", "lydian": "lyd", "loc": "loc", "locrian": "loc",
}  # fmt: skip
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"


@dataclass(frozen=True)
class Key:
    tonic: str  # "G", "Bb", "F#"
    mode: str  # "" (major), "m", "dor", "mix", ...

    def __str__(self) -> str:
        return self.tonic + self.mode

    @property
    def fifths(self) -> int:
        return MAJOR_FIFTHS[self.tonic] + MODE_OFFSET[self.mode]

    def signature(self) -> dict[str, int]:
        """Implicit accidental per letter: +1 sharp, -1 flat."""
        n = self.fifths
        order = SHARP_ORDER if n > 0 else FLAT_ORDER
        return {letter: (1 if n > 0 else -1) for letter in order[: abs(n)]}


_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*([A-Za-z]*)")


def parse_key(value: str) -> Key | None:
    """`E Minor`, `Em`, `Amix`, `A Mixolydian`, `G major` -> Key. None if not a plain key."""
    value = re.sub(r"\s*clef=\S+", "", value)  # clef settings don't change the key
    m = _KEY_RE.match(value)
    if not m:
        return None
    tonic = m.group(1).upper() + m.group(2)
    mode_word = m.group(3).lower()
    mode = MODE_ALIASES.get(mode_word, MODE_ALIASES.get(mode_word[:3]))
    rest = value[m.end() :].strip()
    if mode is None or tonic not in MAJOR_FIFTHS or (rest and not rest.startswith("%")):
        return None
    key = Key(tonic, mode)
    return key if -7 <= key.fifths <= 7 else None


# ------------------------------------------------------------------------------------- headers

# A guitar-chord symbol: root, optional quality and extension, optional bass note.
CHORD = re.compile(r"^[A-G][#b]?(m|min|maj|M|dim|aug|sus|add|\+|o)?[0-9]*(/[A-G][#b]?)?$")

RHYTHM_ALIASES = {"slipjig": "slip jig", "double jig": "jig", "single jig": "single jig"}
METER_ALIASES = {"C": "4/4", "C|": "2/2"}


def normalise_rhythm(value: str) -> str:
    r = " ".join(value.lower().replace("-", " ").split())
    return RHYTHM_ALIASES.get(r, r)


def normalise_meter(value: str) -> str:
    v = value.strip()
    return METER_ALIASES.get(v, v)


def default_unit(meter: str) -> str:
    """ABC's rule when L: is missing: 1/16 if the meter is below 3/4, else 1/8."""
    try:
        num, den = meter.split("/")
        return "1/16" if int(num) / int(den) < 0.75 else "1/8"
    except ValueError:
        return "1/8"


@dataclass
class Tune:
    """One tune, cleaned. `body` is the notes; headers are normalised."""

    ref: str  # "<collection>/<file>#<X>" - stable, unique
    title: str
    rhythm: str | None
    meter: str
    unit: str
    key: str  # normalised when parseable, else the original value
    body: str
    changes_mid_tune: bool  # key/meter/unit change inside the body
    extra: dict[str, str] = field(default_factory=dict)

    def render(self, headers: tuple[str, ...] = ("R", "M", "L", "K")) -> str:
        """The text the model trains on: chosen headers in a fixed order, then the body."""
        values = {"R": self.rhythm, "M": self.meter, "L": self.unit, "K": self.key}
        lines = [f"{h}:{values[h]}" for h in headers if values[h]]
        return "\n".join([*lines, self.body]) + "\n"


_FIELD = re.compile(r"^([A-Za-z]):\s*(.*)$")
# Fields dropped from the training text: X (number), T (title: English, not music), Z/N/B/S/O/C/
# D/H/F (transcriber, notes, book, source, origin, composer, discography, history, file URL).
# Z: carries transcribers' names and email addresses, which must never end up in a model.
_BODY_FIELDS_KEPT = {"K", "M", "L"}  # musically meaningful changes inside the body


def split_tunes(text: str) -> list[str]:
    return [t for t in re.split(r"(?m)^(?=X:)", text) if t.startswith("X:")]


def clean_tune(raw: str, ref_prefix: str) -> Tune | None:
    """Parse one `X:` block. Returns None for blocks without a K: line or without notes."""
    lines = [ln.rstrip() for ln in raw.splitlines()]
    headers: dict[str, str] = {}
    body_start = None
    for i, line in enumerate(lines):
        m = _FIELD.match(line)
        if not m:
            continue
        tag, value = m.group(1), m.group(2).strip()
        headers.setdefault(tag, value)
        if tag == "K":
            body_start = i + 1
            break
    if body_start is None:
        return None

    body_lines = []
    changes = False
    for line in lines[body_start:]:
        if not line.strip():
            continue
        m = _FIELD.match(line)
        if m:
            if m.group(1) in _BODY_FIELDS_KEPT:
                changes = True
                body_lines.append(f"{m.group(1)}:{m.group(2).strip()}")
            continue  # P:, w:, Q:, N: ... inside the body are dropped
        code = line.split("%", 1)[0].rstrip()  # strip comments
        if code:
            body_lines.append(code)
    body = "\n".join(body_lines).strip()
    # Quoted strings that aren't chord symbols are fingerings ("4"), performance notes
    # ("^SEGUE") and section labels ("B MINOR"): annotations for a player, noise for a model,
    # and "B MINOR" would even be played as a chord. Real chord symbols ("G", "Em7") stay.
    body = re.sub(r'"([^"]*)"', lambda m: m.group(0) if CHORD.match(m.group(1)) else "", body)
    if re.search(r"\[[KML]:", body):
        changes = True
    if not re.search(r"[A-Ga-g]", _strip_non_notes(body)):
        return None

    meter = normalise_meter(headers.get("M", "4/4"))
    key = parse_key(headers["K"])
    return Tune(
        ref=f"{ref_prefix}#{headers.get('X', '?').strip()}",
        title=headers.get("T", "").strip(),
        rhythm=normalise_rhythm(headers["R"]) if headers.get("R") else None,
        meter=meter,
        unit=headers.get("L", "").strip() or default_unit(meter),
        key=str(key) if key else headers["K"].strip(),
        body=body,
        changes_mid_tune=changes,
    )


def normalise_title(title: str) -> str:
    t = re.sub(r"\(.*?\)", " ", title.lower())
    words = [w for w in re.findall(r"[a-z0-9]+", t) if w not in {"the", "a", "an"}]
    return " ".join(words)


# ------------------------------------------------------------------------------- note scanning

# Things that look like notes but are not: quoted text/chords, !decorations!, +decorations+,
# inline fields [K:...], and field lines.
_NON_NOTES = re.compile(
    r'"[^"]*"|![^!\n]*!|\+[^+\n]*\+|\[[A-Za-z]:[^\]]*\]|^[A-Za-z]:.*$', re.MULTILINE
)
_NOTE = re.compile(r"(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)")


def _strip_non_notes(body: str) -> str:
    return _NON_NOTES.sub(" ", body)


def _step(letter: str, marks: str) -> int:
    """Diatonic position: C (middle C) = 28, D = 29, ..., c = 35."""
    octave = 5 if letter.islower() else 4
    octave += marks.count("'") - marks.count(",")
    return LETTERS.index(letter.upper()) + 7 * octave


def fingerprint(body: str, n: int = 16) -> tuple[int, ...] | None:
    """The melody's shape: diatonic steps between its first n notes, ignoring accidentals,
    rhythm, grace notes and key. Two settings or transpositions of the same tune share it."""
    text = re.sub(r"\{[^}]*\}", " ", _strip_non_notes(body))  # drop grace notes
    steps = [_step(m.group(2), m.group(3)) for m in _NOTE.finditer(text)]
    if len(steps) < 12:
        return None
    steps = steps[:n]
    return tuple(b - a for a, b in itertools.pairwise(steps))


def stable_unit_hash(*parts: str) -> float:
    """A deterministic number in [0, 1) from strings: reproducible 'randomness' for augmentation."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# -------------------------------------------------------------------------------- transposition


class NotTransposable(ValueError):
    pass


_ACC = {"^^": 2, "^": 1, "=": 0, "_": -1, "__": -2}
_ACC_OUT = {2: "^^", 1: "^", 0: "=", -1: "_", -2: "__"}
_TONIC_SPELLINGS: dict[int, list[str]] = {}
for _name in MAJOR_FIFTHS:
    _TONIC_SPELLINGS.setdefault((NATURAL_PC[_name[0]] + {"#": 1, "b": -1}.get(_name[1:], 0)) % 12,
                                []).append(_name)  # fmt: skip


def transpose_key(key: Key, semitones: int) -> Key:
    """Move the tonic by `semitones`, choosing the spelling with the fewest sharps or flats."""
    pc = (NATURAL_PC[key.tonic[0]] + {"#": 1, "b": -1}.get(key.tonic[1:], 0) + semitones) % 12
    options = [Key(t, key.mode) for t in _TONIC_SPELLINGS[pc]]
    options = [k for k in options if -7 <= k.fifths <= 7]
    if not options:
        raise NotTransposable(f"no spelling of {key} moved {semitones:+d} fits in 7 accidentals")
    return min(options, key=lambda k: (abs(k.fifths), k.fifths < 0))


def _natural_pitch(step: int) -> int:
    octave, index = divmod(step, 7)
    return 12 * octave + NATURAL_PC[LETTERS[index]]


def _render_note(step: int) -> str:
    octave, index = divmod(step, 7)
    letter = LETTERS[index]
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter + "," * (4 - octave)


def _transpose_chord_symbol(symbol: str, letter_shift: int, semitones: int) -> str:
    """Guitar-chord text like "Em" or "D7/F#": move root and bass, keep the quality."""

    def move(m: re.Match[str]) -> str:
        letter, acc = m.group(2), {"#": 1, "b": -1}.get(m.group(3), 0)
        index = LETTERS.index(letter)
        pitch = NATURAL_PC[letter] + acc + semitones
        new_index = (index + letter_shift) % 7
        new_letter = LETTERS[new_index]
        new_acc = (pitch - NATURAL_PC[new_letter] + 6) % 12 - 6
        if abs(new_acc) > 1:
            raise NotTransposable(f"chord {symbol!r} would need a double accidental")
        return new_letter + {1: "#", -1: "b", 0: ""}[new_acc]

    return re.sub(r"(^|/)([A-G])([#b]?)", lambda m: m.group(1) + move(m), symbol)


def transpose(tune: Tune, semitones: int) -> Tune:
    """Rewrite a tune `semitones` higher (or lower). Raises NotTransposable if unsafe.

    Tonal transposition: every note moves by the same number of letter names as the tonic
    (G -> A is one letter), and gets whatever accidental makes the pitch come out right. An
    accidental is written only where the reader of the *new* key would otherwise get it wrong,
    which is how a human engraver would write it.
    """
    old_key = parse_key(tune.key)
    if old_key is None or tune.changes_mid_tune:
        raise NotTransposable(f"{tune.ref}: key {tune.key!r} or a mid-tune change")
    new_key = transpose_key(old_key, semitones)
    letter_shift = (LETTERS.index(new_key.tonic[0]) - LETTERS.index(old_key.tonic[0])) % 7
    if semitones < 0 and letter_shift:
        letter_shift -= 7
    old_sig, new_sig = old_key.signature(), new_key.signature()

    out: list[str] = []
    in_bar: dict[int, int] = {}  # step -> accidental in force (input), reset at bar lines
    out_bar: dict[int, int] = {}  # same, for what has been written (output)
    body = tune.body
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '"':  # chord symbol or annotation
            end = body.index('"', i + 1)
            text = body[i + 1 : end]
            if re.match(r"^[A-G]", text):
                text = _transpose_chord_symbol(text, letter_shift, semitones)
            out.append(f'"{text}"')
            i = end + 1
            continue
        if ch in "!+" and (end := body.find(ch, i + 1)) != -1 and "\n" not in body[i:end]:
            out.append(body[i : end + 1])  # decoration
            i = end + 1
            continue
        if ch == "[" and re.match(r"\[[A-Za-z]:", body[i : i + 3]):  # inline field, e.g. [P:A]
            end = body.index("]", i)
            out.append(body[i : end + 1])
            i = end + 1
            continue
        if ch == "|" or body.startswith("::", i):
            in_bar.clear()
            out_bar.clear()
            out.append(ch)
            i += 1
            continue
        m = _NOTE.match(body, i)
        if m:
            acc_text, letter, marks = m.groups()
            step = _step(letter, marks)
            if acc_text is not None:
                acc = _ACC[acc_text]
                in_bar[step] = acc
            else:
                acc = in_bar.get(step, old_sig.get(letter.upper(), 0))
                if any(s % 7 == step % 7 and s != step and a != acc for s, a in in_bar.items()):
                    # The original itself reads differently in different programs (see below):
                    # there is no single right answer to transpose, so leave the tune alone.
                    raise NotTransposable(f"{tune.ref}: accidental ambiguous across octaves")
            pitch = _natural_pitch(step) + acc + semitones
            new_step = step + letter_shift
            new_acc = pitch - _natural_pitch(new_step)
            if new_acc not in _ACC_OUT:
                raise NotTransposable(f"{tune.ref}: a note would need a triple accidental")
            implied = out_bar.get(new_step, new_sig.get(LETTERS[new_step % 7], 0))
            # Readers disagree on whether an accidental also affects the same letter in other
            # octaves (the ABC 2.1 standard says no; abc2midi's default says yes). Where it
            # would matter, write the accidental out so every reader gets the same pitch.
            ambiguous = any(
                s % 7 == new_step % 7 and s != new_step and a != new_acc for s, a in out_bar.items()
            )
            if new_acc != implied or ambiguous:
                out.append(_ACC_OUT[new_acc])
                out_bar[new_step] = new_acc
            out.append(_render_note(new_step))
            i = m.end()
            continue
        out.append(ch)
        i += 1

    return Tune(
        ref=f"{tune.ref}~t{semitones:+d}",
        title=tune.title,
        rhythm=tune.rhythm,
        meter=tune.meter,
        unit=tune.unit,
        key=str(new_key),
        body="".join(out),
        changes_mid_tune=False,
        extra={**tune.extra, "transposed": str(semitones)},
    )
