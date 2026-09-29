"""ABC music: folk tunes in ABC notation, from public-domain collections (DESIGN 5.1, ADR 0005).

Source: the ABC transcriptions of three public-domain tune books that ship inside the `music21`
package, pinned by `uv.lock`:

    Ryan's Mammoth Collection (1883)      ~1,060 tunes   reels, hornpipes, jigs, clogs
    O'Neill's Music of Ireland (1903)      ~2,010 tunes   the "1850" collection
    Aird's Selection of Airs (1782-1803)   ~1,180 tunes   airs, marches, dances

The Session's data dump was the original plan and was dropped: its licence prohibits training
language models on it (ADR 0005).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slmkit.project_api import Doc, EvalPrompt, Grader, Project, SFTExample, TokenizerSpec
from slmkit.registry import register_project

from .abcnotation import (
    Key,
    NotTransposable,
    Tune,
    clean_tune,
    fingerprint,
    normalise_title,
    parse_key,
    split_tunes,
    stable_unit_hash,
    transpose,
)

COLLECTIONS = ("ryansMammoth", "oneills1850", "airdsAirs")

# SFT requests, in plain words. The vocabulary was fixed at pretraining from ABC text, which has
# no q, no capital J/N/W/X/Y and no "?": a request using them would contain <unk> tokens the
# model has never learned (test_sft_requests_use_only_known_characters). Hence lower-case starts.
MODE_WORDS = {"": "major", "m": "minor", "dor": "dorian", "mix": "mixolydian",
              "lyd": "lydian", "phr": "phrygian", "loc": "locrian"}  # fmt: skip
WITH_RHYTHM = (
    "{a} {rhythm} in {key}",
    "write {a} {rhythm} in {key}",
    "compose {a} {rhythm} in the key of {key}",
    "{a} {rhythm} in {key}, {meter} time",
    "{rhythm} in {key}",
)
WITHOUT_RHYTHM = (
    "a tune in {meter} time in {key}",
    "write a tune in {key}, {meter} time",
    "compose something in {meter} time, key of {key}",
)
# The prompt is an ABC comment line, so request + answer is still a valid ABC file.
REQUEST = "% {}\n"


def key_words(key: Key) -> str:
    return f"{key.tonic} {MODE_WORDS[key.mode]}"


def request(rhythm: str | None, meter: str, key: Key, choice: float = 0.0) -> str:
    """A plain-language request for a tune; `choice` in [0, 1) picks the phrasing."""
    templates = WITH_RHYTHM if rhythm else WITHOUT_RHYTHM
    template = templates[int(choice * len(templates))]
    a = "an" if rhythm and rhythm[0] in "aeiou" else "a"
    return REQUEST.format(template.format(a=a, rhythm=rhythm, meter=meter, key=key_words(key)))


# A title shared by more than this many tunes is generic ("Reel", "Minuet"), not an identity.
MAX_TITLE_SHARE = 3


class AbcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Transpositions added to the *train* split. Kept small: a tune moved to D-flat is out of
    # distribution for fiddle music, so each copy must also land in a common key (below).
    transpose_semitones: list[int] = Field(default_factory=lambda: [-2, -1, 1, 2])
    max_key_accidentals: int = Field(3, ge=0, le=7)
    # Fraction of training documents rendered with some of R:/M:/K: removed, so the base model
    # also learns to write tunes without being told their rhythm, meter or key (DESIGN 5.1).
    header_dropout: float = Field(0.4, ge=0.0, le=1.0)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def load_tunes(raw_dir: Path) -> list[Tune]:
    tunes = []
    for collection in COLLECTIONS:
        for path in sorted((raw_dir / collection).glob("*.abc")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for raw in split_tunes(text):
                tune = clean_tune(raw, f"{collection}/{path.stem}")
                if tune is not None:
                    tunes.append(tune)
    return tunes


def assign_groups(tunes: list[Tune]) -> dict[str, str]:
    """Group tunes that are the same tune: same melody opening, or the same distinctive title.

    Settings of one tune ("3rd Setting") and the same tune in two different books must land
    on the same side of the split, or validation measures memory. Over-merging only makes the
    split lumpier; under-merging leaks. So when in doubt, merge.
    """
    uf = _UnionFind()
    by_print: dict[tuple[int, ...], str] = {}
    titles = Counter(normalise_title(t.title) for t in tunes)
    by_title: dict[str, str] = {}
    for t in tunes:
        uf.find(t.ref)
        fp = fingerprint(t.body)
        if fp is not None:
            if fp in by_print:
                uf.union(t.ref, by_print[fp])
            by_print.setdefault(fp, t.ref)
        title = normalise_title(t.title)
        if title and titles[title] <= MAX_TITLE_SHARE:
            if title in by_title:
                uf.union(t.ref, by_title[title])
            by_title.setdefault(title, t.ref)
    roots = {t.ref: uf.find(t.ref) for t in tunes}
    return {
        ref: "g-" + hashlib.sha256(root.encode()).hexdigest()[:12] for ref, root in roots.items()
    }


@register_project("abc_music")
class AbcMusic(Project):
    Args = AbcArgs
    data_version = 1

    def ingest(self, raw_dir: Path) -> None:
        import music21  # the abc_music extra; imported here so the engine never needs it

        corpus = Path(music21.__file__).parent / "corpus"
        for collection in COLLECTIONS:
            target = raw_dir / collection
            target.mkdir()
            for path in sorted((corpus / collection).rglob("*.abc")):
                (target / path.name).write_bytes(path.read_bytes())
        (raw_dir / "SOURCE.txt").write_text(
            f"music21 {music21.__version__} corpus: {', '.join(COLLECTIONS)}\n"
            "Public-domain tune books; see projects/abc_music/README.md and ADR 0005.\n"
        )

    def documents(self, raw_dir: Path) -> Iterator[Doc]:
        tunes = load_tunes(raw_dir)
        groups = assign_groups(tunes)
        for t in tunes:
            yield Doc(
                id=t.ref,
                group=groups[t.ref],
                text=t.render(),
                meta={
                    "rhythm": t.rhythm, "meter": t.meter, "unit": t.unit, "key": t.key,
                    "body": t.body, "changes_mid_tune": t.changes_mid_tune,
                },
            )  # fmt: skip

    def _tune(self, doc: Doc) -> Tune:
        m = doc.meta
        return Tune(ref=doc.id, title="", rhythm=m["rhythm"], meter=m["meter"], unit=m["unit"],
                    key=m["key"], body=m["body"], changes_mid_tune=m["changes_mid_tune"])  # fmt: skip

    def _render(self, tune: Tune) -> str:
        """Apply header dropout deterministically: the same tune always renders the same way."""
        assert isinstance(self.args, AbcArgs)
        if stable_unit_hash(tune.ref, "dropout") >= self.args.header_dropout:
            return tune.render()
        # Drop a non-empty subset of R/M/K, chosen by hash; L: always stays (it sets durations).
        subset = 1 + int(stable_unit_hash(tune.ref, "subset") * 7)  # 1..7 as a 3-bit mask
        kept = tuple(h for bit, h in ((1, "R"), (2, "M"), (4, "K")) if not subset & bit)
        return tune.render(headers=tuple(h for h in ("R", "M", "L", "K") if h in kept or h == "L"))

    def augment(self, doc: Doc) -> Iterator[Doc]:
        """Train split only: the original plus transpositions into common keys, each possibly
        with some headers dropped."""
        assert isinstance(self.args, AbcArgs)
        tune = self._tune(doc)
        variants = [tune]
        for n in self.args.transpose_semitones:
            try:
                moved = transpose(tune, n)
            except NotTransposable:
                continue
            key = parse_key(moved.key)
            if key is not None and abs(key.fifths) <= self.args.max_key_accidentals:
                variants.append(moved)
        for v in variants:
            yield Doc(id=v.ref, group=doc.group, text=self._render(v),
                      meta={**doc.meta, "key": v.key, "body": v.body})  # fmt: skip

    def tokenizer_spec(self) -> TokenizerSpec:
        return TokenizerSpec(type="char")

    def eval_prompts(self, split: str) -> Iterator[EvalPrompt]:
        # Header prefixes: the model continues with a tune that should fit them. `meta` is what
        # the graders check against (so the same graders measure prompt adherence after SFT).
        for name, rhythm, meter, key in (
            ("reel-D", "reel", "4/4", "D"),
            ("jig-G", "jig", "6/8", "G"),
            ("hornpipe-A", "hornpipe", "4/4", "A"),
            ("air-Em", None, "3/4", "Em"),
        ):
            header = (f"R:{rhythm}\n" if rhythm else "") + f"M:{meter}\nL:1/8\nK:{key}\n"
            yield EvalPrompt(id=name, prompt=header,
                             meta={"rhythm": rhythm, "meter": meter, "key": key})  # fmt: skip

    def sft_examples(self, docs: Iterable[Doc]) -> Iterator[SFTExample]:
        """Each tune (and each transposed copy, on the train split) as request -> full tune.

        The answer always has every header the tune has, whatever header dropout did to the
        pretraining text: the model should state the rhythm, meter and key it was asked for.
        """
        for doc in docs:
            tune = self._tune(doc)
            key = parse_key(tune.key)
            if key is None:
                continue  # can't describe a key we can't parse
            ask = request(tune.rhythm, tune.meter, key, stable_unit_hash(doc.id, "request"))
            yield SFTExample(prompt=ask, completion=tune.render(),
                             meta={"rhythm": tune.rhythm, "meter": tune.meter, "key": tune.key})  # fmt: skip

    def sft_eval_prompts(self, split: str) -> Iterator[EvalPrompt]:
        """The same four requests as `eval_prompts`, in words instead of headers."""
        for p in self.eval_prompts(split):
            m = p.meta
            key = parse_key(m["key"])
            assert key is not None
            yield EvalPrompt(id=p.id, prompt=request(m["rhythm"], m["meter"], key), meta=m)

    def graders(self) -> list[Grader]:
        from .graders import GRADERS

        return list(GRADERS)
