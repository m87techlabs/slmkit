"""Corpus report for abc_music: tunes, groups, and a note-for-note transposition check.

    uv run python projects/abc_music/check_corpus.py            # the whole corpus (~1 minute)
    uv run python projects/abc_music/check_corpus.py --sample 200

Every transposition is played by abc2midi (the reference ABC player) and must sound exactly n
semitones higher than the original, note for note. See docs/concepts/data-preparation.md §4 for
why abc2midi, and not music21, is the judge.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from slmkit.registry import load_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="check only this many tunes")
    args = parser.parse_args()

    import music21
    from music21 import midi as m21midi

    load_project("abc_music", {}, Path("/tmp"))
    proj = sys.modules["slmkit_projects.abc_music.project"]
    abcn = sys.modules["slmkit_projects.abc_music.abcnotation"]
    tunes = proj.load_tunes(Path(music21.__file__).parent / "corpus")
    groups = proj.assign_groups(tunes)
    sizes = Counter(Counter(groups.values()).values())
    print(f"tunes {len(tunes):,}   groups {len(set(groups.values())):,}   "
          f"group sizes {dict(sorted(sizes.items()))}")  # fmt: skip
    print(f"keys not parseable: {sum(abcn.parse_key(t.key) is None for t in tunes)}   "
          f"key/meter changes mid-tune: {sum(t.changes_mid_tune for t in tunes)}")  # fmt: skip

    def played(t: object, tmp: Path) -> list[int]:
        src, out = tmp / "t.abc", tmp / "t.mid"
        src.write_text("X:1\n" + t.render())  # type: ignore[attr-defined]
        out.unlink(missing_ok=True)
        subprocess.run(["abc2midi", str(src), "-o", str(out), "-silent"], capture_output=True,
                       check=False)  # fmt: skip
        mf = m21midi.MidiFile()
        mf.open(str(out))
        mf.read()
        mf.close()
        on = m21midi.ChannelVoiceMessages.NOTE_ON
        return [e.pitch for tr in mf.tracks for e in tr.events if e.type == on and e.velocity]

    random.seed(0)
    chosen = random.sample(tunes, min(args.sample, len(tunes))) if args.sample else tunes
    stats: Counter[str] = Counter()
    failures = []
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for t in chosen:
            for n in (-2, -1, 1, 2):
                try:
                    moved = abcn.transpose(t, n)
                except abcn.NotTransposable:
                    stats["refused (unsafe to transpose)"] += 1
                    continue
                a, b = played(t, Path(tmp)), played(moved, Path(tmp))
                if a and len(a) == len(b) and all(y - x == n for x, y in zip(a, b, strict=True)):
                    stats["exact"] += 1
                else:
                    stats["MISMATCH"] += 1
                    failures.append(f"{t.ref} {n:+d}")
    print(f"transpositions checked with abc2midi in {time.time() - start:.0f}s: {dict(stats)}")
    for f in failures[:20]:
        print("  mismatch:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
