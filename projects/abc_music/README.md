# abc_music

slmkit's first full-lifecycle project (M2): a small model that writes folk tunes in
[ABC notation](https://abcnotation.com/wiki/abc:standard), is fine-tuned to follow requests such as
"a jig in G", is graded by programs, exported, served, and can be listened to.

## Data

| | |
|---|---|
| Source | ABC transcriptions of three public-domain tune books, as shipped in the `music21` package (BSD-3-Clause), version pinned by `uv.lock` |
| Collections | Ryan's Mammoth Collection (1883, 1,059 tunes), O'Neill's Music of Ireland (1903, 2,009), Aird's Selection of Airs (1778–1803, 1,180) |
| After cleaning | 4,248 tunes, ~1.1M characters, 87 distinct characters |
| Licence | The printed books are in the **public domain**. The transcription files state no licence of their own; see [ADR 0005](../../docs/decisions/0005-abc-data-source.md) for the reasoning and the residual risk |
| Not used | The Session's data dump (its licence prohibits training language models), the Essen collection (unclear status), the Nottingham Music Database (GPL / unstated) |

**Transcribers.** These files exist because of volunteers who typed them in: **Ray Davies** and **Bob
Puckette** (Ryan's), **Henrik Norbeck**, **Bob Safranek** and **Trish O'Neil** (O'Neill's), and **Jack
Campin** (Aird's). Their credit lines are removed from the *training text*, because a model has no use
for names or email addresses, and are credited here instead.

## From raw files to documents

`documents()` (in [`project.py`](project.py), using [`abcnotation.py`](abcnotation.py)):

1. **Split** each file into tunes at `X:` lines.
2. **Clean:** keep only `R:` rhythm, `M:` meter, `L:` note length and `K:` key, plus the body. Drop
   titles (`T:`, English rather than music), transcriber and source lines (`Z:`, `B:`, `S:`, `N:`, …),
   lyrics, part labels and comments. Drop quoted strings that are not chord symbols (fingerings like
   `"4"`, notes like `"^SEGUE"`, labels like `"B MINOR"`).
3. **Normalize** headers: `Reel`/`REEL` → `reel`, `slipjig` → `slip jig`, `C|` → `2/2`, `E Minor` →
   `Em`, `A Mixolydian` → `Amix`; a missing `L:` gets ABC's default for the meter.
4. **Group** tunes that are the same tune: an identical melody opening (the diatonic steps between the
   first 16 notes, which survive transposition and small variations in rhythm), or the same distinctive
   title (a title shared by more than 3 tunes, like "Reel", is ignored). 4,248 tunes → 3,677 groups;
   the largest has 6 members, e.g. *The Sailor's Hornpipe* from two books.

Every document renders its headers in a fixed order, `R:`, `M:`, `L:`, `K:`, then the body:

```
R:reel
M:2/4
L:1/16
K:D
u(FG)|\
AFDF AFDF | AGFE .D2(EF) | GECE GECE | GFED .C2(AG) |
```

## Augmentation (train split only)

- **Transposition** by −2, −1, +1, +2 semitones, kept only if the new key has at most 3 sharps or flats
  (a D♭ reel is out of distribution for fiddle music). Transposition rewrites the ABC text: each note
  moves by the same number of letter names as the tonic and gets whatever accidental makes its pitch
  right, written only where the new key would otherwise read it wrong.
- **Header dropout** (40%): a deterministic subset of `R:`/`M:`/`K:` is removed, so the base model also
  learns to write tunes it has not been told the rhythm, meter or key of. `L:` always stays, because it
  sets every note's length.

Result: 3,829 training tunes → 11,316 training documents (2.96×), 2.85M tokens. Validation (419 tunes)
is never augmented.

**How transposition is verified.** Every transposition is played by `abc2midi`, the reference ABC
player, and every note must sound exactly *n* semitones higher. Over the whole corpus: **16,603 of
16,603 exact**, with 389 transpositions refused as unsafe (key changes mid-tune, unspellable keys, and
the ambiguous case below).

## Known pitfalls

- **`music21` is not a reliable judge of ABC pitch.** It applies an accidental only to the note it is
  written on, while the ABC standard carries it to later notes in the bar. `=g2dg` has two G naturals;
  music21 reads the second as G♯. Used as the test oracle, it wrongly flagged 16 correct
  transpositions. `abc2midi` is used instead.
- **Players disagree across octaves.** Whether `=g` (upper octave) also makes a later `G` natural
  differs between the standard (no) and `abc2midi`'s default (yes). Transposed output writes the
  accidental explicitly wherever this could matter, and tunes whose *original* depends on the
  ambiguity are not transposed.
- **Rhythm labels are patchy:** `R:` is on 99% of Ryan's, 44% of O'Neill's and ~0% of Aird's tunes.
- **Titles are not identities:** Aird's has many tunes titled "Minuet" or "Reel"; merging on those
  would glue unrelated tunes together, so titles shared by more than 3 tunes are ignored.

## Graders

*Phase C.* Planned: parses in `abc2midi`; bar durations match `M:`; ends on the tonic of `K:`; prompt
adherence after SFT; n-gram novelty against the training set.

## Experiments

| File | Model | Purpose |
|---|---|---|
| [`baseline.yaml`](experiments/baseline.yaml) | `nano` (0.85M), char | the first run: prove the pipeline (best val 1.2564 after 30M tokens, 0.01 GPU-h) |
