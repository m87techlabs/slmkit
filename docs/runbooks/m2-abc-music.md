# Runbook M2: ABC music, the full lifecycle

*Companion to [`../concepts/data-preparation.md`](../concepts/data-preparation.md) (Phase A), and
later pages per phase. Project details: [`projects/abc_music/README.md`](../../projects/abc_music/README.md).
Built up phase by phase, like the M1 runbook.*

| Phase | Builds | Status |
|---|---|---|
| **A** | The ABC corpus: clean, group, transpose, header dropout | ☑ this page |
| B | BPE tokenizer, compared with char by bits per character | ☐ |
| C | Graders, `slm eval`, `slm runs compare` | ☐ |
| D | SFT: prompts from headers, loss on the answer only | ☐ |
| E | Export, serve, listen on Windows | ☐ |
| F | Sweeps, exit criteria | ☐ |

Run everything from the repo root, with the extras installed (`uv sync --all-extras`: the project
needs `music21`, and the checks need `abc2midi` from the setup script).

---

# Phase A: the ABC corpus

## A.0 The checks

```bash
make test                                       # 116 passed in ~8s
uv run pytest -q projects/abc_music             # 27 passed in <1s
make lint
```

The project tests include the note-for-note `abc2midi` check on 30 real tunes (A.4 runs it on
all of them).

---

## A.1 Build the data artifacts

```bash
uv run slm pack abc_music/baseline
```
```
abc_music/baseline:
  raw       built   raw-3b076cc50c50  ~/slm/raw/abc_music
  dataset   built   ds-f4f721a6c0e4  ~/slm/datasets/abc_music/ds-f4f721a6c0e4
  tokenizer built   tk-1030590a89ca  ~/slm/tokenizers/tk-1030590a89ca
  packed    built   pk-337567abd517  ~/slm/packed/pk-337567abd517
```

About 5 seconds. **Your IDs should match.** The data comes from the `music21` version pinned in
`uv.lock` (no download), and every augmentation choice comes from a hash, not a random seed.

```bash
uv run slm lineage pk-337567abd517
```
```
pk-337567abd517  [packed]
    vocab_size = 87
    train_tokens = 2848661
    train_docs = 11316
    val_tokens = 107031
    val_docs = 419
    train_unk_tokens = 0 · val_unk_tokens = 0
    dataset: ds-f4f721a6c0e4  [dataset]
        train_docs = 3829 · train_groups = 3309 · train_chars = 974765
        val_docs = 419 · val_groups = 368 · val_chars = 106612
        train_docs_after_augment = 11316
        raw: raw-3b076cc50c50  [raw]
            files = SOURCE.txt, airdsAirs/book1.abc, airdsAirs/book2.abc, ... (1105 total)
```

**Read it:**
- 3,829 training tunes became **11,316** documents (2.96×) through transposition. The 419 validation
  tunes are untouched: augmentation happens after the split, on train only.
- Validation is 9.9% of tunes, and exactly 10% of *groups* (368 of 3,677).
- 87 distinct characters, and no `<unk>` anywhere: every character in validation also appears in
  training.

**Why:** these are the first numbers to check on any new corpus, before any training. They show
whether the split is the size you intended and whether the augmentation did what you expected.

---

## A.2 Look at one tune and its augmented copies

```bash
D=~/slm/datasets/abc_music/ds-f4f721a6c0e4
jq -r 'select(.id | startswith("ryansMammoth/AllAboardReel#1")) | "=== \(.id)  \(.group)\n\(.text[:200])"' \
  $D/train.jsonl $D/val.jsonl
```
```
=== ryansMammoth/AllAboardReel#1  g-b8fa3870b2d7
R:reel
M:2/4
L:1/16
K:D
u(FG)|\
AFDF AFDF | AGFE .D2(EF) | GECE GECE | GFED .C2(AG) |
…
.d2.d2 .d2(ed) | d=cAB .=c2.G2 | …

=== ryansMammoth/AllAboardReel#1~t-2  g-b8fa3870b2d7
…
K:C
…
.c2.c2 .c2(dc) | c_BGA .B2.F2 | …

=== ryansMammoth/AllAboardReel#1~t+1  g-b8fa3870b2d7
…
K:Eb
…
.e2.e2 .e2(fe) | e_dBc .d2.A2 | …
```

Check by eye:
- **Same group** for all copies, so they can never be split across train and validation.
- **Headers in a fixed order**, `R:` `M:` `L:` `K:`, normalized (`Reel` → `reel`).
- **The accidentals are musically right.** `d=cAB .=c2` (C natural in D major) became `c_BGA .B2` in C
  major. The second B needs no flat sign because, by the ABC standard, the `_B` earlier in the bar
  still applies. Up a semitone into E♭ it becomes `e_dBc .d2`.
- Title, transcriber and book lines are gone (next section).

---

## A.3 Nothing personal made it into the data

The raw files credit their transcribers by name and email, in forms like
`Z:Contributed by Ray Davies, ray:davies99.freeserve.co.uk` and `Z:Transcribed by …@…`.

```bash
for s in freeserve swipnet gsp.org Davies Norbeck Safranek Transcribed Contributed @; do
  printf '%-12s %s\n' "$s" "$(cat $D/train.jsonl $D/val.jsonl | grep -c -- "$s")"
done
```
```
freeserve    0
swipnet      0
gsp.org      0
Davies       0
Norbeck      0
Safranek     0
Transcribed  0
Contributed  0
@            0
```

**Why:** a language model can reproduce anything it trains on, and names and emails have no place
in a folk-tune model. The transcribers are credited in the project README instead. A unit test
(`test_no_document_contains_an_email`) keeps this true.

---

## A.4 Every transposition, checked note for note

```bash
uv run python projects/abc_music/check_corpus.py
```
```
tunes 4,248   groups 3,677   group sizes {1: 3200, 2: 405, 3: 56, 4: 12, 5: 2, 6: 2}
keys not parseable: 4   key/meter changes mid-tune: 53
transpositions checked with abc2midi in 53s: {'exact': 16603, 'refused (unsafe to transpose)': 389}
```

Every one of the 16,603 transpositions was played by `abc2midi`, and each note sounds exactly *n*
semitones away from the original. The 389 refusals are deliberate: tunes that change key or meter
mid-way, the 4 with bagpipe or unknown keys, and a few whose original notes different ABC players
read differently.

**Why this check, and why `abc2midi`:** a transposition bug would not crash anything; it would quietly
teach the model wrong notes. Getting to 100% took three rounds, and the first "failures" turned out to
be `music21`'s misreading of ABC rather than transposition bugs. That story, and why refusing 2.3% of
cases is the right trade, is in [`data-preparation.md`](../concepts/data-preparation.md) §4.

---

## A.5 First training run: does the pipeline work end to end?

```bash
uv run slm pretrain abc_music/baseline
```

About 35 seconds (`nano`, 0.86M parameters, 30M tokens):

```
  parameters  864,256 (853,120 non-embedding + 11,136 embedding)
eval  step      0  train 4.4333  val 4.4337  (gap +0.0004)       ← ln 87 = 4.47: guessing
  measured    4,239,474 tokens/s  ·  35.3 TFLOPS  ·  MFU 31.5% …
eval  step    250  train 1.5264  val 1.5634  (gap +0.0370)  * best
eval  step    500  train 1.3023  val 1.3526  (gap +0.0503)  * best
eval  step    750  train 1.2165  val 1.2762  (gap +0.0597)  * best
eval  step    916  train 1.1929  val 1.2564  (gap +0.0635)  * best
complete: 30,015,488 tokens, best val 1.2564, 0.01 GPU-h
```

Unlike Shakespeare, validation is **still improving at the end**, with a small gap. The transposed
copies give the model enough variety that 30M tokens don't overfit it. Phase F sizes the runs properly.

The generated jig (prompt `R:jig / M:6/8 / L:1/8 / K:G`):

```
uG/-.B/|\
ABA AGE|(F>E)F f>dB | ABd ecA|BBG G2B|
cde fed|cAG FGA|BAG GAB|{c}BAB A2:|
|: (uc/d/)|\
e2d ecA|B2d f2d|{d}edc Bcd|e2a g2d|
```

`ABA AGE|` is two groups of three eighth notes, exactly 6/8. It has a repeat, a second part, and grace
notes in the right idiom. The reel prompt came out shakier, with bar lengths that don't add up; Phase
C's bar-duration grader will measure that instead of eyeballing it. The jig also plays:

```bash
# paste the sample into jig.abc with an "X:1" line on top, then:
abc2midi jig.abc -o jig.mid
```
```
writing MIDI file jig.mid        ← 178 notes; one warning about a tie cut off where sampling stopped
```

Listening on Windows is Phase E.

---

## Phase A: done when

- [x] `make test`, project tests and `make lint` pass.
- [x] `slm pack abc_music/baseline` produces `pk-337567abd517`, with 0 `<unk>` and the expected
      split sizes.
- [x] No names or emails in any document.
- [x] 16,603 of 16,603 transpositions verified note-for-note with `abc2midi`.
- [x] A `nano` model trains on it and generates recognizable, playable ABC.
- [ ] **You** have run A.1–A.5 and the output matches.
