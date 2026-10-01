# Runbook M2: ABC music, the full lifecycle
<!-- slm-studio: projects=abc_music -->

*Companion to [`../concepts/data-preparation.md`](../concepts/data-preparation.md) (Phase A), and
later pages per phase. Project details: [`projects/abc_music/README.md`](../../projects/abc_music/README.md).
Built up phase by phase, like the M1 runbook.*

| Phase | Builds | Status |
|---|---|---|
| **A** | The ABC corpus: clean, group, transpose, header dropout | ☑ |
| **B** | BPE tokenizer, compared with char by bits per character | ☑ |
| **C** | Graders, `slm eval`, `slm runs compare` | ☑ |
| **D** | SFT: prompts from headers, loss on the answer only | ☑ |
| **E** | Export, serve, listen on Windows | ☑ |
| **F** | Sweeps, exit criteria | ☑ this page |

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

---

# Phase B: BPE, and comparing tokenizers fairly

*Concepts: [`../concepts/tokenization.md`](../concepts/tokenization.md) §6. Code:
[`src/slmkit/tokenizers/bpe.py`](../../src/slmkit/tokenizers/bpe.py).*

## B.0 The checks

```bash
make test                                            # 126 passed in ~8s
uv run pytest -q tests/unit/test_bpe_tokenizer.py    # 10 passed
```

The BPE tests cover a lossless round trip on both music and English, special tokens at the same IDs
as the char tokenizer, deterministic training, no token spanning a newline, and `<unk>` for unseen
characters.

---

## B.1 Build a BPE tokenizer: one line in the experiment

`experiments/bpe512.yaml` is `baseline.yaml` with one section changed:

```yaml
tokenizer:
  type: bpe
  vocab_size: 512
```

```bash
uv run slm pack abc_music/bpe512
```
```
abc_music/bpe512:
  raw       exists  raw-3b076cc50c50  …
  dataset   exists  ds-f4f721a6c0e4  …      ← the same data: only the tokenizer and packing change
  tokenizer built   tk-53942a2f8d7a  …
  packed    built   pk-c0bc3a003af2  …
```
```bash
uv run slm lineage pk-c0bc3a003af2 | head -8
```
```
pk-c0bc3a003af2  [packed]  …
    vocab_size = 512
    train_tokens = 1531524     ← char: 2,848,661 for the same text (both include one <eos> per tune)
    train_docs = 11316
    val_tokens = 57571         ← char: 107,031
    val_docs = 419
```

**Why only the tokenizer and pack rebuild:** the dataset artifact's ID doesn't depend on the tokenizer
config, so it is reused. Only the stages downstream of the change run, the same way Terraform only
touches the resources a change affects.

---

## B.2 Look at what BPE learned

```bash
uv run python -c "
from pathlib import Path; from slmkit.tokenizers import load_tokenizer
t = load_tokenizer(Path.home() / 'slm/tokenizers/tk-53942a2f8d7a')
s = 'R:jig\nM:6/8\nL:1/8\nK:G\n|:ABA AGE|(F>E)F f>dB|'
print([t.token(i) for i in t.encode(s)])
print(sorted((t.token(i) for i in range(2, t.vocab_size)), key=len, reverse=True)[:15])"
```
```
['R:jig', '\n', 'M:6/8', '\n', 'L:1/8', '\n', 'K:G', '\n', '|:', 'AB', 'A', ' A', 'GE', '|(', 'F', '>E', ')', 'F', ' f>', 'dB|']
['R:hornpipe', 'R:hornp', 'L:1/16', 'R:reel', 'L:1/8', 'M:6/8', 'L:1/1', 'M:2/2', 'M:2/4', 'R:ree', 'M:4/4', 'R:jig', ' A2:|', ' G2:|', ' gfed']
```

- Each header line is **one token**. The model can no longer misspell `M:6/8`.
- A space always *leads* a token (`' A'`, `' f>'`) and never ends one: that is the GPT-2-style
  pre-splitting, chosen after measuring three schemes (tokenization.md §6).
- It found **phrase endings** (`' A2:|'`, `' G2:|'`) and a **scale run** (`' gfed'`) on its own,
  purely because ABC repeats them.

---

## B.3 Train the A/B run

```bash
uv run slm pretrain abc_music/bpe512
```

About 25 seconds:

```
  parameters  918,656 (853,120 non-embedding + 65,536 embedding)     ← 512 × 128 embedding
eval  step      0  train 6.1795  val 6.1846  (gap +0.0051, 4.799 bpc)  * best
eval  step    250  train 2.7556  val 2.8440  (gap +0.0884, 2.207 bpc)  * best
eval  step    500  train 2.3538  val 2.5123  (gap +0.1586, 1.950 bpc)  * best
eval  step    750  train 2.1907  val 2.3850  (gap +0.1943, 1.851 bpc)  * best
eval  step    916  train 2.1510  val 2.3531  (gap +0.2021, 1.826 bpc)  * best
complete: 30,015,488 tokens, best val 2.3531, 0.01 GPU-h
```

Every eval line now ends with **bpc**, bits per character. The per-token loss (2.35) can't be compared
with the char model's (1.26); bpc can.

The generated jig (prompt `R:jig / M:6/8 / L:1/8 / K:G`), as well-formed as the char model's:

```
(uc/c/)|d>ed cBA|B2e BGF|Ged cBA|
G>AA A>cB|AGG ABc|dcA AFG|c>BA A2:|
|:(uA/B/)|cfc Acc|Bfb afd|ecA B2A|cAc fdB|
```

---

## B.4 Compare the two, on the full validation split

```bash
uv run slm runs list
uv run python scripts/val_metrics.py run-749d    # char   (use your own IDs from runs list)
uv run python scripts/val_metrics.py run-f79b    # BPE
```
```
abc-baseline (run-749d)   loss 1.2557  ppl   3.51  bpc 1.812  top1  60.8%  top5  90.8%  (vocab 87, 1.00 chars/token)
abc-bpe512   (run-f79b)   loss 2.3504  ppl  10.49  bpc 1.824  top1  42.0%  top5  70.1%  (vocab 512, 1.86 chars/token)
```

**Read the last-but-one column first.** Loss, perplexity and accuracy are *per token*, and a BPE
token is 1.86 characters, so BPE looks three times worse by perplexity while being essentially equal
in bits per character: **1.812 vs 1.824**. One seed each can't separate those; the Phase F sweep runs
three.

**Why this matters beyond ABC:** comparing models across tokenizers by per-token loss or perplexity is
one of the most common mistakes in language-model comparisons. Bits per character (or per byte) is the
fair unit, and slmkit now reports it everywhere a loss is reported.

---

## Phase B: done when

- [x] `make test` (126), `make test-gpu` (4) and `make lint` pass.
- [x] `slm pack abc_music/bpe512` builds a 512-token BPE that round-trips the whole corpus losslessly.
- [x] The trainer reports bpc on every eval; `val_metrics.py` reports it per run.
- [x] Char vs BPE compared at equal compute: 1.812 vs 1.824 bpc.
- [ ] **You** have run B.1–B.4 and the output matches.

---

# Phase C: graders and `slm eval`

*Concepts: [`../concepts/evaluation.md`](../concepts/evaluation.md). Code:
[`src/slmkit/eval/runner.py`](../../src/slmkit/eval/runner.py),
[`src/slmkit/graders/`](../../src/slmkit/graders/),
[`projects/abc_music/graders.py`](../../projects/abc_music/graders.py).*

## C.0 The checks

```bash
make test                                                                      # 151 passed in ~10s
uv run pytest -q tests/unit/test_eval.py projects/abc_music/tests/test_graders.py   # 25 passed
make test-gpu                                                                  # 4 passed
```

The grader tests use hand-made tunes whose correct scores are known: a well-formed jig scores
1/1/1, a tune with two over-long bars scores exactly ⅓ on bars (the pickup and its completion
aren't judged), a jig in G ending on D scores 0 on tonic, and garbage doesn't play.

---

## C.1 Grade the graders first

A grader is only trustworthy if real, human-written tunes score near the top. The calibration
test runs the bar and tonic graders over every tenth corpus tune:

```bash
uv run pytest -q projects/abc_music/tests/test_graders.py -k human_corpus
```

Over the whole corpus: **bar_accuracy 0.990** (92.9% of tunes perfect) and **80.5% end on the
tonic**, because real tunes often end on the third or fifth. So 0.8 is the target for
`ends_on_tonic`, not 1.0. The handful of corpus tunes scoring 0 on bars have wrong headers in the
source files: O'Neill's #316 says 3/4, but every bar holds eight eighth notes. The grader was right.

**Why:** a grader bug looks exactly like a model weakness. Calibrating on known-good data is the
only way to tell them apart. See evaluation.md §3.

---

## C.2 Evaluate a model

```bash
uv run slm eval run-749d          # your char baseline run (IDs from `slm runs list`)
```
```
run-749d028accd4 (abc-baseline) · best · step 916 · 3 seeds × 200 samples
  model    seed 0: ended 0.880  length 287.170  plays 0.630  bar_accuracy 0.712  ends_on_tonic 0.240  novelty 0.998
  …
metric                       model    baseline (token frequencies)
ended                0.907 ± 0.025                   0.923 ± 0.018
length            275.915 ± 14.369                215.237 ± 23.074
plays                0.647 ± 0.014                   0.063 ± 0.018
bar_accuracy         0.708 ± 0.012                   0.047 ± 0.012
ends_on_tonic        0.212 ± 0.030                   0.152 ± 0.006
novelty              0.998 ± 0.001                   1.000 ± 0.000

report: ~/slm/runs/run-749d028accd4/eval/ev-144df0a282.json
```

About 25 seconds on the GPU: 600 samples from the model, 600 from the baseline, every one graded.

**Read it against the baseline column,** which is tokens drawn at random by training frequency:

- `plays` and `bar_accuracy` are 10× and 15× the baseline: real learning.
- `ended` roughly matches the baseline for a trivial reason: random draws hit `<eos>` within 600 tokens ~91%
  of the time. On its own it says nothing.
- `novelty` is ~1.0 for random text too: a guard against copying, not a score.
- `ends_on_tonic` barely beats the baseline's ~1-in-7 (C.4).

**Run it again** and you get `ev-144df0a282 already exists`: the report ID hashes the run, the
checkpoint, the settings, the prompt texts and the graders' source code, so identical evaluations are
reused, and a changed prompt or grader automatically forces a fresh one. Same seeds also give identical
numbers: evaluation is reproducible.

(These are the numbers with the corrected prompts, re-measured in Phase E; see C.6.)

---

## C.3 Compare runs side by side

```bash
uv run slm eval run-f79b                       # the BPE run
uv run slm runs compare run-749d run-f79b --prompts headers
```
```
run                                   run-749d028a                      run-f79bc951
name                                  abc-baseline                        abc-bpe512
stage                                     pretrain                          pretrain
model                                         nano                              nano
tokenizer                                     char                           bpe 512
seed                                          1337                              1337
tokens                                       30.0M                             30.0M
GPU-h                                         0.01                              0.01
best val loss                               1.2564                            2.3531
best val bpc                                     -                             1.826
eval              ev-144df0a282 (headers, 3 seeds)  ev-fedddba9a1 (headers, 3 seeds)
ended                                0.907 ± 0.025                     0.950 ± 0.013
length                            275.915 ± 14.369                  302.323 ± 15.737
plays                                0.647 ± 0.014                     0.492 ± 0.026
bar_accuracy                         0.708 ± 0.012                     0.733 ± 0.008
ends_on_tonic                        0.212 ± 0.030                     0.190 ± 0.043
novelty                              0.998 ± 0.001                     0.987 ± 0.004
```

`--prompts headers` picks each run's latest header-prompt report; without it, `compare` shows the
latest report of any kind.

(The char run shows `-` for bpc only because it was trained before the trainer reported bpc; its
full-validation figure is 1.812.)

Bits per character called char and BPE a tie. The graders don't: char plays far more often (a gap of
about eight times the sampling spread), BPE gets slightly more bars right (about twice the spread, the
edge of what 3 seeds resolve), and BPE copies slightly more (1.3% of 32-character windows). Whether
these survive different *training* seeds is Phase F's question.

---

## C.4 Investigate a suspicious number: ends on the tonic

A 19% tonic rate against 80% for real tunes looked like a possible grader bug, so the endings were
counted directly (80 samples of the char model, with the original 4/4 prompts of C.6):

```
reel-D      [('ended D', 5), ('ended A', 4), ('ended F#', 4), ('ended G', 2), ('ended E', 2), ('cut   G', 1)]
jig-G       [('ended G', 8), ('ended A', 4), ('ended C', 3), ('ended E', 2), ('cut   C', 1), ('ended D', 1)]
hornpipe-A  [('ended E', 5), ('ended F#', 4), ('ended A', 3), ('ended G#', 3), ('ended C#', 2), ('ended D', 2)]
air-Em      [('ended G', 4), ('ended E', 3), ('ended D', 3), ('ended F#', 2), ('cut   C', 2), ('ended A', 1)]
```

76 of 80 finished on their own, and the grader scores each correctly. **It's a real weakness:** a
0.86M-parameter model learns local rules (bar lengths) much better than a long-range one (end on the
key stated at the top).

The trap along the way: every ending above is a note of the key, which suggested "it knows the key,
just not to come home". An `ends_in_key` grader scored the model 0.997, and random characters 0.987.
In ABC, the key signature makes almost any bare note in-key, so the metric measured the notation, not
the model. It was removed. The baseline column is what caught it (evaluation.md §5).

---

## C.5 Why samples fail to play

Of 200 samples (original prompts), 77 (char) and 101 (BPE) don't play. The `abc2midi` errors behind
them are grammar slips, not garbage:

| Cause | Char | BPE |
|---|---|---|
| broken rhythm (`>`) between notes of unequal length | 33 | 45 |
| malformed note (e.g. an accidental with no note after it) | 23 | 40 |
| a repeat opened but never closed | 24 | 24 |
| a tie to nothing | 13 | 14 |

`plays` is strict: any `Error` line fails the sample, even ones `abc2midi` recovers from.

---

## C.6 Found in Phase E: the prompts asked for meters the corpus doesn't have

The first eval prompts asked for a reel and a hornpipe in **4/4**. Listening to exported samples in
Phase E turned up a well-formed 2/4 hornpipe that scored 0 on bars, because the plain-English request
"a hornpipe in A major" never mentioned 4/4 but the grader checked it. Count what the corpus holds:

```bash
uv run python - <<'PY'
import json, collections
from slmkit import artifacts
d = artifacts.find_artifact("ds-f4f721a6c0e4")          # the dataset, from `slm lineage run-749d`
c = collections.Counter()
for line in open(d / "train.jsonl"):
    m = json.loads(line)["meta"]
    c[(m["rhythm"], m["meter"])] += 1
for r in ("reel", "hornpipe", "jig"):
    print(r, {mm: v for (rr, mm), v in c.most_common() if rr == r})
PY
```
```
reel {'2/2': 1173, '2/4': 718, '4/4': 9}
hornpipe {'2/4': 674, '2/2': 580, '4/4': 27}
jig {'6/8': 1267, '9/8': 69, '2/4': 18, '2/2': 13}
```

9 of 1,900 reels are in 4/4. Asked "a reel in D major, 4/4 time", the fine-tuned model wrote `M:2/4` in
48 of 50 samples: it can't be talked out of what its data says reels are. A base model copies an
`M:4/4` header, so header prompts hid the problem.

**Fixed (Phase E):** the prompts ask for each rhythm's most common form (reel 2/2, hornpipe 2/4 with
`L:1/16`), every plain-English request states the meter, and the eval ID hashes the prompt text. Before,
it hashed only the prompt *kind*, so changed prompts would have reused stale reports. Every number in
C.2, C.3 and D.3 was re-measured; C.4 and C.5 describe the original samples.

**Why it matters:** an eval prompt is part of the measurement. Check that the training data contains
what you ask for (evaluation.md §6).

---

## Phase C: done when

- [x] `make test` (151), `make test-gpu` (4) and `make lint` pass.
- [x] Graders calibrated on the human corpus (bars 0.990, tonic 80.5%).
- [x] `slm eval` reports 3 seeds × 200 samples, mean ± spread, next to a no-model baseline.
- [x] `slm runs compare` shows config, training result and eval side by side.
- [x] Both models beat the baseline on `plays` and `bar_accuracy`; the tonic weakness is explained.
- [ ] **You** have run C.1–C.3 and the output matches.

---

# Phase D: SFT, from continuing text to following requests

*Concepts: [`../concepts/sft.md`](../concepts/sft.md). Code:
[`src/slmkit/data/sft.py`](../../src/slmkit/data/sft.py),
[`src/slmkit/train/stage.py`](../../src/slmkit/train/stage.py). Decision:
[ADR 0006](../decisions/0006-training-stages-and-sft-api.md).*

## D.0 The checks

```bash
make test                                                                          # 164 passed
uv run pytest -q tests/unit/test_sft.py projects/abc_music/tests/test_sft_requests.py   # 13 passed
```

The one DESIGN requires: `test_prompt_positions_get_zero_loss_and_zero_gradient`. The prompt's
positions contribute exactly **zero** loss and **zero** gradient; the answer's don't. Also covered:
SFT starts from the pretrained weights (compared tensor by tensor), resumes like pretraining, and
every request template uses only characters the frozen vocabulary knows.

---

## D.1 What the model is trained on

```bash
uv run python -c "
from pathlib import Path
from slmkit.registry import load_project
from slmkit.data.split import read_docs
p = load_project('abc_music', {}, Path('/tmp'))
for e in p.sft_eval_prompts('val'): print(repr(e.prompt))
ex = list(p.sft_examples(read_docs(Path.home() / 'slm/datasets/abc_music/ds-f4f721a6c0e4/train.jsonl')))
print(len(ex), 'examples'); print(ex[5].prompt + ex[5].completion[:120])"
```
```
'% a reel in D major\n'
'% a jig in G major\n'
'% a hornpipe in A major\n'
'% a tune in 3/4 time in E minor\n'
11312 examples
% compose a reel in the key of Bb major
R:reel
M:2/4
L:1/16
K:Bb
vB2(Bd) fBdf | bdfb Td'2(c'b) | gbfb dfbg |\
```

The request is an ABC comment line (`%`), so request plus answer is still valid ABC. The answer
always states every header. The phrasing varies across 8 templates, all lower-case, because the
vocabulary has no capital `W` or `?` (concepts/sft.md §1).

---

## D.2 Fine-tune

The experiment enables SFT in its `sft:` section:

```yaml
sft:
  enabled: true
  lr: 3.0e-4                # a third of pretraining's peak
  epochs: 3
```

That section is **not** part of the pretrained run's identity, so turning SFT on doesn't start a new
pretraining run.

```bash
uv run slm sft abc_music/baseline
```

About 27 seconds:

```
  sft: 407 train examples longer than block_size were skipped
  sft: 8 val examples longer than block_size were skipped
run run-9b890db6dced  (abc_music/baseline, abc-baseline-sft)
  stage       sft, starting from run-749d028accd4 (ckpt/best); lr 0.0003, 3 epochs
  per step    16,384 tokens = 32 x 512
  budget      16,750,080 tokens = 1.40e+14 FLOPs
eval  step      0  train 1.2269  val 1.2809  (gap +0.0540)  * best
eval  step    250  train 1.1287  val 1.1863  (gap +0.0577)  * best
eval  step    500  train 1.0810  val 1.1467  (gap +0.0657)  * best
eval  step   1000  train 1.0401  val 1.1092  (gap +0.0691)  * best
complete: 16,760,832 tokens, best val 1.1092, 0.01 GPU-h
```

- **It starts near 1.28, not 4.4:** it begins from the pretrained model, not random weights.
- The loss is over **answers only**, and the answers' headers are predictable from the request, so it
  isn't comparable with pretraining's 1.256.
- No bpc: bits per character isn't defined for a masked loss.

The sample printed at step 0 (before any fine-tuning) and at the end, for "a jig in G major":

```
step 0 (pretrained model)                  step 1023 (fine-tuned)
% a jig in G major                         % a jig in G major
M:6/8                                      R:jig
L:1/8                                      M:6/8
K:D            ← wrong key, no R:          L:1/8
F EFG A2 B | cde d2 B AFC | …              K:G
                                           D|DFD DFG | ABA ABA | G2 E GED | AFA ABA |
                                           DED GEC | DBB ABd | cBA GEG | AFA A2 :|
```

`slm run abc_music/baseline` now does both stages: it skips pretraining (complete), then skips SFT
(complete).

---

## D.3 Measure it: three ways of asking

```bash
uv run slm eval run-749d                   # base model, header prompts
uv run slm eval run-749d --prompts sft     # base model, asked in words (no SFT)
uv run slm eval run-9b89                   # SFT model, asked in words (auto for an SFT run)
```

| | Base + header prompt | Base + words | **SFT + words** |
|---|---|---|---|
| plays | 0.647 ± 0.014 | 0.418 ± 0.032 | **0.730** ± 0.048 |
| bar_accuracy | 0.708 ± 0.012 | 0.256 ± 0.020 | **0.851** ± 0.023 |
| ends_on_tonic | 0.212 ± 0.030 | 0.137 ± 0.003 | **0.340** ± 0.026 |
| ended | 0.907 ± 0.025 | 0.817 ± 0.068 | **0.982** ± 0.006 |

```bash
uv run slm runs compare run-749d run-9b89 --prompts sft
```
```
run                               run-749d028a                  run-9b890db6
name                              abc-baseline              abc-baseline-sft
stage                                 pretrain                           sft
model                                     nano                          nano
tokenizer                                 char                          char
seed                                      1337                          1337
tokens                                   30.0M                         16.8M
GPU-h                                     0.01                          0.01
best val loss                           1.2564                        1.1092
best val bpc                                 -                             -
eval              ev-b921034aed (sft, 3 seeds)  ev-0ddc32e2a9 (sft, 3 seeds)
ended                            0.817 ± 0.068                 0.982 ± 0.006
length                        356.162 ± 23.809               259.345 ± 4.707
plays                            0.418 ± 0.032                 0.730 ± 0.048
bar_accuracy                     0.256 ± 0.020                 0.851 ± 0.023
ends_on_tonic                    0.137 ± 0.003                 0.340 ± 0.026
novelty                          0.981 ± 0.006                 0.998 ± 0.001
```

**Reading it:**
- **Base + words** collapses: without SFT, the model doesn't understand a request.
- **SFT + words passes parity** with base + headers, the M2 exit criterion (DESIGN §5.1), and goes past
  it on plays and bars. Part of that is more training: SFT adds 16.8M tokens on the same tunes.
- It also **finishes** 98% of the time and ends on the tonic more often. SFT examples are always
  complete tunes.
- These are the Phase E re-measurements (C.6). The first measurement showed bare parity (bars 0.689 vs
  0.677) because its requests asked for 4/4 reels and hornpipes without saying so.

---

## Phase D: done when

- [x] `make test` (164) and `make lint` pass; the mask test proves zero loss and gradient on prompts.
- [x] `slm sft` fine-tunes from the pretrained best checkpoint, resumably; `slm run` chains both.
- [x] SFT + plain-language requests reaches parity with base + headers on plays and bars (re-measured
      in Phase E: it exceeds it).
- [ ] **You** have run D.1–D.3 and the output matches.

---

# Phase E: export, serve, listen

*Concepts: [`../concepts/serving.md`](../concepts/serving.md). Code:
[`src/slmkit/export/`](../../src/slmkit/export/), [`src/slmkit/serve/app.py`](../../src/slmkit/serve/app.py).
Decision: [ADR 0007](../decisions/0007-model-export-and-render-hook.md). File formats:
[`../MODEL.md`](../MODEL.md) §6.*

## E.0 The checks

```bash
make test                                                                       # 188 passed
uv run pytest -q tests/unit/test_export.py tests/unit/test_export_hf.py tests/unit/test_serve.py   # 18 passed
```

The one DESIGN requires is `test_transformers_matches_slmkit_on_the_export`: a trained toy model is
exported, loaded through `AutoModelForCausalLM` and `AutoTokenizer`, and must give the same token IDs
and logits within 1e-4 over a full context of random tokens. `test_versions_are_immutable` and
`test_serve.py` cover the rest of this page.

---

## E.1 Export the fine-tuned model

```bash
uv run slm export run-9b89 --name abc-folk --version 1
```
```
checking parity on 1.tmp ...
  slmkit round trip: max |Δlogit| = 0.0e+00 over 94 tokens
  transformers 5.17.0: max |Δlogit| = 0.0e+00, tokenizer IDs match: True
exported abc-folk:1 -> ~/slm/models/abc-folk/1
```

Before the directory existed under its final name, it was loaded back twice, by slmkit and by
`transformers`, and both gave exactly the checkpoint's logits (serving.md §4). A failed check would
have left nothing behind.

Also export the base model, and the BPE model to prove the other tokenizer type:

```bash
uv run slm export run-749d --name abc-folk-base --version 1
uv run slm export run-f79b --name abc-folk-bpe --version 1
uv run slm models list
```
```
MODEL                STAGE        PARAMS RUN                 STEP  HF PARITY  CREATED
abc-folk:1           sft         864,256 run-9b890db6dced    1000    0.0e+00  2026-09-28T19:43:09-05:00
abc-folk-base:1      pretrain    864,256 run-749d028accd4     916    0.0e+00  2026-09-28T19:43:12-05:00
abc-folk-bpe:1       pretrain    918,656 run-f79bc95199ab     916    0.0e+00  2026-09-28T19:43:14-05:00
```

**Look inside:**

```bash
ls -l ~/slm/models/abc-folk/1
cat ~/slm/models/abc-folk/1/config.json
less ~/slm/models/abc-folk/1/MODEL_CARD.md
```
```
-rw-r--r-- 1 you you    2910 MODEL_CARD.md
-rw-r--r-- 1 you you     641 config.json
-rw-r--r-- 1 you you     122 generation_config.json
-rw-r--r-- 1 you you    1972 manifest.json
-rw-r--r-- 1 you you 3461056 model.safetensors
-rw-r--r-- 1 you you    2068 tokenizer.json
-rw-r--r-- 1 you you     167 tokenizer_config.json
```

- `model.safetensors` is 3.5 MB: 864,256 parameters × 4 bytes. The run's checkpoint is 10 MB,
  because it also holds AdamW's two running averages.
- `config.json` says `"architectures": ["LlamaForCausalLM"]`, `"hidden_size": 128`,
  `"tie_word_embeddings": true`: slmkit's settings under Hugging Face's names.
- The card's example prompt is `% a reel in D major, 2/2 time`, and its evaluation table is the SFT
  eval from D.3 (`ev-0ddc32e2a9`: plays 0.730, bar_accuracy 0.851). A base model's card reports its
  header-prompt eval instead.

**Why:** a checkpoint is the trainer's private state; an export is what everyone else uses, so it has
to be standard, self-describing and checked (serving.md §1).

---

## E.2 Versions are immutable

```bash
uv run slm export run-9b89 --name abc-folk --version 1     # the same checkpoint again
uv run slm export run-749d --name abc-folk --version 1     # a different run, same version
```
```
abc-folk:1 already exported from run-9b890db6dced step 1000
error: abc-folk:1 already exists (from run-9b890db6dced step 1000); versions are immutable, use --version 2
```

The first is a no-op, like a pipeline stage whose artifact exists. The second is refused: a client
that tested against `abc-folk:1` must get the same model tomorrow (serving.md §5).

```bash
uv run slm lineage abc-folk:1
```
```
abc-folk:1  [model]  created 2026-09-28T19:43:09-05:00  git 4deaa10 (dirty)
    stage = sft
    checkpoint = best
    step = 1000
    …
    run: run-9b890db6dced  [run]  created 2026-09-28T19:00:27-05:00  git b58da0b (dirty)
        …
        parent: run-749d028accd4  [run]  created 2026-09-25T11:10:36-05:00  git 9329250 (dirty)
            …
            packed: pk-337567abd517  [packed]  …
                dataset: ds-f4f721a6c0e4  [dataset]  …
                    raw: raw-3b076cc50c50  [raw]  …
```

From the deployed model back to the tune books, through both training stages.

---

## E.3 Load it with Hugging Face `transformers`

No slmkit code involved: this is how anyone else would use the export.

```bash
uv run python - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = os.path.expanduser("~/slm/models/abc-folk/1")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
ids = tok("% a jig in G major, 6/8 time\n", return_tensors="pt", add_special_tokens=False).input_ids
torch.manual_seed(0)
out = model.generate(ids, do_sample=True)  # generation_config.json: temperature 0.8, stop at <eos>
print(tok.decode(out[0], skip_special_tokens=True))
PY
```
```
% a jig in G major, 6/8 time
R:jig
M:6/8
L:1/8
K:G
d|cBA GFE|Bcd ecA|GFE D2(G/A/)|
BGG GAB|cBA GAB|cBc edc|BGE E2:|
||e|fef dcB|cBA GAB|cBA Bcd|ecA ABG|
|efe fed|cBA ABd|cBA GED|GFG AFD:|
```

`model.generate` read the sampling defaults from `generation_config.json` and stopped at `<eos>`
(token 1) by itself. A different library, the same model.

---

## E.4 Serve it

In one terminal (or a tmux pane):

```bash
uv run slm serve --model abc-folk:1
```
```
abc-folk:1: 864,256 params, sft, from run-9b890db6dced step 1000 · device cpu
try: curl -s http://127.0.0.1:8000/info
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

In another:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/info | python3 -m json.tool
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "% a reel in D major, 2/2 time\n", "seed": 0}' | python3 -m json.tool
```
```
{"status":"ok","model":"abc-folk:1"}
…
{
    "model": "abc-folk:1",
    "completion": "R:reel\nM:2/2\nL:1/8\nK:F\n((3GAB) | c2 ((3dcB) A2 ((3def) | …",
    "finished": true,
    "prompt_tokens": 30,
    "new_tokens": 244,
    "unknown_prompt_tokens": 0,
    "seed": 0,
    "settings": {"max_new_tokens": 600, "temperature": 0.8, "top_k": 0, "top_p": 1.0},
    "seconds": 0.5479,
    "tokens_per_second": 445.3
}
```

Half a second on the CPU. Notice `K:F`: asked for D major, this sample is in F. The model follows the
key most of the time, not always, and the graders measure how often. (`"seed": 0` on the CPU; the same
seed on a GPU gives a different tune, because the two random-number generators differ.)

**Try to break it:**

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "% Write a jig?\n", "seed": 0, "max_new_tokens": 60}' | python3 -m json.tool | grep unknown
curl -s -w '\nHTTP %{http_code}\n' localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "% a jig", "temprature": 0.5}'
```
```
    "unknown_prompt_tokens": 2,
{"detail":[{"type":"extra_forbidden","loc":["body","temprature"],"msg":"Extra inputs are not permitted","input":0.5}]}
HTTP 422
```

`W` and `?` aren't in the vocabulary (sft.md §1), and the server says so instead of silently
answering a garbled prompt. A misspelt field is an error, not a silently ignored setting. Open
`http://127.0.0.1:8000/docs` in a Windows browser for the interactive API page (WSL forwards
localhost). Stop the server with Ctrl-C.

**Why so small:** one model, one request at a time, no auth, localhost only. Anything others can
reach gets a gateway in front (M4); see serving.md §6.

---

## E.5 Listen on Windows

```bash
uv run slm export run-9b89 --name abc-folk --version 1 --to-windows
```
```
abc-folk:1 already exported from run-9b890db6dced step 1000
sampling 2 per prompt on cuda ...
  reel-D-0         .abc .mid  ended=1.00 plays=1.00 bar_accuracy=1.00 ends_on_tonic=0.00
  reel-D-1         .abc .mid  ended=1.00 plays=0.00 bar_accuracy=1.00 ends_on_tonic=0.00
  jig-G-0          .abc .mid  ended=1.00 plays=1.00 bar_accuracy=1.00 ends_on_tonic=0.00
  jig-G-1          .abc .mid  ended=1.00 plays=1.00 bar_accuracy=1.00 ends_on_tonic=1.00
  hornpipe-A-0     .abc .mid  ended=1.00 plays=1.00 bar_accuracy=1.00 ends_on_tonic=0.00
  hornpipe-A-1     .abc .mid  ended=1.00 plays=0.00 bar_accuracy=1.00 ends_on_tonic=0.00
  air-Em-0         .abc .mid  ended=1.00 plays=1.00 bar_accuracy=0.33 ends_on_tonic=0.00
  air-Em-1         .abc .mid  ended=1.00 plays=1.00 bar_accuracy=0.67 ends_on_tonic=0.00
wrote /mnt/c/Users/Public/Music/slmkit/abc-folk-v1
on Windows: C:\Users\Public\Music\slmkit\abc-folk-v1
```

On Windows, open `C:\Users\Public\Music\slmkit\abc-folk-v1` in Explorer:

1. **Play `jig-G-0.mid`** (double-click; Media Player plays MIDI). Then `reel-D-0.mid` and
   `hornpipe-A-0.mid`.
2. **Play the samples that scored `plays=0.00`**, `reel-D-1.mid` and `hornpipe-A-1.mid`. `abc2midi`
   still made MIDI files, but reported errors on the way: "Malformed note" in the reel (a stray
   character mid-bar) and "Could not find note to be tied" in the hornpipe (C.5 lists the usual
   causes). Listen for whether you can hear the slip.
3. **Open `SAMPLES.md`** for every file's scores, and an `.abc` file in any ABC app or text editor to
   see the notation: `jig-G-0.abc` starts `X:1`, `T:jig-G (generated)`, then the request as a `%`
   comment and the tune.

The export's `MODEL_CARD.md` is copied alongside. The folder is a convenience copy, rewritten every
time; the model itself stays in `$SLM_HOME`.

**Why:** graders set a floor; a person listening is the other half of the exit criteria. Listening is
also how Phase E found the eval-prompt flaw (C.6): a hornpipe that sounded right scored 0 on bars.

---

## E.6 Play with it in the browser

The playground at `/` builds a request, generates, draws the tune as sheet music and plays it
(serving.md §8, ADR 0008). The viewer is copied into models at export time, so export a version that
carries it:

```bash
uv run slm export run-9b89 --name abc-folk --version 2
ls ~/slm/models/abc-folk/2/ui
uv run slm serve --model abc-folk:2
```
```
viewer.js
abc-folk:2: 864,256 params, sft, from run-9b890db6dced step 1000 · device cpu
playground: http://127.0.0.1:8000/   (API docs: http://127.0.0.1:8000/docs)
```

(`abc-folk:1` was exported before viewers existed, so it has no `ui/`. To use the repo's viewer with it
anyway: `uv run slm serve --model abc-folk:1 --ui projects/abc_music/web`.)

Open **http://localhost:8000/** in a browser on Windows. Then:

1. **Ask for** "a jig (6/8)" **in the key of** "G major": the prompt becomes `% a jig in G major, 6/8
   time`. Generate (or Ctrl+Enter).
2. Press **play** under the result: the notes highlight as they sound. Try other instruments, and the
   BPM box to slow it down.
3. **Reuse this seed** and Generate again: the same tune, note for note. Change the temperature to 0.3,
   then 1.2, with the same seed, and compare.
4. Change the key to "Bb major" or the form to "a slip jig (9/8)", or type a request of your own. Try
   `% Write a jig?`: the page warns that 2 prompt characters aren't in the vocabulary.
5. **Copy link** gives a URL that reproduces the result, e.g.
   `http://localhost:8000/?prompt=%25%20a%20jig%20in%20G%20major%2C%206%2F8%20time%0A&seed=3&go=1`.

Also try the base model, `uv run slm serve --model abc-folk-base:1 --ui projects/abc_music/web`: the
builder writes ABC headers instead of words, because that's what a base model understands.

**Check it without clicking** (this is how the screenshot in serving.md §8 was made): with the server
running, from WSL,

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu \
  --window-size=1400,1000 --virtual-time-budget=20000 --screenshot='C:\Users\Public\playground.png' \
  "http://localhost:8000/?prompt=%25%20a%20jig%20in%20G%20major%2C%206%2F8%20time%0A&seed=3&go=1"
```

and open `C:\Users\Public\playground.png`. (Microsoft Edge works the same way:
`/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`.)

**Measured:** one tune takes about 0.6 s on the CPU. The first version took 3.4 s, because PyTorch's
CPU threads multiplied across server threads; `slm serve --threads` (default 2) fixed it (serving.md §6).

---

## Phase E: done when

- [x] `make test` (188), `make test-gpu` (4) and `make lint` pass.
- [x] `slm export` writes an HF-format model and refuses to publish unless slmkit and `transformers`
      reproduce the checkpoint's logits (measured difference: 0.0, char and BPE).
- [x] Versions are immutable, and `slm lineage` reaches raw data from a model.
- [x] `slm serve` answers `/health`, `/info` and `/generate`, validates requests, and reports unknown
      prompt characters.
- [x] `--to-windows` writes `.abc` and `.mid` files with a graded index.
- [x] `slm serve` has a browser playground; `abc_music`'s viewer draws and plays tunes (E.6, ADR 0008).
- [x] **You** have run E.1–E.5, and listened to a generated tune on Windows: it sounds like a tune.

---

# Phase F: sweeps and exit criteria

*Concepts: [`../concepts/experiments.md`](../concepts/experiments.md). Code:
[`src/slmkit/eval/summary.py`](../../src/slmkit/eval/summary.py), the experiment YAMLs in
[`projects/abc_music/experiments/`](../../projects/abc_music/experiments/), and
[`projects/abc_music/check_sweep.py`](../../projects/abc_music/check_sweep.py).*

## F.0 The checks

```bash
make test                                        # 191 passed
uv run pytest -q tests/unit/test_summary.py      # 3 passed
```

`test_summary_averages_over_training_seeds` trains the toy project with two seeds and checks that the
summary's mean and spread are exactly `statistics.fmean` and `stdev` of the per-run values.
`test_run_trains_then_evaluates_each_stage` checks that `slm run` evaluates what it trained and reuses
the report the second time.

---

## F.1 The experiments are YAML only

```bash
diff projects/abc_music/experiments/baseline.yaml projects/abc_music/experiments/noaug.yaml
```

Besides comments and the run name, the only difference is `transpose_semitones: []` (and `noaug`
has no `sft:` section). `micro.yaml` changes `preset: micro`, and `micro_noaug.yaml` changes both.
No engine or project code changed to add them: the framework check.

---

## F.2 Run the sweep

In tmux: about 13 minutes wall-clock, 0.13 GPU-hours in all. It resumes if interrupted; re-run the
same loop.

```bash
for exp in baseline bpe512 micro noaug micro_noaug; do
  for seed in 1337 1 2; do
    uv run slm run abc_music/$exp --set run.seed=$seed || break 2
  done
done 2>&1 | tee ~/slm/sweep-m2.log
```

Each `slm run` trains (or finds the run complete), fine-tunes if the experiment has `sft.enabled`
(only `baseline`), and evaluates each stage: 3 sampling seeds × 200 samples, or `already exists`.

**Why a loop and not a sweep tool:** every step already resumes and skips completed work, so a shell
loop is a complete, restartable sweep. A scheduler would add a dependency and solve nothing here.

---

## F.3 Read the summary

```bash
uv run slm runs summary --project abc_music
```
```
experiment         abc_music/baseline     abc_music/baseline       abc_music/bpe512        abc_music/micro  abc_music/micro_noaug        abc_music/noaug
stage                        pretrain                    sft               pretrain               pretrain               pretrain               pretrain
runs                  3 (3 evaluated)        3 (3 evaluated)        3 (3 evaluated)        3 (3 evaluated)        3 (3 evaluated)        3 (3 evaluated)
seeds                        1,2,1337               1,2,1337               1,2,1337               1,2,1337               1,2,1337               1,2,1337
GPU-h                            0.02                   0.02                   0.01                   0.03                   0.03                   0.02
eval prompts                  headers                    sft                headers                headers                headers                headers
best val loss           1.218 ± 0.046          1.030 ± 0.076          2.272 ± 0.078          0.996 ± 0.009          0.957 ± 0.055          1.149 ± 0.051
best val bpc            1.757 ± 0.067                      -          1.763 ± 0.060          1.436 ± 0.013          1.381 ± 0.080          1.657 ± 0.074
ended                   0.922 ± 0.014          0.981 ± 0.001          0.963 ± 0.014          0.953 ± 0.012          0.979 ± 0.011          0.954 ± 0.013
length                272.261 ± 9.612        261.168 ± 6.733       292.920 ± 20.561        249.022 ± 5.428       220.609 ± 10.357       257.343 ± 16.348
plays                   0.619 ± 0.038          0.677 ± 0.049          0.548 ± 0.049          0.722 ± 0.009          0.743 ± 0.050          0.662 ± 0.033
bar_accuracy            0.727 ± 0.020          0.800 ± 0.050          0.735 ± 0.012          0.782 ± 0.031          0.813 ± 0.004          0.762 ± 0.013
ends_on_tonic           0.231 ± 0.025          0.328 ± 0.015          0.231 ± 0.049          0.356 ± 0.059          0.416 ± 0.040          0.296 ± 0.033
novelty                 0.998 ± 0.000          0.996 ± 0.002          0.993 ± 0.005          0.997 ± 0.001          0.998 ± 0.001          0.998 ± 0.001
```

Each ± is the spread across the three **training** seeds (`slm eval`'s ± is sampling noise within one
model). The SFT column is scored with plain-language requests, the others with header prompts. Read
across (experiments.md §2–§6):

- **micro beats nano** on everything, by several spreads.
- **char vs BPE:** bits per character tie (1.757 vs 1.763), as in Phase B; char plays more often;
  Phase C's bar-accuracy gap has gone.
- **noaug beats baseline**, and micro_noaug matches or beats micro: the surprise of this phase (F.5).
- **SFT stays ahead of the header prompt** on every grader, across all three seeds.

To see it as a picture: `make figures` writes `docs/images/abc-sweep.png` from the same code (needs
`uv sync --extra docs`).

---

## F.4 Check for overfitting

```bash
uv run python - <<'PY'
import json
from slmkit.train.run import list_runs
for d, st in sorted(list_runs(), key=lambda r: r[1].get("experiment", "")):
    if st.get("experiment", "").startswith("abc_music/") and st.get("stage", "pretrain") == "pretrain":
        ev = [json.loads(x) for x in open(d / "metrics.jsonl") if '"eval"' in x]
        best = min(ev, key=lambda e: e["val_loss"])
        print(f"{st['experiment']:<22} {d.name}  best step {best['step']:>4}  "
              f"end gap {ev[-1]['val_loss'] - ev[-1]['train_loss']:+.3f}")
PY
```
```
abc_music/baseline     run-6b3967166fa0  best step  916  end gap +0.058
abc_music/baseline     run-70f7e9180ffd  best step  916  end gap +0.063
abc_music/baseline     run-749d028accd4  best step  916  end gap +0.064
…
abc_music/micro        run-a9651520abda  best step  500  end gap +0.359
abc_music/micro_noaug  run-c183e88ed53f  best step  750  end gap +0.374
…
```

nano runs end with a train/val gap of about +0.06 and their best checkpoint at the last step: still
improving, **capacity-limited**. micro runs end at +0.31 to +0.36 with validation flat from about step
750: memorizing, **data-limited**. That difference is what motivated `micro_noaug` (experiments.md §4).

---

## F.5 Why no transposition wins: test on transposed tunes

First rule out leakage, then score both ways:

```bash
uv run python projects/abc_music/check_sweep.py overlap
uv run python projects/abc_music/check_sweep.py keys
```
```
validation, original keys   vs baseline train   0.3% of 32-character windows seen; 0 of 412 tunes more than half seen
validation, original keys   vs noaug train      0.2% of 32-character windows seen; 0 of 412 tunes more than half seen
validation, transposed      vs baseline train   0.3% of 32-character windows seen; 0 of 825 tunes more than half seen
validation, transposed      vs noaug train      0.1% of 32-character windows seen; 0 of 825 tunes more than half seen
…
bpc             original keys       transposed  difference
baseline        1.744 ± 0.068    1.756 ± 0.072      +0.012
noaug           1.644 ± 0.076    1.830 ± 0.086      +0.185
micro           1.419 ± 0.016    1.247 ± 0.131      -0.172
micro_noaug     1.361 ± 0.079    1.569 ± 0.101      +0.208
```

No leakage. Models trained without transposition are ~0.19 bits per character worse on the same tunes
moved two semitones; with it, no penalty. Transposition buys robustness to key, and the evaluation's
prompts and validation tunes are all in the tune books' own keys, so it doesn't reward that. (micro
doing *better* on transposed copies is unexplained; experiments.md §5 records it as open.)

**Why:** a surprising result is a question about the measurement before it's a finding about the model.

---

## F.6 What "parse rate" means

```bash
uv run python projects/abc_music/check_sweep.py parse
uv run python projects/abc_music/check_sweep.py temperature
```
```
                                    parses  makes MIDI  plays, no error
baseline run-6b3967166fa0            1.000       1.000            0.560
…
micro_noaug run-a7889b63ebf5         1.000       1.000            0.785
micro_noaug run-c183e88ed53f         1.000       1.000            0.725
random characters (baseline)         0.915       0.915            0.045

run-a7889b63ebf5, 200 samples per temperature, sampling seed 0
T=1.0  plays 0.620  bar_accuracy 0.767  ended 0.995  novelty 1.000
T=0.8  plays 0.785  bar_accuracy 0.815  ended 0.995  novelty 1.000
T=0.6  plays 0.850  bar_accuracy 0.801  ended 0.920  novelty 0.974
T=0.4  plays 0.890  bar_accuracy 0.825  ended 0.840  novelty 0.915
T=0.2  plays 0.945  bar_accuracy 0.846  ended 0.860  novelty 0.957
```

Random characters "parse" 91.5% of the time, because `abc2midi` recovers from almost anything; only
the strict `plays` means something. Lowering the temperature raises `plays` towards 95% by making tunes
loop and copy instead (experiments.md §8).

---

## Phase F: done when

- [x] `make test` (191) and `make lint` pass.
- [x] Five experiments × three training seeds, each evaluated; `slm runs summary` averages them.
- [x] Every new experiment is a YAML file only (the framework check).
- [x] Surprising results checked before being believed: leakage (F.5), and what "parse rate"
      measures (F.6).
- [ ] **You** have run F.1–F.6 and the output matches.

## M2: closed

Every exit criterion is met except `plays` ≥ 95%, which stands at 0.743 for the best configuration
and is carried to M3 as a target for constrained decoding (ROADMAP, decided 2026-09-30).

---

# See it in slm studio

`slm studio` shows everything this runbook built, read straight from `$SLM_HOME`
([`studio.md`](studio.md) for starting it). With **abc_music** selected in the project switcher:

| Page | What of M2 you'll see |
|---|---|
| **Your model** | 18 runs, 500M tokens read, 0.13 GPU-hours; the best model (`micro_noaug`, 1.331 bits per character); this runbook in the list |
| **Lifecycle** | Phase A's raw tune books and the two datasets (with and without transposition), the char and BPE tokenizers (Phase B), every sweep run (Phase F), the SFT runs (Phase D) and the exports (Phase E). Click `abc-folk:1` to trace it back to the tune books |
| **Training** | The Phase F sweep as curves: tick `baseline`, `micro`, `noaug` and `micro_noaug` and compare bits per character; open a `micro` run to see training and validation loss part at about step 500 (F.4) |
| **Playground** | `abc-folk:2` (export it as in E.6) with the request builder, **Try:** examples, sheet music, playback, and **Reading this tune**: what you asked for against what it wrote, in plain words (two-models.md §3) |
