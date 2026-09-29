# Runbook M2: ABC music, the full lifecycle

*Companion to [`../concepts/data-preparation.md`](../concepts/data-preparation.md) (Phase A), and
later pages per phase. Project details: [`projects/abc_music/README.md`](../../projects/abc_music/README.md).
Built up phase by phase, like the M1 runbook.*

| Phase | Builds | Status |
|---|---|---|
| **A** | The ABC corpus: clean, group, transpose, header dropout | ☑ |
| **B** | BPE tokenizer, compared with char by bits per character | ☑ |
| **C** | Graders, `slm eval`, `slm runs compare` | ☑ |
| **D** | SFT: prompts from headers, loss on the answer only | ☑ this page |
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
  model    seed 0: ended 0.920  length 280.840  plays 0.615  bar_accuracy 0.686  ends_on_tonic 0.240  novelty 0.998
  …
metric                       model    baseline (token frequencies)
ended                0.920 ± 0.005                   0.923 ± 0.018
length             279.462 ± 6.439                215.237 ± 23.074
plays                0.643 ± 0.025                   0.063 ± 0.018
bar_accuracy         0.677 ± 0.011                   0.047 ± 0.012
ends_on_tonic        0.188 ± 0.058                   0.152 ± 0.006
novelty              0.999 ± 0.000                   1.000 ± 0.000

report: ~/slm/runs/run-749d028accd4/eval/ev-a844801d43.json
```

About 25 seconds on the GPU: 600 samples from the model, 600 from the baseline, every one graded.

**Read it against the baseline column,** which is tokens drawn at random by training frequency:

- `plays` and `bar_accuracy` are 10× and 14× the baseline: real learning.
- `ended` matches the baseline for a trivial reason: random draws hit `<eos>` within 600 tokens ~91%
  of the time. On its own it says nothing.
- `novelty` is ~1.0 for random text too: a guard against copying, not a score.
- `ends_on_tonic` barely beats the baseline's ~1-in-7 (C.4).

**Run it again** and you get `ev-a844801d43 already exists`: the report ID hashes the settings, the
checkpoint and the graders' source code, so identical evaluations are reused, and a changed grader
automatically forces a fresh one. Same seeds also give identical numbers: evaluation is reproducible.

---

## C.3 Compare runs side by side

```bash
uv run slm eval run-f79b                       # the BPE run
uv run slm runs compare run-749d run-f79b
```
```
run                          run-749d028a             run-f79bc951
name                         abc-baseline               abc-bpe512
tokenizer                            char                  bpe 512
best val loss                      1.2564                   2.3531
best val bpc                            -                    1.826
eval              ev-a844801d43 (3 seeds)  ev-a844801d43 (3 seeds)
plays                       0.643 ± 0.025            0.505 ± 0.005
bar_accuracy                0.677 ± 0.011            0.728 ± 0.007
ends_on_tonic               0.188 ± 0.058            0.203 ± 0.043
novelty                     0.999 ± 0.000            0.985 ± 0.004
…
```

(The char run shows `-` for bpc only because it was trained before the trainer reported bpc; its
full-validation figure is 1.812.)

Bits per character called char and BPE a tie. The graders show a trade-off: char plays more often,
BPE gets more bars right, and BPE copies slightly more (1.5% of 32-character windows). Each gap is
several times the sampling spread. Whether it survives different *training* seeds is Phase F's
question.

---

## C.4 Investigate a suspicious number: ends on the tonic

A 19% tonic rate against 80% for real tunes looked like a possible grader bug, so the endings were
counted directly (80 samples of the char model):

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

Of 200 samples, 77 (char) and 101 (BPE) don't play. The `abc2midi` errors behind them are grammar
slips, not garbage:

| Cause | Char | BPE |
|---|---|---|
| broken rhythm (`>`) between notes of unequal length | 33 | 45 |
| malformed note (e.g. an accidental with no note after it) | 23 | 40 |
| a repeat opened but never closed | 24 | 24 |
| a tie to nothing | 13 | 14 |

`plays` is strict: any `Error` line fails the sample, even ones `abc2midi` recovers from.

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
| plays | 0.643 ± 0.025 | 0.413 ± 0.051 | **0.635** ± 0.059 |
| bar_accuracy | 0.677 ± 0.011 | 0.268 ± 0.025 | **0.689** ± 0.011 |
| ends_on_tonic | 0.188 ± 0.058 | 0.132 ± 0.010 | **0.292** ± 0.046 |
| ended | 0.920 ± 0.005 | 0.830 ± 0.013 | **0.982** ± 0.010 |

```bash
uv run slm runs compare run-749d run-9b89
```
```
run                               run-749d028a                  run-9b890db6
name                              abc-baseline              abc-baseline-sft
stage                                 pretrain                           sft
…
eval              ev-7925f26eaf (sft, 3 seeds)  ev-0a9ec0eef6 (sft, 3 seeds)
plays                            0.413 ± 0.051                 0.635 ± 0.059
bar_accuracy                     0.268 ± 0.025                 0.689 ± 0.011
```

(`compare` shows each run's latest report: for the base run, that's the words-prompt one.)

**Reading it:**
- **Base + words** collapses: without SFT, the model doesn't understand a request.
- **SFT + words reaches parity** with base + headers on plays and bars. That is the M2 exit criterion
  (DESIGN §5.1: parity is the pass mark, not a win). SFT's contribution is the interface.
- It also **finishes** 98% of the time and ends on the tonic more often. SFT examples are always
  complete tunes.

---

## Phase D: done when

- [x] `make test` (164) and `make lint` pass; the mask test proves zero loss and gradient on prompts.
- [x] `slm sft` fine-tunes from the pretrained best checkpoint, resumably; `slm run` chains both.
- [x] SFT + plain-language requests reaches parity with base + headers on plays and bars.
- [ ] **You** have run D.1–D.3 and the output matches.
