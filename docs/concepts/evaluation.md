# Evaluation: graders, baselines, calibration and seeds

*Milestone M2, Phase C. Code: [`src/slmkit/eval/runner.py`](../../src/slmkit/eval/runner.py),
[`src/slmkit/graders/`](../../src/slmkit/graders/),
[`projects/abc_music/graders.py`](../../projects/abc_music/graders.py). Terms:
[`../GLOSSARY.md`](../GLOSSARY.md). Hands-on: [`../runbooks/m2-abc-music.md`](../runbooks/m2-abc-music.md) §C.
Which metric suits which task: [`metrics.md`](metrics.md).*

Validation loss says how well a model predicts held-out text. It does not say whether what the
model *writes* is any good: a generated reel can have bars that don't add up, or end on the wrong
note, while loss keeps improving. Evaluation asks that second question with **graders**: programs
that score each generated sample against a rule. slmkit only builds projects where such rules
exist, so no human judgment is needed.

Four ideas make grader numbers trustworthy: **baselines**, **calibration**, **seeds**, and
**looking at the samples**.

---

## 1. What `slm eval` does

```
for each sampling seed (0, 1, 2):
    for each eval prompt (reel in D 2/2, jig in G 6/8, hornpipe in A 2/4, air in E minor 3/4):
        generate ~50 samples from the best checkpoint
    score every sample with every grader
    average each metric over the seed's 200 samples
report each metric as mean ± standard deviation across the 3 seeds
repeat with the no-model baseline; write runs/<run>/eval/<eval_id>.json
```

The graders for `abc_music`:

| Metric | From | Score | Meaning |
|---|---|---|---|
| `plays` | project | 0/1 | `abc2midi`, the reference ABC player, plays it without errors (≥ 8 notes) |
| `bar_accuracy` | project | 0–1 | share of bars whose length matches the **requested** meter |
| `ends_on_tonic` | project | 0/1 | the last note is the tonic of the **requested** key |
| `novelty` | engine | 0–1 | share of 32-character windows **not** found verbatim in training data |
| `ended` | engine | 0/1 | the model produced `<eos>` itself, instead of hitting the length limit |
| `length` | engine | characters | how much it wrote |

"Requested" means the prompt's meter and key, not the tune's own headers. For header prompts they
coincide. After SFT (Phase D) the prompt is plain English ("a jig in G"), and the same graders then
measure whether the model did what it was asked.

**Graders are pure functions**, `(prompt, output) → {metric: score}`, each tested with known-good and
known-bad fixtures (`projects/abc_music/tests/test_graders.py`): a correct jig scores 1/1/1, a tune
with two over-long bars scores exactly ⅓ on bars, one ending on D in G scores 0 on tonic, and
garbage doesn't play.

---

## 2. Baselines: what does a number mean?

A score is only meaningful next to the score of something that knows nothing. Every `slm eval`
report includes one: **tokens drawn independently, in proportion to their frequency in the training
data**. It produces text with the right character mix and no structure at all.

Measured on the two Phase B models (`nano`, 0.86–0.92M parameters, 30M tokens each), 3 sampling
seeds × 200 samples:

| Metric | Char model | Char baseline | BPE-512 model | BPE baseline | Human corpus (§3) |
|---|---|---|---|---|---|
| plays | **0.647** ± 0.014 | 0.063 ± 0.018 | **0.492** ± 0.026 | 0.140 ± 0.035 | ~1.0 |
| bar_accuracy | **0.708** ± 0.012 | 0.047 ± 0.012 | **0.733** ± 0.008 | 0.053 ± 0.005 | 0.990 |
| ends_on_tonic | 0.212 ± 0.030 | 0.152 ± 0.006 | 0.190 ± 0.043 | 0.148 ± 0.014 | 0.805 |
| novelty | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.987 ± 0.004 | 1.000 ± 0.000 | — |
| ended | 0.907 ± 0.025 | 0.923 ± 0.018 | 0.950 ± 0.013 | 0.988 ± 0.010 | — |

(Measured with the corrected prompts of §6. The first version of this table asked for a 4/4 reel and
hornpipe, and read 0.643 / 0.677 for char and 0.505 / 0.728 for BPE.)

Reading it:

- **plays** and **bar_accuracy**: both models are far above their baselines. They have learned that
  bars must add up, most of the time.
- **ended** is about the same as the baseline, and that is a lesson in itself. The baseline "ends"
  whenever it happens to draw `<eos>`, which at training frequency (1 in ~250 characters) happens
  within 600 draws about 91% of the time. A metric the baseline matches for a trivial reason tells you
  nothing on its own.
- **novelty** is near 1.0 for everything, including random characters, which are perfectly novel. It
  is a guard against copying, not a score to maximize. BPE's 0.987 means 1.3% of its 32-character
  windows appear verbatim in training, against 0.2% for char, consistent with BPE's faster
  overfitting in Phase B.
- **ends_on_tonic** is barely above the baseline for either model (§5).

**Char vs BPE, graded.** Bits per character called the two tokenizers a tie (1.812 vs 1.824). The
graders show what loss can't: char's samples play far more often (0.65 vs 0.49, a gap of about eight
times the spread, so real for *these two models*). BPE gets slightly more bars right (0.73 vs 0.71), but
that gap is only about twice the spread, which is the edge of what 3 sampling seeds can resolve.
Whether either holds across training seeds is what Phase F measures.

**Why samples fail to play.** `plays` is strict: any `Error` line from `abc2midi` fails the sample,
including ones it recovers from. Tallying the errors in 200 samples per model (with the original
prompts) shows grammar slips, not garbage:

| Cause (abc2midi message) | Char | BPE |
|---|---|---|
| broken rhythm (`>`) between notes of unequal length | 33 | 45 |
| malformed note (e.g. an accidental with no note after it) | 23 | 40 |
| a repeat opened but never closed ("Missing :|") | 24 | 24 |
| a tie (`-`) to nothing | 13 | 14 |

BPE's extra failures are mostly malformed notes. A token like `' f>'` bundles a note with the start of
a rhythm mark, and when the next token doesn't fit, the grammar breaks mid-token.

---

## 3. Calibration: grade the graders

Before trusting a grader on model output, run it on data known to be good: the human-transcribed
corpus. A correct grader should score it near-perfectly. If it doesn't, the grader is wrong.

| Grader | On the 4,248 corpus tunes | What that established |
|---|---|---|
| bar_accuracy | mean **0.990**, 92.9% perfect | the duration scanner handles real ABC (broken rhythm, tuplets, chords, grace notes, pickups) |
| ends_on_tonic | **80.5%** | real tunes don't *always* end on the tonic (many end on the third or fifth), so 0.8, not 1.0, is the target |

The calibration also found **errors in the data**. The few corpus tunes scoring 0 on bars have wrong
headers in the source files: O'Neill's #316 is labelled 3/4, but every bar holds eight eighth notes
(4/4). The grader was right. Calibration is how you tell the difference.

**Pickups.** Many tunes start on an upbeat: `D|GAB AGE|…` begins with a one-note bar, completed by a
short final bar. A naive bar check would fail every such tune. The grader skips a *short* bar at the
start or end of a section (after `|:` or `||`), and judges every other bar. Without that rule, the
corpus would have scored ~0.9 and the grader would have been measuring notation style.

**Why not use `abc2midi`'s own bar warnings?** It warned about a short bar but said nothing about an
over-long *first* bar in a probe tune, apparently treating any first bar as a possible pickup. A grader
that misses long bars isn't calibratable, so slmkit uses `abc2midi` only for what it is authoritative
on (does it play?) and a small, tested scanner for bar lengths.

---

## 4. Seeds: how much is luck?

Sampling is random, so the same checkpoint scores differently on every draw of 200 samples. `slm eval`
repeats the whole evaluation with 3 sampling seeds and reports the spread (standard deviation):

```
plays   0.630 · 0.655 · 0.655   → 0.647 ± 0.014
```

A difference between two runs smaller than about twice this spread is not a difference. That is why
DESIGN requires **≥ 3 seeds before believing a comparison**.

There are two kinds of seed, and both matter:

| | Varies | Measures | Where |
|---|---|---|---|
| **Sampling seed** | which samples are drawn from one fixed model | noise in the *measurement* | `slm eval --seeds 0,1,2` |
| **Training seed** | initialization and batch order: a different model | noise in the *result* | separate runs with `run.seed` (Phase F) |

Phase F uses both: three training seeds per configuration, each evaluated with three sampling seeds.

---

## 5. Always read the samples: the tonic puzzle

Both models end on the tonic about 19–21% of the time, against 80.5% for real tunes and 15% for the
baseline. A number that low deserves suspicion of the grader before the model, so the samples were
checked (with the original prompts; §6).

**The grader is right.** The report's examples score correctly by hand: a reel in D ending `d2 :|`
passes, and a hornpipe in A ending on `c` (C♯, the third) fails. The final notes of 80 char-model
samples, by prompt:

```
reel in D       D ×5   A ×4   F# ×4   G ×2   E ×2
jig in G        G ×8   A ×4   C ×3    E ×2   D ×1
hornpipe in A   E ×5   F# ×4  A ×3    G# ×3  C# ×2  D ×2
air in E minor  G ×4   E ×3   D ×3    F# ×2  A ×1
```

76 of the 80 had finished on their own (`<eos>`), so these aren't cut-off tunes. The tonic is often the
single most common ending, but only just.

**A tempting wrong conclusion, and how the baseline stopped it.** Every ending above is a note of the
key, so it looked as if the model "knows which notes belong to the key, just not to come home". An
`ends_in_key` grader was added to measure that. It scored the model 0.997, and **the random-character
baseline 0.987**. In ABC the key signature applies to every bare letter: in D major, a plain `F` *is*
F♯. Almost any ending is in key by construction. The metric measured the notation, not the model, so it
was removed. Without the baseline column, it would have gone into this page as a finding.

**What the tonic result does mean.** The baseline's 0.15 is roughly 1 in 7: a random note of the scale.
The models' 0.19–0.21 is barely better than that. Ending on the tonic is a long-range pattern: the last
note has to refer back to the key stated at the top, hundreds of characters earlier. A 0.86M-parameter
model captures the local rules (bar lengths, note shapes) far better than that kind of long-range
structure. That makes `ends_on_tonic` one of the metrics to watch as models grow in Phase F.

Some samples also fall into a loop (`c2c c2c c2c …`) and never finish; `ended` and `length` together
show how often.

---

## 6. Ask for what exists: a flaw found in Phase E

The first eval prompts asked for a reel and a hornpipe in **4/4**. That looks reasonable: 4/4 is the
most common meter in music generally. It was found wrong in Phase E, while listening to exported
samples, when a fine-tuned model asked for "a hornpipe in A major" wrote a well-formed 2/4 hornpipe
and scored 0 on bars. Counting the training corpus showed why:

| Rhythm | Training tunes | Meters |
|---|---|---|
| reel | 1,900 | 2/2 × 1,173 · 2/4 × 718 · **4/4 × 9** |
| hornpipe | 1,281 | 2/4 × 674 · 2/2 × 580 · **4/4 × 27** |

A 4/4 reel barely exists in these tune books. Two things went wrong because of it:

- **The request didn't state what the grader checked.** The plain-English prompt said "a hornpipe in
  A major" and the grader checked for 4/4. A 2/4 answer is a correct answer to that request.
- **Stating it didn't help.** Asked "a reel in D major, 4/4 time", the fine-tuned model wrote `M:2/4`
  in 48 of 50 samples. It can't be talked out of what its data says reels are (sft.md §5). A base
  model given a `M:4/4` header copies it, so header prompts hid the problem.

The fix: each prompt asks for the corpus's **most common** form of that rhythm (reel 2/2, hornpipe
2/4 with `L:1/16`), and every plain-language request states the meter, since the bars grader checks it
(`test_sft_eval_prompts_state_what_the_graders_check`). The eval report ID now hashes the prompt text
too; before, only the prompt *kind* was hashed, and changed prompts would have silently reused the old
reports. All Phase C and D numbers were re-measured.

**The lesson:** an eval prompt is part of the measurement. Before asking a model for something, check
the training data contains it. Asking for what the corpus doesn't have measures something else:
whether a request can override the data, a harder and different question.

---

## 7. What a grader can't tell you

- **Musicality.** A tune can play, have perfect bars and end on the tonic while being dull or
  directionless. Graders set a floor; listening (Phase E) is still part of the exit criteria.
- **Diversity.** A model that wrote the same good tune 200 times would score perfectly on every grader
  above except novelty. Reading samples and checking novelty together guard against it.
- **Anything the grader's author didn't think of.** Every grader encodes one rule. Calibration shows
  it measures that rule correctly; it can't show the rule is the one that matters.
