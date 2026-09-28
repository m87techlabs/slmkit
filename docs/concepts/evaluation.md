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
    for each eval prompt (reel in D, jig in G, hornpipe in A, air in E minor):
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
| plays | **0.643** ± 0.025 | 0.063 ± 0.018 | **0.505** ± 0.005 | 0.140 ± 0.035 | ~1.0 |
| bar_accuracy | **0.677** ± 0.011 | 0.047 ± 0.012 | **0.728** ± 0.007 | 0.053 ± 0.005 | 0.990 |
| ends_on_tonic | 0.188 ± 0.058 | 0.152 ± 0.006 | 0.203 ± 0.043 | 0.148 ± 0.014 | 0.805 |
| novelty | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.985 ± 0.004 | 1.000 ± 0.000 | — |
| ended | 0.920 ± 0.005 | 0.923 ± 0.018 | 0.955 ± 0.013 | 0.988 ± 0.010 | — |

Reading it:

- **plays** and **bar_accuracy**: both models are far above their baselines. They have learned that
  bars must add up, most of the time.
- **ended** is about the same as the baseline, and that is a lesson in itself. The baseline "ends"
  whenever it happens to draw `<eos>`, which at training frequency (1 in ~250 characters) happens
  within 600 draws about 91% of the time. A metric the baseline matches for a trivial reason tells you
  nothing on its own.
- **novelty** is near 1.0 for everything, including random characters, which are perfectly novel. It
  is a guard against copying, not a score to maximize. BPE's 0.985 means 1.5% of its 32-character
  windows appear verbatim in training, against 0.1% for char, consistent with BPE's faster
  overfitting in Phase B.
- **ends_on_tonic** is barely above the baseline for either model (§5).

**Char vs BPE, graded.** Bits per character called the two tokenizers a tie (1.812 vs 1.824). The
graders show a trade-off that loss can't: char's samples play more often (0.64 vs 0.51), BPE's get more
bars right (0.73 vs 0.68). Both gaps are several times the sampling spread, so they are real for
*these two models*. Whether they hold across training seeds is what Phase F measures.

**Why samples fail to play.** `plays` is strict: any `Error` line from `abc2midi` fails the sample,
including ones it recovers from. Tallying the errors in 200 samples per model shows grammar slips, not
garbage:

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
plays   0.615 · 0.660 · 0.655   → 0.643 ± 0.025
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

Both models end on the tonic about 19–20% of the time, against 80.5% for real tunes and 15% for the
baseline. A number that low deserves suspicion of the grader before the model, so the samples were
checked.

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
The models' 0.19–0.20 is barely better than that. Ending on the tonic is a long-range pattern: the last
note has to refer back to the key stated at the top, hundreds of characters earlier. A 0.86M-parameter
model captures the local rules (bar lengths, note shapes) far better than that kind of long-range
structure. That makes `ends_on_tonic` one of the metrics to watch as models grow in Phase F.

Some samples also fall into a loop (`c2c c2c c2c …`) and never finish; `ended` and `length` together
show how often.

---

## 6. What a grader can't tell you

- **Musicality.** A tune can play, have perfect bars and end on the tonic while being dull or
  directionless. Graders set a floor; listening (Phase E) is still part of the exit criteria.
- **Diversity.** A model that wrote the same good tune 200 times would score perfectly on every grader
  above except novelty. Reading samples and checking novelty together guard against it.
- **Anything the grader's author didn't think of.** Every grader encodes one rule. Calibration shows
  it measures that rule correctly; it can't show the rule is the one that matters.
