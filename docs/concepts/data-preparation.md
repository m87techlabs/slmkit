# Data preparation: duplicates, augmentation, and trusting a transformation

*Milestone M2, Phase A. The worked example is `abc_music` ([its README](../../projects/abc_music/README.md)).
Terms: [`../GLOSSARY.md`](../GLOSSARY.md). Hands-on: [`../runbooks/m2-abc-music.md`](../runbooks/m2-abc-music.md) §A.*

In M1 the data was one clean file. Real corpora need three decisions before any training: **what
counts as the same example**, **how to get more training data without cheating**, and **how to know a
data transformation is right**. Each decision can quietly invalidate every number that comes after
it, which is why the lifecycle puts so many guard rails in data preparation.

---

## 1. Can you use it at all?

Before quality, check the licence. The obvious ABC corpus, The Session's 45,000-tune dump, prohibits
"training Large Language Models" outright, and that restriction covers private use too
([ADR 0005](../decisions/0005-abc-data-source.md)). slmkit uses public-domain tune books instead: 4,248
tunes rather than 45,000.

Two habits worth keeping: **read the licence before writing the ingest**, and **strip what the model
has no business learning**. These files name their transcribers, with email addresses, in `Z:` lines.
Cleaning drops those lines, credit goes in the README, and a test checks that no document contains
an `@`.

---

## 2. Near-duplicates: the leak you can't see

The same tune turns up more than once: *The Sailor's Hornpipe* is in two of the three books, and
O'Neill's prints several "settings" (versions) of some tunes. If one copy lands in training and
another in validation, validation loss rewards **remembering** the tune, and the model looks better
than it is.

M1 split by `Doc.group` to prevent this, but a group only works if it identifies the tune. These
books have no shared tune ID, so the group is built from two signals:

| Signal | How | Catches | Guard against over-merging |
|---|---|---|---|
| **Melody fingerprint** | the diatonic steps between the first 16 notes, ignoring accidentals, rhythm and grace notes | the same tune in another book, another key, or with small rhythmic changes | none needed: two different tunes rarely share 15 identical steps |
| **Distinctive title** | lower-cased, articles and punctuation removed | a later setting whose opening differs | titles shared by more than 3 tunes ("Reel", "Minuet") are ignored |

Tunes linked by either signal join one group (a union-find, i.e. connected components):

```
4,248 tunes  →  3,677 groups     sizes: 1 × 3,200 · 2 × 405 · 3 × 56 · 4 × 12 · 5 × 2 · 6 × 2
```

**Why err towards merging:** merging two different tunes only makes the split slightly lumpier.
Failing to merge two copies of one tune leaks. Asymmetric costs deserve an asymmetric rule.

Infra analogy: deduplicating alerts by fingerprint rather than by exact message text, because the
same incident arrives with different timestamps and hostnames.

---

## 3. Augmentation: more training data, honestly

With ~1M characters, the model sees each tune many times and starts fitting quirks (M1 showed where
that leads). **Augmentation** creates new training examples by transforming existing ones in ways
that keep them valid examples of the same thing.

For music, the natural transformation is **transposition**: the same melody, higher or lower. A reel in
D moved up a tone to E is still a good reel. It teaches the model that *melodic shape* matters, not
absolute pitch.

```
original (K:D)   AFDF AFDF | AGFE .D2(EF) | …   d=cAB .=c2.G2
−2 semitones     GECE GECE | GFED .C2(DE) | …   c_BGA .B2.F2     (K:C)
+1 semitone      BGEG BGEG | BAGF .E2(FG) | …   e_dBc .d2.A2     (K:Eb)
```

Rules that keep it honest:

- **Train split only, and only after splitting** (CONTRIBUTING.md rule 4). Transposing first and splitting
  second would put a validation tune's transposed twin in training: a leak with extra steps. Every
  transposed copy also inherits its original's group.
- **Stay in distribution.** Only ±1–2 semitones, and only into keys with at most 3 sharps or flats. A
  tune moved to D♭ is still music, but not *fiddle* music. Training on it teaches the model something
  the real data never contains.
- **Deterministic.** Which copies exist, and which have headers dropped, comes from a hash of the tune's
  ID, not a random number generator. The same config rebuilds exactly the same dataset, and therefore
  the same artifact ID.

Result: 3,829 training tunes → **11,316 documents** (2.96×), 2.85M tokens.

**Did it help? Not with what the evaluation asks.** The Phase F sweep (experiments.md §5) trained
with and without transposition at equal compute, 3 seeds each. Without it, models scored *better* on
the validation tunes in their original keys (1.657 vs 1.757 bits per character at nano size), and
about 0.19 bits per character *worse* on the same tunes transposed. Transposition buys robustness to
key; an evaluation in common keys doesn't reward it. Correct as a transformation is not the same as
useful for the test.

**Header dropout** is a second, different kind of augmentation: 40% of documents lose some of their
`R:`/`M:`/`K:` lines. It creates no new music; it teaches the model to write a tune without being
told its rhythm, meter or key. It also sets up the M2 SFT comparison: the base model is familiar with
header prefixes, so "base model + header prefix" is a strong baseline for the fine-tuned model to match.

---

## 4. How do you know a transformation is right?

A transposition bug would not crash anything. It would produce plausible-looking ABC with a few wrong
notes, the model would learn from it, and nothing would ever flag it. So it has to be checked against
something **independent**: an **oracle**.

The check: play the original and the transposed tune, and require every note to sound exactly *n*
semitones higher. What happened when building it is the lesson:

| Oracle | Result | What it showed |
|---|---|---|
| `music21` (a music analysis library) | 1,168 exact, **16 mismatches** out of 1,184 | the mismatches were **music21's** errors: it applies an accidental only to the note it is written on, but the ABC standard carries it to later notes in the bar. `=g2dg` means two G naturals; music21 plays G, G♯ |
| `abc2midi` (the reference ABC player) | 1,172 exact, 12 mismatches | real edge cases: players disagree on whether an accidental carries into *other octaves*. Fix: write the accidental explicitly wherever it could matter, and refuse to transpose originals that depend on it |
| `abc2midi`, after fixes | 1,180 exact, 4 mismatches | `"B MINOR"`, a section label, was being treated as a chord symbol and transposed. Fix: a strict chord-symbol pattern, and strip everything else |
| `abc2midi`, **whole corpus** | **16,603 of 16,603 exact**, 389 refused | done |

Three lessons, in increasing order of importance:

1. **Test data transformations against an independent implementation**, not against your own
   understanding of the format. Every one of the three bugs above was a misunderstanding that
   hand-written test cases would have shared.
2. **An oracle can be wrong too.** The first mismatches looked like transposition bugs and were not.
   Checking each disagreement by hand, before "fixing" anything, is what found that.
3. **Refusing is a valid answer.** 389 transpositions (2.3%) are skipped, for key changes mid-tune,
   unspellable keys, or originals that different players read differently. A transformation that is
   right 100% of the time on 97.7% of the data is worth far more than one that is right 99.9% of the
   time on all of it.

---

## 5. Checklist for the next project

| Question | abc_music's answer |
|---|---|
| Does the licence allow training? | The Session: no. Public-domain books: yes (with a stated residual risk) |
| What personal data is in the raw files? | transcribers' names and emails in `Z:`, stripped and tested |
| What counts as "the same example"? | melody fingerprint ∪ distinctive title → 3,677 groups |
| Which transformations keep an example valid? | ±2 semitones into common keys; header dropout |
| Are they applied after splitting, to train only? | yes (engine-enforced) |
| How is each transformation verified? | an independent oracle (`abc2midi`) over the whole corpus |
| Is the dataset reproducible? | deterministic hashes, so same config → same artifact IDs |
