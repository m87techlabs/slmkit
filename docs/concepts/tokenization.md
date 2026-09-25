# Tokenization and the data pipeline

*Milestone M1, Phase A. Terms: [`../GLOSSARY.md`](../GLOSSARY.md). Check it on your own machine:
[`../runbooks/m1-engine.md`](../runbooks/m1-engine.md) §A.*

A model can only do arithmetic on numbers. Before any learning happens, text has to become a
sequence of integers, and those integers have to be stored so the trainer can pull millions of
random slices from them quickly. This page covers both halves: **tokenization** (text → IDs)
and the **data pipeline** around it (split → tokenize → pack → sample).

---

## 1. What a tokenizer is

A tokenizer is a reversible mapping between text and a list of integer **token IDs**, based on a
fixed **vocabulary**: the list of every token the model knows.

With slmkit's character tokenizer on Shakespeare, the vocabulary is every distinct character in
the training text, sorted, after two special tokens:

```
ID:     0      1     2    3   4   5   6   7   8   9  10  11  12  13  14  15  ...  66
token: <unk> <eos>  \n  ' '  !   $   &   '   ,   -   .   3   :   ;   ?   A  ...   z
```

65 characters + 2 specials = **67 tokens**. Encoding is a lookup:

```
"ROMEO:"  →  [32, 29, 27, 19, 29, 12]
"Romeo"   →  [32, 55, 53, 45, 55]        ← lower-case letters are different tokens
```

The model never sees letters, only these numbers. Its first layer (the **embedding**) turns each
ID into a learned vector, so the vocabulary size directly sets how big that layer is:
`vocab_size × d_model` parameters (see [`../MODEL.md`](../MODEL.md) §4).

---

## 2. Three kinds of tokenizer, and which slmkit uses where

| Type | A token is | Vocab size | Shakespeare as tokens | slmkit uses it for |
|---|---|---|---|---|
| **Character** | one character | ~65 | 1,115,394 | M1 Shakespeare; ABC baseline |
| **BPE** (byte-pair encoding) | a frequent chunk of characters, learned from data | ~1K–50K | 338,025 with GPT-2's 50K vocab (~3.3 chars/token) | ABC, compared with char in §6 (M2) |
| **Fixed vocabulary** | one domain unit, listed by hand | exactly what the domain needs | — | chess moves (1,968 UCI moves), cricket ball events |

The trade-off is **vocabulary size against sequence length**:

- A **character** tokenizer has a tiny vocabulary but long sequences. A 256-token context holds
  only ~50 words of Shakespeare, so the model has to learn spelling *and* grammar from a short
  window.
- **BPE** merges frequent character pairs repeatedly ("t"+"h" → "th", "th"+"e" → "the"), so
  common words become one token. Sequences are ~3–4× shorter, but the vocabulary is larger and
  has to be *trained* on the data.
- A **fixed vocabulary** fits when the domain already has natural units. A chess move `e2e4` is
  one idea; splitting it into characters would make the model learn to spell moves before it
  could learn to play them.

**Why character-level for M1.** M1 exists to prove the trainer correct against nanoGPT's
published number, and nanoGPT's reference run is character-level, so matching it is the point.
It is also the easiest tokenizer to trust: every ID maps to one visible character, there is
nothing to train, and a round trip is trivially lossless.

**Rejected for M1:** GPT-2's BPE (50,257 tokens). The embedding alone would be 50,257 × 384 =
**19.3M parameters**, nearly twice the rest of the `ref` model, spent on tokens like
" Microsoft" that never appear in Shakespeare. Byte-level (256 tokens, one per UTF-8 byte) was
also considered; for Shakespeare, which is plain ASCII, it would behave the same as character
level but with ~190 IDs that never occur.

---

## 3. The two special tokens

Every slmkit tokenizer reserves the lowest IDs for the same special tokens, so engine code can
rely on them without asking which tokenizer is in use:

| ID | Token | Meaning | Why |
|---|---|---|---|
| 0 | `<unk>` | "a character this tokenizer never saw" | The vocabulary is built from the **train split only**. A character that appears only in validation gets `<unk>`, not an ID the model never trained. `slm pack` reports the count (`val_unk_tokens`); for Shakespeare it is 0 |
| 1 | `<eos>` | end of document | Appended after each document when packing, so the model learns where a tune or game ends, and generation knows when to stop |

**Why fit on the train split only?** For a character vocabulary it hardly matters. For BPE (M2)
it does: the merges are *statistics* of the text, and computing them on validation data leaks
information about it. Doing it the same way for every tokenizer type keeps the rule simple.

**Why Shakespeare turns `<eos>` off** (`data.append_eos: false`): its documents are 10,000
character pieces of **one** continuous text. An `<eos>` between them would teach the model a
boundary that doesn't exist in the text.

---

## 4. The pipeline around the tokenizer

```
ingest     raw/<project>/                 download once, SHA-256 checked, never modified
   ↓
prepare    datasets/<project>/ds-…/       Docs split by group → train.jsonl, val.jsonl
   ↓
tokenize   tokenizers/tk-…/               vocabulary fitted on train.jsonl
   ↓
pack       packed/pk-…/                   each split encoded into one flat uint16 file
   ↓
(train)    random windows read from packed/…/train.bin
```

Each box is a separate **artifact**: an immutable directory with a `manifest.json` recording its
inputs. It is the same idea as Terraform resources with dependencies: change the tokenizer and
only `tokenize` and `pack` re-run; `ingest` and `prepare` are reused.

### Split before anything else, and by group

The first decision the pipeline makes is which text is **validation**, text the model is never
trained on and is only measured against. It happens before tokenizing and before augmenting,
because anything computed from all the data can leak validation information into training.

It splits by `Doc.group`, not by document. For Shakespeare every 10K block is its own group; for
ABC music every *tune* is a group, so its several settings (near-duplicates) always land on the
same side. The concrete failure this prevents: if one setting of a tune is in training and
another in validation, validation loss rewards **memorising** the tune, and a model that has
learned nothing about music looks great.

The split is exact rather than random: groups are sorted by a salted hash and the first 10% go to
validation. Shakespeare's ~110 blocks give **exactly 11** validation blocks. A coin flip per group
could give anywhere from about 5 to 17.

### Pack into one flat file

Tokenized documents are concatenated end to end and written as raw `uint16` (2 bytes per token):

```
packed/pk-41a610dc1eb1/
  train.bin   2,007,730 bytes = 1,003,865 tokens × 2
  val.bin       223,058 bytes =   111,529 tokens × 2
```

- **One file, not one per document**, because the trainer needs to jump to any position
  instantly. The file is memory-mapped (`np.memmap`), so the OS loads only the pages actually
  read and caches them. Thousands of small files would each cost a lookup.
- **`uint16`** holds IDs up to 65,535 in half the space of a 4-byte integer. Every vocabulary
  slmkit plans fits, and packing refuses one that wouldn't.

### Sample random windows, shifted by one

The trainer never iterates over documents. Each step, the sampler picks random start positions
and reads `block_size + 1` tokens from each:

```
window (9 tokens):   t  h  ' '  h  e  r  ' '  n  a
x  (model input):    t  h  ' '  h  e  r  ' '  n        [60, 48, 3, 48, 45, 58, 3, 54]
y  (targets):           h  ' '  h  e  r  ' '  n  a     [48, 3, 48, 45, 58, 3, 54, 41]
```

`y` is `x` shifted left by one: **at every position, the target is the next token.** That one
line is the entire training objective. One window of 256 tokens is 256 prediction exercises at
once, because the causal mask stops each position from seeing its own answer (see
[`../MODEL.md`](../MODEL.md) §1).

Getting this shift wrong is a classic silent bug: with `y == x`, the model learns to copy its
input, loss drops to nearly zero, and nothing useful is learned. `test_sampler.py` pins it
using a sequence where token *i* has value *i*, so `y == x + 1` must hold exactly.

**Why random windows instead of epochs over fixed chunks?** Shakespeare's train split is ~1M
tokens, which is only about 3,900 non-overlapping 256-token chunks. Random starts give ~1M
different windows. The sampler's random state is saved in every checkpoint, so a resumed run
draws exactly the same batches an uninterrupted run would have (the resume-equivalence test in
Phase C).

---

## 5. What can go wrong, and where it is caught

| Mistake | Symptom if uncaught | Caught by |
|---|---|---|
| Random per-document split | Validation loss suspiciously good; model memorises | `test_split.py`, `test_pipeline.py` |
| Augmenting before splitting | Transformed copies of validation text in training | `test_pipeline.py::test_augmentation_touches_train_only` |
| Off-by-one in targets | Loss near zero, garbage samples | `test_sampler.py` |
| Vocab over 65,535 | IDs silently wrap around in `uint16` | `pack_split` raises |
| Upstream data changed | A "reproducible" run trains on different text | SHA-256 pinned in `project.py` |
| Stale artifact reused after a code change | Old data under a new config | Stage `CODE_VERSION` and project `data_version` in the artifact ID |

---

## 6. BPE, built and measured (M2)

*Code: [`src/slmkit/tokenizers/bpe.py`](../../src/slmkit/tokenizers/bpe.py). Hands-on: runbook M2 §B.*

### How the merges are learned

**Byte-pair encoding** starts with single characters and repeats one step: find the most frequent
pair of adjacent tokens in the training text and merge it into a new token. Every merge adds one
entry to the vocabulary, and training stops at the target size. The real merges learned on ABC
(numbered in the order they were learned):

```
#1   ' ' '|'     → ' |'           the very first merges: a space and a bar line,
#2   ' ' '('     → ' ('           then a space and each common note (' c', ' d', ' e', ...)
#28  ' A' '2'    → ' A2'          a note with its length
#30  'M' ':'     → 'M:'           every tune has a meter line
#121 'M:' '6'    → 'M:6'
#123 'M:6' '/8'  → 'M:6/8'        the whole header is now one token
#292 ' A2' ':|'  → ' A2:|'        "a half note, then repeat": a standard phrase ending
```

Nothing told it these are musical units. They are simply what ABC repeats most. The 15 longest
tokens it learned at a 512 vocabulary include every common header (`R:hornpipe`, `R:reel`, `M:6/8`,
`L:1/16`), phrase endings (`' A2:|'`, `' G2:|'`) and a scale run (`' gfed'`).

The merges are trained on the **train split only**, like the char vocabulary (§3), and with the same
two special tokens at the same IDs: `<unk>` = 0, `<eos>` = 1.

### What merges may cross: pre-splitting

Before merging, text is cut into chunks, and no token can span two chunks. That choice matters more
than it looks. Measured at a 512 vocabulary on the ABC validation split:

| Pre-splitting | Characters per token | What tokens look like |
|---|---|---|
| spaces and newlines are their own tokens | 1.55 | `' '`, `'|'`, `' '`, `'('`: a third of all tokens are single spaces |
| **a space joins the chunk after it** (GPT-2's scheme; newlines alone) | **1.87** | `' |'`, `' ('`, `'c2'`: musical units |
| no pre-splitting at all | 2.16 | `') | ('`, `' \\\n|'`: tokens straddle bar lines and phrases |

slmkit uses the middle row. The last row compresses a little more, but a token like `') | ('` has no
musical meaning, which makes the model's vocabulary harder to read and its behaviour harder to reason
about. **Rejected:** byte-level BPE (GPT-2's 256-byte base alphabet). ABC uses 87 characters, so ~170
IDs would never occur, and there are no non-ASCII characters that need byte fallback.

### Vocabulary size vs compression

| Vocabulary | Characters per token | Validation tokens (106,612 characters, `<eos>` not counted) |
|---|---|---|
| 87 (char) | 1.00 | 106,612 |
| 128 | 1.30 | 81,699 |
| 256 | 1.63 | 65,404 |
| **512** | **1.87** | **57,152** |
| 1,024 | 2.10 | 50,849 |
| 2,048 | 2.31 | 46,120 |

Diminishing returns: doubling from 512 to 1,024 buys 12% more compression, and costs 65,536 extra
embedding parameters (512 more rows of 128) on a `nano` model that has 853K others. That is why DESIGN caps ABC's BPE at
~1K. ABC compresses far less than English (~3–4 characters per token at similar vocabularies) because
it has few long repeated words: most of it is short note groups separated by spaces.

### Char vs BPE: the A/B

Two runs identical except for the tokenizer (`abc_music/baseline` and `abc_music/bpe512`): same
`nano` model, same 512-token context, same 30M-token budget, so the same compute. Each best
checkpoint, scored on the full validation split with `scripts/val_metrics.py`:

| | Char (vocab 87) | BPE (vocab 512) |
|---|---|---|
| Loss per token | 1.256 | 2.350 |
| Perplexity per token | 3.51 | 10.49 |
| Top-1 accuracy per token | 60.8% | 42.0% |
| Characters per token | 1.00 | 1.86 |
| **Bits per character** | **1.812** | **1.824** |

**The trap in the first three rows.** Judged per token, BPE looks far worse: three times the
perplexity, 19 points less accurate. But each BPE token is 1.86 characters, so every guess predicts
more text and is harder. Per-token numbers are only comparable between models with the *same*
tokenizer. Bits per character divides the total loss by characters instead, and there the two are
within 0.7% of each other. The trainer prints bpc on every eval line for exactly this reason:

```
eval  step    916  train 2.1510  val 2.3531  (gap +0.2021, 1.826 bpc)  * best
```

**Reading the result.** At equal compute, on this corpus, the tokenizer barely matters for
prediction quality, and a single seed can't separate 1.812 from 1.824 (Phase F runs three).
Differences that do show:

- **BPE sees 1.86× more text** for the same compute and the same 512-token context. That helps with
  long-range structure (a whole tune and more fits in context), and costs faster overfitting: BPE's
  train/val gap was 0.20 per token (≈ 0.16 bpc) against char's 0.06 (≈ 0.09 bpc).
- **BPE can't make some mistakes a char model can.** It emits `M:6/8` as one token, so it can't
  misspell the header. Whether that shows up in the graders is a Phase C/F question.
- **Char is simpler to inspect.** Every token is one visible character.

Both stay supported; the tokenizer is one line in an experiment YAML (`tokenizer.type`), which is
exactly the kind of choice slmkit makes cheap to compare.
