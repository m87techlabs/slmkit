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
| **BPE** (byte-pair encoding) | a frequent chunk of characters, learned from data | ~1K–50K | 338,025 with GPT-2's 50K vocab (~3.3 chars/token) | ABC comparison (M2) |
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
