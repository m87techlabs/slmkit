# Runbook M1: the engine

*Companion to [`../concepts/tokenization.md`](../concepts/tokenization.md) (Phase A),
`the-model.md` (Phase B) and `the-training-loop.md` (Phase C). This page is built up phase by
phase, and each phase's section is complete before the next phase starts.*

M1 builds the engine and proves it correct by reproducing nanoGPT's character-level Shakespeare
result. It is split into four phases, each ending with this runbook checked by hand:

| Phase | Builds | Status |
|---|---|---|
| **A** | Config, artifacts, Project API, char tokenizer, data pipeline | ☑ this page |
| B | The model (`model/llama.py`) | ☐ |
| C | Trainer, checkpoints/resume, sampling, tracking | ☐ |
| D | The reference run, kill-and-resume | ☐ |

Run everything from the repo root in `Ubuntu-ML`. Expected output is from the reference machine;
paths show `~/slm` as `$SLM_HOME`.

---

# Phase A: from raw text to training tokens

## A.0 The 30-second check

```bash
make test     # 51 passed in ~0.4s
make lint     # All checks passed! / Success: no issues found in 25 source files
```

The tests are CPU-only and use a synthetic "toy" project, so they need no network and never
touch `~/slm`.

---

## A.1 Configuration: presets, experiments, overrides

**What.** An experiment YAML names a model preset and a train preset, then overrides what it
needs. `--set` overrides on the command line. The result is validated by pydantic schemas.

**How.** [`src/slmkit/config/`](../../src/slmkit/config/). Merge order, later wins:

```
presets/model/ref.yaml → presets/train/default.yaml → projects/…/experiments/ref.yaml → --set
```

**Verify: see the resolved config.**

```bash
uv run slm config shakespeare_char/ref
```
```
# shakespeare_char/ref  (…/projects/shakespeare_char/experiments/ref.yaml)
# config hash: c728a2d697f4
run:
  name: shakespeare-ref
  ...
model:
  preset: ref
  n_layers: 6          ← from presets/model/ref.yaml
  d_model: 384
  dropout: 0.2
  ...
train:
  preset: default
  max_tokens: 81920000 ← from the experiment
  grad_clip: 1.0       ← only in presets/train/default.yaml
  betas: [0.9, 0.99]   ← experiment overrides the preset's [0.9, 0.95]
```

**Verify: overrides, and swapping a whole preset.**

```bash
uv run slm config shakespeare_char/ref --set model.preset=nano --set train.lr=6e-4
```
```
# config hash: f36c29a5188e     ← different config, different hash
  preset: nano
  n_layers: 4                   ← the entire nano preset, not just the label
  d_model: 128
  lr: 0.0006
```

**Verify: a typo fails loudly, naming the key.**

```bash
uv run slm config shakespeare_char/ref --set model.dropuot=0.1; echo "exit=$?"
```
```
error: …/experiments/ref.yaml:
1 validation error for ExperimentConfig
model.dropuot
  Extra inputs are not permitted [type=extra_forbidden, input_value=0.1, input_type=float]
exit=2
```

**Why.** It works like Terraform variables. Presets are module defaults you share, the
experiment is a tfvars file, and `--set` is `-var`. Rejecting unknown keys matters more in ML
than elsewhere: a misspelt `dropuot` that was silently ignored would cost an hour-long run
*and* give you a wrong conclusion about dropout. **Rejected:** Hydra, which solves the same merge
with a framework's worth of conventions (CONTRIBUTING.md rule 6). The merge here is about 40 lines in
`load.py`.

---

## A.2 The four data stages

**What.** `ingest → prepare → tokenize → pack`, each producing one immutable artifact under
`$SLM_HOME`.

**How.** [`src/slmkit/pipeline.py`](../../src/slmkit/pipeline.py). Each stage builds any missing
upstream artifacts first, so `slm pack` on a fresh machine runs all four.

**Verify: run the stages one at a time.**

```bash
uv run slm ingest   shakespeare_char/ref
uv run slm prepare  shakespeare_char/ref
uv run slm tokenize shakespeare_char/ref
uv run slm pack     shakespeare_char/ref
```
```
shakespeare_char/ref:
  raw       built   raw-b7c40ca6e54d  ~/slm/raw/shakespeare_char
shakespeare_char/ref:
  raw       exists  raw-b7c40ca6e54d  ~/slm/raw/shakespeare_char
  dataset   built   ds-bdaf3875e330  ~/slm/datasets/shakespeare_char/ds-bdaf3875e330
shakespeare_char/ref:
  ...
  tokenizer built   tk-e4687abe1801  ~/slm/tokenizers/tk-e4687abe1801
shakespeare_char/ref:
  ...
  packed    built   pk-41a610dc1eb1  ~/slm/packed/pk-41a610dc1eb1
```

**Your IDs should be exactly these.** An ID is a hash of the inputs (config, upstream IDs, stage
code version), not of the time or machine, so anyone who runs this commit on this data gets
`pk-41a610dc1eb1`. A different ID means a different input, and `slm lineage` (A.4) shows which.

**Verify: run it again, and everything is skipped.**

```bash
uv run slm pack shakespeare_char/ref
```
```
  raw       exists  raw-b7c40ca6e54d  …
  dataset   exists  ds-bdaf3875e330  …
  tokenizer exists  tk-e4687abe1801  …
  packed    exists  pk-41a610dc1eb1  …
```

**Why.** It works like `make`, with content hashes instead of timestamps. It is what makes
`slm run` safe to re-run after a week away: finished work is never redone and never
overwritten. **Why a per-stage `CODE_VERSION` instead of the git SHA in the hash:** hashing the
SHA would rebuild everything after every commit, including documentation commits. The SHA is
still recorded in each manifest for traceability.

---

## A.3 Look inside the artifacts

```bash
find ~/slm/raw ~/slm/datasets ~/slm/tokenizers ~/slm/packed -type f | sort
```
```
~/slm/datasets/shakespeare_char/ds-bdaf3875e330/manifest.json
~/slm/datasets/shakespeare_char/ds-bdaf3875e330/train.jsonl
~/slm/datasets/shakespeare_char/ds-bdaf3875e330/val.jsonl
~/slm/packed/pk-41a610dc1eb1/manifest.json
~/slm/packed/pk-41a610dc1eb1/meta.json
~/slm/packed/pk-41a610dc1eb1/train.bin
~/slm/packed/pk-41a610dc1eb1/val.bin
~/slm/raw/shakespeare_char/input.txt
~/slm/raw/shakespeare_char/manifest.json
~/slm/tokenizers/tk-e4687abe1801/manifest.json
~/slm/tokenizers/tk-e4687abe1801/tokenizer.json
```

**A manifest** records what the artifact is, what went in, and what came out:

```bash
cat ~/slm/packed/pk-41a610dc1eb1/manifest.json
```
```json
{
  "kind": "packed",
  "id": "pk-41a610dc1eb1",
  "project": "shakespeare_char",
  "created": "2026-09-23T11:42:20-05:00",
  "git_sha": "89c7b65",
  "git_dirty": false,
  "code_version": 1,
  "inputs": {"dataset": "ds-bdaf3875e330", "tokenizer": "tk-e4687abe1801"},
  "config": {"append_eos": false, "dtype": "uint16"},
  "stats": {"vocab_size": 67, "train_tokens": 1003865, "val_tokens": 111529,
            "train_unk_tokens": 0, "val_unk_tokens": 0, ...}
}
```

Check the arithmetic: 1,003,865 + 111,529 = **1,115,394**, exactly the size of the source file.
Nothing was lost or duplicated. `git_dirty: false` means the code that built it is exactly
commit `89c7b65`.

**The dataset** is plain JSON Lines, one document per line:

```bash
head -c 200 ~/slm/datasets/shakespeare_char/ds-bdaf3875e330/val.jsonl
```
```
{"id": "block-0008", "group": "block-0008", "text": "SICINIUS:\nHe's a disease that must be cut away.\n\nMENENIUS:\n…
```

**The tokenizer** is the vocabulary, readable as-is:

```bash
cat ~/slm/tokenizers/tk-e4687abe1801/tokenizer.json
```
```
{"type": "char", "specials": ["<unk>", "<eos>"], "chars": ["\n", " ", "!", "$", "&", "'", …, "z"]}
```

**The packed data** is 2 bytes per token (`train.bin` 2,007,730 bytes = 1,003,865 × 2). Decode a
slice back to text to prove the round trip:

```bash
uv run python -c "
import numpy as np; from pathlib import Path; from slmkit.tokenizers import load_tokenizer
h = Path('~/slm').expanduser()
tok = load_tokenizer(h / 'tokenizers/tk-e4687abe1801')
val = np.memmap(h / 'packed/pk-41a610dc1eb1/val.bin', dtype=np.uint16, mode='r')
print(val[:12].tolist()); print(tok.decode(val[:120].tolist()))"
```
```
[33, 23, 17, 23, 28, 23, 35, 33, 12, 2, 22, 45]      ← S I C I N I U S : \n H e
SICINIUS:
He's a disease that must be cut away.

MENENIUS:
O, he's a limb that has but a disease;
Mortal, to cut it off;
```

**Why these formats.** JSONL and JSON because you can read them with `head` and `jq`. Raw
`uint16` because the trainer memory-maps it and jumps to random offsets. See
`tokenization.md` §4.

---

## A.4 Lineage: where did this come from?

```bash
uv run slm lineage pk-41a610dc1eb1
```
```
pk-41a610dc1eb1  [packed]  created 2026-09-23T11:42:20-05:00  git 89c7b65
    vocab_size = 67
    train_tokens = 1003865
    ...
    dataset: ds-bdaf3875e330  [dataset]  created …  git 89c7b65
        train_groups = 99
        val_groups = 11
        ...
        raw: raw-b7c40ca6e54d  [raw]  created …  git 89c7b65
            files = input.txt
    tokenizer: tk-e4687abe1801  [tokenizer]  created …  git 89c7b65
        vocab_size = 67
        dataset: ds-bdaf3875e330  (shown above)
```

**Why.** When a model behaves oddly in three months, this is how you answer "what data, which
tokenizer, which code?" without guessing. It works like `terraform state show` followed back
through the dependencies. Lineage is a graph, not a tree: the tokenizer and the packed data
both read the same dataset, which is printed once and then referenced.

---

## A.5 Change the config and watch what rebuilds

Use a throwaway `$SLM_HOME` so your real one stays tidy:

```bash
SLM_HOME=/tmp/slm-play uv run slm pack shakespeare_char/ref --set data.val_fraction=0.2
```
```
  raw       built  raw-b7c40ca6e54d      ← same ID: val_fraction doesn't affect the download
  dataset   built  ds-bd4a5bb563bf       ← new: a different split
  tokenizer built  tk-76859af2de2d       ← new: fitted on a different train split
  packed    built  pk-29746cf766ff       ← new
```

The raw ID is identical in both homes because the downloaded bytes are identical. Everything
downstream of the change gets a new ID; the original `pk-41a610dc1eb1` in `~/slm` is untouched.
Clean up the scratch home with `rm -rf /tmp/slm-play`. It was never your real `$SLM_HOME`.

**Why.** "Never overwrite" means two experiments with different splits can coexist and be
compared, and a result can always be traced back to the exact artifact it came from.

---

## A.6 The split: no leakage, exact fraction

```bash
D=~/slm/datasets/shakespeare_char/ds-bdaf3875e330
comm -12 <(jq -r .group $D/train.jsonl | sort -u) <(jq -r .group $D/val.jsonl | sort -u) | wc -l
jq -r .group $D/val.jsonl | tr '\n' ' '
```
```
0
block-0008 block-0033 block-0045 block-0049 block-0072 block-0092 block-0097 block-0099 block-0102 block-0104 block-0105
```

- **0** groups appear in both splits: no leakage.
- **11 of 110** blocks are validation, exactly 10%, spread across the play rather than taken from
  the end.

**Why.** Splitting by group is CONTRIBUTING.md rule 4, and the reason is explained in
`tokenization.md` §4. For Shakespeare it is almost a formality. For ABC music (M2) it is the
single most important correctness detail, and this is the machinery that will enforce it.

---

## A.7 The guard still holds

```bash
SLM_HOME=/mnt/c/Users/Public/slm uv run slm pack shakespeare_char/ref; echo "exit=$?"
```
```
error: /mnt/c/Users/Public/slm is under /mnt/, a Windows drive reached over 9P and 20-130x slower than ext4. …
exit=2
```

Nothing is created on the Windows drive (`ls /mnt/c/Users/Public/slm` → no such file). Before
M1 only `slm doctor` checked this; now `artifacts.py` refuses the path before any stage writes
a byte (CONTRIBUTING.md rule 1).

---

## A.8 The one-token shift

The trainer's targets are the inputs moved by one position. See it on real data:

```bash
uv run python -c "
import numpy as np; from pathlib import Path
from slmkit.tokenizers import load_tokenizer; from slmkit.data.sampler import RandomWindowSampler
h = Path('~/slm').expanduser(); tok = load_tokenizer(h / 'tokenizers/tk-e4687abe1801')
train = np.memmap(h / 'packed/pk-41a610dc1eb1/train.bin', dtype=np.uint16, mode='r')
x, y = RandomWindowSampler(train, block_size=8, batch_size=1, seed=0).next_batch()
print(x[0].tolist(), repr(tok.decode(x[0].tolist())))
print(y[0].tolist(), repr(tok.decode(y[0].tolist())))"
```
```
[60, 48, 3, 48, 45, 58, 3, 54] 'th her n'
[48, 3, 48, 45, 58, 3, 54, 41] 'h her na'
```

Read it column by column: after `t` comes `h`, after `th` comes ` `, after `th ` comes `h`, and so
on. Each position is one "predict the next character" exercise. **Why it matters:** an
off-by-one here trains a model to copy its input. Loss collapses, samples are garbage, and
nothing errors. `test_sampler.py` fails if `y != x + 1` on a counting sequence.

---

## A.9 The engine–project boundary

```bash
uv run pytest -q tests/unit/test_engine_boundary.py
```
```
2 passed
```

The test parses every file in `src/slmkit/` and fails if engine code imports a project, uses a
project's name as an identifier, or compares against it as a string (`if project ==
"chess"`). Prose examples in docstrings are allowed. The second test proves the check really
catches a violation. **Why:** CONTRIBUTING.md rule 2 is what keeps "a new use case = a new directory"
true, and a rule enforced by a test survives in a way a rule in a README does not.

---

## Phase A: done when

- [x] `make test` and `make lint` pass.
- [x] `slm pack shakespeare_char/ref` produces `pk-41a610dc1eb1`; a second run skips all stages.
- [x] Token counts add up to the source file; `unk_tokens` is 0; decoding round-trips.
- [x] The split has zero shared groups and exactly 11 validation blocks.
- [x] `/mnt` paths are refused before anything is written.
- [ ] **You** have run A.1–A.8 and the output matches.
