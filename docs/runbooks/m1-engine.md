# Runbook M1: the engine

*Companion to [`../concepts/tokenization.md`](../concepts/tokenization.md) (Phase A),
`the-model.md` (Phase B) and `the-training-loop.md` (Phase C). This page is built up phase by
phase, and each phase's section is complete before the next phase starts.*

M1 builds the engine and proves it correct by reproducing nanoGPT's character-level Shakespeare
result. It is split into four phases, each ending with this runbook checked by hand:

| Phase | Builds | Status |
|---|---|---|
| **A** | Config, artifacts, Project API, char tokenizer, data pipeline | ☑ |
| **B** | The model (`model/llama.py`) | ☑ |
| **C** | Trainer, checkpoints/resume, sampling, tracking | ☑ |
| **D** | The reference run, kill-and-resume | ☑ this page |

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

---

# Phase B: the model

*Concepts: [`../concepts/the-model.md`](../concepts/the-model.md). Specification:
[`../MODEL.md`](../MODEL.md). Code: [`src/slmkit/model/`](../../src/slmkit/model/).*

## B.0 The checks

```bash
make test        # 72 passed in ~3.5s  (includes the HF parity test, ~2s of it importing transformers)
make test-gpu    # 3 passed in ~10s    (bf16 vs fp32, finite gradients, compiled vs eager)
make lint
```

The first `make test-gpu` on a machine takes longer (~20s) because `torch.compile` builds
kernels into `$TORCHINDUCTOR_CACHE_DIR`; later runs reuse them.

---

## B.1 Inspect the model an experiment builds

**What.** `slm model <experiment>` builds the exact model the experiment would train (vocabulary
from the tokenizer artifact, shape from the preset) and reports its size, cost and starting
loss.

```bash
uv run slm model shakespeare_char/ref
```
```
shakespeare_char/ref   model preset `ref`
  vocab          67 (from tk-e4687abe1801)  ·  context 256 tokens
  architecture   6 layers · d_model 384 · 6 heads x 64 · kv heads 6 · ffn 1024 · dropout 0.2
  parameters       10,647,552 total
                   10,621,824 non-embedding   (per layer 1,770,240 = attention 589,824 + mlp 1,179,648 + norms 768)
                       25,728 embedding       (67 x 384, shared with the output head)
  compute        71.0 MFLOPs per training token
                 this run: 81,920,000 tokens = 5.81e+15 FLOPs ~ 2.3 GPU-min at 43 TFLOPS (small models run 2-3x slower than this)
  memory         weights 40.6 MiB fp32 · training state ~162 MiB + activations · checkpoint ~122 MiB
  sanity check   loss before training: 4.370 on 4 val batches   (a uniform guess scores ln 67 = 4.205)
```

**Check each line against the spec:**

- `10,621,824 non-embedding` is exactly MODEL.md §4's figure for `ref`, and the per-layer split
  matches the hand count there.
- `25,728 embedding` = 67 × 384. The vocabulary is 67, not nanoGPT's 65, because of the two
  special tokens (`tokenization.md` §3).
- `71.0 MFLOPs per token` = 6 × (non-embedding + output head) + attention; see
  `model/stats.py::flops_per_token`.
- **The sanity check is the important line.** An untrained model should be almost equally unsure
  of every character, which scores ln 67 = 4.205. 4.370 is close, as it should be. A number like 8
  or 10 would mean the initialization is broken, and training would start from a bad place.

Try another size without editing anything:

```bash
uv run slm model shakespeare_char/ref --set model.preset=nano
```
```
  architecture   4 layers · d_model 128 · 4 heads x 32 · kv heads 4 · ffn 384 · dropout 0.0
  parameters          861,696 total
                      853,120 non-embedding   (per layer 213,248 = attention 65,536 + mlp 147,456 + norms 256)
```

**Why a command for this.** Before spending GPU time you want to know what you are about to
train, like `terraform plan` before `apply`. It also catches a wrong preset or vocabulary before
the trainer does.

---

## B.2 The module tree

```bash
uv run python -c "
from slmkit.model import CausalLM, ModelArgs
print(CausalLM(ModelArgs(vocab_size=67, block_size=256, n_layers=6, d_model=384,
                         n_heads=6, n_kv_heads=6, ffn_hidden=1024, dropout=0.2)))"
```
```
CausalLM(
  (model): Decoder(
    (embed_tokens): Embedding(67, 384)
    (dropout): Dropout(p=0.2, inplace=False)
    (layers): ModuleList(
      (0-5): 6 x Block(
        (input_layernorm): RMSNorm()
        (self_attn): Attention(
          (q_proj): Linear(in_features=384, out_features=384, bias=False)
          (k_proj): Linear(in_features=384, out_features=384, bias=False)
          (v_proj): Linear(in_features=384, out_features=384, bias=False)
          (o_proj): Linear(in_features=384, out_features=384, bias=False)
          (resid_dropout): Dropout(p=0.2, inplace=False)
        )
        (post_attention_layernorm): RMSNorm()
        (mlp): MLP(
          (gate_proj): Linear(in_features=384, out_features=1024, bias=False)
          (up_proj): Linear(in_features=384, out_features=1024, bias=False)
          (down_proj): Linear(in_features=1024, out_features=384, bias=False)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
    (norm): RMSNorm()
    (rope): RotaryEmbedding()
  )
  (lm_head): Linear(in_features=384, out_features=67, bias=False)
)
```

Compare with the diagram in `MODEL.md` §2: every box is here, and `bias=False` everywhere.
`RotaryEmbedding` has no parameters (its cos/sin tables are recomputed, not learned or saved).

---

## B.3 Same function as Hugging Face's Llama

**What.** slmkit's weights load into HF's own `LlamaForCausalLM` with every name matching, and both
produce the same logits.

```bash
uv run python -c "
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from slmkit.model import CausalLM, ModelArgs
torch.manual_seed(0)
a = ModelArgs(vocab_size=67, block_size=64, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2, ffn_hidden=384)
ours = CausalLM(a).eval()
hf = LlamaForCausalLM(LlamaConfig(vocab_size=67, hidden_size=128, intermediate_size=384,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    rms_norm_eps=a.norm_eps, rope_parameters={'rope_type': 'default', 'rope_theta': a.rope_theta},
    tie_word_embeddings=True)).eval()
print(hf.load_state_dict(ours.state_dict(), strict=True))
x = torch.randint(0, 67, (3, 64))
with torch.no_grad(): print('max abs diff', (ours(x)[0] - hf(x).logits).abs().max().item())"
```
```
<All keys matched successfully>
max abs diff 0.0
```

This uses grouped-query attention (`n_kv_heads=2`) on purpose, the least-travelled code path.
Try changing `'rope_theta': a.rope_theta` to `500.0` in the HF config: the keys still match, but
the difference jumps well above zero. That is how you know the comparison is sensitive.

**Why.** It is the proof behind "export is a rename" (MODEL.md §6) and it pins the RoPE
convention (`the-model.md` §6), a mistake that would otherwise surface only in M2, as an
exported model that quietly produces worse text.

---

## B.4 The past cannot see the future

```bash
uv run python -c "
import torch
from slmkit.model import CausalLM, ModelArgs
torch.manual_seed(0)
model = CausalLM(ModelArgs(vocab_size=67, block_size=32, n_layers=2, d_model=64,
                           n_heads=4, n_kv_heads=4, ffn_hidden=192)).eval()
a = torch.randint(0, 67, (1, 20)); b = a.clone(); b[0, 12:] = (b[0, 12:] + 1) % 67
with torch.no_grad(): diff = (model(a)[0] - model(b)[0]).abs().amax(dim=-1)[0]
print('max |logit change| per position:', [round(d, 3) for d in diff.tolist()])"
```
```
max |logit change| per position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.228, 1.259, 1.015, 1.145, 1.289, 0.992, 1.202, 1.703]
```

Tokens from position 12 on were changed. Positions 0–11 are **exactly** 0.0: their predictions do
not depend on anything after them. **Why:** if the mask leaked, every position could read its own
answer, training loss would collapse toward zero, and generation (where the future does not
exist yet) would produce garbage.

---

## B.5 Overfit one batch

The first thing to run when a trainer misbehaves (CONTRIBUTING.md's debugging order starts here). One
fixed batch, trained on repeatedly, must be memorized:

```bash
uv run python -c "
import torch
from slmkit.model import CausalLM, ModelArgs
torch.manual_seed(0)
model = CausalLM(ModelArgs(vocab_size=67, block_size=32, n_layers=2, d_model=64,
                           n_heads=4, n_kv_heads=4, ffn_hidden=192))
x = torch.randint(0, 67, (4, 32)); y = torch.roll(x, -1, dims=1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(151):
    _, loss = model(x, y)
    if step % 25 == 0: print(f'step {step:3d}  loss {loss.item():.4f}')
    opt.zero_grad(); loss.backward(); opt.step()"
```
```
step   0  loss 4.2558      ← ≈ ln 67: knows nothing
step  25  loss 0.6805
step  50  loss 0.0561
step  75  loss 0.0208
step 100  loss 0.0135
step 125  loss 0.0099
step 150  loss 0.0077      ← memorized
```

**Why.** It needs no data pipeline, no schedule and no GPU, so if it fails, the fault is in the
model, the loss or the optimizer wiring, and nowhere else. It proves the model *can* learn; the
reference run in Phase D proves it learns *the right thing*.

---

## B.6 `torch.compile` and the M0 gap it exposed

```bash
uv run slm doctor | grep compile
```
```
  PASS  torch.compile            Triton kernel built and ran
```

This line is new. The first run of the GPU tests failed with:

```
fatal error: Python.h: No such file or directory
```

The first time `torch.compile` compiles a function, Triton builds a small C helper against Python's
headers. Those headers come from `python3-dev`, which M0's setup script didn't install. Nothing in
M0 compiled anything, so `doctor` passed anyway. Fixed in three places: the setup script now
installs `python3-dev`, `doctor` runs a real compile, and the M0 runbook and STACK.md say why.
**The lesson, again: check behaviour, not presence.**

To see the check fail on purpose (fake compiler, empty caches so nothing is reused):

```bash
CC=/bin/false TRITON_CACHE_DIR=/tmp/tc TORCHINDUCTOR_CACHE_DIR=/tmp/ic uv run slm doctor | grep compile
```
```
  FAIL  torch.compile            CalledProcessError: Command '['/bin/false', '/tmp/…/cuda_utils.c', …
```

---

## Phase B: done when

- [x] `make test`, `make test-gpu` and `make lint` pass.
- [x] `slm model shakespeare_char/ref` reports 10,621,824 non-embedding parameters and an
      initial loss near ln 67.
- [x] HF parity: all keys match, max logit difference 0.0.
- [x] Causality holds exactly; one batch is memorized.
- [ ] **You** have run B.1–B.6 and the output matches.

---

# Phase C: the training loop

*Concepts: [`../concepts/the-training-loop.md`](../concepts/the-training-loop.md). Code:
[`src/slmkit/train/`](../../src/slmkit/train/), [`sampling/`](../../src/slmkit/sampling/),
[`tracking/`](../../src/slmkit/tracking/).*

**Use a scratch `$SLM_HOME` for this phase.** The run ID is a hash of the config and data, so a
short test run of `shakespeare_char/ref` in `~/slm` *would be* the reference run, and Phase D
would resume it instead of starting fresh. A scratch home keeps them apart. Copy `doctor.json`
across so MFU uses your measured peak:

```bash
export SLM_HOME=/tmp/slm-c
mkdir -p $SLM_HOME && cp ~/slm/doctor.json $SLM_HOME/
```

Every command in C.1–C.7 assumes that export. Open a new terminal (or `export SLM_HOME=~/slm`)
to go back to your real home.

## C.0 The checks

```bash
make test                                                  # 89 passed in ~7s
uv run pytest -q tests/unit/test_trainer.py tests/unit/test_train_parts.py   # 17 passed
make test-gpu                                              # 4 passed in ~20s
```

---

## C.1 Watch it learn: 300 steps

```bash
uv run slm pretrain shakespeare_char/ref --max-steps 300 --set train.eval_every_steps=100
```

About 40 seconds, most of it `torch.compile` and evals. Abridged output:

```
shakespeare_char/ref:
  raw       built   raw-b7c40ca6e54d  /tmp/slm-c/raw/shakespeare_char        ← same IDs as ~/slm:
  dataset   built   ds-bdaf3875e330  …                                        content-addressed
  tokenizer built   tk-e4687abe1801  …
  packed    built   pk-41a610dc1eb1  …
run run-a1d5f0a224ff  (shakespeare_char/ref, shakespeare-ref)
  device      cuda  bf16=True  compile=True
  parameters  10,647,552 (10,621,824 non-embedding + 25,728 embedding)
  per step    16,384 tokens = 64 x 256
  budget      81,920,000 tokens = 5.81e+15 FLOPs
  remaining   81,920,000 tokens ~ 2.3 GPU-min (estimate at 43 TFLOPS)
eval  step      0  train 4.3705  val 4.3694  (gap -0.0011)  * best  [3.8s]
── sample @ step 0: ROMEO ─────────────
ROMEO:
Azz;BdL:klq&vJgYbkmB�Q:oHICrqqBo
  measured    812,483 tokens/s  ·  57.7 TFLOPS  ·  MFU 51.4% of 112.1 (measured by slm doctor --bench)
              memory 1.58 GiB peak  ·  remaining 1.7 GPU-min (measured)
step     50  loss 2.4175  lr 5.00e-04  gnorm 0.63  812,483 tok/s    1.0%  remaining 1.7 GPU-min
step    100  loss 1.9501  lr 1.00e-03  gnorm 0.60  809,816 tok/s    2.0%  remaining 1.6 GPU-min
eval  step    100  train 1.8837  val 1.9035  (gap +0.0198)  * best  [2.6s]
── sample @ step 100: ROMEO ─────────────
ROMEO:
As sorn kill'd the came, I rehesed the the lisine:
Why, the carthou as the butly brooust sue surnead
…
eval  step    300  train 1.3924  val 1.4569  (gap +0.0644)  * best  [2.6s]
── sample @ step 300: JULIET ─────────────
JULIET:
How diest not but which come, have together with the own.

GREMIO:
Perful, the Duke of York holds a company mind of Gloucester,
  checkpoint  ckpt/step_0000300  (0.1s)
stopped (session limit (300 steps)) at step 300, 6.0% done. Run the same command to resume.
```

**What to check, line by line:**

| Line | Expect | Why it matters |
|---|---|---|
| `eval step 0` | train ≈ val ≈ 4.37, close to ln 67 = 4.20 | untrained model is guessing uniformly (Phase B, B.1) |
| step 0 sample | random characters | the baseline everything is compared against |
| `measured` | ~810K tokens/s, MFU ~51% | replaces the 43 TFLOPS estimate; `remaining` switches to "(measured)" |
| `lr` | 5.00e-04 at step 50, 1.00e-03 at step 100 | warmup reaches peak exactly at 1,638,400 tokens (the-training-loop.md §3) |
| `gnorm` | falling, ~0.4–0.6 | stable; a spike to 10+ would mean trouble |
| evals | val 4.37 → 1.90 → 1.57 → 1.46 | learning; below the bigram baseline (2.45) by step 100 |
| gap | small and growing (+0.02 → +0.06) | expected: the model is starting to fit the training text specifically |
| samples | format by step 100, mostly real words by 300 | the number and the text agree |

**Why samples, not just loss:** at step 300, validation loss already equals nanoGPT's *final*
result. That looked too good, so it was checked for leakage before being trusted: the causal
mask on the compiled GPU path, and shared text between splits. Both were clean (the-training-loop.md §2).
The samples are what made the number believable.

---

## C.2 Resume: run the same command again

```bash
uv run slm pretrain shakespeare_char/ref --max-steps 100 --set train.eval_every_steps=100
```
```
  RESUMING    from step_0000300 (6.0% done, saved 2026-09-24T09:39:57-05:00, 0.01 GPU-h so far)
  remaining   77,004,800 tokens ~ 1.6 GPU-min (measured)
step    350  loss 1.3979  lr 9.94e-04  gnorm 0.40  800,834 tok/s    7.0%  remaining 1.6 GPU-min
step    400  loss 1.3707  lr 9.92e-04  gnorm 0.37  806,907 tok/s    8.0%  remaining 1.6 GPU-min
eval  step    400  train 1.3189  val 1.3992  (gap +0.0803)  * best  [3.1s]
  checkpoint  ckpt/step_0000400  (0.1s)
```

- No flag was needed: the same command found the same run directory and its checkpoint.
- The loss picks up where it stopped (1.49 at step 300 → 1.40 at 350). There is no jump back
  towards 4.2 and no warmup spike, because the optimizer state and lr schedule came back too.
- `remaining` says "(measured)" straight away: the measured speed was saved in the checkpoint.
- `--set train.eval_every_steps=100` did **not** start a new run. Eval cadence is an
  *operational* setting, excluded from the run ID (`train/run.py::OPERATIONAL`). Try
  `--set train.lr=6e-4` instead and you get a different run ID and a fresh start.

**Why:** the machine is off most nights, so every real run is a resumed run. The exact proof
is a test, `test_resume_is_equivalent_to_never_stopping`: 20 steps straight vs 10 + resume + 10
give identical losses (difference 0.0 on the CPU).

---

## C.3 Ctrl-C: stopping is normal

Start a run with no limit and press **Ctrl-C** once, after a few `step` lines:

```bash
uv run slm pretrain shakespeare_char/ref --set train.eval_every_steps=100000
```
Your step numbers depend on when you press it; this is from a real stop:

```
step   3350  loss 0.8089  lr 3.29e-04  gnorm 0.38  816,589 tok/s   67.0%  remaining 0.6 GPU-min
^C
SIGINT: finishing this step, then checkpointing. Press Ctrl-C again to abort without saving.
  checkpoint  ckpt/step_0003387  (0.2s)
stopped (SIGINT) at step 3387, 67.7% done. Run the same command to resume.
```

You may also see a `KeyboardInterrupt` traceback from `torch/_inductor/…/subprocess.py`. That
is one of `torch.compile`'s worker processes, which shares the terminal's process group and
received the same Ctrl-C. It is harmless. The trainer's own `checkpoint` and `stopped` lines are
what count.

**Why this needed a fix.** Under `uv run`, one Ctrl-C reaches the trainer **twice**: once from
the terminal and once forwarded by `uv`. The first version treated the second arrival as "press
again to abort" and threw the checkpoint away mid-write. A repeat within one second now counts
as the same keypress (the-training-loop.md §7). A deliberate second press, later, still aborts
without saving.

---

## C.4 Where are my runs?

```bash
uv run slm runs list
```
```
RUN               NAME               TOKENS             DONE  GPU-h   LEFT BEST VAL  LAST CHECKPOINT        STATUS
run-a1d5f0a224ff  shakespeare-ref    6.6M/81.9M         8.0%   0.01   0.03   1.3992  step 400 · just now    stopped: session limit (100 steps)
```

After two weeks away, this table is how you remember where you were: how far each run got,
GPU-hours spent and still needed (from *measured* speed), its best validation loss, and how much
work the last checkpoint protects. It reads each run's `status.json`, so nothing needs to be
running.

---

## C.5 Talk to the model

```bash
uv run slm sample run-a1d5 --prompt "JULIET:\n" --tokens 300 --seed 1
```
```
# run-a1d5f0a224ff · best · step 400 · val loss 1.3992
JULIET:
Why, came for the nost. He.

DUKE OF AUMERLE:
The very sir, this is the issue of the proud,
And you go my happing exprise sweet appoised.

KING EDWARD IV:
Nay, again the bear; the valiant night
Did Viclate her king, in the fries,
```

- A unique prefix of the run ID is enough (`run-a1d5`).
- It uses `ckpt/best` (lowest validation loss) by default; `--which latest` uses the newest
  checkpoint.
- Try `--temperature 0` (always the most likely character: it soon loops), `--temperature 1.5`
  (more inventive, more misspelt), `--top-k 5`. Same `--seed`, same text.
- `\n` in `--prompt` is a real newline.

---

## C.6 The curves: TensorBoard

```bash
uv run tensorboard --logdir $SLM_HOME/runs
```

Open http://localhost:6006 in a Windows browser (WSL forwards `localhost`). Under **Scalars**:
`loss/train`, `loss/val`, `lr` (the warmup ramp and the start of the cosine), `grad_norm`,
`tokens_per_s`, `mfu`. Under **Text**: every sample, by step. Stop it with Ctrl-C when done.
Nothing needs to be running *during* training; TensorBoard only reads the event files in `tb/`.

---

## C.7 Inside a run directory

```bash
cd $SLM_HOME/runs/run-a1d5f0a224ff
ls; ls ckpt ckpt/best
jq -c 'select(.kind=="eval") | {step, train_loss, val_loss}' metrics.jsonl
du -sh ckpt/step_0000400
```
```
ckpt  config.resolved.yaml  manifest.json  metrics.jsonl  status.json  tb  train.log
ckpt: best  step_0000300  step_0000400
ckpt/best: model.pt  optimizer.pt  state.pt
{"step":0,"train_loss":4.370459203720093,"val_loss":4.369408130645752}
{"step":100,"train_loss":1.8836883318424225,"val_loss":1.903482329249382}
{"step":200,"train_loss":1.513597030043602,"val_loss":1.5650890219211577}
{"step":300,"train_loss":1.3924472147226334,"val_loss":1.4568572574853897}
{"step":400,"train_loss":1.3189196968078614,"val_loss":1.3991872245073318}
122M	ckpt/step_0000400
```

- `122M` per checkpoint matches MODEL.md §6's estimate of ~12 bytes per parameter (weights and
  two AdamW averages).
- `train.log` is everything printed, flushed line by line, so it survives a power-off.
  `metrics.jsonl` and `status.json` mean no tracking tool is ever *required*.
- The contents of each file are described in the-training-loop.md §9.

When you're done: `rm -rf /tmp/slm-c`. It was only ever a scratch home.

---

## Phase C: done when

- [x] `make test` (89), `make test-gpu` (4) and `make lint` pass.
- [x] A run learns: val 4.37 → 1.46 in 300 steps, samples going from noise to verse.
- [x] The same command resumes with no jump; resume is exact in the test (difference 0.0).
- [x] Ctrl-C under `uv run` checkpoints and exits cleanly.
- [x] `runs list`, `sample` and TensorBoard all read the run without it running.
- [ ] **You** have run C.1–C.7 and the output matches.

Phase D is the real thing: the full reference run in `~/slm`, stopped and resumed, against the
M1 exit criteria.

---

# Phase D: the reference run

*The M1 exit criteria, met on a real run in `~/slm`. Numbers below are from that run.*

## D.1 Run it, stop it, resume it

In `tmux`, with your real `$SLM_HOME`:

```bash
tmux new -s ref
uv run slm pretrain shakespeare_char/ref
# … press Ctrl-C part-way through …
uv run slm pretrain shakespeare_char/ref        # same command: resumes
```

The whole run is ~82M tokens at ~830K tokens/s: **0.05 GPU-hours** (about 3 minutes including
evals and compilation). The first session was stopped with Ctrl-C at 84.9%:

```
step   4200  loss 0.6951  lr 1.58e-04  gnorm 0.42  830,487 tok/s   84.0%  remaining 0.3 GPU-min
SIGINT: finishing this step, then checkpointing. Press Ctrl-C again to abort without saving.
stopped (SIGINT) at step 4247, 84.9% done. Run the same command to resume.
```

and the second session picked it up:

```
  RESUMING    from step_0004247 (84.9% done, saved 2026-09-24T13:10:44-05:00, 0.04 GPU-h so far)
  remaining   12,337,152 tokens ~ 0.2 GPU-min (measured)
step   4250  loss 0.7076  lr 1.51e-04  gnorm 0.42  warming up   85.0%  remaining 0.2 GPU-min
…
complete: 81,920,000 tokens, best val 1.2888, 0.05 GPU-h
```

**Check for a jump.** Training loss before the stop: 0.7506, 0.7216, 0.6951. After: 0.7076. That
is ordinary step-to-step noise (the per-step loss is one batch with dropout on). The learning rate
continues down the same curve, 1.58e-04 → 1.51e-04. A broken resume shows up as a loss back near
1.3 or 4.2, or an lr back at warmup values.

(The first line after a resume says `warming up` instead of a speed: those first 10 steps
include recompiling, and the original version printed a misleading 25,135 tok/s there.)

```bash
uv run slm runs list
```
```
RUN               NAME               TOKENS             DONE  GPU-h   LEFT BEST VAL  LAST CHECKPOINT        STATUS
run-a1d5f0a224ff  shakespeare-ref    81.9M/81.9M      100.0%   0.05   0.00   1.2888  step 5000 · 1m ago     complete
```

Running the command again does nothing: `run already complete; nothing to do`.

---

## D.2 Read the curve

```bash
jq -r 'select(.kind=="eval") | "\(.step)\t\(.train_loss)\t\(.val_loss)"' \
  ~/slm/runs/run-a1d5f0a224ff/metrics.jsonl
```

| Step | Tokens | Train | Val | Gap | |
|---|---|---|---|---|---|
| 0 | 0 | 4.3705 | 4.3694 | 0.00 | guessing (ln 67 = 4.20) |
| 250 | 4.1M | 1.4440 | 1.5074 | +0.06 | past the bigram baseline (2.45) |
| 500 | 8.2M | 1.2710 | 1.3608 | +0.09 | |
| 1000 | 16.4M | 1.1252 | 1.2961 | +0.17 | |
| **1250** | **20.5M** | **1.0710** | **1.2888** | +0.22 | **best → `ckpt/best`** |
| 2000 | 32.8M | 0.9047 | 1.3125 | +0.41 | val rising: overfitting |
| 3000 | 49.2M | 0.6855 | 1.4233 | +0.74 | |
| 4000 | 65.5M | 0.5268 | 1.5400 | +1.01 | |
| 5000 | 81.9M | 0.4424 | 1.6253 | +1.18 | end |

Three phases. Until ~step 1250 both losses fall together: the model is learning things that are
true of Shakespeare in general. After that, training loss keeps falling while validation loss
rises: it is learning things that are true only of *these* 1M characters. By then it has seen the
training text ~20 times, and it will see it ~80 times by the end.

**Why the trainer keeps `ckpt/best`:** the final checkpoint is the *worst* model by validation loss
since step 250. Without a separately kept best checkpoint, a run that overfits would hand you its
most over-confident version. `slm sample` uses `best` by default for this reason.

**Compared with nanoGPT.** nanoGPT reports ~1.47 on this data with a GPT-2-style model of the
same size. slmkit's best is 1.29. The curve has the same shape (fast fall, minimum, overfit), but the
numbers are not directly comparable: slmkit validates on 11 whole scenes spread through the file,
nanoGPT on the last 10% of it, and slmkit's model is Llama-style (RoPE, SwiGLU). Leakage was ruled
out separately: no 50-character passage of validation text occurs in training (the-training-loop.md
§2).

---

## D.3 Is it copying? Measure, don't guess

A rising validation loss is often described as "memorizing". Test that directly. Generate 3,000
characters from the best and from the final checkpoint, then count how much appears verbatim in
the training text:

```bash
uv run slm sample run-a1d5 --which best   --prompt "ROMEO:\n" --tokens 3000 --seed 0 > /tmp/best.txt
uv run slm sample run-a1d5 --which latest --prompt "ROMEO:\n" --tokens 3000 --seed 0 > /tmp/latest.txt
uv run python - <<'PY'
import json
from pathlib import Path
D = Path.home() / "slm/datasets/shakespeare_char/ds-bdaf3875e330"
train = "".join(json.loads(line)["text"] for line in open(D / "train.jsonl"))
for name in ("best", "latest"):
    text = Path(f"/tmp/{name}.txt").read_text().split("\n", 1)[1]
    for n in (20, 40):
        seen = {train[i:i + n] for i in range(len(train) - n)}
        windows = [text[i:i + n] for i in range(len(text) - n)]
        print(f"{name:6s} {n}-char windows found in training text: "
              f"{sum(w in seen for w in windows) / len(windows):.1%}")
PY
```
```
best   20-char windows found in training text: 3.0%
best   40-char windows found in training text: 0.0%
latest 20-char windows found in training text: 3.1%
latest 40-char windows found in training text: 0.0%
```

**It is not copying.** Neither checkpoint reproduces any 40-character passage, and the final
model copies about as much as the best one (3.1% vs 3.0% of 20-character windows, mostly common
phrases). The step-4247 checkpoint scored 2.4%: the same picture. What validation loss measures here is **over-confidence**.
The final model puts very high probability on patterns specific to the training text, and
cross-entropy punishes confident mistakes on unseen text heavily, even though its samples still
read as plausible verse:

```
best (step 1250, val 1.29)                     latest (step 5000, val 1.63)
ROMEO:                                         ROMEO:
No part, faith, that same shows fair friends.  Nay, thou canst not know thou art a man:
                                               Let me be long to say 'tis so. Hast thou now
PERDITA:                                       Some one that is destroyed by thy life;
Commend me at my pains: if you be this palace. To teach him by the other instance of a fear,
```

**Why this matters beyond Shakespeare:** loss and sample quality measure different things, and
"val loss went up, so it must be memorizing" is a hypothesis, not a finding. This n-gram novelty
check is the M2 engine grader (`graders/`), shown here by hand first.

---

## D.4 Beyond the pass mark: under- and overfitting side by side

Two more experiments, each one YAML file differing from `ref` only in the model, show all three fitting
regimes on the same data (under a minute of GPU each):

```bash
uv run slm pretrain shakespeare_char/underfit   # 13K parameters
uv run slm pretrain shakespeare_char/nano       # 0.85M, no dropout
uv run python scripts/val_metrics.py run-f6d4 run-d020 run-a1d5
```
```
baseline: letter frequencies       loss 3.3264  ppl  27.84  bpc 4.799  top1  15.2%  top5  40.3%
shakespeare-underfit (run-f6d4)    loss 1.8890  ppl   6.61  bpc 2.725  top1  43.7%  top5  78.4%
shakespeare-nano (run-d020)        loss 1.3764  ppl   3.96  bpc 1.986  top1  58.3%  top5  86.2%
shakespeare-ref (run-a1d5)         loss 1.2894  ppl   3.63  bpc 1.860  top1  60.7%  top5  87.7%
```

Then regenerate the figures from those runs (`uv sync --extra docs` once):

```bash
make figures
```

What the curves mean is in [`../concepts/fitting.md`](../concepts/fitting.md), and what each metric
means is in [`../concepts/metrics.md`](../concepts/metrics.md).

---

## M1: exit criteria

- [x] Val loss ≤ 1.55, with a curve shaped like nanoGPT's: **1.2888** at step 1250 (D.2).
- [x] Stop mid-run and restart: resumes with no jump in loss or learning rate (D.1). Tested as a
      process restart (Ctrl-C, then a fresh process loading the checkpoint from disk), not a
      machine reboot. A reboot additionally tests that the fsync'd files survive a power cut.
- [x] MFU logged and plausible for the size: 52.9% at `ref` (Phase C, C.1).
- [x] `make test` on the CPU in under 60 s: 89 tests in ~8 s.

---

# See it in slm studio

The reference model can be typed into. Export its best checkpoint (validation loss 1.2888, step 1250),
then open it in the studio's Playground with **shakespeare_char** selected:

```bash
uv run slm export run-a1d5 --name shakespeare --version 1
```
```
checking parity on 1.tmp ...
  slmkit round trip: max |Δlogit| = 0.0e+00 over 71 tokens
  transformers 5.17.0: max |Δlogit| = 0.0e+00, tokenizer IDs match: True
exported shakespeare:1 -> ~/slm/models/shakespeare/1
```

Press **Try: ROMEO**, or type the start of a speech (`JULIET:` and a new line, then `O Romeo, Romeo,`).
It writes verse in Shakespeare's style, made-up words included: 10.6M parameters, one character at a
time, trained on 1 MB. On the GPU the studio writes about 190 characters a second. What to try, and why it
can't chat: [`../concepts/two-models.md`](../concepts/two-models.md) §4.

| Page | What of M1 you'll see |
|---|---|
| **Training** | the reference run's curves: validation loss bottoming out at step 1250 and climbing after (overfitting, fitting.md), and "watch it learn" from random characters to verse |
| **Lifecycle** | the Shakespeare text, its char tokenizer and packed data, and the three M1 runs |
| **Parameters** | the `ref` preset: 10.6M parameters, and what this GPU measured for it |
