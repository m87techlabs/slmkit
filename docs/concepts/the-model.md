# The model, walked through the code

*Milestone M1, Phase B. The specification is [`../MODEL.md`](../MODEL.md); this page explains how
[`src/slmkit/model/llama.py`](../../src/slmkit/model/llama.py) works and why. Terms:
[`../GLOSSARY.md`](../GLOSSARY.md). To check it on your machine:
[`../runbooks/m1-engine.md`](../runbooks/m1-engine.md) §B.*

The whole model is about 200 lines of PyTorch. This page follows one batch through it, using the
real `ref` preset trained on Shakespeare, with the tensor shape at every step.

```
B = 64 sequences per batch     T = 256 tokens each     D = 384 (d_model)
H = 6 heads                    Hd = 64 (head_dim)       F = 1024 (ffn_hidden)     V = 67 (vocab)
```

---

## 1. The journey of one batch

```
token IDs                        (64, 256)          integers in [0, 67)
  │ embed_tokens                 (64, 256, 384)     one learned 384-vector per token
  │
  │  ┌ 6 x Block ───────────────────────────────────────────────────────────┐
  │  │ x = x + Attention(RMSNorm(x))    (64, 256, 384)  mix in earlier tokens │
  │  │ x = x + MLP(RMSNorm(x))          (64, 256, 384)  transform each token  │
  │  └───────────────────────────────────────────────────────────────────────┘
  │ norm                         (64, 256, 384)
  │ lm_head                      (64, 256, 67)      a score for every possible next character
  ▼
logits  →  cross-entropy against targets (64, 256)  →  one number: the loss
```

Two things are worth noticing before any detail:

- **The shape `(64, 256, 384)` never changes inside the stack.** Every block reads a tensor of that
  shape and writes one of the same shape. That fixed shape is the **residual stream**, and it is
  the central idea of the whole architecture (§3).
- **The model makes 64 × 256 = 16,384 predictions per batch**, one for every position, and all of
  them in parallel. Because of the causal mask (§5), position 100 is predicting token 101 using
  only tokens 0–100, so each position is an honest exercise.

---

## 2. Embedding: from IDs to vectors

```python
self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)   # a (67, 384) table
x = self.embed_tokens(idx)                                        # row lookup
```

An embedding is a lookup table: token 32 (`R`) becomes row 32, a vector of 384 numbers. The rows
start random and are learned. After training, characters that behave alike (vowels, say, or
punctuation that ends a line) end up with similar vectors, because that helps prediction.

**Tied embeddings:** `lm_head.weight = embed_tokens.weight`. The same table is used again at the
end to score how well the final vector matches each token. It is one matrix with two jobs, which
saves 67 × 384 parameters here and ~40M at GPT-2 scale.

---

## 3. The residual stream: the model's shared bus

```python
x = x + self.self_attn(self.input_layernorm(x), rope)
x = x + self.mlp(self.post_attention_layernorm(x))
```

Each sub-layer reads the stream, computes something, and **adds** its result back. Nothing
overwrites `x`. A useful infrastructure picture is an append-only event log: each of the 12
sub-layers (6 attention, 6 MLP) reads the current state and contributes a delta.

Why it matters:
- **Training works through depth.** In the backward pass the `+` sends the gradient straight
  through to earlier layers unchanged. Without residuals, a gradient passes through every
  layer's transformation and tends to shrink to nothing or blow up.
- **Layers start as near no-ops.** With small initial weights (§8), each block adds very little at
  first, so the untrained model behaves roughly like "embedding → output" and deepens its
  computation gradually as it learns.

**Pre-norm:** the norm is applied to the *input* of each sub-layer (`attn(norm(x))`), not to the
stream itself. The stream keeps its raw scale, and each sub-layer sees well-scaled input. The
original 2017 transformer normalized *after* adding (post-norm), which is much harder to train
deep.

---

## 4. RMSNorm: keep every vector at a sensible scale

```python
x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
return self.weight * x32.to(dtype)
```

Divide the vector by its root-mean-square, then multiply by a learned per-channel gain. A worked
example with a 2-dimensional vector:

```
x = [3, 4]     rms = sqrt((9 + 16) / 2) = 3.54     x / rms = [0.85, 1.13]     (rms now 1.0)
```

Without it, the scale of the stream drifts as 12 sub-layers add to it, and the next matrix
multiply amplifies the drift. The arithmetic is done in fp32 even under bf16 autocast, because
squaring and averaging 384 numbers is exactly where bf16's ~3 significant digits lose accuracy.

---

## 5. Attention: gather information from earlier tokens

This is the only place tokens exchange information. Everything else processes each position on
its own.

```python
q = self.q_proj(x)   # query: "what am I looking for?"
k = self.k_proj(x)   # key:   "what do I contain?"
v = self.v_proj(x)   # value: "what do I hand over if selected?"
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

For each position, attention scores every earlier position by `q · k / √Hd`, turns the scores into
weights with a softmax, and returns the weighted sum of their values. When predicting the letter
after `ROME`, a query at `E` might score the earlier `R`, `O`, `M` highly and pull in their values:
this looks like the start of a name.

**The causal mask** (`is_causal=True`) sets the score of every *later* position to −∞ before the
softmax, so those positions get weight 0:

```
            key: t0   t1   t2   t3
query t0         ✓    ·    ·    ·
query t1         ✓    ✓    ·    ·        ✓ = may attend    · = masked to −∞
query t2         ✓    ✓    ✓    ·
query t3         ✓    ✓    ✓    ✓
```

This is what makes 256 parallel predictions honest. `test_causal_future_tokens_cannot_change_the_past`
changes tokens 12 onwards and checks that the logits at positions 0–11 are bit-for-bit
unchanged.

**Heads.** The 384 dimensions are split into 6 heads of 64. Each head runs its own attention
with its own queries and keys, so one can track "which speaker is this" while another tracks
"am I inside a word". Their outputs are concatenated and mixed by `o_proj`.

**SDPA** (`scaled_dot_product_attention`) does scores, mask, softmax and weighted sum as one fused
GPU kernel, without ever storing the full 256 × 256 score matrix per head. The result is
identical; it is simply faster and uses less memory than writing the four steps out.

**GQA** (`n_kv_heads < n_heads`): several query heads share one key/value head, shrinking what must
be cached during generation. All presets use plain multi-head attention today; the code path
exists and is tested against Hugging Face.

---

## 6. RoPE: how the model knows word order

Attention on its own is order-blind: `q · k` is the same wherever the two tokens sit. RoPE fixes
this by **rotating** each query and key by an angle proportional to its position before the dot
product:

```python
angles = position * theta ** (-2i / Hd)      # a different speed for each pair of dimensions
x * cos + rotate_half(x) * sin               # rotate each pair by its angle
```

The useful property: rotating `q` by angle *a·m* and `k` by *a·n* makes their dot product depend
only on **m − n**, the distance between the tokens. The model learns patterns like "the token 3
positions back" that hold anywhere in the sequence, without a learned position table.

Dimension pairs rotate at different speeds, fast in the first pairs and slow in the last, like
the hands of a clock. That lets nearby and distant positions both stay distinguishable.

**The convention matters.** slmkit pairs dimension *i* with *i + 32* (`rotate_half`, Hugging Face's
choice), not *2i* with *2i + 1* (Meta's original). Both work, but a model trained with one gives
wrong answers when run with the other. `test_hf_parity.py` catches this: the logits match HF
exactly (difference 0.0), and changing the RoPE base breaks the match, which proves the test is
sensitive.

---

## 7. SwiGLU MLP: where most of the parameters live

```python
self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Each position is processed independently: widen from 384 to 1,024, apply a gated non-linearity,
narrow back to 384. `up_proj` produces candidate features; `silu(gate_proj(x))` decides, per
feature, how much of each passes. With three matrices of 384 × 1,024 each, the MLP holds
1,179,648 of each block's 1,770,240 parameters, **two-thirds**. Attention decides *where* to look;
the MLP is where most of what the model knows is stored.

**Why 8/3 × D wide:** a plain GELU MLP uses two matrices at 4 × D. SwiGLU uses three, so 8/3 × D
keeps the parameter count the same (3 × 8/3 = 2 × 4), rounded up to a multiple of 64 for
GPU-friendly shapes: 8/3 × 384 = 1,024 exactly.

---

## 8. From vectors to a loss

```python
logits = self.lm_head(self.model(idx))                 # (64, 256, 67)
loss = F.cross_entropy(logits.float().view(-1, 67), targets.reshape(-1))
```

**Logits** are raw scores, one per vocabulary entry. A softmax turns them into probabilities, and
**cross-entropy** is −ln(probability the model gave the correct next token), averaged over all
16,384 positions.

The numbers to know for Shakespeare:

| Loss | Meaning | Perplexity (e^loss) |
|---|---|---|
| **4.20** = ln 67 | Uniform guessing: every character equally likely | 67 |
| 3.31 | Knows letter frequencies, nothing else (unigram) | 27 |
| 2.45 | Knows which letter follows which (bigram) | 12 |
| **~1.47** | nanoGPT's reference: spelling, words, speaker format | ~4.3 |

The 3.31 and 2.45 rows are computed exactly from the Shakespeare text: they are the losses of
the best possible model that looks at 0 and 1 previous characters. A trained model that can't beat
2.45 has learned nothing a lookup table couldn't.

Perplexity is "how many characters the model is effectively choosing between". At 1.47 it is
about 4 per position, down from 67.

**Initialization sets the starting point.** Weights start as Normal(0, 0.02), and the two
projections that write into the residual stream start at 0.02/√(2 × 6), so the stream's
variance doesn't grow with depth (details in `model/init.py`). The result: before any training
the model is nearly uniform, and `slm model` measures **4.37** against the ideal 4.20. The small
excess is because random logits aren't exactly equal. A starting loss of 8 or 10 would mean the
initialization is too large, and that is the first thing to check if a run starts badly.

**Targets of `-100` are ignored.** Nothing in pretraining uses that, but SFT (M2) marks prompt
tokens that way so the model is only graded on the answer. `test_ignore_index_is_excluded_from_the_loss`
already pins the behaviour.

---

## 9. Dropout

With `dropout: 0.2` (the `ref` preset), 20% of activations are zeroed at random during training,
at three points: after the embedding, inside attention, and after each sub-layer's output
projection. It only happens in `model.train()` mode; `model.eval()` turns it off. It stops the
model leaning on any single pathway, which matters when a 10.6M-parameter model sees the same
1M characters ~80 times. Where data is plentiful (chess), presets set it to 0.

---

## 10. On the GPU: bf16 and `torch.compile`

- **bf16 autocast.** Inside `torch.autocast("cuda", dtype=torch.bfloat16)` the matrix multiplies run
  in bf16 on the tensor cores, while the weights, RMSNorm and the loss stay fp32. The GPU test
  checks that the bf16 loss matches fp32 within 0.02 and that the weights are still fp32
  afterwards.
- **`torch.compile`** traces the model and fuses small operations (RMSNorm, SiLU × up, the RoPE
  arithmetic) into fewer generated Triton kernels. The model code doesn't change; compiled and
  eager outputs are tested to agree.

**What the GPU tests found.** The first `torch.compile` failed with `Python.h: No such file or
directory`. Triton builds a small C helper at the first compile, and that needs Python's
development headers (`python3-dev`), which M0 never installed because M0 never compiled
anything. The fix went into the setup script, and `slm doctor` now runs a real compile. This is
the general lesson of M0 applied again: **check behaviour, not presence.**

---

## 11. How we know it is right

| Claim | Test | What a failure would mean |
|---|---|---|
| Parameter counts equal MODEL.md for every preset | `test_preset_parameter_counts_match_the_spec` | Architecture drifted from the spec |
| The past cannot see the future | `test_causal_future_tokens_cannot_change_the_past` | Loss looks great, model learned nothing |
| Untrained loss ≈ ln V | `test_initial_loss_is_near_uniform` | Initialization too large or too small |
| It can memorize one batch | `test_overfits_one_batch` (loss 4.2 → < 0.05 in 150 steps) | Wiring bug: shift, mask, loss or optimizer |
| Same function as HF Llama | `test_hf_parity.py` (max difference 0.0) | Export would silently change the model |
| bf16 and compile agree with fp32 eager | `tests/gpu/test_model_gpu.py` | Numerics or compiler problem on this GPU |

**Rejected:** writing the model with HF `transformers`' `LlamaForCausalLM` directly. It would be
correct by definition, but the point of slmkit is to understand what those 200 lines do. HF is
used only as the **reference** the hand-written version is tested against.
