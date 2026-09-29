# The model

*What slmkit builds: the model type, its architecture features, its hyperparameters, the size
presets, and the files a trained model ends up as. It is the specification the code in
`src/slmkit/model/` implements (M1 Phase B). For terms, see [`GLOSSARY.md`](GLOSSARY.md); for the
walk-through of the actual code, `concepts/the-model.md` (lands with Phase B).*

---

## At a glance

| Property | Value |
|---|---|
| **Type** | Decoder-only transformer, autoregressive **causal language model** |
| **Family** | Llama-style: pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings |
| **Compatibility target** | Hugging Face `LlamaForCausalLM`: same parameter names and maths |
| **Sizes** | 6 presets, **0.85M → 304M** parameters (excluding embeddings) |
| **Vocabulary** | Per project: 67 (Shakespeare: 65 characters + 2 special tokens) · 87 char or 512 BPE (ABC music) · ~1,970 (chess moves) |
| **Context length** | 256–1,024 tokens, set per experiment |
| **Training precision** | bf16 mixed precision (fp32 weights and optimizer, bf16 matmuls) |
| **Resume checkpoint** | Directory of PyTorch state files; slmkit-only |
| **Export format** | HF directory: `config.json` + `model.safetensors` + tokenizer + model card |
| **Later formats** | GGUF, for llama.cpp and similar runtimes (M4) |
| **Hardware target** | One 12–24 GB consumer NVIDIA GPU for training; CPU is enough to serve |

**One family only** (CONTRIBUTING.md rule 5). Every project, from Shakespeare to chess, uses this same
architecture at a different size and with a different vocabulary. That is what lets one engine,
one exporter and one serving path cover all of them.

---

## 1. What "decoder-only causal language model" means

The model does exactly one thing: **given a sequence of tokens, predict a probability for every
possible next token.** Generating text is running that prediction in a loop: pick a token, append
it, predict again.

Worked example with the character tokenizer on Shakespeare (vocabulary of 65 characters):

```
input tokens:      "R O M E"              (4 token IDs)
model output:      4 rows × 65 scores     (one row per position: "what comes after this prefix?")
row for "ROME":    'O' 0.91 · 'S' 0.03 · 'L' 0.01 · ...   → sample 'O' → "ROMEO"
```

- **Decoder-only**: there is no separate encoder reading an input. Prompt and answer are one
  stream of tokens. The original 2017 transformer had both halves; GPT, Llama and nearly every
  modern LLM keep only the decoder.
- **Causal**: position *i* may only look at positions ≤ *i*. A mask enforces this during training,
  so the model can't cheat by looking at the token it is supposed to predict. It is what lets one
  training sequence of 256 tokens give 256 prediction exercises at once.
- **Autoregressive**: at inference time, each output token becomes input for the next.

Every capability slmkit's models get, whether writing a jig in E minor or playing a legal chess
move, comes from this next-token objective applied to the right data.

---

## 2. Architecture, block by block

```
token IDs ──► Embedding (vocab × d_model) ─────────────────────────────┐
                                                                        │  x
   ┌──────────────── repeated n_layers times ─────────────────────┐    │
   │  x ──► RMSNorm ──► Attention (causal, RoPE on q/k) ──► + ──►  │ ◄──┘
   │  └──────────────────── residual ──────────────────────┘      │
   │  x ──► RMSNorm ──► SwiGLU MLP ─────────────────────────► + ──►│
   │  └──────────────────── residual ──────────────────────┘      │
   └───────────────────────────────────────────────────────────────┘
        ──► final RMSNorm ──► LM head (d_model × vocab, weight shared with Embedding) ──► logits
```

Each feature, why slmkit uses it, and what it replaces from the original GPT-2 design that
nanoGPT follows:

| Feature | What it does | Why this choice | GPT-2 used instead |
|---|---|---|---|
| **Token embedding** | Looks up a learned vector of length `d_model` for each token ID | The standard way to turn IDs into numbers the network can work with | same |
| **Pre-norm** | Normalizes the *input* to each sub-layer, not its output | Much more stable to train; gradients flow cleanly through the residual path | same (GPT-2 moved to pre-norm too) |
| **RMSNorm** | Rescales a vector by its root-mean-square, then a learned per-channel gain | Simpler and slightly cheaper than LayerNorm (no mean subtraction, no bias) with the same quality | LayerNorm |
| **Residual connections** | Each sub-layer *adds* to `x` rather than replacing it | Deep networks train at all only because of this; the network learns corrections | same |
| **Multi-head causal self-attention** | Each position mixes in information from earlier positions, weighted by learned relevance, in `n_heads` independent heads | The core transformer mechanism | same |
| **RoPE** (rotary position embedding) | Encodes position by rotating query/key vectors by a position-dependent angle | No parameters, encodes *relative* distance directly, and extends more gracefully than learned positions | Learned absolute position table |
| **SDPA** | PyTorch's fused attention kernel (`scaled_dot_product_attention`) | Fast and memory-efficient without the `flash-attn` package (see STACK.md) | hand-written attention |
| **GQA-ready** (`n_kv_heads`) | Lets several query heads share one key/value head | Shrinks the KV cache for inference. All presets currently set `n_kv_heads = n_heads` (plain multi-head attention); the knob exists so no architecture change is needed later | — |
| **SwiGLU MLP** | Two parallel projections, one gated by SiLU, multiplied, then projected back | Consistently better quality per parameter than a plain MLP. Uses three matrices, so the width is 8/3 × `d_model` instead of 4× to keep the parameter count equal | GELU MLP, 4× width |
| **No bias terms** | Linear layers are pure matrix multiplies | Biases add little in transformers; removing them matches Llama and simplifies export | biases everywhere |
| **Tied embeddings** | The output layer reuses the embedding matrix | Halves embedding parameters. Matters for small models, where embeddings can be a large share | same (tied) |
| **Dropout** | Randomly zeroes activations during training | Regularization for tiny datasets only (Shakespeare, ABC); 0.0 wherever data is plentiful | same |

**Why the Llama shape rather than GPT-2's?** Two reasons. It is the architecture nearly all
current open models descend from, so what you learn transfers. And being parameter-compatible
with `LlamaForCausalLM` means the whole ecosystem (HF `transformers`, llama.cpp, vLLM) can load
slmkit models without custom code. **Rejected:** GPT-2 exactly (older design, and export would need
a second architecture), Mixture-of-Experts (routing complexity with no benefit below ~1B
parameters), and encoder-decoder models (no advantage for pure generation).

**An export detail that bites.** Two conventions exist for applying RoPE: interleaved pairs
(Meta's original code) and split halves (Hugging Face's `rotate_half`). They give different
numbers from identical weights. slmkit uses the **HF convention**, so exported logits match
`transformers` exactly. `tests/unit/test_hf_parity.py` loads slmkit weights into HF's
`LlamaForCausalLM` and checks that the logits match; the measured difference is 0.0 in fp32.

### Where training starts: initialization

Every weight matrix and the embedding start as random Normal(0, 0.02). The two projections
that write into the residual stream (`o_proj`, `down_proj`) start smaller, at 0.02 / √(2 ×
n_layers), so the running sum of 2 × n_layers contributions doesn't grow with depth. RMSNorm
gains start at 1. This is GPT-2's scheme, and nanoGPT's, which M1 has to match. The visible
result: an untrained model's loss is close to **ln(vocab_size)** (4.20 for 67 tokens), meaning it
is almost equally unsure of every token. `slm model` measures it (4.37 on real Shakespeare);
a starting loss far above ln(V) means the initialization is broken. Details and rejected
alternatives: `src/slmkit/model/init.py`.

---

## 3. Hyperparameters

Where each value is set determines who can change it: a preset (shared), the tokenizer (derived),
or an experiment YAML (per run).

| Name | Meaning | Set by | Typical |
|---|---|---|---|
| `n_layers` | Number of transformer blocks stacked | model preset | 4–24 |
| `d_model` | Width of the vector carried through the network; the main size knob | model preset | 128–1024 |
| `n_heads` | Attention heads per layer; `head_dim = d_model / n_heads` | model preset | 4–16 (head_dim 32–64) |
| `n_kv_heads` | Key/value heads (GQA); equal to `n_heads` = plain multi-head attention | model preset | = `n_heads` |
| `ffn_hidden` | Inner width of the SwiGLU MLP: `8/3 × d_model`, **rounded up** to a multiple of 64 (GPU-friendly shapes) | model preset | 384–2752 |
| `dropout` | Fraction of activations zeroed in training | model preset, overridable | 0.0–0.2 |
| `tie_embeddings` | Share embedding and output matrices | model preset | `true` |
| `vocab_size` | Number of distinct tokens | **the tokenizer artifact** (never hand-set) | 65–~2K |
| `block_size` | Context length: max tokens the model sees at once | experiment (`data.block_size`) | 256–1,024 |
| `rope_theta` | RoPE base frequency; controls how quickly rotation angles change with position | engine default | 10,000 |
| `norm_eps` | Small constant in RMSNorm to avoid division by zero | engine default | 1e-5 |

`vocab_size` comes from the tokenizer so a model can never disagree with the tokenizer it is
paired with. The same rule applies in Terraform, where an output feeds another module's input
rather than being copied by hand.

---

## 4. Size presets

Parameter counts **exclude embeddings** (the convention used for scaling comparisons, since
embedding size depends on the vocabulary, not the model). Exact values, computed from
`presets/model/*.yaml`:

| Preset | Layers | d_model | Heads (head_dim) | ffn_hidden | **Params (non-emb.)** | Per layer | Intended use |
|---|---|---|---|---|---|---|---|
| `nano` | 4 | 128 | 4 (32) | 384 | **853,120** (0.85M) | 213K | Unit tests; ABC baseline |
| `micro` | 6 | 256 | 8 (32) | 704 | **4,820,224** (4.8M) | 803K | ABC music |
| `ref` | 6 | 384 | 6 (64) | 1024 | **10,621,824** (10.6M) | 1.77M | M1 Shakespeare reference (nanoGPT's dimensions) |
| `tiny` | 8 | 512 | 8 (64) | 1408 | **25,698,816** (25.7M) | 3.2M | Chess |
| `small` | 12 | 768 | 12 (64) | 2048 | **84,953,856** (85.0M) | 7.1M | Chess scaling; GPT-2-small class |
| `medium` | 24 | 1024 | 16 (64) | 2752 | **303,612,928** (303.6M) | 12.7M | Late chess experiments only |

### Worked example: counting `ref` by hand

Per layer, with `d = 384`, `ffn = 1024`:

```
attention   q, k, v, o projections      4 × 384 × 384        =   589,824
SwiGLU MLP  gate, up, down projections  3 × 384 × 1024       = 1,179,648
RMSNorm     two gain vectors            2 × 384              =       768
                                                     per layer 1,770,240
6 layers + final RMSNorm (384)                                10,621,824
embedding (tied, shared with LM head)   67 × 384             =    25,728
                                                     total    10,647,552  (≈ nanoGPT's 10.65M)
```

The vocabulary is 67, not 65: Shakespeare has 65 distinct characters, and every slmkit
tokenizer adds `<unk>` and `<eos>` (see `concepts/tokenization.md` §3). **Check it yourself:**
`uv run slm model shakespeare_char/ref` prints exactly these numbers from the real model.

Two-thirds of every block is the MLP. That ratio holds at every size, and it is why "a model is
mostly matrix multiplies" is literally true.

### Total parameters depend on the vocabulary

| Preset | char (65) | small BPE (1,024) | chess (~1,972) | BPE 8K (for scale) |
|---|---|---|---|---|
| `nano` | 0.86M | 0.98M | 1.11M | 1.90M |
| `micro` | 4.84M | 5.08M | 5.33M | 6.92M |
| `ref` | 10.65M | 11.02M | 11.38M | 13.77M |
| `tiny` | 25.7M | 26.2M | 26.7M | 29.9M |
| `small` | 85.0M | 85.7M | 86.5M | 91.3M |
| `medium` | 303.7M | 304.7M | 305.6M | 312.0M |

At `nano` with an 8K vocabulary, over half the model would be embeddings, which is one reason
slmkit's projects keep vocabularies small.

### Which size for which project

A useful rule of thumb from the "Chinchilla" scaling results: training is compute-efficient at
about **20 tokens of data per parameter**. Use it as a sanity check, not a law:

| Project | Data | Preset(s) | Tokens / param | Consequence |
|---|---|---|---|---|
| Shakespeare (M1) | ~1M chars | `ref` | ~80M seen ÷ 10.6M ≈ 8, *but* only ~1M unique | Many epochs over tiny data → heavy dropout (0.2); memorization is expected, the point is the loss curve |
| ABC music (M2) | 5–20M tokens | `nano`, `micro` | ~1–4 per epoch at `micro` | Data-limited: dropout 0.1–0.2, early stopping, stop by ~4 epochs |
| Chess (M3) | billions available | `tiny` → `small` (→ `medium`) | as many as compute allows | Compute-limited: the budget, not the data, sets the size |
| Cricket (M5) | bounded ball events | `nano`–`tiny` | depends on corpus | Sized when the data is measured |

---

## 5. Compute and memory per size

**Compute.** Training costs about **6 × params × tokens** FLOPs (2 for the forward pass, 4 for
the backward pass), plus 10–15% for attention at 1K context. With the planning figure of ~43
TFLOPS effective:

| Preset | FLOPs per training token | 100M tokens | 1B tokens |
|---|---|---|---|
| `ref` | 64M | ~2.5 GPU-min | ~25 GPU-min |
| `tiny` | 154M | ~6 GPU-min | ~1 GPU-hour |
| `small` | 510M | ~20 GPU-min | ~3.3 GPU-hours |
| `medium` | 1.8G | ~70 GPU-min | ~12 GPU-hours |

Measured MFU (M1, compiled bf16, 16,384 tokens per step) is 24% at `nano`, 42% at `micro`, 51% at
`ref` and 54% at `tiny`. From `micro` up, real throughput **beats** the 43 TFLOPS planning figure
(`ref` runs at 57), so this table is conservative there; only `nano` runs slower. The trainer
replaces the estimate with measured tokens/sec after 50 steps. Details:
`concepts/the-training-loop.md` §8.

**Memory.** Training with AdamW in mixed precision needs about **16 bytes per parameter**: fp32
weights (4), gradients (4) and two Adam moment buffers (4 + 4). Activations (intermediate values
kept for the backward pass) come on top and grow with batch size × context length.

| Preset | Weights (fp32) | Training state (~16 B/param) | Fits in 16 GB? |
|---|---|---|---|
| `ref` | 43 MB | 0.17 GB | trivially |
| `small` | 346 MB | 1.4 GB | yes, large batches |
| `medium` | 1.2 GB | 4.9 GB | yes; activations dominate |

This is why DESIGN says **memory is not the binding constraint, compute is**: even `medium` uses
under a third of the card before activations.

---

## 6. Files a model becomes

A model passes through three file formats in its life. They exist for different readers.

### During training: resume checkpoints (slmkit-only)

```
$SLM_HOME/runs/<run_id>/
├── config.resolved.yaml        the exact merged config (the "tfvars" actually applied)
├── manifest.json               lineage: which packed data and tokenizer artifacts
├── ckpt/step_000500/           one directory per checkpoint, written as *.tmp then renamed
│   ├── model.pt                model weights (fp32)
│   ├── optimizer.pt            AdamW moment buffers
│   └── state.pt                step, tokens seen, scheduler, sampler position, all RNG states
└── tb/                         TensorBoard event files
```

- **Format:** PyTorch `torch.save`, which is Python pickle.
- **Size:** about **12 bytes/param** (weights plus two optimizer moments; gradients aren't saved).
  `ref` ≈ 130 MB, `small` ≈ 1 GB, `medium` ≈ 3.7 GB.
- **Why pickle is acceptable here:** only slmkit writes and reads these, and they hold Python
  state (RNG states, sampler position) that safetensors can't represent. **Never load someone
  else's `.pt` file**, because unpickling can execute code.
- **Why a directory, not one file:** it is written as `ckpt.tmp/`, fsynced, then renamed. A
  power cut mid-save leaves either the old checkpoint or the new one, never a half-written file.

### After training: the export (M2), for everyone else

Written by `slm export <run> --name <name> --version <n>` (`src/slmkit/export/hf.py`):

```
$SLM_HOME/models/<name>/<version>/
├── config.json                 HF LlamaConfig: sizes, vocab_size, rope_theta, norm eps, eos id
├── generation_config.json      default sampling: temperature, top_k, top_p, max_new_tokens
├── model.safetensors           weights only, fp32
├── tokenizer.json              HF tokenizers format (char → WordLevel, BPE unchanged)
├── tokenizer_config.json       lets AutoTokenizer find <unk> and <eos>
├── MODEL_CARD.md               how to prompt it, training, eval results, parity, limitations
└── manifest.json               lineage (run, tokenizer) and the parity check results
```

- **Format:** **safetensors**, a JSON header plus raw tensor bytes. It can't execute code, loads by
  memory-mapping, and is the ecosystem standard.
- **Precision:** fp32. At these sizes the file is small: the `abc_music` nano model is 3.5 MB, `ref`
  would be about 43 MB and `small` about 346 MB. bf16 would halve that; it isn't implemented, because
  nothing here is short of disk and fp32 keeps the parity check exact.
- **Tied head stored once.** With tied embeddings `lm_head.weight` is the embedding tensor. safetensors
  can't store two names for one tensor, so the head is left out and HF re-ties it on load
  (`tie_word_embeddings: true`). The nano export holds 38 tensors: the embedding, the final norm, and 9
  per layer.
- **Versions are immutable.** Exporting the same checkpoint again is a no-op; a different checkpoint
  needs a new version number. Serving addresses models as `name:version`.
- **Parameter names** follow `LlamaForCausalLM`, so loading it needs no custom code:

  | slmkit tensor | HF name |
  |---|---|
  | token embedding | `model.embed_tokens.weight` |
  | per-layer attention norm | `model.layers.{i}.input_layernorm.weight` |
  | q / k / v / o projections | `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` |
  | per-layer MLP norm | `model.layers.{i}.post_attention_layernorm.weight` |
  | SwiGLU gate / up / down | `model.layers.{i}.mlp.{gate,up,down}_proj.weight` |
  | final norm | `model.norm.weight` |
  | output head | `lm_head.weight` (tied: same tensor as the embedding, not stored) |

- **Proof it is right:** every export is loaded back twice before it is published. slmkit must get
  identical logits; `AutoModelForCausalLM` and `AutoTokenizer` must get the same token IDs and logits
  within 1e-4. Measured on the three `abc_music` exports: a difference of exactly 0.0 (same PyTorch
  kernels, same order of operations). See concepts/serving.md.

### Optional: GGUF (M4)

A single file for llama.cpp and runtimes built on it, usually **quantized** (weights stored in
4–8 bits). At Q8_0 it is about 1 byte/param, so `small` is around 90 MB. It is converted from
the HF export with llama.cpp's tools. Character-level and chess vocabularies may need a custom
tokenizer mapping, verified per project.

| Format | Written by | Read by | Contains | Safe to load from strangers? |
|---|---|---|---|---|
| resume checkpoint (`.pt`) | trainer, every 20 min | trainer only | weights + optimizer + RNG + sampler | **No** (pickle) |
| HF export (`.safetensors`) | `slm export` | slmkit serve, `transformers`, vLLM | weights + config + tokenizer | Yes |
| GGUF | llama.cpp converter | llama.cpp, Ollama, etc. | quantized weights + tokenizer | Yes |

---

## 7. Deliberately out of scope

| Not doing | Why |
|---|---|
| Other architectures (GPT-2, Mamba, RWKV, MoE) | One family keeps one exporter and one serving path. Changing it needs an ADR (rule 5) |
| Models above ~350M parameters | Compute: 1B would take ~775 GPU-hours (DESIGN §2) |
| Long context (> 2K tokens) | No project needs it; attention cost grows with the square of the length |
| Fine-tuning someone else's pretrained model | The point is training from random initialization; SFT only ever runs on slmkit's own base models |
| Quantization-aware training | Quantization happens only at export (GGUF), where it is standard practice |
| Multi-GPU | One GPU, one process (rule 7) |
