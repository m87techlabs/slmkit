# Glossary

Every abbreviation and piece of jargon used in this repo, in plain language.

This is written for someone comfortable with software and infrastructure but **new to machine
learning**. Terms are grouped by what they belong to rather than alphabetically, because most of
them only make sense next to their neighbours. If you want alphabetical, use your browser's find.

---

## The big picture

**LLM — Large Language Model.** A neural network trained to predict the next piece of text given
everything before it. That single objective, at scale, is enough to produce something that can
answer questions and write code.

**SLM — Small Language Model.** The same architecture and training method, but small enough for
one consumer GPU — roughly 1M to 3B parameters. An SLM cannot chat about anything, but on a
narrow task with a well-defined output format it can be genuinely good. **slmkit** = "small
language model kit".

**Parameter.** One number inside the model that training adjusts. "A 124M model" has 124 million
of them. Parameter count is the main dial for capability and cost.

**Pretraining.** The first and longest stage: show the model a large corpus and train it to
predict the next token. Produces a **base model**, which continues text but does not follow
instructions.

**SFT — Supervised Fine-Tuning.** A second, much shorter training stage on pairs of
(prompt, desired response). Teaches the base model to respond to an instruction rather than just
continue text. Everything downstream of this — RLHF, DPO — is out of scope for slmkit.

**Inference.** Running a trained model to produce output. Also called serving or generation.

**The lifecycle.** slmkit's pipeline, and the thing this repo exists to teach:
`ingest → prepare → tokenizer → pack → pretrain → SFT → eval → export → serve`.

---

## Turning text into numbers

**Token.** The unit a model actually reads. Depending on the tokenizer, a token is a character,
a word fragment, or a whole domain symbol. Models never see text — only token ID numbers.

**Tokenizer.** The reversible mapping between text and token IDs. Designing one is a real choice
with real consequences, which is why slmkit trains its own rather than borrowing one.

**Vocabulary (vocab).** The full set of tokens a tokenizer knows, and thus the size of the
model's output layer. slmkit stores token IDs as `uint16`, which caps vocab at 65,536.

**Character-level tokenizer.** One token per character. Tiny vocab, no unknown inputs, but
sequences get long. Good for small corpora.

**BPE — Byte Pair Encoding.** Starts from characters and repeatedly merges the most frequent
adjacent pair into a new token, so common strings become single tokens. This is what most
general-purpose LLMs use.

**Fixed vocabulary.** A hand-specified token list for a domain with a known, finite symbol set —
for example one token per legal chess move. Impossible for natural language, ideal when it
applies.

**Context length / block size.** How many tokens the model can see at once. Everything beyond it
is invisible. slmkit calls this `block_size`.

**Embedding.** A lookup table turning each token ID into a vector of numbers. **Tied embeddings**
reuse that same table for the output layer, saving parameters.

---

## Inside the model

**Transformer.** The architecture behind essentially every modern language model. Its core idea
is *attention*.

**Decoder-only.** The transformer variant used for text generation: each position may look at
earlier positions but never later ones. All models in slmkit are decoder-only.

**Attention.** The mechanism letting each position pull in information from earlier positions,
weighted by relevance. "Attention heads" run several of these in parallel, each free to learn a
different relationship.

**SDPA — Scaled Dot-Product Attention.** PyTorch's built-in fused attention
(`torch.nn.functional.scaled_dot_product_attention`). It selects the fastest available backend
automatically, which is why slmkit depends on it instead of the `flash-attn` package.

**FlashAttention.** A memory-efficient attention implementation that avoids materializing the
full attention matrix. Available *through* SDPA; the standalone package is a separate dependency
slmkit deliberately avoids.

**Layer / block.** One attention + feed-forward unit. Models stack many; `n_layers` is the count.

**`d_model`.** The width of the vectors flowing through the model. Together with `n_layers` it
determines nearly all of the parameter count.

**RMSNorm.** A normalization step keeping activations at a stable scale. Cheaper than the older
LayerNorm because it only rescales, without also re-centering.

**RoPE — Rotary Position Embedding.** How the model knows token *order*: position is encoded by
rotating vectors by an angle proportional to position. Extrapolates to longer sequences better
than a learned position table.

**SwiGLU.** The feed-forward design used in Llama-style models. Uses three weight matrices and a
gating activation instead of the classic two, and performs better per parameter.

**KV heads / GQA — Grouped-Query Attention.** Using fewer key/value heads than query heads to
shrink the memory cache during generation. slmkit's presets set `n_kv_heads == n_heads` (no GQA)
because the models are small enough not to need it.

**Logits.** The raw, unnormalized scores the model outputs — one per vocabulary entry. Softmax
turns them into probabilities.

---

## Training

**Loss.** A single number measuring how wrong the model is. Training minimizes it. For language
models this is **cross-entropy**: the negative log probability the model assigned to the token
that actually came next. Lower is better.

**Perplexity.** `exp(loss)` — roughly "how many equally-likely options the model thinks it is
choosing between". A loss of 1.5 means a perplexity of about 4.5.

**Gradient.** The direction and amount each parameter should change to reduce the loss.
Computed by **backpropagation**.

**Step / iteration.** One batch forward, one backward, one parameter update.

**Batch size.** How many sequences are processed per step. Bigger batches give less noisy
gradients and better GPU utilization, up to a memory limit.

**Epoch.** One full pass over the training data. slmkit schedules by *tokens* rather than epochs,
because with augmentation and streaming, "one pass" is not a well-defined unit.

**Optimizer.** The rule for turning gradients into parameter updates. **AdamW** is the standard
choice; it keeps two running statistics per parameter, which is why optimizer state dominates
training memory.

**8-bit Adam.** Storing those statistics in 8 bits instead of 32, roughly halving optimizer
memory. Provided by the `bitsandbytes` library. Verify it against plain AdamW before trusting it.

**LR — Learning Rate.** How big each update is. The single most important hyperparameter: too
high diverges, too low wastes compute. It does not transfer across model sizes.

**Warmup.** Starting at a near-zero learning rate and ramping up over the first fraction of
training. Without it, early updates are large and unstable.

**Cosine schedule.** Decaying the learning rate along a cosine curve from its peak to a floor.
The common default.

**Gradient clipping.** Capping gradient magnitude so one bad batch cannot blow up the weights.

**Gradient accumulation.** Summing gradients over several small batches before updating, to
simulate a large batch that would not otherwise fit in memory.

**Gradient checkpointing.** Trading compute for memory by discarding intermediate activations
and recomputing them during the backward pass. Roughly 30% slower, substantially less memory.

**Checkpoint.** A saved snapshot of a run. slmkit checkpoints **full state** — weights,
optimizer, scheduler, step count, data position and all random number generator states — so a
resumed run is indistinguishable from an uninterrupted one. On a workstation that gets powered
off, this is the most important feature in the trainer.

**Seed.** The number initializing all randomness. Same seed, same run. At small scale the spread
across seeds often exceeds the effect you are trying to measure, so compare several.

---

## Data hygiene (where results go wrong)

**Train / validation split.** Data held back from training, used to measure generalization. If
validation loss rises while training loss falls, the model is memorizing.

**Overfitting.** Learning the training set rather than the pattern behind it. The main risk when
a corpus is small relative to the model.

**Leakage.** When information from validation reaches training — usually via near-duplicates
landing on both sides of the split. It makes results look great and mean nothing.

**Group split.** Splitting by a key that keeps all variants of one thing together (all settings
of a tune, all positions from a game) instead of splitting individual documents. slmkit's
`Doc.group` field exists solely for this.

**Contamination.** Test data present in training data. Easy to cause accidentally when an
eval benchmark derives from the same source as the training corpus.

**Augmentation.** Generating extra training examples by transforming existing ones. Only valid
when the transformation preserves the thing you want learned, and only ever applied to the
training split.

**n-gram.** A sequence of n consecutive tokens. Comparing generated n-grams against the training
set detects regurgitation.

**Early stopping.** Halting when validation loss stops improving, rather than running the full
schedule.

**Baseline.** The trivial approach you must beat for a result to mean anything — a frequency
table, a random choice, or an untrained model. A number without a baseline is not a result.

---

## Numbers and precision

**fp32 / fp16 / bf16.** 32-, 16- and 16-bit floating point. **bf16** ("brain float") keeps fp32's
exponent range with fewer decimal digits, so it resists overflow where fp16 needs extra
machinery. It is the default for modern training.

**TF32.** An NVIDIA format used automatically for fp32 matrix multiplies on recent GPUs. Faster,
slightly less precise, on by default in slmkit.

**Mixed precision.** Computing in bf16 while keeping a master copy of weights in fp32. Nearly
the speed of low precision with nearly the stability of high precision.

**Quantization.** Shrinking a *trained* model by storing weights in fewer bits (8-bit, 4-bit) for
cheaper inference. **Q4** means 4-bit. Distinct from training precision.

**FLOP / FLOPS / TFLOPS.** A floating-point operation; operations per second; trillions per
second. Training cost is approximated as `6 × parameters × tokens` FLOPs.

**MFU — Model FLOPs Utilization.** The fraction of the GPU's theoretical peak your training loop
actually achieves. 100% is unreachable; 35–45% is good for a large model, and small models score
far lower because their matrices are too small to keep the hardware busy.

**VRAM.** Memory on the GPU. Everything the GPU works on must fit: weights, gradients, optimizer
state and activations.

---

## Generating text

**Sampling.** Choosing the next token from the model's probability distribution. **Greedy**
decoding always takes the most likely token, which is repetitive.

**Temperature.** Flattens or sharpens the distribution before sampling. Below 1 is more
conservative, above 1 more random.

**Top-k / top-p (nucleus).** Restrict sampling to the k most likely tokens, or to the smallest
set whose probabilities sum to p. Both suppress implausible choices.

**Logits processor.** A hook modifying the model's scores before sampling — for instance setting
illegal chess moves to negative infinity. Also called **constrained decoding**. It enforces
rules the model was never guaranteed to learn.

---

## Evaluation

**Eval.** Measuring model quality systematically, rather than reading samples and forming an
impression.

**Grader.** In slmkit, a pure function scoring one generated output, returning named metrics.
Every project must have at least one that runs without human judgment — this is the selection
rule for what slmkit builds at all.

**Held-out.** Data the model never trained on.

**Elo.** A relative skill rating from head-to-head games. Used for the chess project by playing
the model against a known engine.

---

## Scale

**Scaling laws.** The empirical finding that loss falls predictably as parameters, data and
compute grow — which is what makes it possible to budget a run before starting it.

**Chinchilla.** The 2022 result that for a fixed compute budget, the best split is roughly
**20 tokens per parameter**. Earlier models were badly undertrained on this measure.

**Overtraining.** Deliberately training past compute-optimal — more tokens per parameter than
Chinchilla suggests. Wasteful of training compute, but produces a smaller model for the same
quality, which is what you want when the model must run cheaply. Most small models do this.

---

## Adapting existing models (context — not what slmkit does)

**Fine-tuning.** Continuing training on someone else's trained model. Far cheaper than
pretraining, and what most people mean by "training a model". slmkit pretrains from scratch
instead, because the goal is to learn the whole lifecycle.

**LoRA — Low-Rank Adaptation.** Freezing the original weights and training small add-on
matrices. **QLoRA** additionally quantizes the frozen base to 4-bit, making 7–14B fine-tuning
possible on one consumer GPU.

---

## Formats and tooling

**HF — Hugging Face.** The company and ecosystem behind the `transformers` library and the model
hub. "HF format" is the de-facto standard model layout, which is why slmkit exports to it.

**safetensors.** A tensor file format that is safe to load from untrusted sources, unlike
Python's `pickle`.

**GGUF.** The model file format used by `llama.cpp`, aimed at efficient CPU and mixed inference.

**llama.cpp / vLLM / Ollama.** Inference engines. `llama.cpp` targets CPU and small machines,
`vLLM` targets high-throughput GPU serving (Linux only), `Ollama` is a friendly local wrapper.

**memmap (memory-mapped file).** Treating a file on disk as if it were an in-memory array; the
OS loads pages on demand. How slmkit reads token datasets far larger than RAM.

**`uint16`.** Unsigned 16-bit integer, the storage type for token IDs — two bytes each, with a
hard ceiling of 65,536 vocabulary entries.

---

## GPU and platform

**CUDA.** NVIDIA's GPU computing platform. Everything in this stack sits on it.

**Compute capability / `sm_120`.** The GPU generation identifier. `sm_120` is Blackwell
(RTX 50-series). Libraries must be compiled for your architecture or they will not run — this is
the most common setup failure on new hardware.

**Tensor cores.** Specialized units for matrix multiplication, the source of nearly all training
throughput. They need large matrices to be efficient, which is why small models get low MFU.

**Kernel.** One GPU program. A training step launches hundreds; at small model sizes, launch
overhead alone can dominate.

**Triton.** A language for writing GPU kernels in Python. `torch.compile` uses it to generate
fused kernels automatically.

**`torch.compile`.** PyTorch's compiler, which fuses operations into faster kernels.
**CUDA graphs** (`mode="reduce-overhead"`) additionally replay a whole step as one unit,
eliminating per-kernel launch cost — worth the most on small models.

**WSL — Windows Subsystem for Linux.** Runs a real Linux kernel in a lightweight VM on Windows.
WSL2 has GPU access; that access works through `/dev/dxg` and a Microsoft-supplied CUDA shim
rather than a Linux NVIDIA driver, which is why you must never install one inside WSL.

**VHDX.** The virtual disk file holding a WSL distro's Linux filesystem. Fast, and a single
point of failure — back it up.

**ext4 vs `/mnt/c`.** `ext4` is the distro's native Linux filesystem. `/mnt/c` is the Windows
drive reached over the **9P** protocol, which pays a per-operation cost and is 20–70× slower on
small files. All pipeline I/O stays on ext4.

**DDA / GPU-P.** Two Hyper-V GPU-passthrough mechanisms. **DDA** (Discrete Device Assignment) is
Windows Server only; **GPU-P** (partitioning) works on Windows client but targets Windows guests.
Neither gives a hand-built Linux VM a GPU — see ADR 0003.

---

## Python tooling

**uv.** A fast Python package and environment manager. `uv.lock` pins exact versions so an
environment can be rebuilt months later.

**pydantic.** Runtime validation of config against a typed schema, so a bad YAML value fails
immediately with a clear message instead of halfway through a run.

**Typer.** The library building slmkit's `slm` command-line interface.

**ruff / mypy / pytest.** Linter and formatter; static type checker; test runner.

---

## slmkit's own vocabulary

**Engine.** Everything under `src/slmkit/` — the reusable machinery. It never imports a project.

**Project.** One LLM, as a self-contained directory under `projects/`. The Terraform
root-module analogue. Chess is a project; ABC music is a project.

**Experiment.** One run configuration, as a YAML file in `projects/<name>/experiments/`.
Addressed as `<project>/<experiment>`.

**Preset.** A shared, reusable config fragment under `presets/` — a model shape or a training
recipe — that experiments reference by name.

**Artifact.** Any output of a pipeline stage: a dataset, a tokenizer, packed tokens, a run, an
exported model. Artifacts are immutable and live under `$SLM_HOME`.

**Content-addressed.** An artifact's ID is a hash of its inputs, so identical inputs produce the
same ID and a completed stage is skipped rather than redone. This is what makes the pipeline
safe to re-run after an interruption.

**Manifest.** The `manifest.json` in every artifact directory, recording what produced it. Walk
manifests backwards with `slm lineage` to get from a model to its raw data.

**ADR — Architecture Decision Record.** A short document in `docs/decisions/` capturing one
decision: the context, what was chosen, the consequences, and what was rejected. It exists so
that in six months nobody re-argues a settled question.

**`slm doctor`.** The environment check that must pass before any training command runs.

**`$SLM_HOME`.** The directory holding all data, artifacts, runs and caches. Deliberately outside
the repo, on a fast Linux filesystem.

**IOPS — I/O Operations Per Second.** How many separate reads or writes a disk completes per
second. For many small files this matters more than MB/s.

**Page cache.** RAM that Linux uses to hold recently read or written file data. A benchmark that
reads a file it has just written measures this cache, not the disk.

**`O_DIRECT`.** A flag that makes file I/O bypass the page cache, so a benchmark measures the
disk itself. It is what `fio --direct=1` sets.

**Runbook.** A page in `docs/runbooks/` you run top to bottom to check a milestone by hand.

**GPU-hours.** How slmkit budgets compute. Not wall-clock time, because the workstation is
assumed to be off for long stretches and a "four-day run" is really a hundred GPU-hours spread
over however many sessions it takes.

---

## Domain formats

**ABC notation.** A plain-text format for melodies, widely used for folk and traditional music.
Header lines carry metadata (`R:` rhythm, `M:` meter, `K:` key) and the body carries notes.

**PGN — Portable Game Notation.** The standard text format for recorded chess games.

**UCI — Universal Chess Interface.** The protocol chess engines speak, and its move notation
(`e2e4` = from square, to square). There are 1,968 possible such moves, which makes a perfect
fixed vocabulary.

**Stockfish.** The strongest open-source chess engine; used as an opponent to measure strength.

**Cricsheet.** A public source of ball-by-ball cricket match data.
