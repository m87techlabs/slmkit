# Changelog

All notable changes to slmkit. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/). Until 1.0, minor versions may change
interfaces; each entry says what moved. Milestones (M0, M1, …) are described in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## [Unreleased]

### Added
- CI on every push and pull request: lint (ruff, mypy) and the CPU test suite.
- Issue templates (bug report with `make doctor` output, feature or project idea), a pull-request
  checklist, and this changelog.
- README: a "Results so far" table with sourced numbers, badges, and plainer hardware requirements.

## [0.1.0] - 2026-10-09

The engine and its first two projects: a character-level Shakespeare model that reproduces a published
reference, and a folk-tune model taken through the whole lifecycle.

### Added

**M0: environment**
- `slm doctor`: checks the GPU (`sm_120`), bf16 and SDPA, 8-bit Adam, `$SLM_HOME` on ext4, memory and
  swap, and runs a real `torch.compile`; `--bench` measures bf16 TFLOPS (112–122 across sessions on the reference RTX 5080).
- Setup script for a dedicated WSL2 distro, uv-managed Python, and PyTorch from the CUDA 12.8 index.

**M1: the engine**
- Content-addressed, immutable artifacts with manifests and `slm lineage`; a guard against pipeline I/O
  on Windows drives.
- The Project API and registry: a new use case is a directory under `projects/`, with no engine changes.
- Data pipeline: ingest, prepare (split by group, augment the training split only), a character
  tokenizer, and packing into flat token files.
- A Llama-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings) with Hugging
  Face `LlamaForCausalLM`-compatible parameter names, checked logit for logit.
- The training loop: token-based warmup and cosine schedule, AdamW, bf16, `torch.compile`, full-state
  atomic checkpoints every 20 minutes, exact resume, guards, TensorBoard, and samples at every eval.
- `shakespeare_char`: the reference run reaches validation loss 1.29 (nanoGPT's published figure is
  about 1.47) in about 3 GPU-minutes.

**M2: ABC folk tunes, the full lifecycle**
- `abc_music` on 4,248 public-domain tunes: melody-fingerprint grouping, transposition verified note
  for note with `abc2midi`, header dropout.
- A BPE tokenizer, and bits-per-character reporting to compare tokenizers fairly.
- `slm eval`: generic graders (completion, n-gram novelty, parse rate) and project graders (plays, bar
  accuracy, ends on the tonic), calibrated on the human corpus, over multiple sampling seeds with a
  no-model baseline; `slm runs compare`.
- Supervised fine-tuning as a training stage, with the loss masked to the answer: the model takes
  requests in plain English.
- `slm export` to Hugging Face format (safetensors, tokenizer, generated model card), refusing to
  publish unless slmkit and `transformers` reproduce the logits; versioned `name:version` models.
- `slm serve`: a FastAPI server with a browser playground; projects can ship a viewer (sheet music and
  playback for ABC); `--to-windows` writes MIDI to listen to.
- Sweeps over three training seeds and `slm runs summary`: model size, tokenizer and augmentation,
  compared at equal compute.

**slm studio: "See it in action"**
- A local web app over everything on disk: your models and the machine (with live GPU readings), the
  lifecycle as real artifacts with lineage, training curves and samples, experiments with spreads, a
  parameter calculator, the playground for every exported model, the docs with glossary hovers, and
  Verify, which runs the runbooks' read-only checks.

**Documentation**
- A concepts page for every stage, a hands-on runbook per milestone with real expected output, ADRs
  0001–0009, a glossary, a stack reference and the model specification.

### Fixed
- Evaluation prompts asked for 4/4 reels and hornpipes, which the corpus almost never has; they now ask
  for each rhythm's common form, and eval report IDs include the prompt text.
- `flops_per_token` counted an untied output head twice (no preset unties it, so no number changed).
- `slm serve` on few CPU cores: PyTorch's per-thread worker pools oversubscribed the CPU; threads are
  now capped (`--threads`).
- The latest eval report is chosen by its own timestamp, so copied run directories don't confuse it.

[Unreleased]: https://github.com/m87techlabs/slmkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m87techlabs/slmkit/releases/tag/v0.1.0
