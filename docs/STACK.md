# The stack

*Every piece of software slmkit uses or plans to use: what it is, where it sits in the stack, and
why slmkit uses it rather than something else. For ML terms, read [`GLOSSARY.md`](GLOSSARY.md).
For pinned versions, `uv.lock` is authoritative; the versions quoted here are from the reference
machine on 2026-09-23.*

**Rule:** a tool that enters the repo (a dependency, a system package, or a tool a doc tells you
to run) gets an entry on this page in the same change.

Status: **●** in use now · **◐** arrives in the milestone shown · **○** optional

---

## The stack at a glance

This reads like a network stack diagram: each layer only talks to the layers directly above
and below it.

```
  your terminal            slm doctor | slm pretrain | slm serve           (typer CLI)
 ──────────────────────────────────────────────────────────────────────────────────────
  slmkit engine            config (pydantic+yaml) · data (numpy) · model · trainer
                           tokenizers (HF tokenizers) · tracking (tensorboard) · serve (fastapi)
 ──────────────────────────────────────────────────────────────────────────────────────
  deep learning framework  PyTorch   — tensors, autograd, optimizers, SDPA
                             ├── torch.compile → TorchInductor → Triton   (generated kernels)
                             └── bitsandbytes                              (8-bit Adam, optional)
 ──────────────────────────────────────────────────────────────────────────────────────
  GPU libraries            cuBLAS · cuDNN · NCCL · CUDA runtime 12.8      (shipped inside pip wheels)
 ──────────────────────────────────────────────────────────────────────────────────────
  driver                   libcuda.so shim (/usr/lib/wsl/lib)  ─/dev/dxg→  Windows NVIDIA driver
 ──────────────────────────────────────────────────────────────────────────────────────
  OS / platform            Ubuntu 24.04 in WSL2 (ext4)  on  Windows
 ──────────────────────────────────────────────────────────────────────────────────────
  hardware                 NVIDIA GPU (Blackwell, sm_120, tensor cores)
```

The key point for someone coming from infrastructure: **everything above the driver line is
ordinary userland you install with `uv`.** That includes the CUDA libraries, which arrive as pip
packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, …) pulled in by the torch wheel. There is
no system-wide "CUDA install" to manage. The only thing outside `uv`'s control is the driver, and
under WSL that lives on Windows.

---

## 1. Platform

### NVIDIA GPU driver (Windows)
**●** `616.92` · [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

**What it is.** The kernel-level software that owns the GPU hardware. On Windows it is the
normal GeForce/Studio driver.

**Why this way.** Under WSL, Linux never gets a real NVIDIA driver. WSL mounts a `libcuda.so`
shim at `/usr/lib/wsl/lib/`, which forwards CUDA calls over `/dev/dxg` to this Windows driver.
Installing a Linux driver inside the distro replaces the shim and breaks GPU access. You update
the driver on the Windows side. See ADR 0003.

### WSL2 (Windows Subsystem for Linux)
**●** `2.7.11` · [docs](https://learn.microsoft.com/windows/wsl/) ·
[`.wslconfig`](https://learn.microsoft.com/windows/wsl/wsl-config)

**What it is.** A lightweight Hyper-V virtual machine that runs a real Linux kernel, tightly
integrated with Windows (shared networking, file access, GPU forwarding).

**Why.** The Linux ML toolchain (Triton, `torch.compile`, bitsandbytes, vLLM) is Linux-first or
Linux-only, and the machine also runs Windows workloads. We rejected a plain Hyper-V VM (no GPU
on client Windows), Windows-native Python (large parts of the toolchain are missing) and
dual-boot (the machine is shared). All distros share one VM and one memory cap, which is set in
`.wslconfig`.

### Ubuntu 24.04 LTS (`Ubuntu-ML` distro)
**●** · [docs](https://ubuntu.com/server/docs)

**What it is.** The Linux distribution inside WSL, in its own ext4 virtual disk (VHDX).

**Why.** A long-term-support release is the best-tested target for NVIDIA's WSL stack and
PyTorch wheels. It gets its own distro so the ML userland can be rebuilt without touching
anything else.

### ext4
**●** · [kernel docs](https://docs.kernel.org/admin-guide/ext4.html)

**What it is.** The Linux filesystem inside the distro's virtual disk.

**Why.** Measured at about 130× faster than `/mnt/c` (the Windows drive over 9P) for small
files. All pipeline I/O lives here, under `$SLM_HOME`. See runbook M0 §7.

---

## 2. GPU compute layer

### CUDA
**●** runtime `12.8` (in the torch wheel) · driver supports up to `13.4` ·
[docs](https://docs.nvidia.com/cuda/) · [programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

**What it is.** NVIDIA's platform for running general-purpose programs on a GPU: a programming
model, a compiler (`nvcc`), a runtime API, and a family of libraries. "Supports CUDA" is what
makes a graphics card usable for ML at all.

**Why.** It is the only mature option for NVIDIA hardware. AMD's ROCm and Apple's Metal are the
equivalents for other vendors. slmkit never writes CUDA code directly; PyTorch calls it.

**Two version numbers, both correct.** `nvidia-smi` shows `CUDA UMD Version: 13.4`, which is the
*newest* CUDA the **driver** can serve. PyTorch shows `12.8`, which is the CUDA **runtime** its
wheel was built against. The rule is driver ≥ runtime, the same way a Kubernetes API server
supports older client versions. It is also why the torch wheel index is pinned to `cu128`:
12.8 was the first runtime with `sm_120` (Blackwell) support.

### Compute capability (`sm_120`)
**●** · [NVIDIA table](https://developer.nvidia.com/cuda-gpus)

**What it is.** A version number for the GPU's instruction set, written `sm_XY` (e.g. `sm_120` =
12.0 = consumer Blackwell). Compiled GPU code targets specific values.

**Why it matters.** A PyTorch wheel without `sm_120` in `torch.cuda.get_arch_list()` imports fine
and then fails at the first kernel launch. `slm doctor` checks for it explicitly.

### cuBLAS, cuDNN, NCCL
**●** cuBLAS `12.8`, cuDNN `9.19`, NCCL `2.28` (pip packages pulled in by torch) ·
[cuBLAS](https://docs.nvidia.com/cuda/cublas/) · [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) ·
[NCCL](https://docs.nvidia.com/deeplearning/nccl/)

**What they are.** NVIDIA's hand-tuned libraries. **cuBLAS** does matrix multiplication (most
of a transformer's compute). **cuDNN** provides neural-network primitives. **NCCL** handles
communication between multiple GPUs.

**Why.** PyTorch calls cuBLAS for every `a @ b`, which is what `slm doctor --bench` measures.
cuDNN is mostly used by convolution-heavy models, so a transformer barely touches it. NCCL is
installed but unused, because slmkit is single-GPU by design (CONTRIBUTING.md rule 7).

### Tensor cores and bf16
**●** · [glossary: fp32 / fp16 / bf16](GLOSSARY.md)

**What they are.** Dedicated matrix-multiply units on the GPU, much faster than its general
cores, but only for reduced-precision types such as **bf16**.

**Why.** Training in bf16 with fp32 accumulation uses the tensor cores and keeps fp32's numeric
range. That is why all of slmkit's training is bf16 mixed precision.

### Triton
**◐ M1** (via `torch.compile`) · `3.6.0` · [docs](https://triton-lang.org/)

**What it is.** A Python-embedded language and compiler for writing GPU kernels, far simpler than
raw CUDA C++. PyTorch's compiler emits Triton code.

**Why.** slmkit doesn't write Triton itself. It appears as the backend of `torch.compile`, and it
is one of the main reasons for choosing Linux (it is Linux-first). Its compiled-kernel cache goes
to `$TRITON_CACHE_DIR` on ext4, because it writes thousands of small files.

### `torch.compile` and TorchInductor
**◐ M1** · [docs](https://pytorch.org/docs/stable/torch.compiler.html)

**What it is.** PyTorch's just-in-time compiler. It traces the model's Python code into a graph,
and **TorchInductor** (the default backend) fuses operations into fewer, larger Triton kernels.

**Why.** It is typically 1.3–2× faster for free, because small operations like RMSNorm or SwiGLU
stop making separate round trips to GPU memory. It is a single-line switch that can be turned
off, so the hand-written model stays readable. The cache goes to `$TORCHINDUCTOR_CACHE_DIR`.

### SDPA (`scaled_dot_product_attention`) vs `flash-attn`
**◐ M1** · [SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) ·
[FlashAttention](https://github.com/Dao-AILab/flash-attention)

**What it is.** PyTorch's built-in attention function. It picks the fastest available
implementation automatically, including a FlashAttention-style fused kernel.

**Why not the `flash-attn` package.** It needs to be built from source for new GPUs, and its newest
version doesn't target `sm_120`. SDPA provides most of the benefit with no extra dependency. This
is a hard rule in CONTRIBUTING.md.

---

## 3. Deep learning framework

### PyTorch
**●** `torch 2.11.0+cu128` · [docs](https://pytorch.org/docs/stable/) ·
[tutorials](https://pytorch.org/tutorials/)

**What it is.** The library everything else in slmkit is built on. It provides:
- **tensors**: n-dimensional arrays that can live on the GPU;
- **autograd**: record the operations of a forward pass, then compute every parameter's
  gradient automatically with `loss.backward()`;
- **`nn` building blocks**: `Linear`, `Embedding`, and the functional ops;
- **optimizers**: AdamW, and LR schedulers;
- the GPU plumbing underneath: CUDA streams, memory allocator, bf16 autocast.

**Why.** It is the de facto standard for research and open models: every reference
implementation slmkit learns from (nanoGPT, Llama, HF `transformers`) is written in it, and it
has first-class consumer-GPU support. We rejected JAX (strongest on TPUs, and its design
philosophy is less suited to a hand-written, eager-mode learning codebase) and TensorFlow
(little new LLM work uses it).

**Where.** Everywhere under `src/slmkit/model/`, `train/`, `sampling/`.

### NumPy
**●** `2.5` · [docs](https://numpy.org/doc/stable/) ·
[`memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

**What it is.** The standard CPU array library for Python.

**Why.** Packed training data is stored as one big `uint16` array on disk and opened with
`np.memmap`, which maps the file into memory. The trainer reads random windows from it without
loading the whole corpus into RAM, and the OS page cache does the caching. `uint16` holds token
IDs up to 65,535, which is enough for every vocabulary slmkit plans, at half the size of
`int32`.

### bitsandbytes
**○** extra `optim8bit` · `0.50.2` · [docs](https://huggingface.co/docs/bitsandbytes)

**What it is.** A library of low-precision GPU kernels, best known for **8-bit optimizers** and
quantized model loading.

**Why (optional).** 8-bit Adam stores optimizer state in 8 bits instead of 32, cutting memory per
parameter from about 16 bytes to about 8. slmkit's models are small enough that memory isn't the
constraint, so it stays optional. It must be A/B tested against fp32 AdamW before it is trusted
(M3). `slm doctor` verifies that it actually runs on `sm_120`.

---

## 4. Tokenization, weights and model formats

### Hugging Face `tokenizers`
**◐ M2** (BPE) · `0.23` · [docs](https://huggingface.co/docs/tokenizers)

**What it is.** A fast Rust-backed library for training and running subword tokenizers such as
BPE.

**Why.** Training a BPE tokenizer is a well-solved, performance-sensitive problem, and teaches
little when written by hand. M1's character tokenizer is hand-written because it is ten lines
long; M2's BPE comes from this library. Its output format is also what HF `transformers` expects,
so exported models are compatible.

### safetensors
**◐ M2** (export) · `0.8` · [docs](https://huggingface.co/docs/safetensors)

**What it is.** A file format for storing tensors: a JSON header plus raw bytes.

**Why.** PyTorch's default `torch.save` uses Python **pickle**, and loading a pickle can execute
arbitrary code, like `eval`-ing a file you downloaded. safetensors has no code execution, loads
via memory-mapping, and is the standard for sharing weights. Exported models use it. Internal
resume checkpoints still hold optimizer and RNG state that only slmkit reads (DESIGN §6.6).

### Hugging Face `transformers`
**○** extra `export` · `5.17` · [docs](https://huggingface.co/docs/transformers) ·
[Llama](https://huggingface.co/docs/transformers/model_doc/llama)

**What it is.** The most widely used library of pretrained model implementations.

**Why, and why only here.** slmkit's model uses the same parameter names as `LlamaForCausalLM`.
The export test loads a slmkit model into `transformers` and checks that the logits match, which
proves the export is correct and makes the whole HF ecosystem able to use it. It is **never** used
for training (CONTRIBUTING.md rule 6). The point of slmkit is to write that part yourself.

### GGUF and llama.cpp
**◐ M4** · [GGUF](https://huggingface.co/docs/hub/gguf) · [llama.cpp](https://github.com/ggml-org/llama.cpp)

**What they are.** llama.cpp is a C/C++ inference engine that runs models efficiently on CPUs and
consumer GPUs. GGUF is its single-file model format, usually quantized. Popular local runtimes
such as Ollama build on it.

**Why.** It is the most portable way to run a finished model anywhere. Because slmkit's model is
Llama-shaped, conversion is mostly mechanical. Character-level and fixed vocabularies may need a
custom tokenizer mapping, which M4 documents per project.

### vLLM
**Not planned** · [docs](https://docs.vllm.ai/)

**What it is.** A high-throughput GPU inference server for LLMs.

**Why it's mentioned.** It is a reason Linux was chosen (it has no native Windows build), and it
can serve slmkit exports because they are Llama-compatible. slmkit's models are small enough to
serve on a CPU, so vLLM is not in the plan.

---

## 5. Configuration and CLI

### pydantic
**◐ M1** · `2.13` · [docs](https://docs.pydantic.dev/latest/)

**What it is.** Data validation from Python type hints: you declare a class with typed fields,
and pydantic parses input into it or rejects it with a precise error.

**Why.** Experiment YAMLs are layered like tfvars (model preset → train preset → experiment →
`--set`). pydantic turns the merged result into a typed, validated object, so `lr: "6e-4 "` or a
typo'd key fails at startup rather than an hour into a run. The resolved config is hashed to form
artifact IDs.

### PyYAML
**◐ M1** · `6.0` · [docs](https://pyyaml.org/wiki/PyYAMLDocumentation)

**What it is.** A YAML parser and emitter.

**Why.** Presets and experiments are YAML because they are hand-edited and diffable. We rejected
Hydra (CONTRIBUTING.md rule 6): it solves the same merge problem with a lot more machinery than about 50
lines of merge code.

### Typer
**●** `0.27` · [docs](https://typer.tiangolo.com/)

**What it is.** A library that builds a CLI from typed Python function signatures.

**Why.** `slm doctor --bench` is just a function with a `bench: bool` parameter. It needs no
argparse boilerplate, and it generates `--help` for free.

---

## 6. Experiment tracking

### TensorBoard
**◐ M1** · `2.21` · [docs](https://www.tensorflow.org/tensorboard) ·
[PyTorch integration](https://pytorch.org/docs/stable/tensorboard.html)

**What it is.** A local web UI that plots training metrics (loss, learning rate, throughput) from
event files written into the run directory.

**Why.** It works with nothing else running: no server to keep up, no account, and the data
lives with the run (CONTRIBUTING.md rule 10). Start it with `uv run tensorboard --logdir $SLM_HOME/runs`
while a run is going.

### Weights & Biases (wandb)
**○** extra `wandb` · `0.30` · [docs](https://docs.wandb.ai/)

**What it is.** A hosted experiment-tracking service, with richer comparison and sharing than
TensorBoard.

**Why optional.** It needs an account and network access. It is useful for comparing many sweep
runs, but slmkit must never depend on it. A self-hosted container version was rejected because
tracking has to work with Docker stopped.

---

## 7. Serving and operations

### FastAPI and Uvicorn
**◐ M2** · FastAPI `0.141`, Uvicorn `0.53` · [FastAPI](https://fastapi.tiangolo.com/) ·
[Uvicorn](https://uvicorn.dev/)

**What they are.** FastAPI is a Python web framework, typed with pydantic. Uvicorn is the ASGI
server that runs it (for comparison, gunicorn fills the same role for WSGI apps).

**Why.** `slm serve` is a single `/generate` endpoint. FastAPI reuses the same pydantic models as
the config, and the models are small enough to serve from the CPU. Auth, TLS and rate limiting
are deliberately left to a gateway in front of it (M4).

### Docker and Kubernetes
**◐ M4** (optional) · [Docker](https://docs.docker.com/) · [Kubernetes](https://kubernetes.io/docs/) ·
[kind](https://kind.sigs.k8s.io/) · [k3s](https://docs.k3s.io/)

**Why.** Containerising `slm serve` (CPU) and optionally running it as a Deployment with a
readiness probe, with `slm eval` as a Job against it. This is where the ML lifecycle meets
familiar ops patterns. Training never runs in a container: it would add GPU passthrough
complexity to no benefit on a single-user workstation.

### rsync
**◐ M4** · [docs](https://download.samba.org/pub/rsync/rsync.1)

**Why.** Backups of `models/` and selected runs go to another disk or a NAS. A lost WSL virtual
disk should cost one evening, not every model you've trained (DESIGN risk table).

---

## 8. Python development tooling

### Python 3.12
**●** · [docs](https://docs.python.org/3.12/)

**Why 3.12.** It is the newest version every dependency (torch, bitsandbytes, music21) has
wheels for, and it supports modern typing syntax (`list[int]`, `X | None`).

### uv
**●** `0.12` · [docs](https://docs.astral.sh/uv/)

**What it is.** A fast Python package and project manager (written in Rust) that handles
environments, dependency resolution and a lockfile.

**Why.** `uv.lock` pins every package, including the CUDA wheels, so `uv sync` rebuilds the exact
environment months later, like `terraform init` with a lock file. It also handles the custom
PyTorch index cleanly (`[tool.uv.sources]`). We rejected pip + venv (no lockfile of its own) and
conda (a second, heavier package universe that tends to fight pip over CUDA libraries). Never
`pip install` into the system Python.

**One behaviour to know:** `uv run` forwards signals to the program it starts. A terminal Ctrl-C
already reaches every process in the foreground group, so the program receives SIGINT **twice**.
The trainer treats a repeat within one second as the same keypress (see
`concepts/the-training-loop.md` §7).

### hatchling
**●** · [docs](https://hatch.pypa.io/latest/)

**What it is.** The build backend declared in `pyproject.toml`.

**Why.** It makes slmkit an installable package, which is how `slm` becomes a command
(`[project.scripts]`). You never call it directly; `uv` does.

### ruff
**●** `0.16` · [docs](https://docs.astral.sh/ruff/)

**What it is.** A linter and formatter in one fast binary, replacing flake8, isort and black.

**Why.** It is the single source of formatting truth, used by both `make lint` and the editor.
Markdown is excluded so that doc sketches keep their hand alignment.

### mypy
**●** `2.3`, strict mode · [docs](https://mypy.readthedocs.io/)

**What it is.** A static type checker for Python.

**Why strict.** ML code passes many shapes and dtypes around, and a type checker catches the
`None`-where-a-tensor-was-expected class of bug before a GPU run does. Untyped third-party
libraries (bitsandbytes) are exempted explicitly in `pyproject.toml`, not globally. Libraries
that ship without type information get a stub package instead (`types-PyYAML`, in the `dev`
extra).

### pytest
**●** `9.1` · [docs](https://docs.pytest.org/)

**Why.** Unit tests are CPU-only and must finish in under 60 s. GPU tests are marked
`@pytest.mark.gpu` and run separately (`make test-gpu`).

### GNU Make
**●** · [manual](https://www.gnu.org/software/make/manual/)

**Why.** It gives short, memorable entry points (`make doctor`, `make lint`) that also document
what each check really runs. It is a thin wrapper; the logic lives in Python.

### Git and GitHub
**●** · [git docs](https://git-scm.com/doc)

**Why.** Code, configs and docs are versioned. Artifacts under `$SLM_HOME` never are: they are
content-addressed and immutable already, which is closer to a Terraform state backend than to
source.

### VS Code (with the WSL extension)
**●** · [VS Code + WSL](https://code.visualstudio.com/docs/remote/wsl)

**Why.** The editor runs on Windows while its language server, Python and terminals run inside
`Ubuntu-ML`. `.vscode/extensions.json` recommends Python/Pylance, Ruff, YAML and TOML support,
and explicitly discourages a second formatter (Black, Prettier) that would fight ruff.

---

## 9. System tools

Installed by `scripts/setup-ml-distro.sh`.

| Tool | What it is | Why slmkit needs it | Docs |
|---|---|---|---|
| `nvidia-smi` | NVIDIA's GPU status tool (comes with the driver) | Check GPU model, driver, temperature, power and memory during a run | [docs](https://docs.nvidia.com/deploy/nvidia-smi/) |
| `tmux` | Terminal multiplexer | A run keeps going when the terminal closes (not across power-off; checkpoints handle that) | [wiki](https://github.com/tmux/tmux/wiki) |
| `fio` | Flexible I/O benchmark | Honest disk numbers with `--direct=1` (runbook M0 §7) | [docs](https://fio.readthedocs.io/) |
| `zstd` | Fast compression format and tool | Lichess dumps are `.pgn.zst`; `zstdcat` streams them without ever writing the decompressed file | [site](https://facebook.github.io/zstd/) |
| `aria2c` | Multi-connection, resumable downloader | Multi-GB corpus downloads that survive interruption | [docs](https://aria2.github.io/manual/en/html/) |
| `jq` | JSON processor for the shell | Reading manifests, `doctor.json` and Cricsheet JSON from the command line | [manual](https://jqlang.org/manual/) |
| `curl` | HTTP client | Installing `uv`; poking `slm serve` | [docs](https://curl.se/docs/) |
| `build-essential` | gcc, make, libc headers | Triton compiles a small C helper at runtime and needs a system C compiler | [package](https://packages.ubuntu.com/noble/build-essential) |
| `python3-dev` | Python's C headers (`Python.h`) | The same Triton helper includes `Python.h`. Without it `torch.compile` fails at the first compile with `Python.h: No such file or directory`. Missed in M0 and caught by the M1 GPU tests; `slm doctor` now checks it | [package](https://packages.ubuntu.com/noble/python3-dev) |
| `abc2midi` | Part of abcMIDI; converts ABC to MIDI | Grading ABC output and listening to it (M2) | [abcMIDI](https://abcmidi.sourceforge.io/) |

---

## 10. Project-specific libraries and data

Each project's dependency is an **extra** in `pyproject.toml`, so the base install stays light and
the engine never imports a project library.

| Item | Project | What it is | Why | Docs |
|---|---|---|---|---|
| **music21** | ABC (M2) | Toolkit for analysing music notation | Parse generated ABC; check meters, keys and cadences in graders | [docs](https://www.music21.org/music21docs/) |
| **ABC notation** | ABC (M2) | Plain-text melody format | The corpus *and* the model's output language | [standard](https://abcnotation.com/wiki/abc:standard) |
| **TheSession data** | ABC (M2) | Dump of ~45k folk tunes | Training corpus; `tune_id` is the split group | [repo](https://github.com/adactio/TheSession-data) |
| **python-chess** | Chess (M3) | Move generation, PGN parsing, legality | Legal-move grader and legal-move-masked decoding | [docs](https://python-chess.readthedocs.io/) |
| **Stockfish** | Chess (M3) | Strongest open-source chess engine | Fixed-strength opponent to measure Elo | [site](https://stockfishchess.org/) |
| **Lichess database** | Chess (M3) | Monthly game dumps + puzzle DB | Effectively unlimited games; puzzles as an eval set (with their source games excluded from training) | [database](https://database.lichess.org/) |
| **Cricsheet** | Cricket (M5) | Ball-by-ball match data (JSON) | Next-ball-outcome modelling | [site](https://cricsheet.org/) |

---

## 11. References (read, not installed)

| Item | Why it matters here | Link |
|---|---|---|
| **nanoGPT** | M1's correctness reference: `shakespeare_char` must reach a val loss close to its ~1.47 | [repo](https://github.com/karpathy/nanoGPT) |
| **Llama architecture** | The one model family slmkit implements (RMSNorm, RoPE, SwiGLU) | [paper](https://arxiv.org/abs/2302.13971) |
| **Attention Is All You Need** | The original transformer paper | [paper](https://arxiv.org/abs/1706.03762) |

---

## 12. Deliberately not used

Recorded here because "why not X?" is often the most useful question.

| Tool | What it would give | Why not |
|---|---|---|
| HF `Trainer`, PyTorch Lightning, Accelerate | A ready-made training loop | The training loop is what slmkit exists to teach. Hiding it defeats the purpose (rule 6) |
| DeepSpeed, FSDP, DDP | Multi-GPU and sharded training | One GPU, one process (rule 7); models fit easily |
| Hydra | Config composition | The same merge in about 50 lines, without a framework's conventions |
| Ray, Airflow, Kubeflow | Orchestration and schedulers | Nothing runs unattended; `slm run` skips existing artifacts, which is enough |
| `flash-attn` package | Fused attention | Not built for `sm_120`; SDPA covers it |
| A Linux NVIDIA driver | — | Breaks CUDA in WSL (ADR 0003) |
| conda | Environments with binary packages | A second package universe; `uv` + wheels already cover CUDA |
| Self-hosted MLflow / wandb server | Tracking UI | Would need Docker running; tracking must work without it (rule 10) |
