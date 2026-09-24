# slmkit — Design

A framework for building **small, task-specific language models from scratch** on one
consumer GPU. Every model is produced by the same pipeline:

```
ingest → prepare → tokenizer → pack → pretrain → SFT → eval → export → serve
```

The framework is the deliverable. Individual **projects** (ABC music, chess, …) exist to
exercise and harden it. Learning the full LLM lifecycle is the explicit goal, so the model and
training loop are hand-written (no HF Trainer / Lightning / Accelerate).

**Contract:**
- A new model on an **existing** project = one YAML file in that project's `experiments/`.
- A new **project** = one directory under `projects/` (~100–300 lines: ingest, split key,
  tokenizer choice, graders) + one YAML. No changes to the engine.

---

## 1. Target platform

slmkit assumes a **single consumer NVIDIA GPU in a Windows workstation, accessed through WSL2**.
That is the platform the design is written against; it is not the only one that could work, but
every trade-off here follows from it.

| Assumption | Why it shapes the design |
|---|---|
| One consumer GPU, 12–24 GB VRAM | No multi-GPU code paths; model sizes capped by compute, not memory (§2) |
| Windows host, training inside WSL2 | The Linux ML stack is available; Windows drives are slow to reach (§4) |
| The workstation is **shared** | Other containers, desktop apps and games compete for RAM and GPU; separation is by time (§3) |
| The workstation is **not always on** | Powered off overnight and for days; nothing may assume uptime (§3, §8) |
| Consumer hardware, no ECC | Long runs need numeric guards and frequent checkpoints (§8) |

**Reference machine.** Concrete numbers throughout this document were measured on an RTX 5080
(16 GB GDDR7, Blackwell `sm_120`, ~960 GB/s, 360 W) with a 24-thread CPU and 31 GB host RAM,
running Windows 11 Pro + WSL2 (Ubuntu 24.04) + Docker Desktop. Treat them as a worked example,
not a requirement — on different hardware the *method* holds and the numbers change. Re-measure
at M0 (see `docs/ROADMAP.md`) and use your own.

**If you have more or less VRAM:** the ceilings in §2 scale roughly linearly. 12 GB shifts every
tier down one preset; 24 GB makes the 350M tier comfortable rather than aspirational. The
compute budget, not the memory budget, is what actually limits you.

### Compute ceiling (measured — see ADR 0001)

`slm doctor --bench` on the reference machine reports **121.6 TFLOPS** for bf16 matrix multiply
at 8192×8192. The spec-sheet derivation predicted ~112 TFLOPS, so the estimate was sound and
slightly conservative.

Two numbers, often confused:

- **121.6 TFLOPS is a ceiling**, measured on nothing but large dense matmuls. It is the best the
  hardware will ever do.
- **~43 TFLOPS is the planning figure** — roughly 35% of the ceiling. Training is not pure
  matmul: attention, normalization, the optimizer step and data movement all consume time the
  benchmark never spends. That ratio is **MFU**, and it falls further on small models whose
  matrices are too small to keep the tensor cores busy (see §6.6).

**Always measure tokens/sec in the first 10 minutes of a run and recompute the estimate from
that.** Even a measured ceiling is not a prediction of your particular model's throughput.

---

## 2. Compute budget

Rule of thumb: training FLOPs ≈ `6 × params × tokens` (+10–15% for attention at 1K context).

| Run | FLOPs | Est. **GPU-hours** @ 43 TFLOPS | Verdict |
|---|---|---|---|
| 1–5M × 20–50M tokens (ABC) | ~1e15 | minutes | Main iteration loop |
| 25–50M × 1–3B tokens (chess) | ~1e18 | 6–20 h | Scaling-lab tier |
| 124M × 3B | 2.2e18 | **~14–16 h** | Feasible across a few sessions |
| 350M × 7B | 1.5e19 | **~95–105 h** | Chess only; late, and deliberately |
| 1B × 20B | 1.2e20 | ~775 h | **Out of scope** |

**Budget in GPU-hours, never in wall-clock dates.** The machine sleeps, so a "4-day run" is
really ~100 GPU-hours spread over however many evenings it takes. Every ETA the trainer prints
is GPU-hours remaining, and `slm runs list` shows GPU-hours consumed against the target.

**Memory is not the binding constraint. Compute is.**
- Mixed-precision AdamW ≈ 16 bytes/param (weights + grads + fp32 master + 2 Adam moments).
- 8-bit Adam ≈ 8 bytes/param. A 1B model *fits* in 16 GB. It just never finishes.

**Calibration point:** Karvonen's 50M chess-GPT reached ~1300 Elo (99.8% legal moves) in
~1 day on 4× RTX 3090 — roughly 3–4 days of GPU time on the 5080.

**The 350M tier only has one home.** ABC is ~20M tokens and cricket ball-events are bounded;
chess is the only project with 7B tokens available. At 350M it is a compute-scaling exercise,
not a capability one — Karvonen reached decent play at 50M. Do it deliberately or not at all.

---

## 3. Environment

### Decision: dedicated WSL distro `Ubuntu-ML` (ADR 0003)

- Triton / `torch.compile`, bitsandbytes and vLLM are Linux-first. Windows-native would need a
  CUDA + MSVC toolchain from scratch, and Triton exists there only as a community fork.
- **Hyper-V was evaluated and rejected.** A hand-built Hyper-V Linux VM gets no GPU at all:
  DDA is Windows Server only, GPU-P targets Windows guests, and WSL's GPU access is the
  WSL-specific `/dev/dxg` + `/usr/lib/wsl/lib/libcuda.so` shim that a stock kernel lacks.
- A dedicated distro isolates the Python/CUDA **userland** only. The VM, kernel and memory are
  shared with any other distro and with Docker Desktop.

### Setup notes

- **Never install a Linux NVIDIA driver inside WSL.** CUDA comes from the Windows driver via
  `/dev/dxg`. `nvidia-smi` works inside WSL.
- Python environment: **`uv`** with a committed `uv.lock`.
- PyTorch: CUDA **12.8 or newer** wheel index, pinned in `pyproject.toml` under
  `[tool.uv.sources]`. Blackwell support landed in PyTorch 2.7 (cu128 wheels, Triton 3.3).
  Verify with `torch.cuda.get_arch_list()`, which must include `sm_120`.
- Attention: **`torch.nn.functional.scaled_dot_product_attention`**. Do not depend on the
  `flash-attn` package (FA3 doesn't target sm_120; FA2 usually needs a source build).
- bitsandbytes: sm_120 kernels ship in current wheels (since ~v0.45.3). Treat 8-bit Adam as
  "verify before trusting": A/B against fp32 AdamW for 10 minutes and compare loss curves.
- Record exact versions in `docs/decisions/0001-stack-versions.md` at install time.

### `.wslconfig` (Windows: `%UserProfile%\.wslconfig`)

```ini
[wsl2]
memory=20GB          # a ceiling, not a reservation
swap=8GB

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

Option names under `[experimental]` change between WSL releases; check against `wsl --version`.
Raise `memory` for training sessions and lower it again afterwards — it is a ceiling, so an
unused allowance costs nothing while WSL is idle, but a cap set too low will push you into swap
during preprocessing.

**All WSL2 distros share one utility VM, one kernel and one memory cap.** A second distro gives
you a separate filesystem and a separate Python userland, but not a separate memory budget. That
is why separation from other workloads is by *time* rather than by allocation.

### Training session checklist (the "time separation" routine)

1. Stop any other containers (or quit Docker Desktop entirely).
2. Pause Windows Update, or set active hours to cover the session. **Update reboots kill runs
   more often than hardware does.**
3. Set the Windows power plan so the machine never sleeps mid-session.
4. Set a GPU power limit of ~280–300 W with MSI Afterburner (Windows side). You lose ~5–10%
   throughput and run much cooler. Watch the GDDR7 memory-junction temperature.
5. Run training inside `tmux` in `Ubuntu-ML`.
6. When you're done for the night, **just stop it.** Resume is the normal path, not recovery.

### Designing for an intermittent machine

`tmux` survives a disconnect, not a power-off. The real continuity mechanism is the checkpoint,
so:

- Checkpoint on a **time** cadence (default 20 min), never only at epoch boundaries.
- `slm pretrain <cfg>` **resumes by default** if the run directory holds a checkpoint. Resuming
  is not a flag and not an error path; it is how the command normally behaves.
- Resume must be exact enough that a run split across ten sessions matches one continuous run —
  hence the resume-equivalence test in §7.
- `slm runs list` shows, for each run: tokens done / target, GPU-hours consumed, last
  checkpoint time, and whether it is complete. After two weeks away, that table is how you
  remember where you were.
- No cron, no daemons, no "start it Friday and check Monday". Every long run is a sequence of
  sessions you deliberately start.

---

## 4. Storage

### Measured

Measured on the reference machine; your absolute numbers will differ, but the *ratio* is the
point and it holds broadly.

| Path | Sequential write | 4K random read | Method |
|---|---|---|---|
| ext4 (WSL VHDX) | **4,655 MB/s** | **631 MB/s (161K IOPS)** | `fio --direct=1`, qd16/qd32 |
| `/mnt/c` (9P) | 229 MB/s | 26 MB/s (~6.3K IOPS) | `dd`, buffered |

The ext4 figures bypass the page cache (`O_DIRECT`), so they are the real device. An earlier
buffered `dd` reported 1.8 GB/s on 4K reads — that was RAM, not disk, and is a good illustration
of why `--direct=1` matters when you benchmark storage.

Even against the honest number, **ext4 random reads are ~24× faster than `/mnt/c`**.

`/mnt/c` pays a per-operation cost because every file operation crosses the VM over 9P.
Windows reading `\\wsl.localhost\...` pays the same cost in the other direction.

### What actually needs bandwidth

- **Training dataloader:** negligible. A 124M model at ~50K tokens/s × 2 bytes ≈ 100 KB/s, and
  the datasets fit in page cache.
- **Hurts on `/mnt/c`:** venv / imports (tens of thousands of small files), Triton and Inductor
  caches, preprocessing, and checkpoint writes. A 350M checkpoint with optimizer state ≈ 5.6 GB
  takes ~25 s on `/mnt/c` versus ~2 s on ext4, and training blocks while it saves.

### Rules (enforced in code — see §6.8)

1. Everything the pipeline touches lives under `$SLM_HOME` on ext4: raw data, packed data,
   tokenizers, runs, checkpoints and caches.
2. Point `HF_HOME`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` under `$SLM_HOME/cache`.
3. Download inside WSL (`aria2c`/`wget`). Stream-decompress: `zstdcat x.zst | parser`.
4. Store tokens in a few large `uint16` memmap files per split, never one file per document.
   Assert `vocab_size < 65536` at pack time.
5. `/mnt/c` is only for hand-carried outputs (MIDI to listen to, PGN for a chess GUI), via
   `slm export --to-windows`.
6. Back up off the hot path: rsync kept checkpoints to C: or a NAS, and run
   `wsl --export Ubuntu-ML` periodically. The VHDX is a single file and a single point of failure.

---

## 5. Projects (use cases)

Selection rule: **only tasks with a free, programmatic grader.**

### 5.0 `shakespeare_char` — reference reproduction (step 0)

Validates the trainer against a published loss curve. Without a known-good reference, you
cannot tell a trainer bug from a hard task.

**Architecture caveat:** nanoGPT's ~1.47 char-level val loss comes from a GPT-2-style model
(LayerNorm, learned positional embeddings, GELU, ~10.65M params: 6L / 384d / 6h, block 256,
dropout 0.2). slmkit is Llama-style (RMSNorm, RoPE, SwiGLU), so the numbers are close but not
directly comparable. Use a `ref` preset matching nanoGPT's dimensions, and treat **≤1.55 with a
matching curve shape** as the pass mark. A Llama-style model should land at or slightly below
1.47; the point is to catch a broken trainer, not to win a benchmark.

### 5.1 `abc_music` — first real project

- **Data:** thesession.org dump (`adactio/TheSession-data`), ~45K tunes with multiple settings
  each, roughly 5–20M characters.
- **Model size:** **1–5M params**, character-level or small BPE (≤1K vocab), dropout 0.1–0.2,
  early stopping. Beyond ~4 epochs repeated data loses value quickly; past ~16 epochs it is
  mostly memorization.
- **Split by tune ID** (the `group`), never by setting. Near-duplicate settings otherwise leak
  into validation and hide memorization. This is the single most important correctness detail
  in the project.
- **Augmentation (train split only, after splitting):** transpose within ±2 semitones or to the
  genre's common keys (D, G, A, C, Em, Bm, Am). Not all 12: a D♭ reel is out of distribution.
  Expect ~1.5–2× effective data, not 12×.
- **Make SFT measurable:** during pretraining, randomly drop or shuffle `R:/M:/K:` headers for
  ~30–50% of documents. SFT then maps a natural-language prompt ("slip jig in E minor") to a
  tune, with **loss masked on the prompt tokens**.
- **Graders:** parses (abc2midi / music21), bar durations match `M:`, ends on the tonic of `K:`,
  requested rhythm/meter/key obeyed (SFT), and n-gram novelty against the training set.

**SFT baseline must be stated precisely.** Header dropout means the base model is already
trained to condition on an `R:/M:/K:` prefix, so prompting it that way is a strong baseline that
may beat SFT on raw adherence. Compare *SFT with a natural-language prompt* against *base with a
header prefix* and expect **parity**, not a win. What SFT buys here is the natural-language
interface, not better constraint-following.

### 5.2 `chess` — scaling and eval lab

- **Data:** Lichess monthly PGN dumps (~30 GB `.zst` each). Stream; filter by Elo and time
  control; parse in parallel across processes (python-chess is CPU-bound).
- **Tokenizer:** fixed vocabulary, one token per UCI move (1,968) plus specials. The model must
  track board state implicitly.
- **Contamination:** Lichess puzzles are derived from Lichess games, and the puzzle CSV includes
  the source game URL. **Exclude those game IDs from training.**
- **Graders:** legal-move rate (python-chess), puzzle accuracy, Elo against Stockfish at fixed
  depth or skill level.
- **Scaling sweeps:** tune LR per model size (LR does not transfer across sizes without µP).

### 5.3 `cricket_nextball` — third project

Generating commentary from Cricsheet data needs English fluency — a separate TinyStories- or
GPT-2-scale problem — and template-generated targets make the grader circular. Reframed as
**next-ball-outcome sequence modeling**, graded by log-loss and calibration against a frequency
baseline. Free, programmatic, and it reuses the chess-style fixed vocabulary and the generic
calibration graders.

---

## 6. Framework architecture

### 6.1 Principles

1. **Stages talk only through artifacts on disk.** Each stage is a CLI command that reads
   artifacts and writes one new one. Engine = provider, `projects/<name>/` = root module,
   `experiments/*.yaml` = tfvars, `$SLM_HOME` = state.
2. **Content-addressed, idempotent artifacts.** An artifact's ID hashes its inputs (config
   section + upstream artifact IDs + code version). If it exists, the stage is skipped.
   Changing a config value produces a new artifact and never overwrites. This is also what
   makes `slm run` safe to re-invoke after a week away.
3. **The engine never imports a project.** `slmkit.*` has zero knowledge of ABC or chess.
   Projects register through a registry and depend on the engine.
4. **One model family.** A Llama-style decoder (RMSNorm, RoPE, SwiGLU, SDPA, optional tied
   embeddings) with parameter names mapping 1:1 to HF `LlamaForCausalLM`. Export is then a
   rename, and llama.cpp / vLLM / HF `generate` work for free.
5. **Single GPU, single process.** No DDP/FSDP abstractions. YAGNI.
6. **Hand-written core.** Model, trainer, sampler and SFT masking are written in this repo.
   HF `tokenizers` (BPE training) and `safetensors` are allowed.

### 6.2 Pipeline and artifacts

```
projects/<name>/experiments/<exp>.yaml
      │
      ▼
[ingest]   → $SLM_HOME/raw/<project>/              downloaded source, immutable
[prepare]  → $SLM_HOME/datasets/<project>/<id>/    docs split by group; augment train only
[tokenize] → $SLM_HOME/tokenizers/<id>/            tokenizer.json + meta.json
[pack]     → $SLM_HOME/packed/<id>/{train,val}.bin uint16 memmap + meta.json
[pretrain] → $SLM_HOME/runs/<run_id>/              ckpt/, metrics, config.resolved.yaml
[sft]      → $SLM_HOME/runs/<run_id>/              parent = pretrain run
[eval]     → $SLM_HOME/runs/<run_id>/eval/<id>.json
[export]   → $SLM_HOME/models/<name>/<version>/    HF safetensors + tokenizer + MODEL_CARD.md
[serve]    ← models/<name>/<version>
```

Every artifact directory contains a **`manifest.json`**:

```json
{
  "kind": "packed",
  "id": "pk-3f9a1c",
  "project": "abc_music",
  "created": "2026-09-22T21:04:00-05:00",
  "git_sha": "abc1234", "git_dirty": false,
  "inputs": {"dataset": "ds-81be02", "tokenizer": "tk-11c0d4"},
  "config": {"...resolved section...": "..."},
  "stats": {"train_tokens": 18234112, "val_tokens": 961220}
}
```

`slm lineage <id>` walks manifests back to the raw data.

### 6.3 Repository layout

```
slmkit/
├── CONTRIBUTING.md
├── pyproject.toml            # uv; entry point slm = slmkit.cli:app; cu128 index pinned
├── uv.lock
├── Makefile                  # doctor, test, test-gpu, lint, fmt, run CFG=...
├── presets/                  # SHARED config fragments referenced by every project
│   ├── model/                # ref nano micro tiny small medium .yaml
│   └── train/                # default.yaml sft.yaml
├── src/slmkit/               # ENGINE — must never import from projects/
│   ├── cli.py                # Typer: doctor ingest prepare tokenize pack pretrain sft
│   │                         #        eval export serve run lineage runs
│   ├── config/               # pydantic schemas, YAML load + preset merge + --set
│   ├── artifacts.py          # hashing, manifests, atomic dir commit, /mnt guard
│   ├── pipeline.py           # data stages: ingest prepare tokenize pack (idempotent)
│   ├── registry.py           # @register_project discovery
│   ├── project_api.py        # Project ABC, Doc, SFTExample, EvalPrompt, Grader
│   ├── data/                 # group split, packing, memmap sampler, SFT collation
│   ├── tokenizers/           # base char bpe fixed_vocab
│   ├── graders/              # GENERIC graders shared across projects
│   ├── model/                # llama.py init.py stats.py (presets live in presets/)
│   ├── train/                # trainer optim schedule checkpoint guards
│   ├── sampling/             # generate(), temperature/top-k/top-p, logits_processor hook
│   ├── eval/                 # runner, multi-seed aggregation, runs compare
│   ├── export/               # hf.py (safetensors + config.json), gguf.md
│   ├── serve/                # app.py (FastAPI), constrained decoding via project hook
│   ├── tracking/             # base tensorboard wandb noop
│   └── doctor.py
├── projects/                 # ONE DIRECTORY PER LLM
│   ├── _template/            # copy this to start a new one
│   │   ├── project.py  graders.py  README.md
│   │   ├── experiments/baseline.yaml
│   │   └── tests/
│   ├── shakespeare_char/
│   ├── abc_music/
│   ├── chess/
│   └── cricket_nextball/
├── tests/                    # ENGINE tests only; project tests live with their project
│   ├── unit/                 # CPU-only, < 60 s total
│   └── gpu/                  # @pytest.mark.gpu
└── docs/
    ├── DESIGN.md  ROADMAP.md
    └── decisions/            # ADRs
```

**Why experiments live inside a project:** a use case was previously split across `tasks/<name>/`
and `configs/runs/<name>/`. Folding them together means one directory is the entire LLM — the
thing you copy, read, or delete as a unit.

**Why `src/slmkit/graders/` exists:** several graders are generic — n-gram novelty (ABC now, any
generative project later), log-loss and calibration (shakespeare and cricket both), and a
parse-rate wrapper taking any parser callable. Without this tier, cricket reimplements what
chess already has. Domain graders (bar duration, legal-move rate) stay in the project.

### 6.4 Project API (the extension point)

```python
# src/slmkit/project_api.py  (sketch — refine at M3 when chess stresses it)
@dataclass(frozen=True)
class Doc:
    id: str
    group: str            # leakage-free split key (tune id, game id, match id)
    text: str             # or a token-ready representation for fixed-vocab projects
    meta: dict

@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str
    meta: dict

class Project(ABC):
    name: ClassVar[str]
    Args: ClassVar[type[BaseModel]]          # pydantic schema for project args in YAML

    def __init__(self, args: BaseModel, home: Path): ...

    @abstractmethod
    def ingest(self, raw_dir: Path) -> None: ...          # fetch source data; idempotent
    @abstractmethod
    def documents(self, raw_dir: Path) -> Iterator[Doc]: ...
    def augment(self, doc: Doc) -> Iterator[Doc]:          # train split only
        yield doc
    def split_exclusions(self) -> set[str]:                # e.g. puzzle game ids
        return set()
    @abstractmethod
    def tokenizer_spec(self) -> TokenizerSpec: ...         # char | bpe(vocab) | fixed(list)
    def sft_examples(self, docs: Iterable[Doc]) -> Iterator[SFTExample] | None:
        return None
    @abstractmethod
    def eval_prompts(self, split: str) -> Iterator[EvalPrompt]: ...
    @abstractmethod
    def graders(self) -> list[Grader]: ...                 # (EvalPrompt, str) -> dict[str,float]
    def logits_processor(self) -> LogitsProcessor | None:  # e.g. legal-move masking
        return None
```

Register with `@register_project("abc_music")`. Discovery imports `projects.*` packages listed
in config/env, so the engine never imports a project by name.

### 6.5 Configuration

Pydantic v2 schemas validate everything. Composition order (later wins):
**model preset → train preset → experiment YAML → CLI `--set a.b=c`**.
The resolved config is saved to `runs/<id>/config.resolved.yaml` and hashed.

Experiments are addressed by name: `slm run abc_music/micro_v1` resolves to
`projects/abc_music/experiments/micro_v1.yaml`. Inside a project directory, the bare name works.

```yaml
# projects/abc_music/experiments/micro_v1.yaml
run:
  name: abc-micro-v1
  seed: 1337
  tracker: tensorboard            # tensorboard | wandb | none
project:
  name: abc_music
  args:
    transpose_semitones: [-2, -1, 0, 1, 2]
    header_dropout: 0.4
tokenizer:
  type: char                      # char | bpe | fixed (fixed comes from the project)
data:
  block_size: 512
  val_fraction: 0.05
model:
  preset: micro                   # ~4.7M params (non-embedding)
  overrides: {dropout: 0.15}
train:
  preset: default
  max_tokens: 60_000_000
  batch_size: 64
  lr: 1.0e-3
  warmup_tokens: 1_000_000
  eval_every_steps: 250
  ckpt_every_minutes: 20
  optimizer: adamw                # adamw | adamw8bit
  compile: true
sft:
  enabled: true
  lr: 3.0e-4
  epochs: 3
eval:
  num_samples: 500
  seeds: [0, 1, 2]
  temperature: 0.8
```

**Model presets.** SwiGLU hidden = `round_to_64(8/3 × d_model)`; params exclude embeddings.

| Preset | Layers | d_model | Heads | ~Params | Use |
|---|---|---|---|---|---|
| ref | 6 | 384 | 6 | ~10.6M | nanoGPT-dimension reference run |
| nano | 4 | 128 | 4 | ~0.85M | tests, ABC baseline |
| micro | 6 | 256 | 8 | ~4.8M | ABC |
| tiny | 8 | 512 | 8 | ~25.7M | chess |
| small | 12 | 768 | 12 | ~84.9M | chess scaling / GPT-2-small class |
| medium | 24 | 1024 | 16 | ~303.6M | late-stage, chess only |

### 6.6 Trainer requirements

- bf16 autocast, TF32 enabled, `torch.compile` (configurable), gradient accumulation, gradient
  clipping, cosine schedule with warmup, scheduled in **tokens** rather than steps.
- **Full-state checkpoint:** model, optimizer, scheduler, step, tokens seen, data-sampler state,
  and all RNG states (python, numpy, torch CPU, torch CUDA).
- **Atomic save:** write `ckpt.tmp/`, fsync, rename. Keep the last N plus the best-by-val-loss.
  Time-based cadence (default 20 min).
- **Resume is the default path.** `slm pretrain <cfg>` resumes automatically when the run dir
  holds a checkpoint, and logs how long the gap was. Stopping a run is normal operation.
- **Guards:** on non-finite loss or grad norm, skip the step and log it; abort after K
  consecutive skips; log grad norm every step.
- **Throughput report:** at step 50 and every eval, log tokens/s, MFU estimate, peak memory, and
  **GPU-hours remaining** (not a wall-clock finish time — the machine sleeps).
- **SFT** reuses the trainer. The only differences are the dataset (prompt/completion with a
  project template) and a loss mask zeroing the prompt tokens. Unit-test the mask.

**MFU expectations are size-dependent.** MFU scales with `d_model`: small matmuls do not
saturate tensor cores, so `nano`/`micro` realistically reach 10–20% while `small`/`medium`
should reach 25–45%. Do not gate a milestone on one band across all sizes. At the smallest
presets, `torch.compile(mode="reduce-overhead")` and CUDA graphs matter more than anything else,
because the loop is partly kernel-launch-bound.

### 6.7 Evaluation

- `slm eval <run_id>` samples `num_samples` outputs per seed from the project's `eval_prompts`,
  runs every grader, and writes `eval/<eval_id>.json` with per-seed metrics plus mean ± std.
- `slm runs compare a b c` prints a metrics table across runs. At small scale, seed variance
  often exceeds the effect of a change: **run ≥3 seeds before believing a difference.**
- Graders are pure functions, unit-tested with hand-made good and bad samples.

### 6.8 `slm doctor` and path guards

`slm doctor` must pass before any training command runs. It checks:

- CUDA available; device name; `sm_120` in `torch.cuda.get_arch_list()`; bf16 matmul works.
- An SDPA smoke test; a bitsandbytes 8-bit Adam step (warn, don't fail, if missing).
- `$SLM_HOME` set, on ext4 (not under `/mnt/`), with free space above a threshold.
- `HF_HOME` / `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` under `$SLM_HOME`.
- `MemTotal` of the WSL VM, and swap usage (warn above 50%).
- Package versions, printed and written to `$SLM_HOME/doctor.json`.

`artifacts.py` refuses to read or write artifacts under `/mnt/`, with an explanatory error. The
one allowed exception is `export --to-windows`.

### 6.9 Export and serving

- `slm export <run_id> --name abc-folk --version 1` writes an HF-format directory
  (`config.json` for `LlamaForCausalLM`, `model.safetensors`, tokenizer files) plus
  `MODEL_CARD.md` (project, data, params, tokens, eval summary, lineage).
- **Acceptance test:** load the export with HF `transformers` and check logits match slmkit's
  within tolerance.
- GGUF: document the llama.cpp conversion in `export/gguf.md`. Char-level and fixed vocabularies
  may need a custom tokenizer mapping, so verify per project.
- `slm serve --model abc-folk:1` runs a small FastAPI app (`/generate`) applying the project's
  `logits_processor`. Models are tiny, so CPU serving is fine.
- For anything worth keeping online, put an existing gateway in front rather than exposing
  `slm serve` directly — it handles auth, rate limiting and TLS, which this app deliberately
  does not.
- Optional MLOps exercise: deploy the serve container to Kubernetes with a readiness probe and
  run `slm eval` as a Job against it. Kubernetes is **not** used for training.

### 6.10 Tracking

A tracker interface with three implementations: `tensorboard` (default, local files under the
run dir), `wandb` (cloud), and `noop`. **Tracking must not depend on Docker Desktop being up**,
because Docker is stopped during training sessions.

---

## 7. Testing strategy

| Test | What it proves |
|---|---|
| Tokenizer round-trip (each type) | encode→decode is lossless |
| Split leakage | no `group` appears in both train and val; exclusions honoured |
| Packing | token counts match `meta.json`; dtype and bounds valid; `vocab_size < 65536` |
| Model shapes and param count | presets produce the expected sizes |
| Overfit one batch | loss → ~0 on a single batch (catches most trainer bugs) |
| **Resume equivalence** | 20 steps straight vs 10 + resume + 10 → same loss (tol 1e-4, CPU, deterministic) |
| SFT mask | prompt positions contribute zero loss and zero grad |
| HF export parity | logits match `transformers` within tolerance |
| Graders | known-good and known-bad fixtures per project |
| GPU smoke (`-m gpu`) | 50 steps on `nano` with compile + bf16; tok/s printed |

Resume equivalence is the highest-value test in the suite, because on this machine every real
run is a resumed run.

---

## 8. Risks and operations

| Risk | Mitigation |
|---|---|
| **Workstation off for days mid-run** | Time-based checkpoints; resume as the default path; `runs list` shows progress; budget in GPU-hours |
| Windows Update reboot mid-session | Pause updates / active hours; auto-resume |
| Silent numerical issues (no ECC) | Non-finite guards, grad-norm logging, eval on resume |
| Thermals and noise (shared workstation) | ~280–300 W power limit; monitor memory-junction temp |
| WSL memory pressure | Stop other containers first; `autoMemoryReclaim`; stream preprocessing |
| VHDX loss | rsync kept checkpoints/models off-box; periodic `wsl --export` |
| Memorization mistaken for learning | Group splits, n-gram novelty grader, early stopping |
| Eval contamination | `split_exclusions()` (e.g. Lichess puzzle source games) |
| Premature abstraction | Stage and artifact boundaries now; refine the Project API at chess (M3) |
| **Motivation loss during M1** | Print generated samples at every eval, not just loss |

---

## 9. Decisions log (full ADRs in `docs/decisions/`)

1. Train and serve in a dedicated WSL distro; Hyper-V rejected (no GPU).
2. PyTorch SDPA for attention; no `flash-attn` dependency.
3. One Llama-compatible model family; HF-format export.
4. Hand-written trainer; no HF Trainer / Lightning / Accelerate.
5. uv + pydantic + YAML + Typer; no Hydra.
6. 1B-scale from-scratch is out of scope; largest planned run is ~350M, chess only.
7. Kubernetes only for the optional serving exercise.
8. Cricket reframed as next-ball-outcome modeling.
9. `projects/` is the unit of a use case, with experiments folded in (ADR 0002).
10. Budget and report compute in GPU-hours, never wall-clock dates.
