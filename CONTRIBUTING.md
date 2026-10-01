# Contributing to slmkit

The rules every change to slmkit follows: what the repo is for, who its documentation is written
for, the hard rules the code enforces, and the conventions for code, docs and commits. The docs
cite them as "CONTRIBUTING.md rule N".

## What this project is

**slmkit** is a framework for building many **small, task-specific language models from
scratch** on one RTX 5080 (16 GB, Blackwell `sm_120`) inside WSL2. Every model goes through the
same pipeline:

`ingest → prepare → tokenizer → pack → pretrain → SFT → eval → export → serve`

- A new model on an existing project = **one YAML file** in `projects/<name>/experiments/`.
- A new use case = **one directory under `projects/`**, with no engine changes.
- The framework is the deliverable; projects (ABC music, chess, cricket) exist to harden it.
- This is a **learning project**: the model, trainer, sampler and SFT masking are hand-written
  on purpose.

**Read before architectural work:** `docs/DESIGN.md` (architecture, contracts, rules),
`docs/MODEL.md` (the model spec: architecture, presets, file formats; keep it in sync with
`src/slmkit/model/` and `presets/model/`) and `docs/ROADMAP.md` (current milestone and exit
criteria). Work on the current milestone only.

## Vocabulary

A **project** is one LLM (`projects/abc_music/`) — the Terraform root-module analogue. The
**engine** is `src/slmkit/`. This repo is always called "slmkit", never "the project".

## Who this is written for

Readers are assumed to be **comfortable with software and infrastructure** — Linux, Docker,
CI, infrastructure-as-code — and **new to the ML training stack**. Write for that reader.

- Explain ML-specific choices briefly (1–3 sentences) wherever they appear: why this learning
  rate, why this initialization, why the mask goes here. Never assume ML background.
- Skip explanations of infrastructure basics. That is not where the gap is.
- Infrastructure analogies are welcome and encouraged: stages = Terraform modules, experiment
  YAML = tfvars, artifacts = immutable state.
- Any abbreviation not in `docs/GLOSSARY.md` must be added there the first time it is used.
- Any tool, library or service that enters the repo (dependency, system package, or something a
  doc tells you to run) gets an entry in `docs/STACK.md` in the same change: what it is, why
  slmkit uses it, what was rejected, and a link to its documentation.

## Documentation is a deliverable, not an afterthought

This repo exists to be **read** as much as run. Someone learning the LLM lifecycle should be
able to follow it without prior ML knowledge.

- **Every milestone ships its explainer.** When you implement a stage, write the `docs/concepts/`
  page explaining what it does and why it works that way — in the same change, not later.
- **Every milestone ships its runbook** in `docs/runbooks/mN-<name>.md`: for each piece built,
  *what* it is, *how* it was done, *how to verify it by hand* (commands plus real expected
  output captured by running them), and *why* it was done that way, including what was
  rejected. Update it whenever a verification step changes.
- **Figures are generated, never hand-drawn.** Charts in `docs/images/` come from
  `scripts/make_figures.py` (`make figures`) reading real runs; diagrams are Mermaid in the Markdown.
  Regenerate figures when the runs they show change.
- Explain the *why*, not the *what*. `docs/concepts/` is for understanding; docstrings are for
  interface.
- Prefer a worked example with real numbers over an abstract description.
- When you make a non-obvious choice, say what you rejected and why. That is usually the most
  useful sentence on the page.
- Keep everything **generic**. This repo is public: no personal machine names, no private
  infrastructure, no "my setup". Platform assumptions belong in DESIGN §1, stated as
  assumptions.

## The machine is not always on

The target platform is a **shared workstation that is powered off regularly** — overnight and
often for days. It is not a dedicated training box, and there is no deadline.

- **Never design anything that assumes uptime.** No cron, no daemons, no unattended multi-day
  runs, no "start it and check tomorrow".
- Budget and report compute in **GPU-hours, never wall-clock dates.** A "4-day run" is ~100
  GPU-hours spread over however many evenings it takes.
- **Resume is the default path**, not recovery. `slm pretrain` resumes automatically when the
  run dir holds a checkpoint. Stopping a run is normal operation, not failure.
- Checkpoint on a time cadence (20 min), never only at epoch boundaries.
- `tmux` survives a disconnect, not a power-off. The checkpoint is the continuity mechanism.
- Favour correctness over speed of delivery — there is no deadline.

## Environment facts (do not "fix" these)

- Code runs in the WSL distro `Ubuntu-ML` (Ubuntu 24.04). Python via **`uv`** only
  (`uv add`, `uv run`). Never `pip install` into the system Python.
- **Never install or suggest a Linux NVIDIA driver inside WSL.** CUDA comes from the Windows
  driver via `/dev/dxg`. See ADR 0003.
- PyTorch comes from the **CUDA 12.8+** index pinned in `pyproject.toml`; `sm_120` must appear
  in `torch.cuda.get_arch_list()`.
- Attention = `torch.nn.functional.scaled_dot_product_attention`. **Do not add `flash-attn`.**
- bitsandbytes 8-bit Adam is optional and must be A/B-verified against fp32 AdamW before use.
- `$SLM_HOME` (default `~/slm`, ext4) holds all data, artifacts, runs, models and caches.

## Hard rules

1. **No pipeline I/O under `/mnt/`** (Windows drives via 9P are 20–70× slower — measured).
   Artifacts, datasets, caches and checkpoints live under `$SLM_HOME`. `artifacts.py` enforces
   this; don't bypass it. The only exception is `slm export --to-windows`.
2. **The engine never imports a project.** Nothing in `src/slmkit/` may reference `abc_music`,
   `chess`, etc. Project-specific behavior goes through the `Project` API
   (`src/slmkit/project_api.py`). If a project needs something the API lacks, extend the API
   generically and write an ADR.
3. **Artifacts are immutable and content-addressed.** Stages never overwrite; a changed config
   produces a new ID. Every artifact dir gets a `manifest.json` (DESIGN §6.2).
4. **Splits use `Doc.group`**, never random per-document splits. Augmentation happens only on
   the train split, after splitting. Honor `Project.split_exclusions()`.
5. **One model family** (Llama-style, HF `LlamaForCausalLM`-compatible parameter names). No
   alternative architectures without an ADR.
6. **Don't add frameworks** without an ADR: no HF Trainer, Lightning, Accelerate, DeepSpeed,
   Hydra, Ray, Airflow or Kubeflow. Allowed: torch, numpy, pydantic, typer, pyyaml,
   safetensors, HF `tokenizers`, HF `transformers` (export parity tests only), tensorboard,
   wandb (optional), fastapi/uvicorn, project-specific libs (python-chess, music21).
7. **Single GPU, single process.** No DDP/FSDP code paths.
8. **Checkpoints are full-state and atomic:** model, optimizer, scheduler, step, tokens_seen,
   sampler state and all RNG states. Write to `*.tmp`, fsync, rename.
9. **Long GPU runs are started by a person, in tmux.** Anything that launches runs on its own
   (a script, a sweep helper, an automated tool) stops at about 10 minutes of GPU work and prints
   the command, the config path and a GPU-hour estimate from measured tokens/sec instead.
10. Tracking must work with Docker stopped (tensorboard files or wandb cloud, never a
    containerized tracking server).

## Commands

```bash
make doctor                 # env checks; must pass before training
make test                   # CPU unit tests, < 60 s
make test-gpu               # pytest -m gpu smoke tests
make lint                   # ruff check + format --check + mypy

uv run slm run abc_music/micro_v1            # full pipeline; skips existing artifacts
uv run slm pretrain abc_music/micro_v1 --set train.lr=6e-4
uv run slm eval <run_id>
uv run slm runs list                          # tokens done/target, GPU-hours, last checkpoint
uv run slm runs compare <run_a> <run_b>
uv run slm export <run_id> --name <model> --version <n>
uv run slm serve --model <model>:<n>
uv run slm studio start | stop | status | run  # "See it in action": the local web studio
uv run slm lineage <artifact_id>
```

Experiments are addressed as `<project>/<experiment>`, resolving to
`projects/<project>/experiments/<experiment>.yaml`. If a command above doesn't exist yet, it's
part of the current milestone — implement it rather than inventing a different interface.

## Code conventions

- Python 3.12, full type hints, `ruff` (format + lint), `mypy` on `src/`.
- Config: pydantic v2 in `src/slmkit/config/`. Merge order: model preset → train preset →
  experiment YAML → `--set`. Persist `config.resolved.yaml` in every run dir.
- Schedule training in **tokens**, not steps (`max_tokens`, `warmup_tokens`).
- Log on startup: param count (embedding and non-embedding), tokens per step, estimated FLOPs,
  and **GPU-hours remaining**. After step 50, log measured tokens/s, MFU and peak memory.
- **Print generated samples at every eval**, not just loss.
- Seed everything via one `seed_everything(seed)` helper. Aim for statistical reproducibility,
  not bit-exact determinism. Resume-equivalence tests run on CPU with deterministic settings.
- Paths: `pathlib.Path`, resolved from `$SLM_HOME`; no hard-coded home dirs.
- Tests: CPU-only by default and small (`nano` preset, tiny synthetic data). GPU tests are
  marked `@pytest.mark.gpu`. Project tests live in `projects/<name>/tests/`.
- Graders are pure functions `(EvalPrompt, output: str) -> dict[str, float]`, each with
  known-good and known-bad fixtures.
- Licensed Apache-2.0. Per-file copyright headers are not used; `LICENSE` and `NOTICE` at the
  root cover the repo. Do not vendor code under a copyleft licence (GPL/AGPL), and record the
  source and licence of any third-party code or data in the relevant `README.md`.

## Adding a new project — checklist

1. `cp -r projects/_template projects/<name>`.
2. `project.py`: subclass `Project`, decorate `@register_project("<name>")`, define an `Args`
   pydantic model.
3. Implement `ingest`, `documents` (with a meaningful `group`), `tokenizer_spec`,
   `eval_prompts`, `graders`. Optionally `augment`, `split_exclusions`, `sft_examples`,
   `logits_processor`.
4. `README.md`: data source + license, grader definitions, known pitfalls.
5. Tests: split leakage, tokenizer round-trip, grader fixtures.
6. `experiments/baseline.yaml` using the `nano` preset first.
7. Add an optional-dependency extra in `pyproject.toml` if it needs a domain library.
8. **Verify no engine file changed**, or write an ADR explaining the generic API extension.

## Commits

- Conventional commit subjects (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).
- The body explains *why*, not what the diff already shows. Say what was rejected when the
  choice was not obvious.
- **No co-author or generated-by trailers.** Commits carry the repository author only.
- Never commit anything under `$SLM_HOME`; artifacts stay out of git by design.

## Workflow expectations

- **Plan first** for anything touching more than two files: list files, interfaces and tests,
  then implement.
- Keep changes small and milestone-scoped. Tick `docs/ROADMAP.md` checkboxes when exit criteria
  are met, not when code is merely written.
- Record decisions in `docs/decisions/NNNN-title.md` (context, decision, consequences,
  alternatives).
- When results look surprisingly good, suspect leakage or memorization first: check splits,
  n-gram novelty and the val/train gap.
- When loss misbehaves, debug in this order: overfit one batch → LR/warmup → data/label shift
  (off-by-one in targets) → mask → numerics (bf16, NaN guards).
- Never delete run directories or artifacts; propose a cleanup command instead.

## Key numbers (for sanity checks)

- Peak BF16 (FP32 accumulate) ≈ 112 TFLOPS (spec-derived). Plan with **~40 TFLOPS effective**
  until M0 measures the real number, then use that.
- Training FLOPs ≈ 6 × params × tokens. 124M × 3B ≈ 16–18 GPU-hours; 350M × 7B ≈ 100–110
  GPU-hours (chess only); 1B is out of scope (~840).
- AdamW mixed precision ≈ 16 bytes/param; 8-bit Adam ≈ 8 bytes/param. **Memory is not the
  binding constraint — compute is.**
- MFU is size-dependent. Measured in M1: ~24% at `nano`, ~42% at `micro`, ~51% at `ref`, ~54% at
  `tiny` (compiled bf16). The planning figure of ~43 TFLOPS is conservative from `micro` up.
- ABC corpus ≈ 5–20M tokens → 1–5M param models. Chess → up to ~25–124M.
