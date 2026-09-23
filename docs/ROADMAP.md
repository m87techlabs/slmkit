# slmkit — Roadmap

Each milestone has **exit criteria**. Don't start the next one until they pass. The framework
is the product; each project exists to harden it.

Effort is given in **sessions** (an evening's work) and **GPU-hours** separately, because the
machine is off most of the time and wall-clock dates mean nothing here.

Status legend: ☐ not started · ◐ in progress · ☑ done

---

## M0 — Environment ☑

Goal: a reproducible, verified ML environment in the `Ubuntu-ML` WSL distro.
*~1–2 sessions, minutes of GPU.*

- [x] Create the distro: `wsl --install -d Ubuntu-24.04 --name Ubuntu-ML` (or export/import a
      clean Ubuntu rootfs if your WSL version lacks `--name`).
- [x] `scripts/setup-ml-distro.sh` — packages, `uv`, `$SLM_HOME`, cache env vars. Idempotent.
- [x] Update `.wslconfig` (DESIGN §3); `wsl --shutdown`; verify with `free -h` (19.5 GiB).
- [x] Measure real disk throughput with `fio --direct=1`. Recorded in ADR 0001.
- [x] `uv sync`; confirm `sm_120` appears in `torch.cuda.get_arch_list()`.
- [x] Implement `slm doctor` (DESIGN §6.8).
- [x] Write `docs/concepts/environment.md`: why WSL, why these versions, what `doctor` checks
      and what each check protects against.
- [x] Write `docs/runbooks/m0-environment.md`: every M0 step with how to verify it by hand,
      expected output, and why it was done that way.

**Exit criteria — met**
- [x] `slm doctor` all green, including an 8-bit Adam step (bitsandbytes 0.50.2 on `sm_120`).
- [x] Achieved bf16: **121.6 TFLOPS**, against a ~112 spec-sheet estimate. Planning figure is
      now ~43 TFLOPS (≈35% MFU); every GPU-hour budget in DESIGN §2 was updated.
- [x] ADR 0001 records every version and measurement.

---

## M1 — Engine + reference reproduction ☐

Goal: prove the trainer is correct against a published result before trusting any novel project.
*The longest stretch before anything interesting comes out — ~6–10 sessions, a few GPU-hours.*

- [ ] `config/` — pydantic schemas, preset merge, `--set` overrides, resolved-config hashing.
- [ ] `artifacts.py` — manifests, content-addressed IDs, atomic commit, `/mnt/` guard.
- [ ] `registry.py` + `project_api.py` — minimal version, enough for one project.
- [ ] `tokenizers/char.py`.
- [ ] `data/` — group split, packing to `uint16` memmap, random-offset batch sampler.
- [ ] `model/llama.py` — RMSNorm, RoPE, SwiGLU, SDPA, tied embeddings, presets.
- [ ] `train/` — trainer, AdamW, cosine+warmup in tokens, clipping, guards, full-state
      checkpoint/resume, throughput and GPU-hours-remaining logging.
- [ ] `tracking/` — tensorboard + noop.
- [ ] `sampling/` — `generate()` with temperature / top-k / top-p.
- [ ] **Print generated samples at every eval, not just loss.** Non-negotiable: it is the only
      visible payoff in this milestone.
- [ ] `projects/shakespeare_char` with the `ref` preset (6L/384d/6h, block 256, dropout 0.2).
- [ ] Unit tests: tokenizer round-trip, split leakage, packing, shapes, overfit-one-batch,
      **resume equivalence**.
- [ ] Write `docs/concepts/`: `tokenization.md`, `the-model.md` (what a decoder-only
      transformer does, walked through slmkit's own code), `the-training-loop.md` (loss, LR
      schedule, why warmup, what the guards catch).
- [ ] Write `docs/runbooks/m1-engine.md`: how to verify each M1 piece by hand (run the
      reference, watch loss and samples, kill and resume a run), with expected output and why.

**Exit criteria**
- `slm run shakespeare_char/ref` reaches val loss **≤ 1.55 with a loss curve shaped like
  nanoGPT's** (reference ≈1.47 on a GPT-2-style model; slmkit is Llama-style, so the numbers
  are close but not identical — see DESIGN §5.0).
- Kill the process mid-run, reboot the machine, restart: it resumes and the loss curve continues
  without a jump.
- Achieved MFU is logged and plausible **for the preset size** (10–20% at `nano`/`micro`,
  25–45% at `small`+ — DESIGN §6.6).
- `make test` passes on CPU in under 60 s.

---

## M2 — First real project: ABC music (full lifecycle) ☐

Goal: exercise every stage end-to-end — tokenizer, pretrain, SFT, eval, export, serve.
*~6–10 sessions, tens of GPU-hours across all sweeps.*

- [ ] `projects/abc_music`: ingest TheSession dump; parse settings; `group = tune_id`.
- [ ] Augmentation: transposition within ±2 semitones (train split only); header dropout.
- [ ] Tokenizers: `char` baseline; add `bpe.py` (HF `tokenizers`) and compare.
- [ ] `graders/` (engine): n-gram novelty, parse-rate wrapper.
- [ ] Project graders: bar durations vs `M:`, tonic cadence, prompt adherence.
- [ ] SFT: prompt templates, prompt-token loss masking, plus a unit test.
- [ ] `eval/` runner with multi-seed aggregation; `slm runs compare`.
- [ ] `export/hf.py` plus the HF parity test; `MODEL_CARD.md` generation.
- [ ] `serve/app.py` (FastAPI `/generate`).
- [ ] `export --to-windows` for MIDI (abc2midi) so you can listen on Windows.
- [ ] Sweep nano vs micro, char vs BPE, with and without augmentation (3 seeds each).
- [ ] Write `docs/runbooks/m2-abc-music.md` (verify by hand: ingest, splits, graders, listen to output).
- [ ] Write `docs/concepts/`: `sft.md` (why loss masking, how base and instruct models differ)
      and `evaluation.md` (why programmatic graders, why ≥3 seeds, what a baseline is for).

**Exit criteria**
- The best model beats a trivial baseline on every grader; parse rate ≥ 95%.
- The n-gram novelty grader shows it isn't regurgitating training tunes.
- SFT with a natural-language prompt reaches **at least parity** with the base model given a
  header prefix, on meter and key adherence (DESIGN §5.1 — parity is the pass mark, not a win).
- An exported model loads in HF `transformers` with matching logits and serves via `slm serve`.
- **Framework check:** a new ABC experiment is a YAML file only.
- You have listened to a generated tune on Windows and it sounds like a tune.

---

## M3 — Second project: chess (API hardening + scaling lab) ☐

Goal: stress the abstractions with a very different project, then refactor the engine.
*~6–10 sessions; the scaling sweep is the first real GPU-hour spend (tens of hours).*

- [ ] `tokenizers/fixed_vocab.py` (1,968 UCI moves + specials).
- [ ] Streaming ingest: `zstdcat | multiprocess parse`, filtered by Elo and time control,
      sharded packing.
- [ ] `split_exclusions()`: game IDs referenced by the Lichess puzzle DB.
- [ ] Project graders: legal-move rate, puzzle accuracy, Elo vs Stockfish (fixed skill/depth).
- [ ] `logits_processor` for legal-move-masked decoding (serve and eval, both togglable).
- [ ] **Refactor:** anything chess needed that the Project API lacked gets generalized into the
      engine. Write an ADR for each change.
- [ ] Scaling sweep: nano → micro → tiny (→ small if the GPU-hours are there), LR swept per size.
- [ ] `adamw8bit` option with an A/B check against fp32 AdamW.

**Exit criteria**
- The `tiny` model reaches ≥ 99% legal-move rate without masking.
- A scaling plot (loss vs FLOPs across ≥3 sizes) generated by a script from run manifests.
- **Framework check:** zero `if project == "chess"` branches in `src/slmkit/`.

---

## M4 — Serving and MLOps polish ☐

*~2–4 sessions, negligible GPU.*

- [ ] GGUF conversion documented and tested for at least one project.
- [ ] Container image for `slm serve` (CPU); Compose file.
- [ ] Put a gateway in front of `slm serve` (auth, rate limiting, TLS) rather than exposing it.
- [ ] Optional: Kubernetes Deployment + Service + readiness probe; `slm eval` as a Job against
      the service. Any local cluster works (Docker Desktop's Kubernetes, kind, k3s).
- [ ] `slm lineage` and `slm runs list` polished; model registry at `$SLM_HOME/models/index.json`.
- [ ] Backup script: rsync `models/` and selected runs off-box.

**Exit criteria:** any exported model can be served and evaluated by `name:version` with one
command, and `docs/concepts/serving.md` explains the export formats and when each is used.

---

## M5 — Third project + larger runs ☐

- [ ] `projects/cricket_nextball`: Cricsheet ingest, fixed vocab of ball events, log-loss and
      calibration graders against a frequency baseline.
- [ ] Optional: a 124M run (~16–18 GPU-hours, i.e. several sessions) on chess, with the full ops
      checklist (DESIGN §3).
- [ ] Optional: a 350M run (~100–110 GPU-hours) only if the 124M run shows clear headroom.

**Exit criteria:** the third project was added **without modifying the engine** (or with a small,
ADR-documented generic extension).

---

## Out of scope

- 1B-parameter from-scratch pretraining (~840 GPU-hours).
- Multi-GPU / distributed training.
- Fine-tuning third-party base models.
- Kubernetes for training.
- Any workflow assuming the machine is on unattended.
