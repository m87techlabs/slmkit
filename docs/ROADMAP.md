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

## M1 — Engine + reference reproduction ☑

Goal: prove the trainer is correct against a published result before trusting any novel project.
*The longest stretch before anything interesting comes out — ~6–10 sessions, a few GPU-hours.*

- [x] `config/` — pydantic schemas, preset merge, `--set` overrides, resolved-config hashing.
- [x] `artifacts.py` — manifests, content-addressed IDs, atomic commit, `/mnt/` guard.
- [x] `registry.py` + `project_api.py` — minimal version, enough for one project.
- [x] `tokenizers/char.py`.
- [x] `data/` — group split, packing to `uint16` memmap, random-offset batch sampler.
- [x] `model/llama.py` — RMSNorm, RoPE, SwiGLU, SDPA, tied embeddings, presets.
- [x] `train/` — trainer, AdamW, cosine+warmup in tokens, clipping, guards, full-state
      checkpoint/resume, throughput and GPU-hours-remaining logging.
- [x] `tracking/` — tensorboard + noop.
- [x] `sampling/` — `generate()` with temperature / top-k / top-p.
- [x] **Print generated samples at every eval, not just loss.** Non-negotiable: it is the only
      visible payoff in this milestone.
- [x] `projects/shakespeare_char` with the `ref` preset (6L/384d/6h, block 256, dropout 0.2).
- [x] Unit tests: tokenizer round-trip, split leakage, packing, shapes, overfit-one-batch,
      **resume equivalence**.
- [x] Write `docs/concepts/`: `tokenization.md`, `the-model.md` (what a decoder-only
      transformer does, walked through slmkit's own code), `the-training-loop.md` (loss, LR
      schedule, why warmup, what the guards catch).
- [x] Write `docs/runbooks/m1-engine.md`: how to verify each M1 piece by hand (run the
      reference, watch loss and samples, kill and resume a run), with expected output and why.

**Exit criteria — met**
- [x] `slm run shakespeare_char/ref` reaches val loss **≤ 1.55 with a loss curve shaped like
  nanoGPT's**: best **1.2888** at step 1250, then overfits as expected (runbook D.2). nanoGPT
  reports ≈1.47 on a GPT-2-style model and a different validation split — see DESIGN §5.0.
- [x] Kill the process mid-run and restart: it resumes and the loss curve continues without a
  jump (0.6951 → 0.7076 across the stop, runbook D.1). Verified as a process restart, not a
  machine reboot.
- [x] Achieved MFU is logged and plausible **for the preset size** (52.9% at `ref`). Measured in Phase C: ~24% at
  `nano`, ~42% at `micro`, ~51% at `ref`, ~54% at `tiny` (the design's 10–20% guess for small
  presets was pessimistic; DESIGN §6.6).
- [x] `make test` passes on CPU in under 60 s (89 tests, ~8 s).

---

## M2 — First real project: ABC music (full lifecycle) ☑

Goal: exercise every stage end-to-end — tokenizer, pretrain, SFT, eval, export, serve.
*~6–10 sessions, tens of GPU-hours across all sweeps.*

- [x] `projects/abc_music`: ingest public-domain tune books (ADR 0005; The Session's licence forbids LLM training); clean; group by melody fingerprint + title.
- [x] Augmentation: transposition within ±2 semitones into ≤3-accidental keys (train split only, verified note-for-note with `abc2midi`); header dropout.
- [x] Tokenizers: `char` baseline; add `bpe.py` (HF `tokenizers`) and compare: 1.812 vs 1.824 bpc at equal compute, one seed each (tokenization.md §6); the 3-seed comparison is in the Phase F sweep.
- [x] `graders/` (engine): n-gram novelty, parse-rate wrapper.
- [x] Project graders: plays in `abc2midi`, bar durations vs the requested `M:`, ends on the requested tonic (prompt adherence via `EvalPrompt.meta`); calibrated on the human corpus (evaluation.md §3).
- [x] SFT: prompt templates, prompt-token loss masking, plus a unit test (zero loss and zero gradient on prompt positions); `slm sft`, and `slm run` continuing into SFT (ADR 0006).
- [x] `eval/` runner with multi-seed aggregation and a no-model baseline; `slm runs compare`.
- [x] `export/hf.py` plus the HF parity test (checked on every export: logit difference 0.0 for char and BPE); `MODEL_CARD.md` generation; immutable `name:version` (ADR 0007).
- [x] `serve/app.py` (FastAPI `/health`, `/info`, `/generate`).
- [x] `export --to-windows` for MIDI (abc2midi) so you can listen on Windows (`Project.render_sample`, ADR 0007).
- [x] A browser playground in `slm serve`: build a request, generate, see sheet music and play it
      (`Project.web_viewer`, ADR 0008). Added after M2's plan, at the developer's request.
- [x] Sweep nano vs micro, char vs BPE, with and without augmentation (3 seeds each), plus the
      micro × no-augmentation interaction cell; `slm runs summary` (experiments.md). micro wins; at
      equal compute transposition doesn't help on original-key tests, it buys robustness to key.
- [x] Write `docs/runbooks/m2-abc-music.md` (verify by hand: ingest, splits, graders, listen to output).
- [x] Write `docs/concepts/`: `sft.md` (why loss masking, how base and instruct models differ)
      and `evaluation.md` (why programmatic graders, why ≥3 seeds, what a baseline is for).

**Exit criteria — met, except the parse rate, which is carried to M3 (decided 2026-09-30)**
- [ ] The best model beats a trivial baseline on every grader; parse rate ≥ 95%. **Baseline: met**
      (every grader, e.g. plays 0.743 vs 0.063). **Parse rate: not met.** The only meaningful
      definition is the strict `plays` (no `abc2midi` error at all): 0.743 ± 0.050 for the best
      configuration (`micro_noaug`). A lenient "it parses" is passed by random characters 91.5% of the
      time, so it measures nothing, and lowering the temperature to reach 95% makes tunes loop and
      copy instead (experiments.md §8). Carried to M3 as an `abc_music` target for constrained
      decoding.
- [x] The n-gram novelty grader shows it isn't regurgitating training tunes (0.998: 0.2% of
      32-character windows appear in training).
- [x] SFT with a natural-language prompt reaches **at least parity** with the base model given a
      header prefix, on meter and key adherence (DESIGN §5.1 — parity is the pass mark, not a win).
      Bars 0.851 vs 0.708, tonic 0.340 vs 0.212 (runbook D.3, re-measured in Phase E).
- [x] An exported model loads in HF `transformers` with matching logits (difference 0.0) and serves
      via `slm serve`.
- [x] **Framework check:** a new ABC experiment is a YAML file only (`micro`, `noaug` and
      `micro_noaug` needed no code).
- [x] You have listened to a generated tune on Windows and it sounds like a tune.

---

## Studio — `slm studio`: "See it in action" ◐

Goal: everything built and learnt so far, in one local, interactive web app, before M3 adds a
second project. It reads only what is on disk ($SLM_HOME and the repo), so it can't drift from what
the pipeline did. One studio for all projects, with a project switcher: M3's chess must appear with no
studio change. Decided 2026-09-30: no Node (plain ES modules, served by the existing FastAPI);
third-party browser libraries are downloaded to `$SLM_HOME/studio/vendor/` and verified by hash; run
buttons execute read-only commands from a fixed list only. *~3–4 sessions, no GPU.*

- [x] **Phase 1:** `slm studio start | stop | status | run`; pages **Your model** (what you built,
      the machine, live GPU/CPU stats), **Lifecycle** (the pipeline as your real artifacts, with
      lineage), **Training** (loss curves, samples at every eval, configs), **Playground** (any
      exported model). ADR 0009, `runbooks/studio.md`, a "See it in slm studio" section per project runbook.
- [x] **Phase 2:** **Learn** (concepts, ADRs and runbooks rendered, diagrams, glossary on hover),
      **Experiments** (sweeps, seeds, spreads), **Parameters** (size → params, FLOPs, GPU-hours).
- [x] **Phase 3:** **Verify**: each runbook check with a Run button (read-only commands) and your output
      next to the expected output.

**Exit criteria**
- [x] Starting, stopping and checking the studio each take one command, and nothing it writes lands
      in the repo.
- [x] Every number on every page comes from a file on disk, and a test proves it for each API.
- [ ] A project added in M3 appears in the switcher, with its runs, artifacts, models and runbook,
      without changing studio code. *(Rehearsed with `shakespeare_char`, which the studio was never
      written for: every page works.)*

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
- [ ] The chess runbook carries a "See it in slm studio" section, and chess shows up in the studio.
- [ ] Carried from M2: apply the same mechanism to `abc_music` (grammar-constrained decoding) and
      reach `plays` ≥ 95% at the eval temperature without losing `ended` or novelty.
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
