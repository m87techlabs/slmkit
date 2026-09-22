# Concepts

Explainers written as the code is built, for a reader who is strong on software and
infrastructure but **new to machine learning**.

These pages answer *why*, not *what*. For the interface, read the docstrings; for the
architecture, read `../DESIGN.md`; for any unfamiliar term, read `../GLOSSARY.md`.

Each milestone in `../ROADMAP.md` lists the pages it must produce. A milestone is not complete
until its pages exist — the documentation is part of the deliverable, not a follow-up.

| Page | Milestone | Covers |
|---|---|---|
| [`environment.md`](environment.md) | M0 ☑ | Why WSL, how the GPU reaches Linux, why `sm_120` matters, storage, reading the benchmark |
| `tokenization.md` | M1 | Turning text into tokens, and why the choice matters |
| `the-model.md` | M1 | What a decoder-only transformer does, walked through this repo's code |
| `the-training-loop.md` | M1 | Loss, learning-rate schedules, warmup, and what the guards catch |
| `sft.md` | M2 | Base vs instruction-tuned models, and why loss is masked on prompts |
| `evaluation.md` | M2 | Programmatic graders, seeds, baselines, and detecting memorization |
| `serving.md` | M4 | Export formats and when each is used |
