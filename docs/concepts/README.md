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
| [`tokenization.md`](tokenization.md) | M1 ☑ | Turning text into tokens, splitting by group, packing, and the one-token shift |
| [`the-model.md`](the-model.md) | M1 ☑ | What a decoder-only transformer does, walked through this repo's code |
| [`the-training-loop.md`](the-training-loop.md) | M1 ☑ | One step, the lr schedule, evals and samples, checkpoints and resume, MFU |
| [`learning-paradigms.md`](learning-paradigms.md) | M1 ☑ | Supervised, unsupervised, self-supervised; classification vs regression; where slmkit sits |
| [`fitting.md`](fitting.md) | M1 ☑ | Underfitting and overfitting: expected curves next to three real runs, plus an animation |
| [`metrics.md`](metrics.md) | M1 ☑ | Which metric when: loss, perplexity, bpc, accuracy, precision/recall, MAE/RMSE, calibration, MFU |
| [`data-preparation.md`](data-preparation.md) | M2 ◐ | Licences, near-duplicates, honest augmentation, and verifying a transformation against an oracle |
| `sft.md` | M2 | Base vs instruction-tuned models, and why loss is masked on prompts |
| `evaluation.md` | M2 | Programmatic graders, seeds, baselines, and detecting memorization |
| `serving.md` | M4 | Export formats and when each is used |
