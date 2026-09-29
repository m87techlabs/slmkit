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
| [`tokenization.md`](tokenization.md) | M1 ☑ · M2 | Turning text into tokens, splitting by group, packing, the one-token shift; BPE built and compared with char by bits per character (§6) |
| [`the-model.md`](the-model.md) | M1 ☑ | What a decoder-only transformer does, walked through this repo's code |
| [`the-training-loop.md`](the-training-loop.md) | M1 ☑ | One step, the lr schedule, evals and samples, checkpoints and resume, MFU |
| [`learning-paradigms.md`](learning-paradigms.md) | M1 ☑ | Supervised, unsupervised, self-supervised; classification vs regression; where slmkit sits |
| [`fitting.md`](fitting.md) | M1 ☑ | Underfitting and overfitting: expected curves next to three real runs, plus an animation |
| [`metrics.md`](metrics.md) | M1 ☑ | Which metric when: loss, perplexity, bpc, accuracy, precision/recall, MAE/RMSE, calibration, MFU |
| [`data-preparation.md`](data-preparation.md) | M2 ◐ | Licences, near-duplicates, honest augmentation, and verifying a transformation against an oracle |
| [`evaluation.md`](evaluation.md) | M2 ◐ | Graders, baselines, calibrating graders on real data, sampling vs training seeds, reading the samples, and asking only for what the data contains |
| [`sft.md`](sft.md) | M2 ◐ | Base vs fine-tuned models, masking the loss to the answer, the frozen vocabulary, and parity as the pass mark |
| [`serving.md`](serving.md) | M2 ◐ · M4 | Checkpoint vs export, the HF format, parity checks, immutable versions, what `slm serve` does per request, CPU vs GPU latency, listening on Windows; GGUF and packaging in M4 |
