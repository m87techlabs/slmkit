# slmkit

[![CI](https://github.com/m87techlabs/slmkit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/m87techlabs/slmkit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**A framework for building small language models from scratch on a single consumer GPU — and a
guided way to learn how language models actually work.**

```
ingest → prepare → tokenizer → pack → pretrain → SFT → eval → export → serve
```

![A 10.6M-parameter model learning Shakespeare from scratch](docs/images/watch-it-learn.gif)

*The first reference model learning, trained from random weights in about 3 GPU-minutes: loss on
the left, what it writes on the right. The end-to-end picture is in
[`docs/LIFECYCLE.md`](docs/LIFECYCLE.md).*

Most tutorials either fine-tune somebody else's model or stop at a toy training loop. slmkit
covers the whole lifecycle: you train your own tokenizer, pretrain a model from random
initialization, instruction-tune it, evaluate it with graders that cannot be fooled, export it
to a standard format, and serve it.

## Results so far

Measured on one RTX 5080 (16 GB) under WSL2. Each number links to where it was measured.

| | Shakespeare (M1) | Folk tunes (M2) |
|---|---|---|
| **Learned from** | 1 MB of Shakespeare's plays, one character at a time | 4,248 public-domain folk tunes, written as text in ABC notation |
| **Model** | 10.6M parameters | 4.8M parameters (the best of five configurations) |
| **Training** | 81.9M tokens · 3 GPU-minutes · 1.6 GiB of GPU memory | 30M tokens · about 36 GPU-seconds per run · 2.0 GiB |
| **Result** | validation loss **1.29**; nanoGPT's published reference is about 1.47 | **74%** of generated tunes play without errors and **81%** of bars have the right length (mean of 3 training seeds; real tunes: ~100% and 99%) |
| **Then** | | fine-tuned to take requests in plain English, exported in Hugging Face format, served to a browser that draws and plays the tunes |
| **Sources** | [the training loop](docs/concepts/the-training-loop.md), [runbook M1](docs/runbooks/m1-engine.md) | [experiments](docs/concepts/experiments.md), [runbook M2 §F](docs/runbooks/m2-abc-music.md) |

Asked to continue `JULIET:` / `O Romeo, Romeo,` the Shakespeare model writes *"that cohangeth me to his
act / Of God and first that company thee to the bawd."*: the look of Shakespeare, invented words included,
from 1 MB of text. Asked for `a jig in G major, 6/8 time`, the folk-tune model writes a jig in G major in
6/8 whose every bar adds up. [How to read both, with no music background](docs/concepts/two-models.md).

## Who this is for

People who are **comfortable with software and infrastructure but new to machine learning**.
Everything ML-specific is explained; nothing infrastructure-specific is. Every abbreviation is
defined in [`docs/GLOSSARY.md`](docs/GLOSSARY.md) — start there if a term is unfamiliar.

The model, training loop, sampler and SFT masking are **written out in full on purpose**, with no
training framework. There is no `Trainer` class hiding the interesting parts.

## The idea

The framework is the deliverable. Individual models exist to exercise and harden it.

- A new model on an existing project = **one YAML file**.
- A new use case = **one directory** under `projects/`, with no changes to the engine.

Planned projects: character-level Shakespeare (a correctness reference), ABC music notation,
chess, and cricket ball-outcome prediction. Each was chosen for one reason — **it can be graded
by a program**, with no human judgment in the loop.

## Layout

| Path | Role | Terraform analogy |
|---|---|---|
| `src/slmkit/` | Engine. Never imports a project. | provider |
| `presets/` | Shared model and training config fragments | reusable module |
| `projects/<name>/` | One model: ingest, graders, experiments | root module |
| `projects/<name>/experiments/*.yaml` | One run each | tfvars |
| `$SLM_HOME` (default `~/slm`) | All artifacts, content-addressed and immutable | state |

## Requirements

- **GPU:** one NVIDIA GPU with CUDA 12.8 support; 12–24 GB of VRAM is the design target, and every
  model so far fit in about 2 GB.
- **Tested on:** Windows with WSL2 (Ubuntu 24.04) and an RTX 5080. **Native Linux** with an NVIDIA
  driver is expected to work but hasn't been tested; reports are welcome. The WSL-aware parts have
  plain-Linux fallbacks (the studio opens your default browser) or are optional (`slm export
  --to-windows`).
- **Not supported:** macOS and AMD GPUs.
- **Without a GPU:** the CPU test suite (`make test`) runs anywhere Python 3.12 and
  [uv](https://docs.astral.sh/uv/) do, and CI runs it on every push.

[`docs/DESIGN.md`](docs/DESIGN.md) §1 lists the platform assumptions and how the numbers change on
other hardware. Nothing here needs a cloud account, a cluster, or a machine that stays powered on.

## Getting started

```bash
uv sync
make doctor          # environment checks; must pass before any training command
make test            # CPU unit tests, under 60s
uv run slm run shakespeare_char/ref
uv run slm studio start   # "See it in action": everything you built, in your browser
```

## Documentation

| | |
|---|---|
| [`docs/concepts/two-models.md`](docs/concepts/two-models.md) | **No music background?** What the two models do, and how to read their output |
| `slm studio` | **See it in action.** A local web app over your runs, artifacts, models and machine ([runbook](docs/runbooks/studio.md)) |
| [`docs/LIFECYCLE.md`](docs/LIFECYCLE.md) | **Start here.** The model development lifecycle, with diagrams of the pipeline, architecture and roadmap |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Every abbreviation and term, in plain language |
| [`docs/MODEL.md`](docs/MODEL.md) | The model: type, architecture features, hyperparameters, size presets, file formats |
| [`docs/STACK.md`](docs/STACK.md) | Every tool and library: what it is, why it's used, what was rejected |
| [`docs/concepts/`](docs/concepts/) | Explainers written as each stage is built |
| [`docs/runbooks/`](docs/runbooks/) | Per-milestone: verify it yourself, expected output, and why |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture, contracts, and the rules the code enforces |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones and their exit criteria |
| [`docs/decisions/`](docs/decisions/) | ADRs — what was decided, and what was rejected |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | The rules every change follows: hard rules, conventions, commits |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed in each release |

## Status

M0 (environment) and M1 (engine, validated by reproducing a published character-level
Shakespeare result: best validation loss 1.29) are complete. So is M2, the first full-lifecycle
project: folk tunes in ABC notation, trained from scratch, fine-tuned to take requests in words,
graded, compared across 3 training seeds, exported in Hugging Face format and served over HTTP.
M3 (chess) is next. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Apache-2.0 includes an express patent grant from contributors to users, and terminates that
grant for anyone who brings a patent claim over the software. Contributions are accepted under
the same terms.
