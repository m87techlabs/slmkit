# slmkit

**A framework for building small language models from scratch on a single consumer GPU — and a
guided way to learn how language models actually work.**

```
ingest → prepare → tokenizer → pack → pretrain → SFT → eval → export → serve
```

Most tutorials either fine-tune somebody else's model or stop at a toy training loop. slmkit
covers the whole lifecycle: you train your own tokenizer, pretrain a model from random
initialization, instruction-tune it, evaluate it with graders that cannot be fooled, export it
to a standard format, and serve it.

## Who this is for

People who are **comfortable with software and infrastructure but new to machine learning**.
Everything ML-specific is explained; nothing infrastructure-specific is. Every abbreviation is
defined in [`docs/GLOSSARY.md`](docs/GLOSSARY.md) — start there if a term is unfamiliar.

The model, training loop, sampler and SFT masking are **hand-written on purpose**. There is no
`Trainer` class hiding the interesting parts.

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

A single NVIDIA GPU with 12–24 GB of VRAM, on Windows with WSL2. See
[`docs/DESIGN.md`](docs/DESIGN.md) §1 for the full set of platform assumptions and how the
numbers change on different hardware. Nothing here needs a cloud account, a cluster, or a
machine that stays powered on.

## Getting started

```bash
uv sync
make doctor          # environment checks; must pass before any training command
make test            # CPU unit tests, under 60s
uv run slm run shakespeare_char/ref
```

## Documentation

| | |
|---|---|
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Every abbreviation and term, in plain language |
| [`docs/concepts/`](docs/concepts/) | Explainers written as each stage is built |
| [`docs/runbooks/`](docs/runbooks/) | Per-milestone: verify it yourself, expected output, and why |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture, contracts, and the rules the code enforces |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones and their exit criteria |
| [`docs/decisions/`](docs/decisions/) | ADRs — what was decided, and what was rejected |

## Status

Early. The scaffold and design are in place; the engine is being implemented milestone by
milestone. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Apache-2.0 includes an express patent grant from contributors to users, and terminates that
grant for anyone who brings a patent claim over the software. Contributions are accepted under
the same terms.
