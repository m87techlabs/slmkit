# ADR 0004 — Documentation figures are generated from runs; diagrams are Mermaid

**Status:** accepted (after M1)

## Context

slmkit's docs are a deliverable for readers new to ML, and some ideas (overfitting, the lifecycle,
the architecture) are far easier to see than to read. Pictures in docs have a known failure mode:
they are made once, the code or results change, and the picture quietly stops being true. CONTRIBUTING.md
rule 6 also limits new dependencies.

## Decision

- **Charts** (loss curves, schedules, throughput) are drawn by `scripts/make_figures.py` from real
  runs' `metrics.jsonl`, or from slmkit's own functions (the lr schedule), and committed as PNG/GIF in
  `docs/images/`. `make figures` regenerates them. A figure that is an illustration rather than data
  says so in its title.
- **matplotlib** is the plotting library, as an **optional** extra (`uv sync --extra docs`). It is a
  library, not a framework, and nothing in training imports it.
- **Diagrams** (pipeline, architecture, roadmap) are **Mermaid** blocks inside the Markdown, which
  GitHub renders natively.
- Chart style follows one fixed palette and set of mark specs (series order: train = blue,
  validation = orange), so a colour means the same thing on every page.

## Consequences

- Regenerating figures requires the runs they show to exist in `$SLM_HOME`. They are cheap to
  recreate (the M1 set is under 5 GPU-minutes); the experiment YAMLs are in the repo.
- Committed images add ~1 MB to the repo per milestone. Acceptable; revisit if a milestone's figures
  exceed a few MB.
- VS Code's built-in Markdown preview doesn't render Mermaid; `.vscode/extensions.json` recommends an
  extension that does.

## Alternatives considered

- **Hand-drawn diagrams / screenshots** (draw.io, Excalidraw exports): nicer to look at, but they
  can't be diffed, and go stale silently. Rejected for charts entirely; could be revisited for a
  single hero diagram if one is ever needed.
- **Interactive charts** (Plotly, Vega): GitHub Markdown does not run scripts, so they would need a
  separate site. Rejected.
- **TensorBoard screenshots**: tied to a UI version, and not reproducible from a command. Rejected.
