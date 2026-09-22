# ADR 0002 — `projects/` is the unit of a use case

**Status:** accepted (2026-09-22)

## Context

An earlier draft split each use case across `tasks/<name>/` (code) and `configs/runs/<name>/`
(experiment YAML). Adding a use case meant touching two trees, and copying one meant copying
from two places. The mental model for this repo is Terraform: reusable modules plus a root
module per thing you actually build.

## Decision

`projects/<name>/` holds everything specific to one LLM — `project.py`, `graders.py`,
`experiments/*.yaml`, `tests/`, `README.md`. The class is `Project`; registration is
`@register_project("<name>")`; the engine's extension point is `src/slmkit/project_api.py`.

Mapping:

| Terraform | slmkit |
|---|---|
| provider / engine | `src/slmkit/` |
| reusable module | `presets/`, `src/slmkit/{model,tokenizers,graders}` |
| root module | `projects/<name>/` |
| tfvars / workspace | `projects/<name>/experiments/*.yaml` |
| state | `$SLM_HOME` artifacts |

A third tier, `src/slmkit/graders/`, holds generic graders (n-gram novelty, log-loss,
calibration, parse-rate wrapper) so that cricket does not reimplement what chess already has.
Domain graders stay in their project.

## Consequences

- One directory is the whole LLM: copy `projects/_template/` to start a new one.
- "Project" is overloaded — it collides with `[project]` in `pyproject.toml` and with "repo" in
  speech. Discipline: the repo is always "slmkit", never "the project".
- Experiments are addressed by name (`abc_music/micro_v1`), not by path.

## Alternatives considered

Keeping `tasks/` + `configs/runs/` — rejected: two places to touch, weaker copy-to-start story.
