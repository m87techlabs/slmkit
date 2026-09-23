"""The `slm` command-line interface.

Each pipeline stage is a subcommand that reads artifacts and writes one new one (see
docs/DESIGN.md 6.2). Stages build any missing upstream artifacts and skip ones that already
exist, so running `slm pack` on a fresh machine does ingest -> prepare -> tokenize -> pack.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import typer

from slmkit import artifacts, pipeline
from slmkit import doctor as _doctor
from slmkit.config import ConfigError, dump_resolved, load_experiment
from slmkit.registry import ProjectNotFound

app = typer.Typer(
    add_completion=False,
    help="Build small language models from scratch on a single consumer GPU.",
)

EXPERIMENT = typer.Argument(..., help="<project>/<experiment>, e.g. shakespeare_char/ref")
SET = typer.Option(None, "--set", help="Override a config value: --set train.lr=6e-4 (repeatable)")


def _friendly_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Show configuration and path mistakes as one clear line, not a traceback."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except (ConfigError, ProjectNotFound, artifacts.ArtifactPathError) as exc:
            typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None

    return wrapper


@app.callback()
def _main() -> None:
    """Typer collapses a single-command app into the root command; this callback
    keeps subcommands addressable as `slm <command>` as the CLI grows."""


@app.command()
def doctor(
    bench: bool = typer.Option(
        False, "--bench", help="Also measure achieved bf16 TFLOPS (takes a few seconds)."
    ),
) -> None:
    """Check the environment. Must pass before any training command."""
    raise typer.Exit(code=_doctor.main(bench=bench))


@app.command()
@_friendly_errors
def config(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Print the fully resolved config: presets merged, overrides applied."""
    exp = load_experiment(experiment, set_)
    typer.echo(f"# {exp.address}  ({exp.path})")
    typer.echo(f"# config hash: {exp.config_hash()[:12]}")
    typer.echo(dump_resolved(exp.config), nl=False)


def _stage(
    experiment: str, set_: list[str] | None, fn: Callable[..., pipeline.StageResult]
) -> None:
    exp = load_experiment(experiment, set_)
    typer.echo(f"{exp.address}:")
    fn(exp, typer.echo)


@app.command()
@_friendly_errors
def ingest(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Download the project's source data into $SLM_HOME/raw/<project>/."""
    _stage(experiment, set_, pipeline.ensure_raw)


@app.command()
@_friendly_errors
def prepare(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Split documents by group into train/val (augmenting train only)."""
    _stage(experiment, set_, pipeline.ensure_dataset)


@app.command()
@_friendly_errors
def tokenize(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Fit the tokenizer on the train split."""
    _stage(experiment, set_, pipeline.ensure_tokenizer)


@app.command()
@_friendly_errors
def pack(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Encode both splits into flat uint16 files for training."""
    _stage(experiment, set_, pipeline.ensure_packed)


@app.command()
@_friendly_errors
def lineage(
    artifact: str = typer.Argument(..., help="An artifact ID, e.g. pk-3f9a1c2b4d5e"),
) -> None:
    """Trace an artifact back to its raw data through the manifests."""
    try:
        rows = artifacts.lineage(artifact)
    except FileNotFoundError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    for depth, via, m, seen in rows:
        pad = "    " * depth
        label = f"{via}: " if via else ""
        if seen:
            typer.echo(f"{pad}{label}{m['id']}  (shown above)")
            continue
        dirty = " (dirty)" if m.get("git_dirty") else ""
        typer.echo(
            f"{pad}{label}{m['id']}  [{m['kind']}]  "
            f"created {m['created']}  git {m.get('git_sha')}{dirty}"
        )
        for key, value in m.get("stats", {}).items():
            if isinstance(value, dict):
                value = ", ".join(value)  # e.g. raw file names
            typer.echo(f"{pad}    {key} = {value}")
