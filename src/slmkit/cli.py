"""The `slm` command-line interface.

Each pipeline stage is a subcommand that reads artifacts and writes one new
one (see docs/DESIGN.md 6.2). Only `doctor` exists so far; the rest arrive with
their milestones.
"""

from __future__ import annotations

import typer

from slmkit import doctor as _doctor

app = typer.Typer(
    add_completion=False,
    help="Build small language models from scratch on a single consumer GPU.",
)


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
