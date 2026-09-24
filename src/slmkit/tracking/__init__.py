"""Experiment tracking: where loss curves go so you can look at them (DESIGN 6.10).

`tensorboard` writes event files into the run directory. Nothing needs to be running while
training; view them any time with `uv run tensorboard --logdir $SLM_HOME/runs`. `none` writes
nothing extra. Either way the trainer also keeps `metrics.jsonl` in the run directory, which is
what `slm runs list` reads, so no tracker is ever required to know how a run went.

wandb (cloud) is allowed by CONTRIBUTING.md but not implemented yet; it arrives when a milestone needs
sweep comparisons. A self-hosted tracking server was rejected outright: it would need Docker,
and Docker is stopped during training sessions (CONTRIBUTING.md rule 10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def scalar(self, name: str, value: float, step: int) -> None: ...
    def text(self, name: str, value: str, step: int) -> None: ...
    def close(self) -> None: ...


class NoopTracker:
    def scalar(self, name: str, value: float, step: int) -> None:
        pass

    def text(self, name: str, value: str, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardTracker:
    def __init__(self, logdir: Path, purge_step: int | None = None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        # purge_step discards events logged after the checkpoint we resumed from: those steps
        # are about to be re-run, and without purging the curve would show them twice.
        self._writer = SummaryWriter(log_dir=str(logdir), purge_step=purge_step)

    def scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def text(self, name: str, value: str, step: int) -> None:
        # Markdown in TensorBoard's text tab; indenting keeps line breaks in generated text.
        self._writer.add_text(name, "    " + value.replace("\n", "\n    "), step)

    def close(self) -> None:
        self._writer.close()


def make_tracker(kind: str, run_dir: Path, resume_step: int | None) -> Tracker:
    if kind == "tensorboard":
        return TensorBoardTracker(run_dir / "tb", purge_step=resume_step)
    if kind == "none":
        return NoopTracker()
    raise ValueError(f"tracker {kind!r} is not implemented yet; use tensorboard or none")
