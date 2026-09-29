"""Load a trained run's checkpoint for sampling and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slmkit import artifacts
from slmkit.config.load import ConfigError
from slmkit.config.schema import ModelConfig
from slmkit.model import CausalLM, ModelArgs
from slmkit.tokenizers import Tokenizer, load_tokenizer
from slmkit.train import checkpoint
from slmkit.train.run import load_run_config, runs_root


@dataclass
class LoadedRun:
    run_dir: Path
    config: dict[str, Any]  # the run's config.resolved.yaml
    tokenizer: Tokenizer
    model: CausalLM
    state: dict[str, Any]  # the checkpoint's trainer state (step, val loss, ...)
    checkpoint: str  # "best" or "step_0000900"
    stage: str = "pretrain"  # "sft" for a fine-tuned run

    @property
    def run_id(self) -> str:
        return self.run_dir.name


def resolve_run(prefix: str) -> Path:
    """A run ID or a unique prefix of one (as printed by `slm runs list`)."""
    matches = sorted(p for p in runs_root().glob(f"{prefix}*") if p.is_dir())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise ConfigError(f"run {prefix!r} matches {len(matches)} runs ({found})")
    return matches[0]


def load_run(prefix: str, which: str = "best", device: torch.device | None = None) -> LoadedRun:
    """Load weights from `ckpt/best` (lowest validation loss) or the latest step checkpoint."""
    device = device or torch.device("cpu")
    run_dir = resolve_run(prefix)
    cfg = load_run_config(run_dir)
    manifest = artifacts.read_manifest(run_dir)
    tokenizer = load_tokenizer(artifacts.find_artifact(manifest["inputs"]["tokenizer"]))
    ckpt_dir = run_dir / checkpoint.CKPT_DIR / checkpoint.BEST
    if which == "latest" or not ckpt_dir.is_dir():
        latest = checkpoint.latest(run_dir)
        if latest is None:
            raise ConfigError(f"{run_dir.name} has no checkpoint yet")
        ckpt_dir = latest.path
    args = ModelArgs.from_config(
        ModelConfig.model_validate(cfg["model"]), tokenizer.vocab_size, cfg["data"]["block_size"]
    )
    model = CausalLM(args).to(device).eval()
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True))
    state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    stage = manifest.get("stats", {}).get("stage", "pretrain")
    return LoadedRun(run_dir, cfg, tokenizer, model, state, ckpt_dir.name, stage)
