"""Configuration: pydantic schemas plus preset merging. See docs/DESIGN.md 6.5."""

from slmkit.config.load import (
    ConfigError,
    Experiment,
    dump_resolved,
    load_experiment,
    projects_dir,
    repo_root,
    stable_hash,
)
from slmkit.config.schema import ExperimentConfig

__all__ = [
    "ConfigError",
    "Experiment",
    "ExperimentConfig",
    "dump_resolved",
    "load_experiment",
    "projects_dir",
    "repo_root",
    "stable_hash",
]
