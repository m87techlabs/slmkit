"""Load an experiment: resolve its address, merge presets, apply `--set`, validate.

Merge order, later wins (DESIGN 6.5):

    model preset -> train preset -> experiment YAML -> --set a.b=c

The Terraform analogy: presets are module defaults, the experiment YAML is a tfvars file, and
`--set` is `-var` on the command line.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slmkit.config.schema import ExperimentConfig

# Keys under `--set` that choose a preset are applied before presets are expanded, so
# `--set model.preset=tiny` swaps the whole preset rather than only relabelling it.
_PRESET_KEYS = ("model.preset", "train.preset")


class ConfigError(ValueError):
    """An experiment could not be found, merged or validated."""


def repo_root() -> Path:
    """The slmkit checkout: where `presets/` and `projects/` live.

    `SLM_REPO` overrides it; otherwise it is found relative to this installed package,
    which `uv sync` installs in editable mode from `src/`.
    """
    if env := os.environ.get("SLM_REPO"):
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def presets_dir() -> Path:
    return repo_root() / "presets"


def projects_dir() -> Path:
    return repo_root() / "projects"


@dataclass(frozen=True)
class Experiment:
    """A resolved experiment plus where it came from."""

    address: str  # "<project>/<experiment>"
    path: Path
    config: ExperimentConfig

    @property
    def project(self) -> str:
        return self.config.project.name

    def config_hash(self) -> str:
        return stable_hash(self.config.model_dump(mode="json"))


def stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object, independent of key order."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def resolve_address(address: str) -> tuple[str, Path]:
    """`abc_music/micro_v1` -> projects/abc_music/experiments/micro_v1.yaml.

    A path to a YAML file is also accepted, for experiments kept outside the repo.
    """
    as_path = Path(address).expanduser()
    if as_path.suffix in (".yaml", ".yml") and as_path.is_file():
        project = as_path.resolve().parent.parent.name
        return f"{project}/{as_path.stem}", as_path.resolve()

    parts = address.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ConfigError(f"experiment address must be <project>/<experiment>, got {address!r}")
    project, name = parts
    path = projects_dir() / project / "experiments" / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"no experiment at {path}")
    return address.strip("/"), path


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _load_preset(kind: str, name: str) -> dict[str, Any]:
    path = presets_dir() / kind / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (presets_dir() / kind).glob("*.yaml"))
        raise ConfigError(f"unknown {kind} preset {name!r}; available: {', '.join(available)}")
    return _read_yaml(path)


def parse_set(items: list[str]) -> list[tuple[str, Any]]:
    """Parse `a.b=c` overrides. Values are YAML, so `true`, `3` and `[1, 2]` keep their type."""
    out = []
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        out.append((key.strip(), yaml.safe_load(raw)))
    return out


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    node = tree
    *parents, leaf = dotted.split(".")
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"--set {dotted}: {part!r} is not a mapping")
        node = child
    node[leaf] = value


def merge(experiment: dict[str, Any], overrides: list[tuple[str, Any]]) -> dict[str, Any]:
    """Apply the merge order to a raw experiment mapping and return the resolved mapping."""
    exp = copy.deepcopy(experiment)
    for key, value in overrides:
        if key in _PRESET_KEYS:
            _set_path(exp, key, value)

    model = dict(exp.get("model") or {})
    model_preset = model.pop("preset", None)
    if model_preset is None:
        raise ConfigError("experiment must set model.preset")
    model_overrides = model.pop("overrides", None) or {}
    resolved_model = {"preset": model_preset, **_load_preset("model", model_preset)}
    resolved_model.update(model)  # direct keys under `model:` also override
    resolved_model.update(model_overrides)

    train = dict(exp.get("train") or {})
    train_preset = train.pop("preset", "default")
    resolved_train = {"preset": train_preset, **_load_preset("train", train_preset)}
    resolved_train.update(train)

    resolved = {**exp, "model": resolved_model, "train": resolved_train}
    for key, value in overrides:
        if key not in _PRESET_KEYS:
            _set_path(resolved, key, value)
    return resolved


def load_experiment(address: str, overrides: list[str] | None = None) -> Experiment:
    """Resolve, merge and validate an experiment. Raises ConfigError with a readable message."""
    addr, path = resolve_address(address)
    raw = _read_yaml(path)
    resolved = merge(raw, parse_set(overrides or []))
    try:
        config = ExperimentConfig.model_validate(resolved)
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigError(f"{path}:\n{exc}") from exc

    expected_project = addr.split("/")[0]
    if config.project.name != expected_project:
        raise ConfigError(
            f"{path} declares project.name={config.project.name!r}, "
            f"but lives under projects/{expected_project}/"
        )
    return Experiment(address=addr, path=path, config=config)


def dump_resolved(config: ExperimentConfig) -> str:
    """The resolved config as YAML, as persisted to `config.resolved.yaml` in every run."""
    return str(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
