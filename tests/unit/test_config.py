"""Config merge order, overrides and validation (DESIGN 6.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slmkit.config import ConfigError, load_experiment
from slmkit.config.load import merge

REF = "shakespeare_char/ref"


def test_ref_resolves_presets_and_experiment_values() -> None:
    cfg = load_experiment(REF).config
    assert (cfg.model.n_layers, cfg.model.d_model, cfg.model.n_heads) == (6, 384, 6)  # ref preset
    assert cfg.model.dropout == 0.2
    assert cfg.train.lr == 1e-3  # experiment
    assert cfg.train.grad_clip == 1.0  # only in the train preset
    assert cfg.train.betas == (0.9, 0.99)  # experiment beats preset


def test_merge_order_later_wins() -> None:
    exp = {
        "model": {"preset": "nano", "dropout": 0.1, "overrides": {"dropout": 0.3}},
        "train": {"preset": "default", "lr": 5e-4},
    }
    resolved = merge(exp, [("train.lr", 1e-4), ("model.n_layers", 2)])
    assert resolved["model"]["dropout"] == 0.3  # overrides beat direct keys beat preset
    assert resolved["model"]["n_layers"] == 2  # --set beats everything
    assert resolved["train"]["lr"] == 1e-4
    assert resolved["train"]["weight_decay"] == 0.1  # untouched preset value survives


def test_set_preset_swaps_the_whole_preset() -> None:
    cfg = load_experiment(REF, ["model.preset=nano"]).config
    assert (cfg.model.preset, cfg.model.d_model, cfg.model.n_layers) == ("nano", 128, 4)


def test_set_values_are_parsed_as_yaml() -> None:
    cfg = load_experiment(REF, ["train.compile=false", "train.batch_size=32"]).config
    assert cfg.train.compile is False
    assert cfg.train.batch_size == 32


def test_unknown_key_is_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match="lrr"):
        load_experiment(REF, ["train.lrr=1"])


def test_inconsistent_model_shape_is_rejected() -> None:
    with pytest.raises(ConfigError, match="not divisible"):
        load_experiment(REF, ["model.n_heads=5"])


def test_bad_address_and_unknown_preset() -> None:
    with pytest.raises(ConfigError, match="<project>/<experiment>"):
        load_experiment("just-a-name")
    with pytest.raises(ConfigError, match="unknown model preset"):
        load_experiment(REF, ["model.preset=gigantic"])


def test_project_name_must_match_directory(toy_repo: Path) -> None:
    exp = toy_repo / "projects" / "toy" / "experiments" / "wrong.yaml"
    exp.write_text(
        (toy_repo / "projects" / "toy" / "experiments" / "base.yaml")
        .read_text()
        .replace("name: toy,", "name: other,")
    )
    with pytest.raises(ConfigError, match="lives under projects/toy"):
        load_experiment("toy/wrong")


def test_config_hash_is_stable_and_sensitive() -> None:
    a = load_experiment(REF).config_hash()
    assert a == load_experiment(REF).config_hash()
    assert a != load_experiment(REF, ["train.lr=2e-3"]).config_hash()
