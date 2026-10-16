"""`slm export`: HF-format files, exact round trip, immutability, lineage, --to-windows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file
from typer.testing import CliRunner

from slmkit import artifacts
from slmkit.cli import app
from slmkit.config import ConfigError, load_experiment
from slmkit.export.card import front_matter
from slmkit.export.hf import export_run, load_export, parse_ref
from slmkit.export.windows import to_windows
from slmkit.inference import load_run
from slmkit.registry import load_project
from slmkit.tokenizers import CharTokenizer
from slmkit.train.trainer import Trainer

QUIET = {"log": lambda _: None}


def test_export_writes_hugging_face_files(toy_run: str, slm_home: Path) -> None:
    path = export_run(toy_run, "toy-model", 1, **QUIET)
    assert path == slm_home / "models" / "toy-model" / "1"
    for name in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "MODEL_CARD.md", "manifest.json"):  # fmt: skip
        assert (path / name).is_file(), name
    cfg = json.loads((path / "config.json").read_text())
    assert cfg["architectures"] == ["LlamaForCausalLM"] and cfg["eos_token_id"] == 1
    # Tied embeddings: the head is the embedding matrix, stored once.
    assert "lm_head.weight" not in load_file(str(path / "model.safetensors"))
    manifest = artifacts.read_manifest(path)
    assert manifest["kind"] == "model" and manifest["inputs"]["run"] == toy_run
    assert manifest["stats"]["parity"]["slmkit_max_abs_diff"] == 0.0


def test_export_loads_back_with_identical_logits(toy_run: str) -> None:
    export_run(toy_run, "toy-model", 1, **QUIET)
    run = load_run(toy_run, "best", torch.device("cpu"))
    exported = load_export("toy-model:1")
    assert isinstance(exported.tokenizer, CharTokenizer)
    ids = torch.tensor([exported.tokenizer.encode("abc 1 2 x")[:8]])
    with torch.no_grad():
        assert torch.equal(exported.model(ids)[0], run.model(ids)[0])


def test_versions_are_immutable(toy_run: str) -> None:
    first = export_run(toy_run, "toy-model", 1, **QUIET)
    log: list[str] = []
    assert export_run(toy_run, "toy-model", 1, log=log.append) == first  # same checkpoint: no-op
    assert "already exported" in log[0]

    exp = load_experiment("toy/base", ["run.tracker=none", "train.compile=false",
                                       "train.max_tokens=320", "run.seed=7"])  # fmt: skip
    trainer = Trainer(exp, device="cpu", log=lambda _: None)
    trainer.run()
    with pytest.raises(FileExistsError, match="--version 2"):  # another run: refuse
        export_run(trainer.run_id, "toy-model", 1, **QUIET)


def test_model_reference_parsing() -> None:
    assert parse_ref("abc-folk:12") == ("abc-folk", 12)
    for bad in ("abc-folk", "abc-folk:v1", ":1"):
        with pytest.raises(ConfigError):
            parse_ref(bad)


def test_lineage_walks_from_the_model_to_raw_data(toy_run: str) -> None:
    export_run(toy_run, "toy-model", 1, **QUIET)
    kinds = [m["kind"] for _, _, m, _ in artifacts.lineage("toy-model:1")]
    assert kinds[0] == "model" and {"run", "tokenizer", "packed", "raw"} <= set(kinds)


def test_model_card_describes_prompt_and_missing_eval(toy_run: str) -> None:
    card = (export_run(toy_run, "toy-model", 1, **QUIET) / "MODEL_CARD.md").read_text()
    assert "# toy-model:1" in card and "```\nabc\n```" in card  # the toy eval prompt
    assert "Not evaluated" in card


def test_model_card_opens_with_hub_metadata(toy_run: str) -> None:
    card = (export_run(toy_run, "toy-model", 1, **QUIET) / "MODEL_CARD.md").read_text()
    assert card.startswith("---\n")
    meta = yaml.safe_load(card.split("---\n")[1])
    # The toy project states no licence (the default): the card mustn't invent one.
    assert "license" not in meta
    assert meta["library_name"] == "transformers" and meta["pipeline_tag"] == "text-generation"
    assert meta["tags"][:3] == ["slmkit", "llama", "from-scratch"] and "toy" in meta["tags"]
    assert "projects/toy/README.md" in card  # no data statement: point to the README


def test_front_matter_uses_the_projects_licence_and_tags() -> None:
    meta = yaml.safe_load(
        front_matter("abc_music", {"license": "apache-2.0", "tags": ["music", "slmkit"]}).strip(
            "-\n"
        )
    )
    assert meta["license"] == "apache-2.0"
    assert meta["tags"] == ["slmkit", "llama", "from-scratch", "abc-music", "music"]  # no repeats


def test_to_windows_writes_rendered_samples(toy_run: str, tmp_path: Path, slm_home: Path) -> None:
    export_run(toy_run, "toy-model", 1, **QUIET)
    exported = load_export("toy-model:1")
    project = load_project("toy", {}, slm_home)
    dest = to_windows(exported, project, tmp_path / "win", per_prompt=2,
                      device=torch.device("cpu"), **QUIET)  # fmt: skip
    # The toy project has no render_sample override: the default writes plain text.
    assert sorted(p.name for p in dest.iterdir()) == [
        "MODEL_CARD.md", "SAMPLES.md", "p-0.txt", "p-1.txt"
    ]  # fmt: skip
    assert (dest / "p-0.txt").read_text().startswith("abc")


def test_export_cli_and_models_list(toy_run: str) -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["export", toy_run[:12], "--name", "toy-model", "--version", "1"])
    assert out.exit_code == 0, out.output
    assert "exported toy-model:1" in out.output
    listed = runner.invoke(app, ["models", "list"])
    assert listed.exit_code == 0 and "toy-model:1" in listed.output


def test_models_list_filters_by_project(toy_run: str) -> None:
    """Filter manifest ownership, not the export name, without hiding unfiltered models."""
    export_run(toy_run, "first-model", 1, **QUIET)
    other = export_run(toy_run, "second-model", 1, **QUIET)
    manifest_path = other / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["project"] = "other"
    manifest_path.write_text(json.dumps(manifest))
    runner = CliRunner()

    for project, included, excluded in (
        ("toy", "first-model:1", "second-model:1"),
        ("other", "second-model:1", "first-model:1"),
    ):
        result = runner.invoke(app, ["models", "list", "--project", project])
        assert result.exit_code == 0, result.output
        assert included in result.output and excluded not in result.output

    missing = runner.invoke(app, ["models", "list", "--project", "to"])
    assert missing.exit_code == 0
    assert missing.output == "no models found for project to\n"

    unfiltered = runner.invoke(app, ["models", "list"])
    assert unfiltered.exit_code == 0
    assert "first-model:1" in unfiltered.output and "second-model:1" in unfiltered.output


@pytest.mark.parametrize("project", [None, "missing"])
def test_models_list_empty(slm_home: Path, project: str | None) -> None:
    """Distinguish an empty store from a project with no matches."""
    args = ["models", "list"] + ([] if project is None else ["--project", project])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    expected = (
        "no exported models yet under $SLM_HOME/models"
        if project is None
        else f"no models found for project {project}"
    )
    assert result.output == expected + "\n"


def test_models_list_reads_manifest_projects(slm_home: Path) -> None:
    """List stored manifests, including legacy exports without project ownership."""
    for name, owner in (("first", "one"), ("second", "two"), ("legacy", None)):
        path = slm_home / "models" / name / "1"
        path.mkdir(parents=True)
        manifest = {
            "id": f"{name}:1",
            "stats": {"stage": "pretrain", "params": 100, "step": 10},
            "inputs": {"run": "test-run"},
            "created": "2026-01-01T00:00:00+00:00",
        }
        if owner is not None:
            manifest["project"] = owner
        (path / "manifest.json").write_text(json.dumps(manifest))

    runner = CliRunner()
    for owner, included, excluded in (
        ("one", "first:1", "second:1"),
        ("two", "second:1", "first:1"),
    ):
        result = runner.invoke(app, ["models", "list", "--project", owner])
        assert result.exit_code == 0, result.output
        assert included in result.output and excluded not in result.output
        assert "legacy:1" not in result.output

    missing = runner.invoke(app, ["models", "list", "--project", "on"])
    assert missing.exit_code == 0 and missing.output == "no models found for project on\n"
    unfiltered = runner.invoke(app, ["models", "list"])
    assert unfiltered.exit_code == 0
    assert all(ref in unfiltered.output for ref in ("first:1", "second:1", "legacy:1"))


def test_a_project_viewer_is_copied_into_the_export(
    toy_run: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slm_home: Path
) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "viewer.js").write_text("export function render() {}\n")
    project_cls = type(load_project("toy", {}, slm_home))
    monkeypatch.setattr(project_cls, "web_viewer", lambda self: web)
    path = export_run(toy_run, "toy-model", 1, **QUIET)
    assert (path / "ui" / "viewer.js").read_text() == (web / "viewer.js").read_text()
    assert artifacts.read_manifest(path)["stats"]["viewer"] == ["viewer.js"]
