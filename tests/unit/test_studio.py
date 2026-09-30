"""slm studio: every API reads what is on disk, the playground reuses `slm serve`, vendored files
are hash-checked, and the process record survives a dead process. CPU only, toy project."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from slmkit.export.hf import export_run
from slmkit.model import CausalLM, ModelArgs
from slmkit.model.stats import count_parameters, parameters_from_args
from slmkit.studio import machine, process, vendor
from slmkit.studio.app import create_app

ROADMAP = "## M0 — Environment ☑\n\n## M1 — Engine ◐\n\ntext\n\n## M2 — Next ☐\n"
RUNBOOK_INDEX = textwrap.dedent("""\
    | Runbook | Milestone | Status |
    |---|---|---|
    | [`m0-engine.md`](m0-engine.md) | M0 | ☑ |
    | [`m2-toy.md`](m2-toy.md) | M2 | ◐ |
    """)


@pytest.fixture
def studio(toy_run: str, toy_repo: Path, slm_home: Path) -> TestClient:
    docs = toy_repo / "docs" / "runbooks"
    docs.mkdir(parents=True)
    (toy_repo / "docs" / "ROADMAP.md").write_text(ROADMAP)
    (docs / "README.md").write_text(RUNBOOK_INDEX)
    (docs / "m0-engine.md").write_text("# Runbook M0: engine\n")
    (docs / "m2-toy.md").write_text("# Runbook M2: toy\n<!-- slm-studio: projects=toy -->\n")
    (docs / "m3-other.md").write_text("# Runbook M3: other\n<!-- slm-studio: projects=other -->\n")
    export_run(toy_run, "toy-model", 1, log=lambda _: None)
    return TestClient(create_app(slm_home, toy_repo))


def test_projects_and_overview_come_from_disk(studio: TestClient, toy_run: str) -> None:
    (toy,) = studio.get("/api/projects").json()
    assert toy["name"] == "toy" and toy["runs"] == 1 and toy["models"] == 1
    o = studio.get("/api/overview?project=toy").json()
    assert (
        o["totals"]["complete"] == 1
        and o["totals"]["tokens"] > 0
        and o["best"]["run_id"] == toy_run
    )
    assert [m["status"] for m in o["milestones"]] == ["☑", "◐", "☐"]


def test_runbooks_are_matched_to_projects_by_their_marker(studio: TestClient) -> None:
    books = {b["file"]: b for b in studio.get("/api/runbooks?project=toy").json()}
    # engine runbooks (no marker) apply everywhere; project runbooks only to their project
    assert set(books) == {"docs/runbooks/m0-engine.md", "docs/runbooks/m2-toy.md"}
    assert books["docs/runbooks/m2-toy.md"]["status"] == "◐"
    assert studio.get("/api/doc?path=docs/runbooks/m2-toy.md").text.startswith("# Runbook M2")
    for outside in ("pyproject.toml", "docs/../conftest.py", "../../etc/passwd"):
        assert studio.get(f"/api/doc?path={outside}").status_code == 404


def test_lifecycle_graph_links_every_stage(studio: TestClient, toy_run: str) -> None:
    g = studio.get("/api/graph?project=toy").json()
    columns = {n["column"] for n in g["nodes"]}
    assert {"raw", "dataset", "tokenizer", "packed", "pretrain", "model"} <= columns
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["from"] in ids and e["to"] in ids for e in g["edges"])
    assert {"from": toy_run, "to": "toy-model:1", "as": "run"} in g["edges"]


def test_run_detail_has_curves_samples_and_config(studio: TestClient, toy_run: str) -> None:
    (row,) = studio.get("/api/runs?project=toy").json()
    assert row["run_id"] == toy_run and row["complete"] and row["params"] > 0
    d = studio.get(f"/api/runs/{toy_run}").json()
    # The toy run is 20 steps, shorter than the train-loss logging interval: evals only.
    assert (
        d["eval"] and d["samples"] and "model:" in d["config_yaml"] and isinstance(d["train"], list)
    )
    assert studio.get("/api/runs/run-nope").status_code == 404


def test_playground_is_slm_serve_under_a_prefix(studio: TestClient) -> None:
    (m,) = studio.get("/api/models?project=toy").json()
    assert m["ref"] == "toy-model:1" and m["viewer"] is None  # the toy project has no viewer
    assert "slmkit playground" in studio.get("/play/toy-model/1/").text
    assert studio.get("/play/toy-model/1/info").json()["model"] == "toy-model:1"
    out = studio.post(
        "/play/toy-model/1/generate", json={"prompt": "abc", "seed": 1, "max_new_tokens": 4}
    )
    assert out.status_code == 200 and out.json()["seed"] == 1
    assert studio.get("/play/toy-model/9/").status_code == 404
    assert studio.get("/play/toy-model/1/ui/../manifest.json").status_code == 404


def test_machine_readings_are_optional_but_well_formed(slm_home: Path) -> None:
    slm_home.mkdir(parents=True, exist_ok=True)
    live = machine.live()
    assert {"gpu_util", "cpu_percent", "ram_used_gib"} <= set(live)
    assert machine.facts(slm_home)["cores"] >= 1


def test_vendor_files_are_hash_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = vendor.VendorFile(
        "lib.js", "https://example.invalid/lib.js", vendor.digest(b"ok"), "lib 1", "MIT"
    )
    monkeypatch.setattr(vendor, "FILES", (good,))
    assert vendor.ensure(tmp_path, fetch=lambda url: b"ok", log=lambda _: None) == []
    assert (tmp_path / "lib.js").read_bytes() == b"ok"
    (tmp_path / "lib.js").unlink()
    with pytest.raises(vendor.IntegrityError):
        vendor.ensure(tmp_path, fetch=lambda url: b"tampered", log=lambda _: None)
    assert not (tmp_path / "lib.js").exists()

    def offline(url: str) -> bytes:
        raise OSError("no network")

    assert vendor.ensure(tmp_path, fetch=offline, log=lambda _: None) == ["lib.js"]


def test_a_dead_process_record_is_cleared(slm_home: Path) -> None:
    assert process.status() is None
    (process.state_dir() / "studio.json").write_text(json.dumps({"pid": 2**22 + 12345, "port": 1}))
    assert process.status() is None
    assert not (process.state_dir() / "studio.json").exists()


@pytest.mark.parametrize("kv", [4, 2])
def test_parameter_count_from_the_architecture_alone(kv: int) -> None:
    args = ModelArgs(vocab_size=87, block_size=64, n_layers=3, d_model=128, n_heads=4,
                     n_kv_heads=kv, ffn_hidden=352, tie_embeddings=kv == 4)  # fmt: skip
    assert parameters_from_args(args) == count_parameters(CausalLM(args)).total
    torch.manual_seed(0)
