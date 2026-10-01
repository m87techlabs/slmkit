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
from slmkit.model.stats import (
    count_parameters,
    flops_per_token,
    flops_per_token_from_args,
    parameters_from_args,
)
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


# ------------------------------------------------------------------------------ phase 2

GLOSSARY = textwrap.dedent("""\
    # Glossary

    ## The big picture

    **LLM — Large Language Model.** A network that predicts the next token.

    **Top-k / top-p (nucleus).** Restrict sampling to likely tokens.

    **Base model vs fine-tuned model.** A comparison, not a term to match.
    """)


@pytest.fixture
def docs_repo(studio: TestClient, toy_repo: Path) -> TestClient:
    concepts = toy_repo / "docs" / "concepts"
    concepts.mkdir()
    (concepts / "README.md").write_text("| [`b.md`](b.md) | M1 |\n| [`a.md`](a.md) | M2 |\n")
    (concepts / "a.md").write_text("# Concept A\n\nAn LLM reads tokens.\n")
    (concepts / "b.md").write_text("# Concept B\n\n## Sampling\n\nuse top-k here\n")
    (toy_repo / "docs" / "GLOSSARY.md").write_text(GLOSSARY)
    (toy_repo / "docs" / "LIFECYCLE.md").write_text("# The lifecycle\n")
    return studio


def test_doc_tree_follows_reading_order(docs_repo: TestClient) -> None:
    tree = {
        s["title"]: [d["path"] for d in s["docs"]]
        for s in docs_repo.get("/api/docs?project=toy").json()
    }
    assert tree["Start here"] == ["docs/LIFECYCLE.md"]
    assert tree["Concepts"] == ["docs/concepts/b.md", "docs/concepts/a.md"]  # the index's order
    assert (
        "docs/runbooks/m2-toy.md" in tree["Runbooks"]
        and "docs/runbooks/m3-other.md" not in tree["Runbooks"]
    )


def test_glossary_entries_and_their_names(docs_repo: TestClient) -> None:
    entries = {e["term"]: e for e in docs_repo.get("/api/glossary").json()}
    assert entries["LLM — Large Language Model"]["names"] == ["LLM", "Large Language Model"]
    assert entries["Top-k / top-p (nucleus)"]["names"] == ["Top-k", "top-p", "nucleus"]
    assert entries["Base model vs fine-tuned model"]["names"] == []  # comparisons aren't matched
    assert entries["LLM — Large Language Model"]["section"] == "The big picture"


def test_search_finds_lines_with_their_heading(docs_repo: TestClient) -> None:
    hits = docs_repo.get("/api/search?q=TOP-K&project=toy").json()  # case-insensitive
    assert {"path": "docs/concepts/b.md", "heading": "Sampling"} in [
        {"path": h["path"], "heading": h["heading"]} for h in hits
    ]
    assert any(h["path"] == "docs/GLOSSARY.md" for h in hits)


def test_source_and_images_stay_inside_the_readable_repo(
    docs_repo: TestClient, toy_repo: Path
) -> None:
    (toy_repo / ".git").mkdir(exist_ok=True)
    (toy_repo / ".git" / "config").write_text("secret")
    (toy_repo / "docs" / "images").mkdir()
    (toy_repo / "docs" / "images" / "x.png").write_bytes(b"\x89PNG")
    assert docs_repo.get("/api/source?path=projects/toy/project.py").status_code == 200
    for bad in (".git/config", "../outside.txt", "docs/images/x.png", "/etc/passwd"):
        assert docs_repo.get(f"/api/source?path={bad}").status_code == 404, bad
    assert docs_repo.get("/api/image?path=docs/images/x.png").headers["content-type"] == "image/png"
    assert docs_repo.get("/api/image?path=docs/LIFECYCLE.md").status_code == 404


def test_experiments_group_runs_over_seeds(studio: TestClient, toy_run: str) -> None:
    (g,) = [
        g for g in studio.get("/api/experiments?project=toy").json() if g["stage"] == "pretrain"
    ]
    assert g["experiment"] == "toy/base" and [r["run_id"] for r in g["runs"]] == [toy_run]
    assert "best val loss" in g["stats"]


def test_estimate_uses_the_trainers_formulas(studio: TestClient) -> None:
    shape = {"n_layers": 4, "d_model": 128, "n_heads": 4, "n_kv_heads": 4, "ffn_hidden": 384,
             "vocab_size": 87, "block_size": 512, "tie_embeddings": True, "tokens": 30_000_000}  # fmt: skip
    e = studio.post("/api/estimate", json=shape).json()
    args = ModelArgs(**{k: v for k, v in shape.items() if k != "tokens"})
    assert e["params"]["total"] == parameters_from_args(args) == 864_256
    assert e["flops_per_token"] == flops_per_token(CausalLM(args))
    assert e["training_flops"] == e["flops_per_token"] * 30_000_000
    bad = studio.post("/api/estimate", json={**shape, "n_heads": 3})
    assert bad.status_code == 422 and "n_heads" in bad.json()["detail"]
    names = [p["name"] for p in studio.get("/api/presets").json()]
    assert {"nano", "micro", "ref"} <= set(names)


@pytest.mark.parametrize("tie", [True, False])
def test_flops_from_the_architecture_match_the_model(tie: bool) -> None:
    args = ModelArgs(vocab_size=87, block_size=64, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                     ffn_hidden=176, tie_embeddings=tie)  # fmt: skip
    model = CausalLM(args)
    assert flops_per_token(model) == flops_per_token_from_args(args)
    # The output head is one matmul whether or not it shares the embedding's weights.
    matmul = count_parameters(model).total - args.vocab_size * args.d_model * (1 if tie else 2)
    assert (
        flops_per_token(model) == 6 * (matmul + args.vocab_size * args.d_model) + 12 * 2 * 64 * 64
    )


# ------------------------------------------------------------------------------ phase 3

VERIFY_RUNBOOK = textwrap.dedent("""\
    # Runbook: verify
    <!-- slm-studio: projects=toy -->

    ## Look

    ```bash
    ls $SLM_HOME   # where everything lives
    ```
    ```
    runs
    ```

    ## Train

    ```bash
    uv run slm pretrain toy/base
    ```

    ## Script

    ```bash
    uv run python - <<'PY'
    print(1)
    PY
    ```

    ## Pipe

    ```bash
    uv run slm runs list | head
    ```
    """)


@pytest.mark.parametrize(
    ("command", "allowed"),
    [("uv run slm runs summary --project x", True), ("uv run slm lineage pk-1", True),
     ("uv run slm pretrain x/y", False), ("uv run slm eval run-1", False),
     ("uv run slm studio stop", False), ("make test", True), ("make doctor", False),
     ("uv run ruff check .", True), ("uv run ruff check --fix .", False), ("uv run ruff format .", False),
     ("uv run python projects/toy/check_x.py keys", True), ("uv run python evil.py", False),
     ("git log --oneline -3", True), ("git diff --output=/tmp/x", False), ("git push", False),
     ("ls $SLM_HOME", True), ("cat /etc/passwd", False), ("ls $SLM_HOME/../..", False),
     ("rm -rf $SLM_HOME", False), ("uv run slm runs list | head", False),
     ("echo $(whoami)", False), ("A=1 uv run slm runs list", False)],
)  # fmt: skip
def test_only_read_only_commands_may_run(
    command: str, allowed: bool, slm_home: Path, toy_repo: Path
) -> None:
    from slmkit.studio import verify

    slm_home.mkdir(parents=True, exist_ok=True)
    assert verify.classify(command, slm_home, toy_repo)[0] is allowed


@pytest.fixture
def verify_book(studio: TestClient, toy_repo: Path) -> TestClient:
    (toy_repo / "docs" / "runbooks" / "m9-verify.md").write_text(VERIFY_RUNBOOK)
    return studio


def _checks(client: TestClient) -> dict[str, dict]:  # type: ignore[type-arg]
    books = client.get("/api/verify?project=toy").json()["runbooks"]
    (book,) = [b for b in books if b["file"].endswith("m9-verify.md")]
    return {c["heading"]: c for c in book["checks"]}


def test_runbook_checks_are_parsed_with_expected_output(verify_book: TestClient) -> None:
    checks = _checks(verify_book)
    assert checks["Look"]["commands"] == ["ls $SLM_HOME"] and checks["Look"]["expected"] == "runs"
    assert checks["Look"]["runnable"]
    assert not checks["Train"]["runnable"] and "terminal" in checks["Train"]["reason"]
    assert not checks["Script"]["runnable"] and "heredoc" in checks["Script"]["reason"]
    assert "    print(1)" not in checks["Script"]["text"] and "print(1)" in checks["Script"]["text"]
    assert not checks["Pipe"]["runnable"]


def test_a_check_runs_only_when_asked_by_the_studio_itself(
    verify_book: TestClient, slm_home: Path
) -> None:
    import time

    look = _checks(verify_book)["Look"]
    body = {"runbook": "docs/runbooks/m9-verify.md", "check_id": look["id"]}
    assert verify_book.post("/api/verify/run", json=body).status_code == 403  # no header
    evil = verify_book.post("/api/verify/run", json=body,
                            headers={"x-slm-studio": "1", "origin": "https://evil.example"})  # fmt: skip
    assert evil.status_code == 403
    ok = verify_book.post("/api/verify/run", json=body, headers={"x-slm-studio": "1"})
    assert ok.status_code == 200
    for _ in range(100):
        p = verify_book.get(f"/api/verify/runs/{ok.json()['run_id']}").json()
        if p["done"]:
            break
        time.sleep(0.05)
    assert p["exit_code"] == 0 and "runs" in p["output"]
    saved = json.loads((slm_home / "studio" / "verify.json").read_text())
    assert saved[look["id"]]["exit_code"] == 0
    train = _checks(verify_book)["Train"]
    refused = verify_book.post("/api/verify/run", headers={"x-slm-studio": "1"},
                               json={"runbook": "docs/runbooks/m9-verify.md", "check_id": train["id"]})  # fmt: skip
    assert refused.status_code == 403
    unknown = verify_book.post("/api/verify/run", headers={"x-slm-studio": "1"},
                               json={"runbook": "docs/runbooks/m9-verify.md", "check_id": "0" * 12})  # fmt: skip
    assert unknown.status_code == 404
