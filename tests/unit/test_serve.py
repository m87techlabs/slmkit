"""`slm serve`'s HTTP API, in process (FastAPI's TestClient; no port is opened)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from slmkit.export.hf import export_run, load_export
from slmkit.serve.app import create_app


@pytest.fixture
def client(toy_run: str) -> TestClient:
    export_run(toy_run, "toy-model", 1, log=lambda _: None)
    return TestClient(create_app(load_export("toy-model:1"), torch.device("cpu")))


def test_health_and_info(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "toy-model:1"}
    info = client.get("/info").json()
    assert info["stage"] == "pretrain" and info["example_prompt"] == "abc"
    assert info["defaults"]["eos_token_id"] == 1
    assert info["examples"] == [{"id": "p", "prompt": "abc"}]  # every eval prompt, as Try: buttons


def test_generate_is_reproducible_by_seed(client: TestClient) -> None:
    body = {"prompt": "abc", "seed": 3, "max_new_tokens": 12}
    a, b = client.post("/generate", json=body).json(), client.post("/generate", json=body).json()
    assert a["completion"] == b["completion"]
    assert a["text"] == "abc" + a["completion"] and a["new_tokens"] <= 12
    assert a["seed"] == 3 and a["settings"]["max_new_tokens"] == 12


def test_random_seed_is_returned_so_the_sample_can_be_repeated(client: TestClient) -> None:
    first = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 8}).json()
    again = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 8,
                                           "seed": first["seed"]}).json()  # fmt: skip
    assert again["completion"] == first["completion"]


def test_unknown_prompt_characters_are_reported(client: TestClient) -> None:
    out = client.post("/generate", json={"prompt": "abc é€", "max_new_tokens": 2}).json()
    assert out["unknown_prompt_tokens"] == 2


def test_empty_prompt_starts_a_new_document(client: TestClient) -> None:
    out = client.post("/generate", json={"max_new_tokens": 4, "seed": 0})
    assert out.status_code == 200 and out.json()["prompt_tokens"] == 0


@pytest.mark.parametrize(
    "body",
    [{"prompt": "a", "temprature": 0.5},  # misspelt field
     {"prompt": "a", "max_new_tokens": 0},
     {"prompt": "a", "max_new_tokens": 100_000},
     {"prompt": "a", "top_p": 0.0}],
)  # fmt: skip
def test_bad_requests_are_rejected(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/generate", json=body).status_code == 422


def test_playground_page_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "slmkit playground" in page.text
    assert client.get("/info").json()["viewer"] is None  # the toy project ships no viewer


def test_a_viewer_directory_is_served_under_ui(toy_run: str, tmp_path: Path) -> None:
    export_run(toy_run, "toy-model", 1, log=lambda _: None)
    ui = tmp_path / "web"
    ui.mkdir()
    (ui / "viewer.js").write_text("export function render() {}\n")
    app = create_app(load_export("toy-model:1"), torch.device("cpu"), ui)
    c = TestClient(app)
    assert c.get("/info").json()["viewer"] == "ui/viewer.js"
    js = c.get("/ui/viewer.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
