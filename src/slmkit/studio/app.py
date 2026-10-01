"""The studio's web app: a JSON API over `data` and `machine`, the pages, and a playground for
every exported model.

    /                               the studio (web/index.html and its ES modules under /static)
    /api/…                          read-only JSON; every endpoint takes `project=` where it applies
    /vendor/…                       third-party browser libraries, from $SLM_HOME/studio/vendor
    /play/<name>/<version>/         `slm serve`'s playground for that model, loaded on first use

The playground is the same page and the same `ModelServer` as `slm serve`, so there is one
implementation of generation. Up to two models stay loaded (they are a few MB each); older ones
are dropped.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slmkit import artifacts
from slmkit.config.load import ConfigError, repo_root
from slmkit.export.hf import load_export
from slmkit.model import ModelArgs
from slmkit.registry import ProjectNotFound, load_project
from slmkit.serve.app import PAGE, GenerateRequest, GenerateResponse, ModelServer
from slmkit.studio import data, learn, machine, sizing, verify

WEB = Path(__file__).parent / "web"
KEEP_MODELS = 2
ROOT_DOCS = ("README.md", "CONTRIBUTING.md")


def fallback_viewer(model_path: Path, manifest: dict[str, Any]) -> Path | None:
    """For an export made before its project had a viewer, the project's current one from the
    repo (what `slm serve --ui` does by hand). None if the export carries its own, or the project
    has none. The studio runs from the repo, so it may ask the project; `slm serve` never does."""
    if (model_path / "ui" / "viewer.js").is_file():
        return None
    project = manifest.get("config", {}).get("project", {})
    try:
        web = load_project(
            project["name"], project.get("args", {}), artifacts.slm_home()
        ).web_viewer()
    except (KeyError, ConfigError, ProjectNotFound):
        return None
    return web if web is not None and (web / "viewer.js").is_file() else None


class RunRequest(BaseModel):
    """Which check to run: named by ID, never by command. The command comes from the runbook."""

    runbook: str
    check_id: str = Field(pattern=r"^[0-9a-f]{12}$")


def same_origin(request: Request) -> None:
    """Refuse a run request from any other web page. Without this, a site you visit could make
    your browser POST here: the custom header forces a CORS preflight, which this server never
    approves, and a mismatched Origin is refused outright."""
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    allowed = {f"http://{host}", f"http://localhost:{host.rpartition(':')[2]}",
               f"http://127.0.0.1:{host.rpartition(':')[2]}"}  # fmt: skip
    if request.headers.get("x-slm-studio") != "1" or (origin is not None and origin not in allowed):
        raise HTTPException(403, "run requests are only accepted from the studio's own page")


class SizeQuery(BaseModel):
    """A model shape for the Parameters page. Bounds keep the arithmetic meaningful, not small."""

    n_layers: int = Field(ge=1, le=256)
    d_model: int = Field(ge=8, le=32768)
    n_heads: int = Field(ge=1, le=512)
    n_kv_heads: int = Field(ge=1, le=512)
    ffn_hidden: int = Field(ge=8, le=262144)
    vocab_size: int = Field(ge=2, le=1_000_000)
    block_size: int = Field(ge=8, le=1_000_000)
    tie_embeddings: bool = True
    tokens: int = Field(ge=1, le=10**15)


class Models:
    """Exported models loaded on demand, the most recently used kept."""

    def __init__(self, home: Path, device: torch.device) -> None:
        self.home, self.device = home, device
        self._servers: OrderedDict[str, ModelServer] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str, version: int) -> ModelServer:
        ref = f"{name}:{version}"
        with self._lock:
            if ref in self._servers:
                self._servers.move_to_end(ref)
                return self._servers[ref]
            if not data.model_dir(self.home, name, version).is_dir():
                raise HTTPException(404, f"no exported model {ref}")
            exported = load_export(ref, self.device)
            server = ModelServer(
                exported, self.device, fallback_viewer(exported.path, exported.manifest)
            )
            self._servers[ref] = server
            while len(self._servers) > KEEP_MODELS:
                self._servers.popitem(last=False)
            return server


def create_app(home: Path | None = None, repo: Path | None = None, device: str = "cpu") -> FastAPI:
    """`device`: where playground models run. The studio defaults to the GPU when there is one
    (`slm studio start`): a 10M-parameter model generates ~15x faster there, and a few seconds of
    generation barely disturbs a training run. `slm serve` stays on the CPU (serving.md §6)."""
    home = home or artifacts.slm_home()
    repo = repo or repo_root()
    vendor = home / "studio" / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    # Generation runs on server worker threads: cap PyTorch's CPU threads (serving.md §6).
    torch.set_num_threads(2)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = Models(home, torch.device(device))
    runner = verify.Runner(home, repo)
    app = FastAPI(title="slm studio", description="See it in action: read-only views of $SLM_HOME.")
    app.mount("/static", StaticFiles(directory=WEB), name="static")
    app.mount("/vendor", StaticFiles(directory=vendor), name="vendor")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB / "index.html").read_text()

    # ---------------------------------------------------------------- API

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "slm_home": str(home), "repo": str(repo)}

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return data.projects(repo, home)

    @app.get("/api/overview")
    def overview(project: str) -> dict[str, Any]:
        return data.overview(repo, home, project)

    @app.get("/api/machine")
    def machine_facts() -> dict[str, Any]:
        return machine.facts(home)

    @app.get("/api/machine/live")
    def machine_live() -> dict[str, Any]:
        return machine.live()

    @app.get("/api/graph")
    def graph(project: str) -> dict[str, Any]:
        return data.graph(home, project)

    @app.get("/api/runs")
    def runs(project: str) -> list[dict[str, Any]]:
        return data.runs(home, project)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        try:
            return data.run_detail(home, run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    @app.get("/api/models")
    def models_(project: str) -> list[dict[str, Any]]:
        out = data.models(home, project)
        for m in out:
            path = data.model_dir(home, m["name"], m["version"])
            fallback = fallback_viewer(path, data.index_one(path))
            m["viewer"] = "export" if m["has_viewer"] else ("project" if fallback else None)
        return out

    @app.get("/api/runbooks")
    def runbooks(project: str | None = None) -> list[dict[str, Any]]:
        return data.runbooks(repo, project)

    @app.get("/api/doc", response_class=PlainTextResponse)
    def doc(path: str) -> str:
        try:
            if path in ROOT_DOCS:  # the two Markdown files at the repo root
                return (repo / path).read_text()
            return data.read_doc(repo, path)
        except FileNotFoundError:
            raise HTTPException(404, f"no document {path}") from None

    @app.get("/api/experiments")
    def experiments(project: str, prompts: str = "auto") -> list[dict[str, Any]]:
        return data.experiments(home, project, prompts)

    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        return sizing.presets(repo)

    @app.post("/api/estimate")
    def estimate(q: SizeQuery) -> dict[str, Any]:
        args = ModelArgs(**q.model_dump(exclude={"tokens"}))
        try:
            return sizing.estimate(home, args, q.tokens)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/docs")
    def docs(project: str | None = None) -> list[dict[str, Any]]:
        return learn.doc_tree(repo, project)

    @app.get("/api/glossary")
    def glossary() -> list[dict[str, Any]]:
        return learn.glossary(repo)

    @app.get("/api/search")
    def search(q: str, project: str | None = None) -> list[dict[str, Any]]:
        return learn.search(repo, q, project)

    @app.get("/api/source", response_class=PlainTextResponse)
    def source(path: str) -> str:
        try:
            return learn.source(repo, path)
        except FileNotFoundError:
            raise HTTPException(404, f"not readable: {path}") from None

    @app.get("/api/image")
    def image(path: str) -> FileResponse:
        try:
            file, media = learn.image(repo, path)
        except FileNotFoundError:
            raise HTTPException(404, f"no image {path}") from None
        return FileResponse(file, media_type=media)

    @app.get("/api/verify")
    def verify_checks(project: str | None = None) -> dict[str, Any]:
        return {"runbooks": verify.checks_for(repo, home, project), "results": runner.results()}

    @app.post("/api/verify/run")
    def verify_run(req: RunRequest, request: Request) -> dict[str, str]:
        same_origin(request)
        if req.runbook not in {b["file"] for b in data.runbooks(repo)}:
            raise HTTPException(404, f"no runbook {req.runbook}")
        check = next(
            (c for c in verify.parse(repo, req.runbook, home) if c.id == req.check_id), None
        )
        if check is None:
            raise HTTPException(404, "no such check (has the runbook changed? reload the page)")
        try:
            ex = runner.start(check)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from None
        except BlockingIOError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"run_id": ex.id}

    @app.get("/api/verify/runs/{run_id}")
    def verify_progress(run_id: str) -> dict[str, Any]:
        ex = runner.get(run_id)
        if ex is None:
            raise HTTPException(404, "unknown run")
        with ex.lock:
            return {"check_id": ex.check_id, "output": ex.output, "done": ex.done,
                    "exit_code": ex.exit_code, "seconds": ex.seconds}  # fmt: skip

    # ---------------------------------------------------------------- playgrounds

    @app.get("/play/{name}/{version}")
    def play_redirect(name: str, version: int) -> RedirectResponse:
        return RedirectResponse(f"/play/{name}/{version}/")  # relative URLs need the slash

    @app.get("/play/{name}/{version}/", response_class=HTMLResponse)
    def play_page(name: str, version: int) -> str:
        if not data.model_dir(home, name, version).is_dir():
            raise HTTPException(404, f"no exported model {name}:{version}")
        return PAGE.read_text()

    @app.get("/play/{name}/{version}/info")
    def play_info(name: str, version: int) -> dict[str, Any]:
        return models.get(name, version).info()

    @app.post("/play/{name}/{version}/generate")
    def play_generate(name: str, version: int, req: GenerateRequest) -> GenerateResponse:
        try:
            return models.get(name, version).generate(req)
        except ConfigError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/play/{name}/{version}/ui/{file_path:path}")
    def play_ui(name: str, version: int, file_path: str) -> FileResponse:
        ui = models.get(name, version).viewer_dir
        if ui is None:
            raise HTTPException(404, "this model has no viewer")
        target = (ui / file_path).resolve()
        if ui.resolve() not in target.parents or not target.is_file():
            raise HTTPException(404, file_path)
        return FileResponse(target)

    return app
