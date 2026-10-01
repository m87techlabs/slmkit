"""What the studio shows, read from disk: $SLM_HOME's artifacts, runs and models, and the repo's
projects, roadmap and runbooks.

Read-only, and nothing here is typed in by hand: every number on every page traces back to a
manifest, a status file, a metrics log or an eval report, so the studio can't drift from what the
pipeline did. Everything is per project, found by scanning, so a new project (M3's chess)
appears without changing this file.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from slmkit.config.schema import ModelConfig
from slmkit.eval.summary import _val_chars_per_token
from slmkit.model import ModelArgs
from slmkit.model.stats import parameters_from_args
from slmkit.tokenizers import load_tokenizer

# Where each artifact kind's manifests live (DESIGN §4), and its column in the lifecycle view.
LAYOUT = (
    ("raw", "raw/*/manifest.json"),
    ("dataset", "datasets/*/*/manifest.json"),
    ("tokenizer", "tokenizers/*/manifest.json"),
    ("packed", "packed/*/manifest.json"),
    ("run", "runs/*/manifest.json"),
    ("model", "models/*/*/manifest.json"),
)
COLUMNS = ("raw", "dataset", "tokenizer", "packed", "pretrain", "sft", "model")
RUNBOOK_MARK = re.compile(r"<!--\s*slm-studio:\s*projects=([\w,\s]*)-->")


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    path: Path
    manifest: dict[str, Any]

    @property
    def project(self) -> str | None:
        p = self.manifest.get("project")
        return str(p) if p else None


def index(home: Path) -> dict[str, Artifact]:
    """Every artifact under $SLM_HOME, by ID. Explicit globs, not a recursive walk: checkpoint
    directories are large and hold no manifests."""
    out = {}
    for kind, pattern in LAYOUT:
        for path in sorted(home.glob(pattern)):
            try:
                manifest = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            out[manifest["id"]] = Artifact(manifest["id"], kind, path.parent, manifest)
    return out


def _status(run_dir: Path) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads((run_dir / "status.json").read_text())
        return data
    except (OSError, ValueError):
        return {}


def _metrics(run_dir: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x]
    except (OSError, ValueError):
        return []


def _eval_reports(run_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for path in (run_dir / "eval").glob("ev-*.json"):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda r: r.get("created", ""))


@lru_cache(maxsize=64)
def _vocab_size(tokenizer_dir: str) -> int:
    return load_tokenizer(Path(tokenizer_dir)).vocab_size


def _args(run: Artifact, arts: dict[str, Artifact]) -> ModelArgs | None:
    """The run's architecture from its resolved config and its tokenizer, without loading weights."""
    try:
        cfg = yaml.safe_load((run.path / "config.resolved.yaml").read_text())
        tok = arts[run.manifest["inputs"]["tokenizer"]]
        return ModelArgs.from_config(ModelConfig.model_validate(cfg["model"]),
                                     _vocab_size(str(tok.path)), cfg["data"]["block_size"])  # fmt: skip
    except (OSError, KeyError, ValueError):
        return None


# ------------------------------------------------------------------------------ projects, docs


def projects(repo: Path, home: Path) -> list[dict[str, Any]]:
    """Every project with code in the repo or artifacts on disk. `_template` is not a project."""
    arts = index(home)
    names = {p.name for p in (repo / "projects").glob("*/project.py") for p in [p.parent]}
    names = {n for n in names if not n.startswith("_")} | {
        a.project for a in arts.values() if a.project
    }
    out = []
    for name in sorted(names):
        mine = [a for a in arts.values() if a.project == name]
        readme = repo / "projects" / name / "README.md"
        title = (
            readme.read_text().splitlines()[0].lstrip("# ").strip() if readme.is_file() else name
        )
        out.append({
            "name": name,
            "title": title,
            "runs": sum(a.kind == "run" for a in mine),
            "models": sum(a.kind == "model" for a in mine),
            "has_code": (repo / "projects" / name / "project.py").is_file(),
            "last_activity": max((a.manifest.get("created", "") for a in mine), default=None),
        })  # fmt: skip
    return out


def milestones(repo: Path) -> list[dict[str, str]]:
    """The roadmap's milestone headings and their status marks (☑ done, ◐ in progress, ☐ not yet)."""
    out: list[dict[str, str]] = []
    roadmap = repo / "docs" / "ROADMAP.md"
    if not roadmap.is_file():
        return out
    for line in roadmap.read_text().splitlines():
        m = re.match(r"^## (\S+) — (.+?)\s*([☑◐☐])\s*$", line)
        if m:
            out.append({"id": m.group(1), "title": m.group(2).strip(), "status": m.group(3)})
    return out


def runbooks(repo: Path, project: str | None = None) -> list[dict[str, Any]]:
    """Runbooks, each tagged with the projects it covers. A runbook declares them with a marker
    near its top, `<!-- slm-studio: projects=abc_music -->`; without one it is about the engine and
    applies to every project. Status comes from the runbooks index table."""
    folder = repo / "docs" / "runbooks"
    if not folder.is_dir():
        return []
    status = {}
    index_md = folder / "README.md"
    for line in index_md.read_text().splitlines() if index_md.is_file() else []:
        m = re.match(r"^\| \[`([^`]+)`\]\([^)]*\) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            status[m.group(1)] = {"milestone": m.group(2).strip(), "status": m.group(3).strip()}
    out = []
    for path in sorted(folder.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text()
        mark = RUNBOOK_MARK.search("\n".join(text.splitlines()[:12]))
        covers = [p.strip() for p in mark.group(1).split(",") if p.strip()] if mark else None
        if project and covers is not None and project not in covers:
            continue
        out.append({
            "file": f"docs/runbooks/{path.name}",
            "title": text.splitlines()[0].lstrip("# ").strip(),
            "projects": covers,
            **status.get(path.name, {"milestone": "", "status": ""}),
        })  # fmt: skip
    return out


def read_doc(repo: Path, relative: str) -> str:
    """A Markdown file from docs/ or a project README, and nothing else."""
    path = (repo / relative).resolve()
    docs = (repo / "docs").resolve()
    projects_root = (repo / "projects").resolve()
    allowed = docs in path.parents or (projects_root in path.parents and path.name == "README.md")
    if not allowed or path.suffix != ".md" or not path.is_file():
        raise FileNotFoundError(relative)
    return path.read_text()


# ------------------------------------------------------------------------------ runs and models


def _fill_bpc(run: Artifact, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs trained before the trainer logged bits per character get it computed here, with the
    formula the trainer and `slm runs summary` use. SFT runs have none: their loss covers answers
    only, so it isn't per character of the text."""
    evals = [r for r in records if r["kind"] == "eval"]
    stage = run.manifest.get("stats", {}).get("stage", "pretrain")
    if stage != "pretrain" or not evals or all(r.get("val_bpc") is not None for r in evals):
        return records
    try:
        cpt = _val_chars_per_token(
            run.path, yaml.safe_load((run.path / "config.resolved.yaml").read_text())
        )
    except (OSError, KeyError, FileNotFoundError):
        return records
    if not cpt:
        return records
    for r in evals:
        if r.get("val_bpc") is None:
            r["val_bpc"] = r["val_loss"] / cpt / math.log(2)
    return records


def _best_eval(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    evals = [r for r in records if r["kind"] == "eval"]
    return min(evals, key=lambda r: r["val_loss"]) if evals else None


def _run_row(run: Artifact, arts: dict[str, Artifact]) -> dict[str, Any]:
    st = _status(run.path)
    records = _fill_bpc(run, _metrics(run.path))
    best = _best_eval(records)
    reports = _eval_reports(run.path)
    stage = st.get("stage") or run.manifest.get("stats", {}).get("stage", "pretrain")
    kind = "sft" if stage == "sft" else "headers"
    latest = next((r for r in reversed(reports) if r.get("prompts_kind", "headers") == kind), None)
    cfg = run.manifest.get("config", {})
    args = _args(run, arts)
    return {
        "run_id": run.id,
        "experiment": st.get("experiment") or run.manifest.get("stats", {}).get("experiment"),
        "name": st.get("name", cfg.get("run", {}).get("name")),
        "stage": stage,
        "seed": cfg.get("run", {}).get("seed"),
        "preset": cfg.get("model", {}).get("preset"),
        "tokenizer": cfg.get("tokenizer", {}).get("type"),
        "params": parameters_from_args(args) if args else None,
        "vocab_size": args.vocab_size if args else None,
        "block_size": args.block_size if args else None,
        "complete": bool(st.get("complete")),
        "tokens_seen": st.get("tokens_seen", 0),
        "max_tokens": st.get("max_tokens"),
        "gpu_hours": round(st.get("gpu_seconds", 0) / 3600, 4),
        "tok_per_s": st.get("tok_per_s"),
        "best_val_loss": best["val_loss"] if best else None,
        "best_val_bpc": best.get("val_bpc") if best else None,
        "best_step": best["step"] if best else None,
        "created": run.manifest.get("created"),
        "updated": st.get("updated"),
        "evals": len(reports),
        "eval": (
            {
                "eval_id": latest["eval_id"],
                "prompts_kind": kind,
                "aggregate": {m: s["mean"] for m, s in latest["model"]["aggregate"].items()},
            }
            if latest
            else None
        ),
        "parent": run.manifest.get("inputs", {}).get("parent"),
    }


def runs(home: Path, project: str) -> list[dict[str, Any]]:
    arts = index(home)
    rows = [_run_row(a, arts) for a in arts.values() if a.kind == "run" and a.project == project]
    return sorted(rows, key=lambda r: r["created"] or "")


def run_detail(home: Path, run_id: str) -> dict[str, Any]:
    arts = index(home)
    run = arts.get(run_id)
    if run is None or run.kind != "run":
        raise KeyError(run_id)
    records = _fill_bpc(run, _metrics(run.path))
    return {
        **_run_row(run, arts),
        "config_yaml": (run.path / "config.resolved.yaml").read_text(),
        "inputs": run.manifest.get("inputs", {}),
        "git": {"sha": run.manifest.get("git_sha"), "dirty": run.manifest.get("git_dirty")},
        "train": [
            {k: r.get(k) for k in ("step", "tokens", "loss", "lr", "grad_norm", "tok_per_s")}
            for r in records
            if r["kind"] == "train"
        ],
        "eval": [
            {k: r.get(k) for k in ("step", "tokens", "val_loss", "train_loss", "val_bpc")}
            for r in records
            if r["kind"] == "eval"
        ],
        "samples": [
            {"step": r["step"], "tokens": r.get("tokens"), "samples": r["samples"]}
            for r in records
            if r["kind"] == "eval" and r.get("samples")
        ],
        "reports": [
            {
                "eval_id": r["eval_id"],
                "created": r.get("created"),
                "prompts_kind": r.get("prompts_kind", "headers"),
                "settings": r.get("settings"),
                "model": r["model"]["aggregate"],
                "baseline": r.get("baseline", {}).get("aggregate"),
            }
            for r in _eval_reports(run.path)
        ],
    }


def models(home: Path, project: str) -> list[dict[str, Any]]:
    out = []
    for a in index(home).values():
        if a.kind != "model" or a.project != project:
            continue
        s = a.manifest.get("stats", {})
        out.append({
            "ref": a.id,
            "name": a.manifest["config"]["name"],
            "version": a.manifest["config"]["version"],
            "stage": s.get("stage"),
            "params": s.get("params"),
            "source_run": a.manifest["inputs"].get("run"),
            "step": s.get("step"),
            "created": a.manifest.get("created"),
            "has_viewer": (a.path / "ui" / "viewer.js").is_file(),
            "hf_parity": s.get("parity", {}).get("hf_max_abs_diff"),
            "eval": s.get("eval"),
        })  # fmt: skip
    return sorted(out, key=lambda m: m["created"] or "")


def index_one(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((path / "manifest.json").read_text())
    return manifest


def model_dir(home: Path, name: str, version: int) -> Path:
    return home / "models" / name / str(version)


# ------------------------------------------------------------------------------ lifecycle graph


def _describe(a: Artifact) -> str:
    s = a.manifest.get("stats", {})
    cfg = a.manifest.get("config", {})
    if a.kind == "raw":
        return f"{len(s)} source file{'s' if len(s) != 1 else ''}" if s else "downloaded source"
    if a.kind == "dataset":
        return f"{s.get('train_docs_after_augment', s.get('train_docs', '?')):,} train · {s.get('val_docs', '?'):,} val docs"
    if a.kind == "tokenizer":
        return f"{cfg.get('type', '?')} · {s.get('vocab_size', '?')} tokens"
    if a.kind == "packed":
        return f"{s.get('train_tokens', 0):,} train tokens"
    if a.kind == "model":
        return f"{s.get('stage', '')} · {s.get('params', 0):,} params"
    return ""


def _column(a: Artifact) -> str:
    if a.kind == "run":
        return "sft" if a.manifest.get("stats", {}).get("stage") == "sft" else "pretrain"
    return a.kind


def graph(home: Path, project: str) -> dict[str, Any]:
    """The project's artifacts as a graph: one node per artifact, one edge per manifest input.
    Shared inputs (a tokenizer read by many runs) are single nodes, so lineage is a graph, not a
    tree (DESIGN §6.2)."""
    arts = index(home)
    mine = {k: a for k, a in arts.items() if a.project == project}
    rows = {r["run_id"]: r for r in (_run_row(a, arts) for a in mine.values() if a.kind == "run")}
    nodes = []
    for a in mine.values():
        node = {"id": a.id, "kind": a.kind, "column": _column(a), "created": a.manifest.get("created"),
                "summary": _describe(a), "config": a.manifest.get("config"),
                "stats": a.manifest.get("stats"), "inputs": a.manifest.get("inputs", {}),
                "git": a.manifest.get("git_sha"), "path": str(a.path)}  # fmt: skip
        if a.id in rows:
            r = rows[a.id]
            node["summary"] = (f"{(r['experiment'] or '').split('/')[-1]} · seed {r['seed']} · "
                               + (f"val {r['best_val_loss']:.3f}" if r["best_val_loss"] else "no eval"))  # fmt: skip
            node["run"] = r
        nodes.append(node)
    edges = [{"from": src, "to": a.id, "as": name}
             for a in mine.values() for name, src in a.manifest.get("inputs", {}).items()
             if src in mine]  # fmt: skip
    return {"columns": list(COLUMNS), "nodes": nodes, "edges": edges}


# ------------------------------------------------------------------------------ overview


def _all_runs(home: Path) -> Iterator[tuple[Artifact, dict[str, Any]]]:
    for a in index(home).values():
        if a.kind == "run":
            yield a, _status(a.path)


def overview(repo: Path, home: Path, project: str) -> dict[str, Any]:
    run_rows = runs(home, project)
    done = [r for r in run_rows if r["complete"]]
    pre = [r for r in done if r["stage"] == "pretrain" and r["best_val_bpc"]]
    best = min(pre, key=lambda r: r["best_val_bpc"]) if pre else None
    everything = list(_all_runs(home))
    arts = index(home)
    return {
        "project": project,
        "totals": {
            "runs": len(run_rows),
            "complete": len(done),
            "experiments": len({r["experiment"] for r in run_rows}),
            "tokens": sum(r["tokens_seen"] or 0 for r in run_rows),
            "gpu_hours": round(sum(r["gpu_hours"] for r in run_rows), 3),
            "largest_params": max((r["params"] or 0 for r in run_rows), default=0),
            "models": len(models(home, project)),
            "evals": sum(r["evals"] for r in run_rows),
            "first": min((r["created"] for r in run_rows if r["created"]), default=None),
        },
        "best": best,
        "all_projects": {
            "runs": len(everything),
            "tokens": sum(st.get("tokens_seen", 0) for _, st in everything),
            "gpu_hours": round(sum(st.get("gpu_seconds", 0) for _, st in everything) / 3600, 3),
            "artifacts": len(arts),
        },
        "recent": sorted(run_rows, key=lambda r: r["updated"] or "", reverse=True)[:5],
        "milestones": milestones(repo),
        "runbooks": runbooks(repo, project),
    }


# ------------------------------------------------------------------------------ experiments


def experiments(home: Path, project: str, prompts: str = "auto") -> list[dict[str, Any]]:
    """`slm runs summary`'s groups for one project, with every run's own values and its sampling
    spread (the ± of its eval report), so the page can show both kinds of noise side by side."""
    from slmkit.eval.runner import latest_report
    from slmkit.eval.summary import summarize

    out = []
    for g in summarize(prompts, project):
        runs: list[dict[str, Any]] = []
        for rid, seed, values in zip(g.run_ids, g.seeds, g.per_run, strict=True):
            report = latest_report(home / "runs" / rid, g.eval_kind) if g.eval_kind else None
            spread = (
                {m: s["std"] for m, s in report["model"]["aggregate"].items()} if report else {}
            )
            runs.append({"run_id": rid, "seed": seed, "values": values, "sampling_std": spread})
        out.append({"experiment": g.experiment, "stage": g.stage, "eval_kind": g.eval_kind,
                    "gpu_hours": round(g.gpu_hours, 4), "stats": g.stats,
                    "runs": sorted(runs, key=lambda r: int(r["seed"]))})  # fmt: skip
    return out
