"""What the studio's Learn page reads: the repo's documentation, its glossary, full-text search,
the images docs embed, and the source files they link to.

Everything is read from the repo as it is, so the studio shows exactly what's committed. Reading
is confined to the documentation and source directories, by resolved path, so a crafted link
can't reach anything else (`.git`, `.venv`, `$SLM_HOME`, the rest of the disk).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from slmkit.studio.data import runbooks

READABLE_DIRS = ("docs", "src", "projects", "scripts", "tests", "presets")
READABLE_ROOT_FILES = ("README.md", "CONTRIBUTING.md", "Makefile", "pyproject.toml", "LICENSE", "NOTICE",
                       "conftest.py")  # fmt: skip
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".js", ".css", ".html", ".json", ".sh",
                 ".txt", ""}  # fmt: skip
IMAGE_SUFFIXES = {".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml",
                  ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}  # fmt: skip


def _inside(repo: Path, relative: str) -> Path:
    """`relative` resolved inside the repo's readable areas, or FileNotFoundError."""
    root = repo.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise FileNotFoundError(relative)
    rel = path.relative_to(root)
    top = rel.parts[0]
    hidden = any(p.startswith(".") or p in {"__pycache__", "node_modules"} for p in rel.parts)
    if hidden or not (top in READABLE_DIRS or (len(rel.parts) == 1 and top in READABLE_ROOT_FILES)):
        raise FileNotFoundError(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def source(repo: Path, relative: str) -> str:
    path = _inside(repo, relative)
    if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > 1_000_000:
        raise FileNotFoundError(relative)
    return path.read_text(errors="replace")


def image(repo: Path, relative: str) -> tuple[Path, str]:
    path = _inside(repo, relative)
    media = IMAGE_SUFFIXES.get(path.suffix.lower())
    if media is None:
        raise FileNotFoundError(relative)
    return path, media


def _title(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _entry(repo: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(repo).as_posix(), "title": _title(path)}


def doc_tree(repo: Path, project: str | None) -> list[dict[str, Any]]:
    """The documentation, grouped the way a reader meets it. Concepts follow the order of their
    index table, which is the order the milestones produced them."""
    docs = repo / "docs"
    sections: list[dict[str, Any]] = []

    def add(title: str, paths: list[Path]) -> None:
        found = [_entry(repo, p) for p in paths if p.is_file()]
        if found:
            sections.append({"title": title, "docs": found})

    add("Start here", [docs / "LIFECYCLE.md", repo / "README.md"])
    index = docs / "concepts" / "README.md"
    order = re.findall(r"\[`([\w-]+\.md)`\]", index.read_text()) if index.is_file() else []
    concepts = [docs / "concepts" / n for n in order]
    concepts += sorted(
        p for p in (docs / "concepts").glob("*.md") if p not in concepts and p.name != "README.md"
    )
    add("Concepts", concepts)
    if project:
        add(f"This project: {project}", [repo / "projects" / project / "README.md"])
    add("Runbooks", [repo / b["file"] for b in runbooks(repo, project)])
    add(
        "Reference",
        [docs / n for n in ("MODEL.md", "GLOSSARY.md", "STACK.md", "DESIGN.md", "ROADMAP.md")],
    )
    add("Decisions (ADRs)", sorted((docs / "decisions").glob("*.md")))
    return sections


_TERM = re.compile(r"^\*\*(.+?)\.\*\*\s*(.*)$")


def _plain(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links → their text
    return re.sub(r"[`*]", "", text).strip()


def glossary(repo: Path) -> list[dict[str, Any]]:
    """GLOSSARY.md's entries: `**Term — Expansion.** definition`, `**A / B.** …`, or
    `**term (long form).** …`. Each gives the names it can be recognised by in other docs."""
    path = repo / "docs" / "GLOSSARY.md"
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    section = ""
    paragraphs = path.read_text().split("\n\n")
    for para in paragraphs:
        first = para.strip().splitlines()[0] if para.strip() else ""
        if first.startswith("## "):
            section = first[3:].strip()
            continue
        m = _TERM.match(para.strip().replace("\n", " "))
        if not m:
            continue
        head, definition = m.group(1).replace("`", ""), _plain(m.group(2))
        names: list[str] = []
        if " vs " not in head:
            main, _, long = head.partition(" — ")
            for part in main.split(" / "):
                base, _, paren = part.partition(" (")
                names += [base.strip(), paren.rstrip(")").strip()] if paren else [base.strip()]
            if long:
                names.append(long.strip())
        entries.append({"term": head, "names": [n for n in names if len(n) >= 2],
                        "definition": definition, "section": section})  # fmt: skip
    return entries


def search(repo: Path, query: str, project: str | None, limit: int = 30) -> list[dict[str, Any]]:
    """Case-insensitive search over every doc in the tree: matching lines, with their heading."""
    q = query.strip().lower()
    if len(q) < 2:
        return []
    hits = []
    for section in doc_tree(repo, project):
        for doc in section["docs"]:
            heading = ""
            for line in (repo / doc["path"]).read_text().splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("# ").strip()
                if q in line.lower():
                    hits.append({**doc, "heading": heading, "line": _plain(line)[:200]})
                    if len(hits) >= limit:
                        return hits
    return hits
