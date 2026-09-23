"""The engine never imports or depends on a project (CONTRIBUTING.md rule 2).

Checked by parsing the engine's code, so the rule is enforced rather than remembered. Prose in
docstrings and help text may use a project as an example; code may not import one, name one as
an identifier, or special-case one by string (`if project == "chess"`).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "src" / "slmkit"
# Existing projects plus the planned ones, so the boundary holds before they exist.
NAMES = {
    p.name for p in (REPO / "projects").iterdir() if p.is_dir() and not p.name.startswith("_")
} | {"abc_music", "chess", "cricket_nextball"}


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            root = module.split(".")[0]
            if root in {"projects", "slmkit_projects"} or root in NAMES:
                found.append(f"line {node.lineno}: imports {module}")

        if isinstance(node, ast.Name) and node.id in NAMES:
            found.append(f"line {node.lineno}: identifier {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in NAMES:
            found.append(f"line {node.lineno}: attribute .{node.attr}")
        if isinstance(node, ast.Constant) and node.value in NAMES:
            found.append(f"line {node.lineno}: string literal {node.value!r}")
    return found


def test_engine_code_never_depends_on_a_project() -> None:
    offenders = [
        f"{path.relative_to(REPO)} {v}" for path in ENGINE.rglob("*.py") for v in _violations(path)
    ]
    assert not offenders, "engine depends on a project:\n" + "\n".join(offenders)


def test_the_check_catches_a_violation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('import projects.chess\nif name == "abc_music":\n    pass\n')
    assert len(_violations(bad)) == 2
