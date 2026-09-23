"""Project registration and discovery. The engine never imports a project by name.

Projects live outside the installed package, in `projects/<name>/`. They are loaded on demand
from that directory under the synthetic package `slmkit_projects`, so a project can use
relative imports (`from .graders import ...`) and the engine has no import-time dependency on
any of them (CONTRIBUTING.md rule 2).
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path

from slmkit.config.load import projects_dir
from slmkit.project_api import Project

_PACKAGE = "slmkit_projects"
_REGISTRY: dict[str, type[Project]] = {}


class ProjectNotFound(LookupError):
    pass


def register_project(name: str) -> Callable[[type[Project]], type[Project]]:
    def decorator(cls: type[Project]) -> type[Project]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(f"project {name!r} registered twice: {existing} and {cls}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_package(root: Path) -> None:
    pkg = sys.modules.get(_PACKAGE)
    if pkg is None:
        pkg = types.ModuleType(_PACKAGE)
        pkg.__path__ = []
        sys.modules[_PACKAGE] = pkg
    if str(root) not in pkg.__path__:
        pkg.__path__.append(str(root))


def project_class(name: str, root: Path | None = None) -> type[Project]:
    """Return the registered class for `name`, importing `projects/<name>/project.py` if needed."""
    if name not in _REGISTRY:
        root = root or projects_dir()
        if not (root / name / "project.py").is_file():
            available = sorted(
                p.name for p in root.iterdir() if (p / "project.py").is_file() and p.name[0] != "_"
            )
            raise ProjectNotFound(
                f"no project {name!r} in {root}; available: {', '.join(available) or 'none'}"
            )
        _ensure_package(root)
        importlib.import_module(f"{_PACKAGE}.{name}.project")
    if name not in _REGISTRY:
        raise ProjectNotFound(
            f"projects/{name}/project.py was imported but never called @register_project({name!r})"
        )
    return _REGISTRY[name]


def load_project(
    name: str, args: dict[str, object], home: Path, root: Path | None = None
) -> Project:
    """Instantiate a project with its YAML args validated against the project's `Args`."""
    cls = project_class(name, root)
    return cls(cls.Args.model_validate(args), home)
