"""Content-addressed artifacts: IDs, manifests, atomic commit, and the /mnt guard.

An artifact is an immutable directory under $SLM_HOME produced by one pipeline stage. Its ID
is a hash of everything that determines its contents:

    id = hash(kind + stage code version + config section + upstream artifact IDs)

so the same inputs always give the same ID (the stage can be skipped), and any change gives a
new ID (nothing is ever overwritten). The Terraform analogy is a plan: if nothing that feeds
a resource changed, there is nothing to do.

Why a per-stage code version rather than the git SHA: hashing the git SHA would invalidate
every artifact on every commit, including documentation commits. Instead each stage declares
an integer that is bumped when its *output* would change. The git SHA is still recorded in the
manifest for traceability, just not hashed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from slmkit.config.load import repo_root, stable_hash

MANIFEST = "manifest.json"

# Short prefixes make IDs self-describing in logs: `pk-3f9a1c...` is obviously packed data.
PREFIX = {"raw": "raw", "dataset": "ds", "tokenizer": "tk", "packed": "pk", "run": "run"}
_ID_HEX = 12  # 48 bits: collisions are not a practical concern at this scale


class ArtifactPathError(RuntimeError):
    """An artifact path is somewhere it must not be (a Windows drive under /mnt/)."""


def guard_path(path: Path, *, allow_windows: bool = False) -> Path:
    """Refuse pipeline I/O on Windows drives.

    /mnt/c is reached over the 9P protocol and measured 20-130x slower than ext4 for the
    small-file I/O that datasets, caches and checkpoints produce (docs/runbooks/m0-environment.md
    section 7). The only legitimate writer there is `slm export --to-windows`.
    """
    resolved = path.expanduser().resolve()
    if not allow_windows and (resolved == Path("/mnt") or Path("/mnt") in resolved.parents):
        raise ArtifactPathError(
            f"{resolved} is under /mnt/, a Windows drive reached over 9P and 20-130x slower "
            "than ext4. Keep $SLM_HOME on the Linux filesystem (see docs/DESIGN.md section 4)."
        )
    return resolved


def slm_home() -> Path:
    """$SLM_HOME, guarded. Defaults to ~/slm."""
    home = guard_path(Path(os.environ.get("SLM_HOME", "~/slm")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def artifact_id(kind: str, *, code_version: int, config: Any, inputs: dict[str, str]) -> str:
    payload = {"kind": kind, "code_version": code_version, "config": config, "inputs": inputs}
    return f"{PREFIX[kind]}-{stable_hash(payload)[:_ID_HEX]}"


def _git_state() -> tuple[str | None, bool | None]:
    try:
        root = repo_root()
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_manifest(
    *,
    kind: str,
    artifact: str,
    project: str,
    code_version: int,
    config: Any,
    inputs: dict[str, str],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha, dirty = _git_state()
    return {
        "kind": kind,
        "id": artifact,
        "project": project,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "code_version": code_version,
        "inputs": inputs,
        "config": config,
        "stats": stats or {},
    }


def _fsync_tree(root: Path) -> None:
    """Flush every file, then every directory entry, to disk before the rename."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            fd = os.open(Path(dirpath) / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def commit_dir(final: Path, manifest: dict[str, Any]) -> Iterator[Path]:
    """Build an artifact in a temporary sibling directory, then publish it atomically.

    Usage:
        with commit_dir(final, manifest) as tmp:
            write files into tmp
        # final now exists, complete, with manifest.json

    `manifest` is written on exit, so the block may still fill in fields it can only know
    after writing (ingest sets its ID from the hashes of what it downloaded).

    A power cut at any point leaves either no artifact or the complete one, never a partial
    directory under the final name. Leftover `.tmp` directories are incomplete by definition,
    so they are removed on the next attempt; they were never artifacts.
    """
    final = guard_path(final)
    if final.exists():
        raise FileExistsError(f"artifact already exists, refusing to overwrite: {final}")
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
        _fsync_tree(tmp)
        os.rename(tmp, final)
        fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(fd)  # make the rename itself durable
        finally:
            os.close(fd)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_manifest(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((path / MANIFEST).read_text())
    return data


def find_artifact(artifact: str) -> Path:
    """Locate an artifact directory by ID anywhere under $SLM_HOME."""
    home = slm_home()
    for manifest in home.rglob(MANIFEST):
        if manifest.parent.name == artifact:
            return manifest.parent
    # raw artifacts live at raw/<project>/, not raw/<id>/, so match on content.
    for manifest in (home / "raw").glob(f"*/{MANIFEST}"):
        if json.loads(manifest.read_text()).get("id") == artifact:
            return manifest.parent
    raise FileNotFoundError(f"no artifact {artifact!r} under {home}")


def lineage(artifact: str) -> list[tuple[int, str, dict[str, Any], bool]]:
    """Walk manifests back to raw data.

    Returns (depth, input_name, manifest, seen_before) rows. Lineage is a graph, not a tree
    (the tokenizer and the packed data both read the same dataset), so an artifact reached a
    second time is marked rather than expanded again.
    """
    rows: list[tuple[int, str, dict[str, Any], bool]] = []
    seen: set[str] = set()

    def walk(aid: str, depth: int, via: str) -> None:
        manifest = read_manifest(find_artifact(aid))
        rows.append((depth, via, manifest, aid in seen))
        if aid in seen:
            return
        seen.add(aid)
        for name, upstream in manifest.get("inputs", {}).items():
            walk(upstream, depth + 1, name)

    walk(artifact, 0, "")
    return rows
