"""Third-party browser libraries for the studio, fetched once into $SLM_HOME/studio/vendor/.

Nothing third-party is committed to the repo (ADR 0009). Each file is pinned to one version and
one SHA-384 hash; a download that doesn't match is refused, the same guarantee as the
`integrity=` attribute browsers check (subresource integrity). After the first start the studio
works offline.
"""

from __future__ import annotations

import base64
import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VendorFile:
    name: str  # served as /vendor/<name>
    url: str
    sha384: str  # base64, as in an integrity= attribute
    library: str
    licence: str


FILES = (
    VendorFile(
        "uPlot.iife.min.js",
        "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js",
        "Gx3t0zdBAuQOuvvmaLZj7HKEiSgWTAs+VdtNY7wt19QDPTDQjFIwAuXDj0zeN00c",
        "uPlot 1.6.32",
        "MIT",
    ),
    VendorFile(
        "uPlot.min.css",
        "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css",
        "IfV0B7MIOYuO95kO9G5ySKPz/85zqFNOAs8iy4tkK5zd9izhJAB8b7lHrwYqqmYE",
        "uPlot 1.6.32",
        "MIT",
    ),
)

Fetch = Callable[[str], bytes]


class IntegrityError(RuntimeError):
    """A downloaded file doesn't match its pinned hash."""


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        data: bytes = response.read()
        return data


def digest(data: bytes) -> str:
    return base64.b64encode(hashlib.sha384(data).digest()).decode()


def ensure(directory: Path, fetch: Fetch = _fetch, log: Callable[[str], None] = print) -> list[str]:
    """Download every missing or mismatched file. Returns the names that are unavailable
    (offline, say); the studio still starts, and pages that need them say so."""
    directory.mkdir(parents=True, exist_ok=True)
    missing = []
    for f in FILES:
        path = directory / f.name
        if path.is_file() and digest(path.read_bytes()) == f.sha384:
            continue
        try:
            data = fetch(f.url)
        except OSError as exc:
            log(f"  vendor: could not download {f.name} ({exc}); charts will be unavailable")
            missing.append(f.name)
            continue
        if digest(data) != f.sha384:
            raise IntegrityError(f"{f.url} does not match its pinned SHA-384; refusing to use it")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        log(f"  vendor: {f.library} ({f.licence}) -> {path}")
    return missing
