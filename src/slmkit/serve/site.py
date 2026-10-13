"""`slm site`: an exported model as a static web site, with the model running in the browser.

    $SLM_HOME/sites/site-<id>/
        index.html          slm serve's playground page, unchanged
        model.js            the model in JavaScript (the same network as model/llama.py)
        engine-worker.js    runs model.js off the page's main thread
        info                what GET /info would return, plus "engine": "browser"
        model/              the export's config, generation settings, tokenizer and weights
        ui/                 the project's viewer, if the export has one (ADR 0008)
        manifest.json       lineage: which model, which code

Any static file host serves it: a Hugging Face static Space, GitHub Pages, or
`python3 -m http.server`. There is no server code at all, so nothing to run, scale or pay for:
each visitor's browser does the work (ADR 0011). That suits slmkit's models, which are small
enough that a browser generates a tune in about a second.

A site is an artifact like any other: its ID is a hash of the model and the code version, it is
written once, atomically, and never changed. Building the same model again is a no-op.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from slmkit import artifacts
from slmkit.config.load import ConfigError
from slmkit.export.hf import WEIGHTS, load_export, model_dir, parse_ref
from slmkit.serve.app import PAGE, model_info

CODE_VERSION = 1  # bump when the files a site contains would change
STATIC = Path(__file__).parent / "static"
ENGINE_FILES = ("model.js", "engine-worker.js")
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", WEIGHTS)


def build_site(ref: str, *, log: Callable[[str], None] = print) -> Path:
    """Write the static site for exported model `ref` (name:version). Returns its directory."""
    path = model_dir(*parse_ref(ref))
    if not path.is_dir():
        raise ConfigError(f"no exported model {ref} (looked in {path}); see `slm models list`")
    tokenizer = json.loads((path / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "WordLevel":
        # model.js implements slmkit's char tokenizer; BPE merges in JavaScript are future work.
        raise ConfigError(
            f"{ref} uses a {tokenizer['model']['type']} tokenizer; only character-level "
            "models run in the browser so far"
        )
    exported = load_export(ref)
    site_id = artifacts.artifact_id(
        "site", code_version=CODE_VERSION, config={"engine": "browser"}, inputs={"model": ref}
    )
    dest = artifacts.slm_home() / "sites" / site_id
    if dest.exists():
        log(f"{ref}: site already built at {dest}")
        return dest

    viewer = exported.path / "ui"
    info = model_info(
        exported,
        device="your browser",
        viewer="ui/viewer.js" if (viewer / "viewer.js").is_file() else None,
    )
    info |= {"engine": "browser", "files": "model/"}
    manifest = artifacts.build_manifest(
        kind="site",
        artifact=site_id,
        project=exported.manifest["project"],
        code_version=CODE_VERSION,
        config={"engine": "browser"},
        inputs={"model": ref},
        stats={"weights_bytes": (exported.path / WEIGHTS).stat().st_size},
    )
    with artifacts.commit_dir(dest, manifest) as tmp:
        shutil.copy(PAGE, tmp / "index.html")
        for name in ENGINE_FILES:
            shutil.copy(STATIC / name, tmp / name)
        (tmp / "info").write_text(json.dumps(info, indent=2) + "\n")
        (tmp / "model").mkdir()
        for name in MODEL_FILES:
            shutil.copy(exported.path / name, tmp / "model" / name)
        if (viewer / "viewer.js").is_file():
            shutil.copytree(viewer, tmp / "ui")
    for path in [dest, *sorted(dest.rglob("*"))]:
        path.chmod(0o755 if path.is_dir() else 0o644)  # world-readable, like any web root
    log(f"built {site_id} for {ref} -> {dest}")
    return dest
