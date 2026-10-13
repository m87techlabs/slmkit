"""`slm site`: a static site whose model runs in the browser, and proof that it is the same model.

The parity tests run model.js in headless Chrome and compare with PyTorch, the way export compares
with `transformers`: logits at every position within 1e-4, and greedy generation token for token,
past the context length. They are skipped when no Chrome or Chromium is found (set SLM_CHROME to
point at one); CI's Ubuntu runners have Chrome.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from slmkit import artifacts
from slmkit.config import ConfigError
from slmkit.export.hf import (
    WEIGHTS,
    export_run,
    export_state_dict,
    generation_config,
    hf_config,
    load_export,
)
from slmkit.model import CausalLM, ModelArgs
from slmkit.sampling import generate
from slmkit.serve.site import STATIC, build_site
from slmkit.tokenizers import EOS_ID, CharTokenizer

QUIET = {"log": lambda _: None}
WSL_BROWSERS = (
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
)


def find_chrome() -> str | None:
    if os.environ.get("SLM_CHROME"):
        return os.environ["SLM_CHROME"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    return next((p for p in WSL_BROWSERS if Path(p).is_file()), None)


CHROME = find_chrome()
needs_chrome = pytest.mark.skipif(CHROME is None, reason="no Chrome/Chromium (set SLM_CHROME)")


# ---------------------------------------------------------------------------- the site


def test_site_holds_the_page_the_engine_and_the_model(toy_run: str, slm_home: Path) -> None:
    export = export_run(toy_run, "toy-model", 1, **QUIET)
    site = build_site("toy-model:1", **QUIET)
    assert site.parent == slm_home / "sites" and site.name.startswith("site-")
    for name in ("index.html", "model.js", "engine-worker.js", "info", "manifest.json"):
        assert (site / name).is_file(), name
    for name in ("config.json", "generation_config.json", "tokenizer.json", WEIGHTS):
        assert (site / "model" / name).read_bytes() == (export / name).read_bytes(), name
    info = json.loads((site / "info").read_text())
    assert info["engine"] == "browser" and info["files"] == "model/"
    assert info["model"] == "toy-model:1" and info["viewer"] is None  # the toy has no viewer
    manifest = artifacts.read_manifest(site)
    assert manifest["kind"] == "site" and manifest["inputs"] == {"model": "toy-model:1"}
    assert [m["kind"] for _, _, m, _ in artifacts.lineage(site.name)][:2] == ["site", "model"]


def test_building_again_is_a_no_op(toy_run: str) -> None:
    export_run(toy_run, "toy-model", 1, **QUIET)
    first = build_site("toy-model:1", **QUIET)
    log: list[str] = []
    assert build_site("toy-model:1", log=log.append) == first
    assert "already built" in log[0]


def test_bpe_models_are_refused_for_now(toy_run: str, slm_home: Path) -> None:
    export = export_run(toy_run, "toy-model", 1, **QUIET)
    fake = slm_home / "models" / "toy-bpe" / "1"
    shutil.copytree(export, fake)
    tok = json.loads((fake / "tokenizer.json").read_text())
    tok["model"]["type"] = "BPE"
    (fake / "tokenizer.json").write_text(json.dumps(tok))
    with pytest.raises(ConfigError, match="character-level"):
        build_site("toy-bpe:1", **QUIET)


# ---------------------------------------------------------------------------- parity in Chrome


PROBE = """<!doctype html><pre id="out">pending</pre>
<script type="module">
import { loadModel, generate } from "./model.js";
const out = document.getElementById("out");
try {
  const m = await loadModel(new URL("model/", location.href).href);
  const ids = %(ids)s;
  m.model.reset();
  const logits = ids.map((id) => Array.from(m.model.step(id)));
  const greedy = generate(m, { prompt: %(prompt)s, temperature: 0, max_new_tokens: %(new)d });
  out.textContent = JSON.stringify({ encoded: m.tokenizer.encode(%(prompt)s), logits, greedy });
} catch (e) { out.textContent = JSON.stringify({ error: String(e.stack || e) }); }
</script>
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def run_in_chrome(directory: Path, page: str) -> dict[str, Any]:
    """Serve `directory` over HTTP (modules can't load from file://) and return the page's
    JSON from <pre id="out"> after the scripts have run."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert CHROME is not None
    args = [CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=60000", "--dump-dom"]
    if not CHROME.endswith(".exe"):
        # Ubuntu 24.04 blocks the user namespaces Chrome's sandbox needs; the page is our own.
        args += ["--no-sandbox", f"--user-data-dir={directory.parent / 'chrome-profile'}"]
    try:
        url = f"http://localhost:{server.server_address[1]}/{page}"
        dom = subprocess.run(
            [*args, url], capture_output=True, text=True, timeout=180, check=True
        ).stdout
    finally:
        server.shutdown()
    match = re.search(r'<pre id="out">(.*?)</pre>', dom, re.DOTALL)
    assert match, dom[:500]
    result: dict[str, Any] = json.loads(html.unescape(match.group(1)))
    assert "error" not in result, result.get("error")
    return result


def write_site(directory: Path, model: CausalLM, tokenizer: CharTokenizer) -> None:
    """The browser-side files of a site, for a model that was never exported."""
    (directory / "model").mkdir(parents=True)
    for name in ("model.js",):
        shutil.copy(STATIC / name, directory / name)
    (directory / "model" / "config.json").write_text(json.dumps(hf_config(model.args)))
    (directory / "model" / "generation_config.json").write_text(json.dumps(generation_config({})))
    save_file(export_state_dict(model), str(directory / "model" / WEIGHTS))
    tokenizer.to_hf().save(str(directory / "model" / "tokenizer.json"))


@needs_chrome
@pytest.mark.parametrize("n_kv_heads", [4, 2])  # plain multi-head, and grouped-query
def test_browser_logits_match_pytorch(tmp_path: Path, n_kv_heads: int) -> None:
    torch.manual_seed(0)
    tokenizer = CharTokenizer("abcdefgh |:\n")
    args = ModelArgs(vocab_size=tokenizer.vocab_size, block_size=32, n_layers=2, d_model=64,
                     n_heads=4, n_kv_heads=n_kv_heads, ffn_hidden=192)  # fmt: skip
    model = CausalLM(args).eval()
    # The default init gives near-uniform logits, which would pass too easily. Scaled x3, logits
    # span about ±5, like a trained model's. (Much larger, and float32 PyTorch disagrees with
    # itself in float64 by more than the tolerance: x8 gives ±29 and 3e-4.)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    write_site(tmp_path / "site", model, tokenizer)
    ids = torch.randint(2, args.vocab_size, (args.block_size,)).tolist()  # a full context
    page = PROBE % {"ids": json.dumps(ids), "prompt": json.dumps("ab"), "new": 1}
    (tmp_path / "site" / "probe.html").write_text(page)

    got = run_in_chrome(tmp_path / "site", "probe.html")
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0][0].double()
    got_logits = torch.tensor(got["logits"], dtype=torch.float64)
    torch.testing.assert_close(got_logits, expected, atol=1e-4, rtol=0)
    assert got["encoded"] == tokenizer.encode("ab")


@needs_chrome
def test_browser_greedy_generation_matches_pytorch_past_the_context(
    toy_run: str, tmp_path: Path
) -> None:
    """The trained toy model, block size 8: 24 new tokens make the browser re-read its window
    the way generate.py crops the context."""
    export_run(toy_run, "toy-model", 1, **QUIET)
    exported = load_export("toy-model:1")
    site = build_site("toy-model:1", **QUIET)
    work = tmp_path / "copy"
    shutil.copytree(site, work)  # sites are read-only artifacts; the probe page goes in a copy
    work.chmod(0o755)
    prompt = "abc 1"
    page = PROBE % {"ids": "[2, 3]", "prompt": json.dumps(prompt), "new": 24}
    (work / "probe.html").write_text(page)

    got = run_in_chrome(work, "probe.html")
    ids = exported.tokenizer.encode(prompt)
    out = generate(exported.model, torch.tensor([ids]), 24, temperature=0.0, stop_token=EOS_ID)
    new = out[0, len(ids) :].tolist()
    finished = EOS_ID in new
    expected = exported.tokenizer.decode(new[: new.index(EOS_ID)] if finished else new)
    assert got["greedy"]["completion"] == expected
    assert got["greedy"]["finished"] == finished
    assert got["greedy"]["new_tokens"] == len(new)
