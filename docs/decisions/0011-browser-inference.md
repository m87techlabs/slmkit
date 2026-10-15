# ADR 0011 — `slm site`: the model runs in the browser, served as static files

**Status:** accepted (after v0.1.0). Replaces the demo part of ADR 0010 (decision 5): the online demo is
a static site, not a Docker Space running `slm serve`.

## Context

ADR 0010 planned the online demo as a Hugging Face Docker Space running `slm serve`. Creating it failed
with `402 Payment Required`. Hugging Face charges for Docker and Gradio Spaces, even on the free CPU tier:
organisations need a Team plan, personal accounts need PRO. Static Spaces, which serve files only, stay
free for everyone.

A static page can still serve a model if the browser does the computing. slmkit's models are small: the
demo model, abc-folk, has 864,256 parameters (3.5 MB of weights) and needs under a million multiply-adds
per token.

## Decision

1. **`src/slmkit/serve/static/model.js`: the network, written out in JavaScript.** It reads an export's
   own files (`config.json`, `tokenizer.json`, `model.safetensors`) and computes what `model/llama.py`
   computes: RMSNorm, rotate-half RoPE with float32 angle tables, grouped-query attention, SwiGLU, a tied
   head. It has a key/value cache, and when the context passes `block_size` it re-reads the last block,
   as `generate.py` crops. Sampling (temperature, top-k, top-p) mirrors `filter_logits`. No library, no
   build step.
2. **The playground page gains one branch.** When its `info` says `"engine": "browser"`, it generates in
   a Web Worker (`engine-worker.js`) and receives the same JSON as `POST /generate`. Everything else on
   the page, and every project viewer (ADR 0008), works unchanged.
3. **`slm site <name:version>`** writes `$SLM_HOME/sites/site-<id>/`: the page, `model.js`, the worker,
   a static `info`, the model's files and its viewer. It is an artifact, content-addressed by model and
   code version, immutable, with lineage back to the raw data. Character-level models only for now; BPE
   models are refused.
4. **Parity is a test, as for export.** `tests/unit/test_site.py` runs `model.js` in headless Chrome
   against PyTorch: logits at every position within 1e-4 (multi-head and grouped-query), and greedy
   generation token for token past the context length. The test is skipped when no Chrome or Chromium is
   found; CI's Ubuntu runners have Chrome.
5. **The demo is a static Space under the organisation**, holding the site that `slm site abc-folk:3`
   built.

## Consequences

- The demo costs nothing and has nothing to run, sleep, scale or patch. Each visitor's browser does the
  work: ~1,500 tokens/s measured in Chrome, faster than `slm serve` on a CPU (~450 tokens/s), where
  PyTorch's per-operation overhead dominates at this size.
- **slmkit now has two implementations of its model.** `llama.py` stays the specification (MODEL.md),
  and `model.js` has to follow any change to it. The parity test fails if it doesn't: a change to
  RoPE, norms or attention in one and not the other moves logits far beyond 1e-4.
- With sampling, a seed reproduces a result within the browser, not across engines: JavaScript can't
  replay PyTorch's random number generator. Greedy output is identical everywhere.
- The site is the same files whatever hosts it: a static Space, GitHub Pages, or `python3 -m http.server`.

## Alternatives considered

- **Pay for a Docker Space** (PRO or Team): rejected. It's a recurring cost for a demo whose model fits
  in a browser, and it adds a server to keep running.
- **transformers.js with an ONNX export:** rejected. It adds a second model format, a conversion step and
  a large library, all standing in for the part slmkit writes out on purpose (CONTRIBUTING.md rule 6 in
  spirit).
- **WebGPU:** rejected for now. It's needless at under 10M parameters, and not every browser has it.
- **No key/value cache, like `generate.py`:** rejected. A 300-token tune would cost ~45,000 positions
  instead of 300, which is seconds rather than milliseconds in JavaScript.
- **An embedded JavaScript engine (V8 for Python) for the parity test:** rejected. It's a new
  dependency, and it would test an engine visitors don't use. Headless Chrome is the real target and is
  already how slmkit checks its pages (STACK.md).
- **Downloading the weights from the model repository at page load,** instead of copying them into the
  site: rejected. A site that carries its files works on any host and offline, and its parity was tested
  on exactly those bytes.
