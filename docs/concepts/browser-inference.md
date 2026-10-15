# The model in a browser: inference with no server

*After v0.1.0. Code: [`src/slmkit/serve/static/model.js`](../../src/slmkit/serve/static/model.js) (the
network), [`engine-worker.js`](../../src/slmkit/serve/static/engine-worker.js) (runs it off the page's
thread), [`src/slmkit/serve/site.py`](../../src/slmkit/serve/site.py) (`slm site`). Tests:
[`tests/unit/test_site.py`](../../tests/unit/test_site.py). Decision:
[ADR 0011](../decisions/0011-browser-inference.md). Hands-on:
[`../runbooks/hf-hub.md`](../runbooks/hf-hub.md) §H.5. Terms: [`../GLOSSARY.md`](../GLOSSARY.md).*

`slm serve` runs a model on a server: a browser sends a prompt, Python and PyTorch compute, the text
comes back. For the models slmkit trains, there is a simpler option. The **browser** can do the
computing itself, so the "server" is just a folder of static files. `slm site abc-folk:3` builds that
folder.

The infrastructure analogy: moving work from an API server to the client, the way a single-page app
moved rendering into the browser. It only works when the client can afford the work, and here it can:
abc-folk has 864,256 parameters and its weights are a 3.5 MB file, smaller than many web pages' images.

---

## 1. Why it's possible here, and not for big models

Generating one token means multiplying the current state through every weight matrix once: about **one
multiply-add per parameter**, plus attention over the tokens so far. For abc-folk that is under a million
multiply-adds per token, which plain JavaScript does in well under a millisecond.

| | abc-folk | a 7-billion-parameter chat model |
|---|---|---|
| weights to download | 3.5 MB | ~14 GB (16-bit) |
| work per token | ~0.9 M multiply-adds | ~7,000 M |
| in a browser, plain JavaScript | **~1,500 tokens/s, measured** | impractical |

Big models in browsers exist, but they need the GPU (WebGPU), weights squeezed to 4 bits, and a large
download. slmkit's models need none of that.

**Measured:** in Chrome on the reference machine, one jig (189 tokens) in 0.12 s, about **1,500
tokens/s**. `slm serve` on the CPU makes the same tune at about **450 tokens/s**. The browser is faster
because at this size PyTorch's per-operation overhead, not arithmetic, dominates (serving.md §6).

---

## 2. The same network, written a second time

`model.js` is `src/slmkit/model/llama.py` again, in plain JavaScript with no library. It reads an
export's own files: `config.json` for the sizes, `tokenizer.json` for the characters, and
`model.safetensors` for the weights. That last format is simple enough to read by hand: an 8-byte
length, a JSON header naming each tensor's shape and byte range, then raw float32 numbers
(MODEL.md §6).

Every piece maps to a class in `llama.py`, and each has to match exactly, or the logits differ:

| llama.py | model.js | the detail that has to match |
|---|---|---|
| `RMSNorm` | `rmsnorm()` | the epsilon from `config.json`, the learned gain, no mean subtraction |
| `RotaryEmbedding` | `rope()`, tables in the constructor | pairs `(i, i + Hd/2)` ("rotate_half"), and angles rounded to float32 as PyTorch computes them |
| `Attention` | the head loop in `step()` | query head `h` reads key/value head `h ÷ (H/K)` (grouped-query) and scales by `1/√Hd` |
| `MLP` | the SwiGLU lines in `step()` | `down(silu(gate(x)) × up(x))` |
| tied `lm_head` | `this.head = this.embed` | the output layer *is* the embedding matrix, stored once |
| `filter_logits` | `filterLogits()` | temperature, then top-k, then top-p, in that order |

A linear layer is the one operation everything else is built from. In PyTorch it is `x @ W.T`; here it
is a loop:

```js
function matvec(W, x, out, rows, cols) {   // W is [rows, cols], row-major, as nn.Linear stores it
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += W[r * cols + c] * x[c];
    out[r] = sum;
  }
}
```

The weights stay float32, as stored. The arithmetic runs in JavaScript's float64, which is at least as
precise as PyTorch's float32, so results agree to rounding.

---

## 3. The key/value cache: why each token costs one position

`llama.py`'s sampler re-runs the whole context for every new token. On a GPU that costs nothing worth
saving, and the loop stays easy to read. In JavaScript it would be wasteful. For a 300-token tune, the
network would process 1 + 2 + … + 300 ≈ **45,000 positions** to produce 300 tokens.

The fix is the standard one. In attention, a position's *key* and *value* vectors depend only on that
position and the ones before it. They never change once computed, so `model.js` stores them. Each new
token then runs the network for **one position**, and reads the stored keys and values of all earlier
ones. That is 300 positions instead of 45,000.

The cache is `layers × positions × kv heads × head size` numbers for keys, and again for values. For
abc-folk that is 4 × 512 × 4 × 32 × 2 × 8 bytes ≈ **4 MB**. It is kept in float64, like the rest of the
arithmetic, so it adds no rounding of its own.

**At the context limit.** abc-folk sees at most 512 tokens. When the text passes that, `generate.py`
keeps the last 512 tokens, and they take positions 0 to 511 again. Every cached key was rotated for
its old position, so the cache is no longer valid. `model.js` does the same as `generate.py`: it
re-reads the last 512 tokens from scratch. That costs 512 positions instead of 1 for each token past
the limit, but most tunes end well before it.

---

## 4. Proving it is the same model

A second implementation is only useful if it computes the same function. The test does what
`slm export` does with `transformers` (serving.md §4), in a real browser. `tests/unit/test_site.py`
serves a model to headless Chrome, runs `model.js`, and compares with PyTorch:

- **logits at every position of a full context, within 1e-4**, for plain multi-head and for
  grouped-query attention;
- **greedy generation, token for token**, for 24 tokens on a model with an 8-token context, so the
  re-reading in §3 is exercised.

On the real abc-folk:3, the largest logit difference is **2.4e-6**. Greedy output from the Hub's
`transformers`, from `slm serve` and from the browser is the same text.

**A test that can't fail proves nothing.** A freshly initialised model gives nearly uniform logits: every
token scores about the same, so even a wrong implementation would land within 1e-4. The test scales the
random weights so that the logits span about ±5, like a trained model's. It doesn't scale them much
further. At ×8, logits reach ±29, and PyTorch itself, run in float32 and in float64, disagrees by 3e-4,
more than the tolerance.

---

## 5. Seeds: the same seed, a different tune

With temperature 0 (greedy), the browser and the server write the same text. With sampling, they don't:
`seed 3` in the browser gives a different tune than `seed 3` from `slm serve`.

A seed selects a sequence of random numbers, and the sequence comes from a specific generator. PyTorch's
is a Mersenne Twister with its own way of turning bits into a sampled token (`torch.multinomial`).
`model.js` uses a small generator (mulberry32) and samples by walking the cumulative probabilities.
Reproducing PyTorch's generator bit for bit in JavaScript would be possible, but it would be a
reimplementation of PyTorch internals for no benefit to anyone. So a seed **reproduces a result within
the browser** (the playground's "Copy link" still works), and the distribution the tokens are drawn from
is the same.

---

## 6. The page doesn't change

The playground page (serving.md §8) asks for `info`, then posts to `generate`. A site built by
`slm site` has a static file named `info` with two extra fields: `"engine": "browser"` and
`"files": "model/"`. When the page sees them, it starts `engine-worker.js` (a **Web Worker**: a
background thread for a page, so the page stays responsive while it computes) and sends the request
there instead. The worker returns the same JSON that `POST /generate` would, field for field, so the
page, its statistics and the project's viewer (sheet music and playback for ABC) work unchanged.

```
$SLM_HOME/sites/site-5c053f5e701f/
├── index.html          the playground page, as slm serve sends it
├── model.js            the network (§2, §3)
├── engine-worker.js    runs model.js off the page's thread
├── info                what /info returns, plus "engine": "browser"
├── model/              config.json, generation_config.json, tokenizer.json, model.safetensors
├── ui/viewer.js        the project's viewer, from the export
└── manifest.json       which model, which code
```

A site is an artifact: its ID hashes the model and the code version, it is written once, and
`slm lineage site-5c053f5e701f` walks from it back to the raw data. Any static host serves it: a
Hugging Face static Space (§H.5 of the runbook), GitHub Pages, or `python3 -m http.server`.

---

## 7. What it doesn't do yet

- **BPE models.** `model.js` reads slmkit's character-level tokenizer only. A BPE tokenizer needs its
  merges applied in JavaScript; `slm site` refuses such models with a clear message until then.
- **Constrained decoding.** As with `slm serve`, a project's `logits_processor` is Python and doesn't
  run here (serving.md §10).
- **Models much larger than ~10M parameters.** Nothing stops them, but the download and the work per
  token grow with the parameter count; beyond tens of millions, a server or WebGPU is the better tool.

**Rejected:** transformers.js with an ONNX export. It would add a second model format and a library
standing in for exactly the part slmkit exists to write out. **Rejected:** WebGPU. It's needless at this
size, and not every browser has it. **Rejected:** a Docker Space running `slm serve`, the first plan.
It works, but Hugging Face now charges for Docker Spaces, and a server is more than a 3.5 MB model needs.
