// slmkit's model, run in the browser: the same Llama-style network as src/slmkit/model/llama.py,
// written out in plain JavaScript, reading an export's own files. No library, no build step.
//
// Why it exists: a static web page can then serve a model with no server behind it (`slm site`,
// ADR 0011). For the small models slmkit trains, a CPU in a browser is fast enough: abc-folk
// (0.86M parameters) costs about 2 million multiply-adds per token.
//
// What has to match llama.py exactly, or the logits differ (docs/concepts/browser-inference.md):
//   - RMSNorm with the export's epsilon, a learned gain, no bias;
//   - RoPE in the "rotate_half" convention, pairs (i, i + Hd/2), angles computed in float32;
//   - grouped-query attention: query head h reads key/value head floor(h / (H / K));
//   - SwiGLU: down(silu(gate(x)) * up(x));
//   - the output head tied to the embedding when tie_word_embeddings is true.
//
// One difference from llama.py, on purpose: a key/value cache. llama.py re-runs the whole context
// for every new token (cheap on a GPU, and simpler to read). Here each token's keys and values are
// kept, so a new token costs one pass of one position. Logits are the same either way.
//
// Weights are float32 (Float32Array); arithmetic runs in JavaScript's float64, which is at least as
// precise. Parity with PyTorch is tested to 1e-4 (tests/unit/test_site.py).

export const UNK_ID = 0;
export const EOS_ID = 1;

// ---------------------------------------------------------------------------- files

/** Parse a .safetensors file: an 8-byte little-endian header length, a JSON header, raw tensors. */
export function readSafetensors(buffer) {
  const view = new DataView(buffer);
  const headerLength = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const start = 8 + headerLength;
  const tensors = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (meta.dtype !== "F32") throw new Error(`${name}: only float32 weights are supported, not ${meta.dtype}`);
    const [from, to] = meta.data_offsets;
    // slice() copies, so the Float32Array starts 4-byte aligned whatever the header length was.
    tensors[name] = { shape: meta.shape, data: new Float32Array(buffer.slice(start + from, start + to)) };
  }
  return tensors;
}

/** The character-level tokenizer an export carries (tokenizer.json: WordLevel, one char per token). */
export class CharTokenizer {
  constructor(tokenizerJson) {
    const model = tokenizerJson.model;
    const split = tokenizerJson.pre_tokenizer;
    if (model?.type !== "WordLevel" || split?.type !== "Split") {
      throw new Error("only slmkit's character-level tokenizer runs in the browser so far (not BPE)");
    }
    this.stoi = new Map(Object.entries(model.vocab));
    this.itos = [];
    for (const [token, id] of this.stoi) this.itos[id] = token;
  }
  get vocabSize() { return this.itos.length; }
  /** Like CharTokenizer.encode in Python: one ID per code point, unknown characters -> <unk>. */
  encode(text) {
    const ids = [];
    for (const ch of text) {
      const id = this.stoi.get(ch);
      ids.push(id === undefined || id < 2 ? UNK_ID : id);  // "<eos>" in text is characters, not the token
    }
    return ids;
  }
  /** Special tokens have no text; <unk> shows as the replacement character, as in Python. */
  decode(ids) {
    let out = "";
    for (const id of ids) {
      if (id >= 2) out += this.itos[id];
      else if (id === UNK_ID) out += "�";
    }
    return out;
  }
}

/** Fetch an export's files from `base` (a directory URL ending in "/"). */
export async function loadModel(base) {
  const get = async (name) => {
    const response = await fetch(new URL(name, base));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response;
  };
  const [config, generation, tokenizer, weights] = await Promise.all([
    get("config.json").then((r) => r.json()),
    get("generation_config.json").then((r) => r.json()),
    get("tokenizer.json").then((r) => r.json()),
    get("model.safetensors").then((r) => r.arrayBuffer()),
  ]);
  return { model: new Model(config, readSafetensors(weights)), tokenizer: new CharTokenizer(tokenizer), generation };
}

// ---------------------------------------------------------------------------- the network

/** y = W x for a weight W of shape [out, in], stored row-major (nn.Linear's layout: y = x Wᵀ). */
function matvec(W, x, out, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    const row = r * cols;
    for (let c = 0; c < cols; c++) sum += W[row + c] * x[c];
    out[r] = sum;
  }
  return out;
}

/** RMSNorm: scale x to unit root-mean-square, then multiply by the learned gain. */
function rmsnorm(x, gain, eps, out) {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = gain[i] * (x[i] * scale);
  return out;
}

export class Model {
  constructor(config, tensors) {
    const c = config;
    this.vocab = c.vocab_size;
    this.blockSize = c.max_position_embeddings;
    this.nLayers = c.num_hidden_layers;
    this.D = c.hidden_size;
    this.H = c.num_attention_heads;
    this.K = c.num_key_value_heads;
    this.Hd = c.head_dim ?? this.D / this.H;
    this.F = c.intermediate_size;
    this.eps = c.rms_norm_eps;
    const theta = c.rope_parameters?.rope_theta ?? c.rope_theta ?? 10000;
    const w = (name) => {
      if (!tensors[name]) throw new Error(`missing tensor ${name}`);
      return tensors[name].data;
    };
    this.embed = w("model.embed_tokens.weight");
    this.head = c.tie_word_embeddings ? this.embed : w("lm_head.weight");
    this.finalNorm = w("model.norm.weight");
    this.layers = [];
    for (let l = 0; l < this.nLayers; l++) {
      const p = `model.layers.${l}.`;
      this.layers.push({
        inNorm: w(p + "input_layernorm.weight"),
        q: w(p + "self_attn.q_proj.weight"), k: w(p + "self_attn.k_proj.weight"),
        v: w(p + "self_attn.v_proj.weight"), o: w(p + "self_attn.o_proj.weight"),
        postNorm: w(p + "post_attention_layernorm.weight"),
        gate: w(p + "mlp.gate_proj.weight"), up: w(p + "mlp.up_proj.weight"),
        down: w(p + "mlp.down_proj.weight"),
      });
    }
    // RoPE tables, as RotaryEmbedding builds them: float32 at every step (Math.fround), because
    // PyTorch computes them in float32 and the angles at late positions are large numbers.
    const half = this.Hd / 2;
    this.cos = new Float32Array(this.blockSize * half);
    this.sin = new Float32Array(this.blockSize * half);
    for (let i = 0; i < half; i++) {
      const invFreq = Math.fround(1 / Math.fround(Math.pow(theta, Math.fround((2 * i) / this.Hd))));
      for (let pos = 0; pos < this.blockSize; pos++) {
        const angle = Math.fround(pos * invFreq);
        this.cos[pos * half + i] = Math.cos(angle);
        this.sin[pos * half + i] = Math.sin(angle);
      }
    }
    this.reset();
    // Scratch buffers, allocated once.
    const { D, H, K, Hd, F } = this;
    this.x = new Float64Array(D);
    this.h = new Float64Array(D);
    this.qv = new Float64Array(H * Hd);
    this.kv = new Float64Array(K * Hd);
    this.vv = new Float64Array(K * Hd);
    this.att = new Float64Array(H * Hd);
    this.proj = new Float64Array(D);
    this.g = new Float64Array(F);
    this.u = new Float64Array(F);
    this.scores = new Float64Array(this.blockSize);
    this.logits = new Float64Array(this.vocab);
  }

  /** Forget the context: the key/value cache starts empty. */
  reset() {
    const size = this.blockSize * this.K * this.Hd;
    // float64, like the rest of the arithmetic, so the cache adds no rounding of its own. Sharp
    // attention amplifies small errors in keys. abc-folk's cache is 4 MB.
    this.keys = this.layers.map(() => new Float64Array(size));
    this.values = this.layers.map(() => new Float64Array(size));
    this.length = 0;
  }

  /** Rotate each head's vector by its position (rotate_half convention: pairs i and i + Hd/2). */
  rope(vec, heads, pos) {
    const half = this.Hd / 2;
    for (let h = 0; h < heads; h++) {
      const o = h * this.Hd;
      for (let i = 0; i < half; i++) {
        const c = this.cos[pos * half + i], s = this.sin[pos * half + i];
        const a = vec[o + i], b = vec[o + i + half];
        vec[o + i] = a * c - b * s;
        vec[o + i + half] = b * c + a * s;
      }
    }
  }

  /** Feed one token at the next position; returns the logits for the token after it. */
  step(token) {
    const pos = this.length;
    if (pos >= this.blockSize) throw new Error("context full: reset and re-feed the last block");
    const { D, H, K, Hd, F, x, h } = this;
    for (let i = 0; i < D; i++) x[i] = this.embed[token * D + i];
    const group = H / K;  // query heads per key/value head
    const scale = 1 / Math.sqrt(Hd);
    for (let l = 0; l < this.nLayers; l++) {
      const L = this.layers[l];
      // Attention: x += o_proj(attend(rope(q), rope(k), v)) on the normed residual.
      rmsnorm(x, L.inNorm, this.eps, h);
      matvec(L.q, h, this.qv, H * Hd, D);
      matvec(L.k, h, this.kv, K * Hd, D);
      matvec(L.v, h, this.vv, K * Hd, D);
      this.rope(this.qv, H, pos);
      this.rope(this.kv, K, pos);
      const keys = this.keys[l], values = this.values[l];
      keys.set(this.kv, pos * K * Hd);
      values.set(this.vv, pos * K * Hd);
      for (let head = 0; head < H; head++) {
        const kvHead = Math.floor(head / group);
        const q = head * Hd;
        // Causal by construction: only positions 0..pos are in the cache.
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          const k = (t * K + kvHead) * Hd;
          let dot = 0;
          for (let i = 0; i < Hd; i++) dot += this.qv[q + i] * keys[k + i];
          this.scores[t] = dot * scale;
          if (this.scores[t] > max) max = this.scores[t];
        }
        let total = 0;
        for (let t = 0; t <= pos; t++) total += (this.scores[t] = Math.exp(this.scores[t] - max));
        for (let i = 0; i < Hd; i++) this.att[q + i] = 0;
        for (let t = 0; t <= pos; t++) {
          const p = this.scores[t] / total, v = (t * K + kvHead) * Hd;
          for (let i = 0; i < Hd; i++) this.att[q + i] += p * values[v + i];
        }
      }
      matvec(L.o, this.att, this.proj, D, H * Hd);
      for (let i = 0; i < D; i++) x[i] += this.proj[i];
      // SwiGLU MLP: x += down(silu(gate(h)) * up(h)) on the normed residual.
      rmsnorm(x, L.postNorm, this.eps, h);
      matvec(L.gate, h, this.g, F, D);
      matvec(L.up, h, this.u, F, D);
      for (let i = 0; i < F; i++) this.g[i] = (this.g[i] / (1 + Math.exp(-this.g[i]))) * this.u[i];
      matvec(L.down, this.g, this.proj, D, F);
      for (let i = 0; i < D; i++) x[i] += this.proj[i];
    }
    rmsnorm(x, this.finalNorm, this.eps, h);
    matvec(this.head, h, this.logits, this.vocab, D);
    this.length = pos + 1;
    return this.logits;
  }

  /** Logits after a whole sequence, from an empty cache (what model(idx)[:, -1] gives). */
  prefill(ids) {
    this.reset();
    let logits = null;
    for (const id of ids) logits = this.step(id);
    return logits;
  }
}

// ---------------------------------------------------------------------------- sampling

/** A small seeded random number generator (mulberry32). PyTorch's generator can't be reproduced
 *  here, so a seed repeats a result in the browser, not across engines. */
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Temperature, top-k and top-p, as filter_logits in src/slmkit/sampling/generate.py. */
export function filterLogits(logits, temperature, topK, topP) {
  const scaled = Array.from(logits, (z) => z / temperature);
  if (topK > 0 && topK < scaled.length) {
    const kth = [...scaled].sort((a, b) => b - a)[topK - 1];
    for (let i = 0; i < scaled.length; i++) if (scaled[i] < kth) scaled[i] = -Infinity;
  }
  if (topP < 1) {
    const order = scaled.map((_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
    const probs = softmax(order.map((i) => scaled[i]));
    let before = 0;  // probability of the tokens ranked above this one
    for (let r = 0; r < order.length; r++) {
      if (before >= topP) scaled[order[r]] = -Infinity;  // always keeps the first
      before += probs[r];
    }
  }
  return scaled;
}

function softmax(z) {
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / total);
}

function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;  // the first of equal maxima, as torch.argmax
}

/**
 * Generate like POST /generate (src/slmkit/serve/app.py): same request fields, same defaults
 * from generation_config.json, and a response with the same fields, so the page and a project's
 * viewer can't tell the engines apart.
 */
export function generate({ model, tokenizer, generation }, request, name = "model") {
  const r = request || {};
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const prompt = r.prompt ?? "";
  check(typeof prompt === "string" && prompt.length <= 20000, "prompt: a string of at most 20,000 characters");
  const maxNew = r.max_new_tokens ?? generation.max_new_tokens;
  check(Number.isInteger(maxNew) && maxNew >= 1 && maxNew <= 4096, "max_new_tokens: 1 to 4096");
  const temperature = r.temperature ?? generation.temperature;
  check(temperature >= 0 && temperature <= 5, "temperature: 0 to 5");
  const topK = r.top_k ?? generation.top_k ?? 0;
  const topP = r.top_p ?? generation.top_p ?? 1;
  const seed = r.seed ?? Math.floor(Math.random() * 2 ** 31);
  check(Number.isInteger(seed) && seed >= 0, "seed: a whole number, 0 or more");

  const ids = tokenizer.encode(prompt);
  // An empty prompt starts a fresh document: in training, every document followed <eos>.
  let context = ids.length ? ids.slice() : [EOS_ID];
  const random = seededRandom(seed);
  const block = model.blockSize;
  const start = performance.now();
  // The model sees at most its last `block` tokens, as in generate.py.
  let logits = model.prefill(context.slice(-block));
  const out = [];
  for (let n = 0; n < maxNew; n++) {
    let next;
    if (temperature === 0) {
      next = argmax(logits);
    } else {
      const probs = softmax(filterLogits(logits, temperature, topK, topP));
      let u = random(), cumulative = 0;
      next = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (u < cumulative) { next = i; break; }
      }
    }
    out.push(next);
    context.push(next);
    if (next === EOS_ID) break;
    // Past the context length, positions restart: re-read the last `block` tokens (slow, rare).
    logits = model.length < block ? model.step(next) : model.prefill(context.slice(-block));
  }
  const seconds = (performance.now() - start) / 1000;
  const finished = out.includes(EOS_ID);
  const completion = tokenizer.decode(finished ? out.slice(0, out.indexOf(EOS_ID)) : out);
  return {
    model: name,
    completion,
    text: prompt + completion,
    finished,
    prompt_tokens: ids.length,
    new_tokens: out.length,
    unknown_prompt_tokens: ids.filter((id) => id === UNK_ID).length,
    seed,
    settings: { max_new_tokens: maxNew, temperature, top_k: topK, top_p: topP },
    seconds: Math.round(seconds * 1e4) / 1e4,
    tokens_per_second: seconds > 0 ? Math.round((out.length / seconds) * 10) / 10 : 0,
  };
}
