// Parameters: from a model's shape to its size, compute and memory, computed by the same Python
// the trainer uses (/api/estimate), next to what this GPU measured for runs of that shape.

import { api, cssVar, fmt, h } from "../lib/ui.js";

const FIELDS = [
  ["n_layers", "layers", "transformer blocks stacked"],
  ["d_model", "width (d_model)", "size of every token's vector"],
  ["n_heads", "attention heads", "d_model is split across them"],
  ["n_kv_heads", "key/value heads", "fewer than heads = grouped-query attention"],
  ["ffn_hidden", "MLP width", "SwiGLU hidden size, ~2.75× d_model"],
  ["vocab_size", "vocabulary", "tokens the tokenizer knows"],
  ["block_size", "context", "tokens the model sees at once"],
];
const BUDGETS = [3e6, 30e6, 300e6, 3e9];

async function estimate(body) {
  const r = await fetch("/api/estimate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; "));
  return data;
}

function bytes(n) {
  for (const [v, u] of [[2 ** 30, "GiB"], [2 ** 20, "MiB"], [2 ** 10, "KiB"]]) if (n >= v) return `${(n / v).toFixed(n / v < 10 ? 2 : 1)} ${u}`;
  return `${n} B`;
}
function hours(hh) { return hh == null ? "–" : hh < 1 / 60 ? `${(hh * 3600).toFixed(0)} GPU-s` : hh < 1 ? `${(hh * 60).toFixed(1)} GPU-min` : `${hh.toFixed(1)} GPU-h`; }

function breakdown(p) {
  const parts = [["embedding", p.embedding, 1], ["attention", p.attention, 2], ["MLP", p.mlp, 3], ["norms", p.norms, 4]];
  if (p.head) parts.push(["output head", p.head, 5]);
  const bar = h("div", { class: "stack" }, parts.map(([name, v, slot]) => h("div", {
    style: `flex:${v};background:${cssVar(`--s${slot}`)}`, title: `${name}: ${fmt.int(v)} (${((100 * v) / p.total).toFixed(1)}%)` })));
  const legend = h("div", { class: "stack-legend" }, parts.map(([name, v, slot]) => h("span", {},
    h("i", { style: `background:${cssVar(`--s${slot}`)}` }), `${name} ${((100 * v) / p.total).toFixed(1)}% (${fmt.compact(v)})`)));
  return h("div", {}, bar, legend);
}

export async function render({ project, root }) {
  const [presets, runs] = await Promise.all([api("/api/presets"), api(`/api/runs?project=${encodeURIComponent(project)}`)]);
  const pre = runs.filter((r) => r.stage === "pretrain" && r.vocab_size);
  const last = pre[pre.length - 1];
  const state = { ...presets.find((p) => p.name === (last?.preset ?? "nano")) ?? presets[0],
    vocab_size: last?.vocab_size ?? 256, block_size: last?.block_size ?? 512, tokens: last?.max_tokens ?? 30e6 };
  delete state.name; delete state.comment;

  const out = h("div", { class: "grid" });
  const compare = h("section", { class: "panel scroll" });
  const presetPick = h("select", { "aria-label": "preset" }, h("option", { value: "" }, "custom"),
    presets.map((p) => h("option", { value: p.name }, `${p.name} · ${p.comment}`)));
  const inputs = {};
  const form = h("div", { class: "param-form" }, FIELDS.map(([key, text, hint]) => {
    inputs[key] = h("input", { type: "number", min: 1, value: state[key], "aria-label": text });
    return h("label", {}, h("span", {}, text), inputs[key], h("small", { class: "muted" }, hint));
  }));
  inputs.tie = h("input", { type: "checkbox", checked: state.tie_embeddings ? true : null });
  inputs.tokens = h("input", { type: "number", min: 1, value: state.tokens, step: 1e6, "aria-label": "training tokens" });
  const budgets = h("div", { class: "chips-row" }, BUDGETS.map((b) => h("button", { type: "button", onclick: () => { inputs.tokens.value = b; update(); } }, fmt.compact(b))));
  presetPick.value = presets.find((p) => FIELDS.slice(0, 5).every(([k]) => p[k] === state[k]))?.name ?? "";

  presetPick.addEventListener("change", () => {
    const p = presets.find((x) => x.name === presetPick.value);
    if (!p) return;
    for (const [k] of FIELDS.slice(0, 5)) inputs[k].value = p[k];
    inputs.tie.checked = p.tie_embeddings;
    update();
  });
  for (const el of [...Object.values(inputs)]) el.addEventListener("input", () => { presetPick.value = ""; schedule(); });

  let timer = null;
  function schedule() { clearTimeout(timer); timer = setTimeout(update, 250); }
  function body(overrides = {}) {
    const b = Object.fromEntries(FIELDS.map(([k]) => [k, Number(inputs[k].value)]));
    return { ...b, tie_embeddings: inputs.tie.checked, tokens: Number(inputs.tokens.value), ...overrides };
  }

  async function update() {
    let e;
    try { e = await estimate(body()); } catch (err) { out.replaceChildren(h("p", { class: "error-text" }, err.message)); return; }
    const p = e.params, m = e.measured;
    out.replaceChildren(
      h("div", { class: "tiles", style: "margin:0" },
        h("div", { class: "tile" }, h("div", { class: "value" }, fmt.compact(p.total)), h("div", { class: "label" }, "parameters"),
          h("div", { class: "sub" }, `${fmt.compact(p.total - p.embedding)} excluding the embedding`)),
        h("div", { class: "tile" }, h("div", { class: "value" }, fmt.compact(e.flops_per_token)), h("div", { class: "label" }, "FLOPs per token"),
          h("div", { class: "sub" }, "forward + backward, ≈ 6 × matmul parameters")),
        h("div", { class: "tile" }, h("div", { class: "value" }, e.training_flops.toExponential(2)), h("div", { class: "label" }, "training FLOPs"),
          h("div", { class: "sub" }, `${fmt.compact(e.tokens)} tokens · ${e.tokens_per_param.toFixed(1)} per parameter`)),
        h("div", { class: "tile" }, h("div", { class: "value" }, hours(m.gpu_hours_whole_run ?? e.planning.gpu_hours)),
          h("div", { class: "label" }, m.gpu_hours_whole_run ? "on this GPU, as your runs took" : "planning estimate"),
          h("div", { class: "sub" }, m.gpu_hours_whole_run ? `training steps alone ${hours(m.gpu_hours)}; planning said ${hours(e.planning.gpu_hours)}` : `at ${e.planning.tflops} TFLOPS effective`))),
      h("section", { class: "panel" }, h("h2", {}, "Where the parameters are"), breakdown(p),
        h("p", { class: "legend-note" }, `Each layer: ${fmt.int(p.per_layer)} parameters. ` + (p.embedding / p.total > 0.3
          ? `The embedding table is ${(100 * p.embedding / p.total).toFixed(0)}% of this model: a large vocabulary costs parameters that do almost no computation (a lookup).`
          : `With ${fmt.int(Number(inputs.vocab_size.value))} tokens the embedding table is small; the layers hold nearly everything, and the MLP's three matrices are the largest part (MODEL.md §4).`))),
      h("div", { class: "grid two" },
        h("section", { class: "panel" }, h("h2", {}, "Memory and files"), h("dl", { class: "facts" },
          h("dt", {}, "weights (fp32)"), h("dd", {}, bytes(e.memory_bytes.weights_fp32)),
          h("dt", {}, "training state"), h("dd", {}, `${bytes(e.memory_bytes.training_state)} (weights, gradients, AdamW's two averages; plus activations)`),
          h("dt", {}, "checkpoint on disk"), h("dd", {}, bytes(e.memory_bytes.checkpoint)),
          h("dt", {}, "exported model"), h("dd", {}, bytes(e.memory_bytes.export)))),
        h("section", { class: "panel" }, h("h2", {}, "This GPU, at this size"), m.runs
          ? h("dl", { class: "facts" },
            h("dt", {}, "runs measured"), h("dd", {}, `${m.runs} complete pretraining run(s) of exactly this shape`),
            h("dt", {}, "throughput"), h("dd", {}, `${fmt.compact(m.tok_per_s)} tokens/s (median)`),
            h("dt", {}, "achieved"), h("dd", {}, `${m.tflops.toFixed(1)} TFLOPS${m.mfu ? `, ${(100 * m.mfu).toFixed(0)}% of the ${m.peak_tflops} measured peak (MFU)` : ""}`),
            h("dt", {}, "this budget"), h("dd", {}, `${hours(m.gpu_hours)} of training steps; ${hours(m.gpu_hours_whole_run)} as a whole run, with evaluations and compilation`))
          : h("p", { class: "muted" }, `No run of exactly this shape yet, so only the planning estimate: ${e.planning.tflops} TFLOPS effective. `
            + "Small models run below it (MFU is size-dependent, CONTRIBUTING.md), large ones near it."))));
    drawCompare();
  }

  async function drawCompare() {
    const rows = await Promise.all(presets.map(async (p) => {
      try { return [p, await estimate(body({ n_layers: p.n_layers, d_model: p.d_model, n_heads: p.n_heads, n_kv_heads: p.n_kv_heads, ffn_hidden: p.ffn_hidden, tie_embeddings: p.tie_embeddings }))]; }
      catch { return [p, null]; }
    }));
    compare.replaceChildren(h("h2", {}, `Every preset, with this vocabulary, context and ${fmt.compact(Number(inputs.tokens.value))} tokens`), h("table", {},
      h("thead", {}, h("tr", {}, ["preset", "layers × width", "parameters", "FLOPs/token", "planning", "measured here", "checkpoint"].map((c, i) => h("th", { class: i > 1 ? "num" : null }, c)))),
      h("tbody", {}, rows.map(([p, e]) => h("tr", {},
        h("td", {}, h("strong", {}, p.name), h("div", { class: "muted small" }, p.comment)),
        h("td", {}, `${p.n_layers} × ${p.d_model}`),
        h("td", { class: "num" }, e ? fmt.compact(e.params.total) : "–"),
        h("td", { class: "num" }, e ? fmt.compact(e.flops_per_token) : "–"),
        h("td", { class: "num" }, e ? hours(e.planning.gpu_hours) : "–"),
        h("td", { class: "num" }, e?.measured.gpu_hours_whole_run ? hours(e.measured.gpu_hours_whole_run) : "–"),
        h("td", { class: "num" }, e ? bytes(e.memory_bytes.checkpoint) : "–"))))),
      h("p", { class: "legend-note" }, "Planning: FLOPs ÷ 43 TFLOPS. Measured here: what your runs of that exact shape took on this machine, per token, including evaluations. "
        + "Compute, not memory, is what limits a single consumer GPU (DESIGN §2)."));
  }

  root.replaceChildren(h("h1", {}, "Parameters"),
    h("p", { class: "lede" }, "Choose a model's shape and a training budget; see its size, its cost and how much memory it needs. "
      + "Numbers come from the same functions the trainer prints at startup, and from what your runs measured."),
    h("div", { class: "grid two" },
      h("section", { class: "panel" }, h("h2", {}, "Shape"), h("label", {}, h("span", {}, "start from a preset"), presetPick), form,
        h("label", { class: "inline" }, inputs.tie, " tie the output head to the embedding (one matrix, two jobs)"),
        h("label", {}, h("span", {}, "training tokens"), inputs.tokens), budgets),
      out),
    compare);
  await update();
}
