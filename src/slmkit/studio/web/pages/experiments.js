// Experiments: every configuration of the project averaged over its training seeds, the way
// `slm runs summary` computes it, plus the two questions that make a comparison trustworthy:
// is a difference bigger than the noise, and which noise (experiments.md §3)?

import { api, cssVar, fmt, h } from "../lib/ui.js";

const CLEAR = 2; // a difference under ~2 spreads isn't one (evaluation.md §4)
const W = 900, ROW = 44, LEFT = 210, RIGHT = 30, TOP = 30;

function label(g) { return `${g.experiment.split("/").pop()}${g.stage === "sft" ? " (SFT)" : ""}`; }

// How far apart two groups are, in units of their typical spread (null with fewer than 2 runs each).
function separation(a, b) {
  if (!a || !b || a.n < 2 || b.n < 2) return null;
  const spread = Math.sqrt((a.std ** 2 + b.std ** 2) / 2);
  return spread > 0 ? (a.mean - b.mean) / spread : null;
}

function niceTicks(lo, hi, count = 5) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(6));
  return out;
}

function dotPlot(groups, metric, reference, navigate) {
  const ns = "http://www.w3.org/2000/svg";
  const rows = groups.filter((g) => g.stats[metric]);
  const values = rows.flatMap((g) => g.runs.map((r) => r.values[metric]).filter((v) => v != null));
  if (!values.length) return h("p", { class: "muted" }, "No run has this metric.");
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  lo -= pad; hi += pad;
  const height = TOP + rows.length * ROW + 20;
  const x = (v) => LEFT + ((v - lo) / (hi - lo)) * (W - LEFT - RIGHT);
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
  svg.setAttribute("class", "dotplot");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${metric} for each experiment; table below`);
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.append(e);
    return e;
  };
  for (const t of niceTicks(lo, hi)) {
    el("line", { x1: x(t), x2: x(t), y1: TOP - 8, y2: height - 20, stroke: cssVar("--line") });
    el("text", { x: x(t), y: TOP - 14, "text-anchor": "middle", class: "tick" }, fmt.num(t, Math.abs(t) < 10 ? 3 : 1));
  }
  const ref = reference?.stats[metric];
  if (ref) el("line", { x1: x(ref.mean), x2: x(ref.mean), y1: TOP - 8, y2: height - 20, stroke: cssVar("--ink-2"), "stroke-dasharray": "4 3" });
  rows.forEach((g, i) => {
    const y = TOP + i * ROW + ROW / 2;
    const s = g.stats[metric];
    el("text", { x: LEFT - 12, y: y + 4, "text-anchor": "end", class: g === reference ? "row ref" : "row" }, label(g));
    if (g.runs.length > 1) el("line", { x1: x(s.mean - s.std), x2: x(s.mean + s.std), y1: y, y2: y, stroke: cssVar("--s1"), "stroke-width": 3, "stroke-linecap": "round", opacity: 0.35 });
    for (const r of g.runs) {
      const v = r.values[metric];
      if (v == null) continue;
      const dot = el("circle", { cx: x(v), cy: y, r: 6, fill: cssVar("--muted"), stroke: cssVar("--panel"), "stroke-width": 2, class: "dot", tabindex: 0 });
      const t = document.createElementNS(ns, "title");
      t.textContent = `seed ${r.seed} · ${r.run_id} · ${metric} ${fmt.num(v, 4)} (click to open)`;
      dot.append(t);
      dot.addEventListener("click", () => navigate("training", { run: r.run_id }));
    }
    const d = 7;
    el("path", { d: `M${x(s.mean)},${y - d} L${x(s.mean) + d},${y} L${x(s.mean)},${y + d} L${x(s.mean) - d},${y} Z`, fill: cssVar("--s1"), stroke: cssVar("--panel"), "stroke-width": 2 });
  });
  return svg;
}

export async function render({ project, params, root, navigate }) {
  const groups = await api(`/api/experiments?project=${encodeURIComponent(project)}`);
  root.replaceChildren(h("h1", {}, "Experiments"));
  if (!groups.length) { root.append(h("p", { class: "muted" }, "No complete runs yet.")); return; }

  const metrics = [...new Set(groups.flatMap((g) => Object.keys(g.stats)))];
  let metric = params.get("metric") ?? (metrics.includes("best val bpc") ? "best val bpc" : metrics[0]);
  let reference = groups.find((g) => g.experiment.endsWith("/baseline") && g.stage === "pretrain") ?? groups[0];

  const plot = h("div", { class: "panel" });
  const table = h("div", { class: "panel scroll", style: "margin-top:20px" });
  const noise = h("div", { class: "panel", style: "margin-top:20px" });

  const metricPick = h("select", { "aria-label": "metric", onchange: (e) => { metric = e.target.value; draw(); } },
    metrics.map((m) => h("option", { value: m, selected: m === metric ? true : null }, m)));
  const refPick = h("select", { "aria-label": "reference", onchange: (e) => { reference = groups[Number(e.target.value)]; draw(); } },
    groups.map((g, i) => h("option", { value: i, selected: g === reference ? true : null }, label(g))));

  function draw() {
    plot.replaceChildren(
      h("div", { class: "slider-row" }, h("h2", { style: "margin:0" }, "Each training seed"), metricPick,
        h("span", { class: "small muted" }, "compared with"), refPick),
      dotPlot(groups, metric, reference, navigate),
      h("p", { class: "legend-note" }, "Grey dots: one training seed each (hover for the run, click to open it). Blue diamond: the mean; "
        + "pale bar: ± one standard deviation across seeds. Dashed line: the reference's mean."));

    const ref = reference.stats;
    table.replaceChildren(h("h2", {}, "Mean ± spread across training seeds, against ", label(reference)), h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "experiment"), h("th", { class: "num" }, "runs"), metrics.map((m) => h("th", { class: "num" }, m)))),
      h("tbody", {}, groups.map((g) => h("tr", { "aria-selected": String(g === reference) },
        h("td", {}, label(g)), h("td", { class: "num" }, g.runs.length),
        metrics.map((m) => {
          const s = g.stats[m];
          if (!s) return h("td", { class: "num muted" }, "–");
          const z = g === reference ? null : separation({ ...s, n: g.runs.length }, ref[m] && { ...ref[m], n: reference.runs.length });
          const verdict = z == null ? null : Math.abs(z) >= CLEAR
            ? h("span", { class: "chip good", title: `${Math.abs(z).toFixed(1)} spreads apart: a clear difference` }, `${z > 0 ? "▲" : "▼"} clear`)
            : h("span", { class: "chip", title: `${Math.abs(z).toFixed(1)} spreads apart: within the noise` }, "≈ noise");
          return h("td", { class: "num" }, g.runs.length > 1 ? `${fmt.num(s.mean)} ± ${fmt.num(s.std)}` : `${fmt.num(s.mean)} (1 run)`, verdict ? " " : null, verdict);
        }))))),
      h("p", { class: "legend-note" }, `"Clear" means the means differ by at least ${CLEAR}× the typical spread of the two experiments; `
        + "▲/▼ is the direction, not good or bad (lower is better for loss and bpc, higher for most graders). "
        + "Loss is per token, so it only compares runs with the same tokenizer; bpc compares across tokenizers. "
        + "The SFT row is scored with plain-language requests, the others with header prompts."));

    // Only eval metrics have a sampling spread: for loss or bpc, show the first grader metric instead.
    const hasSampling = (m) => groups.some((g) => g.runs.some((r) => r.sampling_std[m] != null));
    const noiseMetric = hasSampling(metric) ? metric : metrics.find((m) => hasSampling(m) && !["ended", "length", "novelty"].includes(m)) ?? metric;
    const rows = groups.filter((g) => g.stats[noiseMetric] && g.runs.some((r) => r.sampling_std[noiseMetric] != null));
    noise.replaceChildren(h("h2", {}, `Two kinds of noise, for ${noiseMetric}`),
      noiseMetric !== metric ? h("p", { class: "muted small" }, `${metric} has no sampling spread (it isn't sampled), so this shows ${noiseMetric}.`) : null,
      rows.length ? h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "experiment"), h("th", { class: "num" }, "sampling spread (one model, re-sampled)"),
          h("th", { class: "num" }, "training spread (re-trained with another seed)"))),
        h("tbody", {}, rows.map((g) => {
          const sampling = g.runs.map((r) => r.sampling_std[noiseMetric]).filter((v) => v != null);
          return h("tr", {}, h("td", {}, label(g)), h("td", { class: "num" }, fmt.num(sampling.reduce((a, b) => a + b, 0) / sampling.length)),
            h("td", { class: "num" }, g.runs.length > 1 ? fmt.num(g.stats[noiseMetric].std) : "–"));
        })))
        : h("p", { class: "muted" }, "Only eval metrics have a sampling spread; pick plays, bar accuracy or another grader above."),
      h("p", { class: "legend-note" }, "Sampling spread is how much one model's score moves between batches of samples (slm eval's ±). "
        + "Training spread is how much the result moves when the model is trained again with a different seed. "
        + "A comparison of single runs that only beats the sampling spread can still be luck (experiments.md §3)."));
  }
  root.append(h("p", { class: "lede" }, `Every configuration of ${project}, each the mean of its training seeds. `
    + "Change one thing per experiment, keep the compute equal, and believe only differences bigger than the spread."), plot, table, noise);
  draw();
}
