// Training: every run of the project, compared on one chart, and one run in detail: its curves,
// what it wrote at each evaluation, its eval reports and the exact config it ran with.

import { api, fmt, h, lineChart, seriesColor } from "../lib/ui.js";

const MAX_COMPARE = 8; // one fixed colour slot each; beyond eight, colours stop being distinguishable

const METRICS = {
  val_loss: { label: "validation loss (nats per token)", pick: (e) => e.val_loss,
              note: "Lower is better. Comparable only between runs with the same tokenizer." },
  val_bpc: { label: "validation bits per character", pick: (e) => e.val_bpc,
             note: "Lower is better. Divides by characters, so char and BPE runs compare fairly." },
  gap: { label: "val − train loss (overfitting)", pick: (e) => (e.val_loss != null && e.train_loss != null ? e.val_loss - e.train_loss : null),
         note: "A widening gap means the model is memorizing its training data (fitting.md)." },
};

// The project's own graders become table columns (whatever they are called), not a fixed list:
// abc_music has plays and bar_accuracy, chess will have legal-move rate. Generic metrics are left out.
const GENERIC = new Set(["ended", "length", "novelty"]);
function graderColumns(runs) {
  const keys = [];
  for (const r of runs) for (const k of Object.keys(r.eval?.aggregate ?? {})) if (!GENERIC.has(k) && !keys.includes(k)) keys.push(k);
  return keys.slice(0, 4);
}

function short(r) { return `${(r.experiment ?? r.name ?? r.run_id).split("/").pop()}${r.stage === "sft" ? " · sft" : ""} · seed ${r.seed}`; }

export async function render({ project, params, root, navigate }) {
  const runs = await api(`/api/runs?project=${project}`);
  if (!runs.length) { root.replaceChildren(h("h1", {}, "Training"), h("p", { class: "muted" }, "No runs yet for this project.")); return; }

  // Default comparison: one run per experiment (the first seed), up to eight.
  const selected = new Set();
  const seen = new Set();
  // SFT runs aren't in the default comparison: their loss covers answers only (sft.md §2).
  for (const r of runs) {
    if (r.stage === "pretrain" && !seen.has(r.experiment) && selected.size < MAX_COMPARE) { seen.add(r.experiment); selected.add(r.run_id); }
  }
  let focus = params.get("run") && runs.some((r) => r.run_id === params.get("run")) ? params.get("run") : [...selected][0];
  if (selected.size < MAX_COMPARE) selected.add(focus);
  let metric = "val_bpc";
  const cache = new Map();
  const detailOf = async (id) => { if (!cache.has(id)) cache.set(id, await api(`/api/runs/${id}`)); return cache.get(id); };

  const compareChart = h("div", { class: "chart" });
  const compareNote = h("p", { class: "legend-note" });
  const metricPicker = h("select", { onchange: (e) => { metric = e.target.value; drawCompare(); } },
    Object.entries(METRICS).map(([k, m]) => h("option", { value: k, selected: k === metric ? true : null }, m.label)));
  const table = h("tbody");
  const graders = graderColumns(runs);
  const detail = h("section", { class: "panel", style: "margin-top:20px" });

  function row(r) {
    const box = h("input", { type: "checkbox", "aria-label": `compare ${short(r)}`, checked: selected.has(r.run_id) ? true : null,
      onclick: (e) => {
        e.stopPropagation();
        if (e.target.checked) {
          if (selected.size >= MAX_COMPARE) { e.target.checked = false; compareNote.textContent = `At most ${MAX_COMPARE} runs at once.`; return; }
          selected.add(r.run_id);
        } else selected.delete(r.run_id);
        drawCompare();
      } });
    const ev = r.eval?.aggregate ?? {};
    return h("tr", { class: "clickable", "aria-selected": String(r.run_id === focus), onclick: () => { focus = r.run_id; drawTable(); drawDetail(); } },
      h("td", {}, box), h("td", {}, r.experiment?.split("/").pop() ?? r.name), h("td", {}, r.stage), h("td", { class: "num" }, r.seed),
      h("td", {}, r.preset), h("td", {}, r.tokenizer), h("td", { class: "num" }, fmt.compact(r.params)),
      h("td", { class: "num" }, fmt.compact(r.tokens_seen)), h("td", { class: "num" }, fmt.num(r.best_val_loss, 4)),
      h("td", { class: "num" }, fmt.num(r.best_val_bpc)), graders.map((k) => h("td", { class: "num" }, fmt.num(ev[k]))),
      h("td", {}, r.complete ? "complete" : "resumable"));
  }
  function drawTable() { table.replaceChildren(...runs.map(row)); }

  async function drawCompare() {
    const ids = runs.filter((r) => selected.has(r.run_id)).map((r) => r.run_id);
    const details = await Promise.all(ids.map(detailOf));
    const m = METRICS[metric];
    const series = details.map((d, i) => ({ label: short(d), color: seriesColor(i),
      points: d.eval.filter((e) => m.pick(e) != null && e.step > 0).map((e) => [e.tokens, m.pick(e)]) }));
    const without = series.filter((s) => !s.points.length).map((s) => s.label);
    compareNote.textContent = `${m.note} Step 0 (random weights) is left out so the scale shows the differences. `
      + "Drag across the chart to zoom; double-click to reset."
      + (without.length ? ` Not shown, no ${m.label.split(" (")[0]} recorded: ${without.join(", ")}.` : "");
    lineChart(compareChart, { series: series.filter((s) => s.points.length), xLabel: "tokens read", yLabel: m.label.split(" (")[0] });
  }

  async function drawDetail() {
    const d = await detailOf(focus);
    const curves = h("div", { class: "chart" });
    const lr = h("div", { class: "chart" });
    const sampleBox = h("pre", {});
    const steps = d.samples;
    const prompts = steps.length ? Object.keys(steps[steps.length - 1].samples) : [];
    let promptId = prompts[0];
    const stepLabel = h("span", { class: "small" });
    const slider = h("input", { type: "range", min: 0, max: Math.max(steps.length - 1, 0), value: Math.max(steps.length - 1, 0),
      "aria-label": "evaluation step", oninput: () => showSample() });
    const promptPick = h("select", { "aria-label": "prompt", onchange: (e) => { promptId = e.target.value; showSample(); } },
      prompts.map((p) => h("option", { value: p }, p)));
    function showSample() {
      const s = steps[Number(slider.value)];
      if (!s) { sampleBox.textContent = "No samples were recorded for this run."; return; }
      const ev = d.eval.find((e) => e.step === s.step);
      stepLabel.textContent = `step ${s.step} · ${fmt.compact(s.tokens)} tokens read · val loss ${fmt.num(ev?.val_loss, 3)}`;
      sampleBox.textContent = s.samples[promptId] ?? "(no sample for this prompt at this step)";
    }

    const reports = d.reports.length ? h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "eval report"), h("th", {}, "prompts"), h("th", {}, "samples"),
        ...Object.keys(d.reports[d.reports.length - 1].model).map((k) => h("th", { class: "num" }, k)))),
      h("tbody", {}, d.reports.map((r) => h("tr", {}, h("td", { class: "mono" }, r.eval_id), h("td", {}, r.prompts_kind),
        h("td", {}, r.settings ? `${r.settings.seeds.length} seeds × ${r.settings.num_samples}` : "–"),
        ...Object.values(r.model).map((s) => h("td", { class: "num" }, `${fmt.num(s.mean)} ± ${fmt.num(s.std)}`)))))))
      : h("p", { class: "muted" }, "Not evaluated yet (slm eval).");

    detail.replaceChildren(
      h("h3", {}, short(d), " ", h("span", { class: "mono muted small" }, d.run_id)),
      h("p", { class: "small" },
        h("span", { class: `chip ${d.complete ? "good" : "warn"}` }, d.complete ? "complete" : "resumable"),
        h("span", { class: "chip" }, `${fmt.int(d.params)} parameters`), h("span", { class: "chip" }, `${d.preset} · ${d.tokenizer}`),
        h("span", { class: "chip" }, `${fmt.compact(d.tokens_seen)} tokens`), h("span", { class: "chip" }, fmt.gpuHours(d.gpu_hours)),
        d.tok_per_s ? h("span", { class: "chip" }, `${fmt.compact(d.tok_per_s)} tokens/s`) : null,
        h("span", { class: "chip" }, `best at step ${d.best_step ?? "–"}`),
        d.parent ? h("a", { href: "#", onclick: (e) => { e.preventDefault(); focus = d.parent; drawTable(); drawDetail(); } }, `fine-tuned from ${d.parent}`) : null,
        " ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); navigate("lifecycle", { id: d.run_id }); } }, "lineage →")),
      h("div", { class: "grid two" },
        h("div", {}, h("h2", {}, "Loss while training"), curves,
          h("p", { class: "legend-note" }, "Train loss is logged every few steps; validation loss at each evaluation. When validation stops falling while training keeps falling, the model has started memorizing.")),
        h("div", {}, h("h2", {}, "Learning rate"), lr,
          h("p", { class: "legend-note" }, "Warmup, then cosine decay, scheduled in tokens (the-training-loop.md §3)."))),
      h("h2", { style: "margin-top:16px" }, "Watch it learn: what it wrote at each evaluation"),
      steps.length ? h("div", {}, h("div", { class: "slider-row" }, promptPick, slider, stepLabel), sampleBox)
        : h("p", { class: "muted" }, "No samples recorded."),
      h("h2", { style: "margin-top:16px" }, "Evaluation reports"), reports,
      h("details", { style: "margin-top:16px" }, h("summary", {}, "The exact config it ran with (config.resolved.yaml)"), h("pre", {}, d.config_yaml)));
    lineChart(curves, { xLabel: "tokens read", yLabel: "loss", series: [
      { label: "train", color: seriesColor(0), points: d.train.map((r) => [r.tokens, r.loss]) },
      { label: "validation", color: seriesColor(1), points: d.eval.filter((e) => e.step > 0).map((e) => [e.tokens, e.val_loss]) }] });
    lineChart(lr, { height: 200, xLabel: "tokens read", yLabel: "learning rate",
      series: [{ label: "lr", color: seriesColor(0), points: d.train.map((r) => [r.tokens, r.lr]) }] });
    showSample();
  }

  root.replaceChildren(
    h("h1", {}, "Training"),
    h("p", { class: "lede" }, `Every run for ${project}. Tick runs to compare them (up to ${MAX_COMPARE}); click a row to open it below.`),
    h("section", { class: "panel" }, h("div", { class: "slider-row" }, h("h2", { style: "margin:0" }, "Compare"), metricPicker), compareChart, compareNote),
    h("section", { class: "panel scroll", style: "margin-top:20px" }, h("table", {},
      h("thead", {}, h("tr", {}, ["", "experiment", "stage", "seed", "preset", "tokenizer"].map((c, i) => h("th", { class: i === 3 ? "num" : null }, c)),
        ["params", "tokens", "best val loss", "bpc", ...graders].map((c) => h("th", { class: "num", title: graders.includes(c) ? "latest eval report, mean over sampling seeds" : null }, c.replace(/_/g, " "))),
        h("th", {}, "status"))), table)),
    detail);
  drawTable();
  await drawCompare();
  await drawDetail();
}
