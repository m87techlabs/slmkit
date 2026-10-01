// Your model: what you built on this PC, the machine it ran on (with live readings), the journey
// so far, and where to read more.

import { api, fmt, h, sparkline, tile } from "../lib/ui.js";

const LIVE_EVERY_MS = 2000;
const HISTORY = 60; // two minutes of readings

const GAUGES = [
  { key: "gpu_util", label: "GPU load", unit: "%", max: () => 100 },
  { key: "vram_used_mib", label: "GPU memory", unit: "GiB", scale: 1 / 1024, max: (r) => r.vram_total_mib / 1024,
    of: (r) => r.vram_total_mib && `of ${(r.vram_total_mib / 1024).toFixed(1)}` },
  { key: "gpu_temp_c", label: "GPU temperature", unit: "°C", max: () => 100 },
  { key: "gpu_power_w", label: "GPU power", unit: "W", max: (r) => r.gpu_power_limit_w,
    of: (r) => r.gpu_power_limit_w && `of ${Math.round(r.gpu_power_limit_w)}` },
  { key: "cpu_percent", label: "CPU load (WSL)", unit: "%", max: () => 100 },
  { key: "ram_used_gib", label: "Memory (WSL)", unit: "GiB", max: (r) => r.ram_total_gib,
    of: (r) => r.ram_total_gib && `of ${r.ram_total_gib.toFixed(1)}` },
];

function machinePanel(facts) {
  const rows = [
    ["GPU", facts.gpu && `${facts.gpu}${facts.vram_gib ? ` · ${Number(facts.vram_gib).toFixed(1)} GiB` : ""}`],
    ["Driver / CUDA", [facts.driver, facts.cuda && `CUDA ${facts.cuda}`, facts.capability].filter(Boolean).join(" · ")],
    ["Measured bf16", facts.bf16_tflops && `${facts.bf16_tflops} TFLOPS (slm doctor --bench)`],
    ["CPU", facts.cpu && `${facts.cpu} · ${facts.cores} threads visible to WSL`],
    ["Memory (WSL)", facts.ram_gib && `${facts.ram_gib} GiB`],
    ["System", facts.os],
    ["PyTorch / Python", [facts.torch, facts.python].filter(Boolean).join(" · ")],
    ["$SLM_HOME", `${facts.slm_home} · ${facts.slm_home_gib} GiB used · ${facts.disk_free_gib} of ${facts.disk_total_gib} GiB free`],
  ].filter(([, v]) => v);
  return h("dl", { class: "facts" }, rows.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

export async function render({ project, root, navigate }) {
  const [o, facts, models] = await Promise.all([
    api(`/api/overview?project=${project}`), api("/api/machine"), api(`/api/models?project=${project}`),
  ]);
  const t = o.totals;
  const best = o.best;

  const hero = h("section", {},
    h("h1", {}, t.complete ? `You built ${t.complete} language models for ${project}, on this PC.` : `${project}: nothing trained yet.`),
    h("p", { class: "lede" },
      t.complete
        ? `From scratch: every weight started random and learned from ${fmt.compact(t.tokens)} tokens of reading, `
          + `using ${fmt.gpuHours(t.gpu_hours)} of your GPU. ${t.experiments} experiments, since ${fmt.date(t.first)}.`
        : "Run the pipeline for this project (its runbook shows how), then come back."),
    best ? h("p", { class: "lede" }, "Best so far: ",
      h("strong", {}, best.experiment), ` (seed ${best.seed}, ${fmt.int(best.params)} parameters): `,
      `${fmt.num(best.best_val_bpc)} bits per character on text it never saw. `,
      models.length ? h("a", { href: `#/playground?project=${project}` }, "Play with an exported model →") : "Export one to play with it.")
      : null);

  const tiles = h("div", { class: "tiles" },
    tile("models trained", fmt.int(t.complete), `${t.runs} runs · ${t.experiments} experiments`),
    tile("tokens read", fmt.compact(t.tokens), "across all runs"),
    tile("GPU time", fmt.gpuHours(t.gpu_hours), "measured, not wall-clock"),
    tile("largest model", fmt.compact(t.largest_params), "parameters"),
    tile("exported", fmt.int(t.models), "Hugging Face-format models"),
    tile("evaluations", fmt.int(t.evals), "graded sample sets"));

  const gauges = h("div", { class: "gauges" });
  const history = Object.fromEntries(GAUGES.map((g) => [g.key, []]));
  const liveNote = h("p", { class: "muted small" }, "Live, every 2 seconds.");
  async function poll() {
    let r;
    try { r = await api("/api/machine/live"); } catch { liveNote.textContent = "Live readings unavailable."; return; }
    gauges.replaceChildren(...GAUGES.map((g) => {
      const raw = r[g.key];
      const v = raw == null ? null : raw * (g.scale ?? 1);
      history[g.key] = [...history[g.key], v].slice(-HISTORY);
      return h("div", { class: "gauge" },
        h("div", { class: "value" }, v == null ? "–" : `${v.toFixed(v < 10 && g.unit !== "%" ? 1 : 0)} ${g.unit}`),
        h("div", { class: "label" }, g.label, g.of?.(r) ? ` (${g.of(r)})` : ""),
        sparkline(history[g.key], g.max(r) ?? undefined));
    }));
  }
  await poll();
  const timer = setInterval(poll, LIVE_EVERY_MS);

  const machine = h("section", { class: "panel" }, h("h2", {}, "The machine"), machinePanel(facts), gauges, liveNote);

  const journey = h("section", { class: "panel" }, h("h2", {}, "The journey"),
    h("ul", { class: "journey" }, o.milestones.map((m) => h("li", {},
      h("span", { class: "status", title: { "☑": "done", "◐": "in progress", "☐": "not started" }[m.status] }, m.status),
      h("span", {}, h("strong", {}, m.id), ` ${m.title.replace(/`/g, "")}`)))),
    h("p", { class: "muted small" }, "From docs/ROADMAP.md."));

  const recent = h("section", { class: "panel" }, h("h2", {}, "Recent runs"),
    o.recent.length ? h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "experiment"), h("th", {}, "stage"), h("th", { class: "num" }, "seed"),
        h("th", { class: "num" }, "best val loss"), h("th", {}, "updated"))),
      h("tbody", {}, o.recent.map((r) => h("tr", { class: "clickable", onclick: () => navigate("training", { run: r.run_id }) },
        h("td", {}, r.experiment), h("td", {}, r.stage), h("td", { class: "num" }, r.seed),
        h("td", { class: "num" }, fmt.num(r.best_val_loss, 4)), h("td", {}, fmt.ago(r.updated)))))))
      : h("p", { class: "muted" }, "No runs yet."));

  const books = h("section", { class: "panel" }, h("h2", {}, "Runbooks for this project"),
    h("p", { class: "muted small" }, "What was built, how to check it by hand, and why. They open in Learn; Run buttons come in studio phase 3."),
    h("ul", { class: "journey" }, o.runbooks.map((b) => h("li", {},
      h("span", { class: "status" }, (b.status || "").slice(0, 1)),
      h("span", {}, h("a", { href: `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(b.file)}` }, b.title),
        h("span", { class: "muted small" }, ` · ${b.milestone}${b.projects ? "" : " · engine"}`))))));

  const everywhere = o.all_projects;
  const all = h("p", { class: "muted small" },
    `Across every project: ${fmt.int(everywhere.runs)} runs, ${fmt.compact(everywhere.tokens)} tokens, `
    + `${fmt.gpuHours(everywhere.gpu_hours)}, ${fmt.int(everywhere.artifacts)} artifacts on disk.`);

  root.replaceChildren(hero, tiles, h("div", { class: "grid two" }, machine, h("div", { class: "grid" }, journey, recent)),
    h("div", { class: "grid two", style: "margin-top:20px" }, books, h("section", { class: "panel" },
      h("h2", {}, "How to read this studio"),
      h("ul", {}, h("li", {}, h("strong", {}, "Lifecycle"), ": every artifact your pipeline made, and what fed what."),
        h("li", {}, h("strong", {}, "Training"), ": loss curves, what each model wrote as it learned, and its config."),
        h("li", {}, h("strong", {}, "Experiments"), ": every configuration over its training seeds, and which differences are real."),
        h("li", {}, h("strong", {}, "Parameters"), ": from a model's shape to its size, cost and memory, against what this GPU measured."),
        h("li", {}, h("strong", {}, "Playground"), ": ask an exported model for something, and see (and hear) it."),
        h("li", {}, h("strong", {}, "Learn"), ": every concept, runbook and decision, with the glossary on hover.")),
      all)));
  return () => clearInterval(timer);
}
