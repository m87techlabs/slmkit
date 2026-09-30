// Lifecycle: the pipeline drawn as your real artifacts. One column per stage, one box per artifact,
// one line per manifest input. Click a box to trace its lineage and read its manifest.

import { api, fmt, h } from "../lib/ui.js";

const STAGES = {
  raw: ["Raw data", "slm ingest", "the source files, untouched"],
  dataset: ["Dataset", "slm prepare", "cleaned documents, split by group"],
  tokenizer: ["Tokenizer", "slm tokenize", "text ↔ token IDs, fitted on train"],
  packed: ["Packed", "slm pack", "token IDs in one flat file per split"],
  pretrain: ["Pretraining runs", "slm pretrain", "a model learning to continue text"],
  sft: ["Fine-tuning runs", "slm sft", "a pretrained model taught requests"],
  model: ["Exported models", "slm export", "Hugging Face format, versioned"],
};

function related(id, edges) {
  // Ancestors (what it was made from) and descendants (what was made from it).
  const up = new Set([id]), down = new Set([id]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const e of edges) {
      if (up.has(e.to) && !up.has(e.from)) { up.add(e.from); grew = true; }
      if (down.has(e.from) && !down.has(e.to)) { down.add(e.to); grew = true; }
    }
  }
  return new Set([...up, ...down]);
}

// Edges implied by a longer path are not drawn (a transitive reduction): a run reads the tokenizer
// directly, but tokenizer → packed → run already shows that. Lineage still uses every edge.
function drawnEdges(edges) {
  const out = new Map();
  for (const e of edges) out.set(e.from, [...(out.get(e.from) ?? []), e.to]);
  const reachable = (from, to, skip) => {
    const stack = (out.get(from) ?? []).filter((n) => n !== skip);
    const seen = new Set(stack);
    while (stack.length) {
      const n = stack.pop();
      if (n === to) return true;
      for (const m of out.get(n) ?? []) if (!seen.has(m)) { seen.add(m); stack.push(m); }
    }
    return false;
  };
  return edges.filter((e) => !reachable(e.from, e.to, e.to));
}

export async function render({ project, params, root, navigate }) {
  const g = await api(`/api/graph?project=${project}`);
  const shown = drawnEdges(g.edges);
  const columns = g.columns.filter((c) => g.nodes.some((n) => n.column === c));
  const board = h("div", { class: "lifecycle" });
  const detail = h("section", { class: "panel", style: "margin-top:20px" },
    h("p", { class: "muted" }, "Select an artifact to see its lineage and manifest."));
  const boxes = new Map();

  for (const col of columns) {
    const [title, cmd, what] = STAGES[col] ?? [col, "", ""];
    const nodes = g.nodes.filter((n) => n.column === col).sort((a, b) => (a.created ?? "").localeCompare(b.created ?? ""));
    board.append(h("div", { class: "col" }, h("h3", {}, title), h("div", { class: "cmd" }, h("code", {}, cmd), ` · ${what}`),
      nodes.map((n) => {
        const box = h("button", { class: "node", "data-id": n.id, onclick: () => select(n.id) },
          h("div", { class: "id" }, n.id), h("div", { class: "summary" }, n.summary));
        boxes.set(n.id, box);
        return box;
      })));
  }

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "edges");
  board.prepend(svg);
  const paths = [];

  function draw() {
    const origin = board.getBoundingClientRect();
    svg.setAttribute("width", board.scrollWidth);
    svg.setAttribute("height", board.scrollHeight);
    svg.replaceChildren();
    paths.length = 0;
    for (const e of shown) {
      const a = boxes.get(e.from), b = boxes.get(e.to);
      if (!a || !b) continue;
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const x1 = ra.right - origin.left + board.scrollLeft, y1 = ra.top + ra.height / 2 - origin.top;
      const x2 = rb.left - origin.left + board.scrollLeft, y2 = rb.top + rb.height / 2 - origin.top;
      const mid = (x1 + x2) / 2;
      const p = document.createElementNS(ns, "path");
      p.setAttribute("d", `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
      p.dataset.from = e.from; p.dataset.to = e.to;
      svg.append(p);
      paths.push(p);
    }
    highlight(current);
  }

  let current = null;
  function highlight(id) {
    const keep = id ? related(id, g.edges) : null;
    for (const [nid, box] of boxes) {
      box.classList.toggle("dim", !!keep && !keep.has(nid));
      box.classList.toggle("on", nid === id);
    }
    for (const p of paths) {
      const on = keep && keep.has(p.dataset.from) && keep.has(p.dataset.to);
      p.classList.toggle("on", !!on);
      p.classList.toggle("dim", !!keep && !on);
    }
  }

  function select(id) {
    current = current === id ? null : id;
    highlight(current);
    const n = g.nodes.find((x) => x.id === current);
    if (!n) { detail.replaceChildren(h("p", { class: "muted" }, "Select an artifact to see its lineage and manifest.")); return; }
    const lineage = [...related(n.id, g.edges)].filter((x) => x !== n.id);
    detail.replaceChildren(
      h("h3", {}, h("span", { class: "mono" }, n.id), ` · ${STAGES[n.column]?.[0] ?? n.kind}`),
      h("p", { class: "muted small" }, `Created ${fmt.date(n.created)} · git ${n.git ?? "–"} · ${n.path}`),
      n.run ? h("p", {}, h("button", { class: "primary", onclick: () => navigate("training", { run: n.id }) }, "Open in Training →")) : null,
      h("div", { class: "grid two" },
        h("div", {}, h("h2", {}, "Made from"), Object.keys(n.inputs).length
          ? h("ul", {}, Object.entries(n.inputs).map(([k, v]) => h("li", {}, `${k}: `, h("a", { href: "#", onclick: (e) => { e.preventDefault(); select(v); } }, v))))
          : h("p", { class: "muted" }, "Nothing: this is where the pipeline starts."),
          h("h2", {}, "Connected artifacts"), h("p", { class: "small" }, `${lineage.length} upstream or downstream.`)),
        h("div", {}, h("h2", {}, "Stats"), h("pre", {}, JSON.stringify(n.stats ?? {}, null, 2)))),
      h("details", {}, h("summary", {}, "Config (what determines this artifact's ID)"), h("pre", {}, JSON.stringify(n.config ?? {}, null, 2))));
    detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  root.replaceChildren(
    h("h1", {}, "The lifecycle, as it happened"),
    h("p", { class: "lede" },
      `Every box is an artifact on disk for ${project}; every line is an input recorded in a manifest. `
      + "Artifacts are immutable and content-addressed: change a setting and you get a new box, never an overwritten one. "
      + "Click a box to trace what it was made from and what was made from it. "
      + "(A line that a longer path already implies is left out, e.g. tokenizer → run, since tokenizer → packed → run shows it.)"),
    g.nodes.length ? board : h("p", { class: "muted" }, "No artifacts yet for this project."),
    detail);
  requestAnimationFrame(draw);
  const resize = new ResizeObserver(() => draw());
  resize.observe(board);
  board.addEventListener("scroll", draw);
  if (params.get("id")) select(params.get("id"));
  return () => resize.disconnect();
}
