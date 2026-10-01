// Playground: pick one of the project's exported models and use it. The model's own playground
// (the same page as `slm serve`, with its viewer) runs in a frame at /play/<name>/<version>/.

import { api, fmt, h } from "../lib/ui.js";

export async function render({ project, params, root }) {
  const models = await api(`/api/models?project=${project}`);
  root.replaceChildren(h("h1", {}, "Playground"));
  if (!models.length) {
    root.append(h("p", { class: "lede" }, "No exported models for this project yet. Export one from a finished run:"),
      h("pre", {}, "uv run slm runs list\nuv run slm export <run_id> --name <name> --version 1"));
    return;
  }
  // Prefer what was asked for, else the newest model with a viewer, else the newest.
  const wanted = params.get("model");
  const newest = [...models].reverse();
  let current = models.find((m) => m.ref === wanted) ?? newest.find((m) => m.viewer && m.stage === "sft")
    ?? newest.find((m) => m.viewer) ?? newest[0];
  const frame = h("iframe", { class: "play", title: "model playground" });
  const cards = h("div", { class: "cards" });

  function draw() {
    cards.replaceChildren(...models.map((m) => h("button", { class: "card", "aria-pressed": String(m.ref === current.ref),
      onclick: () => { current = m; draw(); } },
      h("strong", {}, m.ref), h("div", { class: "small" }, `${m.stage === "sft" ? "fine-tuned: takes requests" : "base: continues its prompt"} · ${fmt.int(m.params)} parameters`),
      h("div", { class: "small muted" }, `from ${m.source_run} step ${m.step} · ${fmt.date(m.created)}`),
      m.viewer === "project" ? h("div", { class: "small muted" }, "Viewer from the project (this export predates viewers).")
        : m.viewer ? null : h("div", { class: "small muted" }, "No viewer: raw text only."))));
    frame.src = `/play/${encodeURIComponent(current.name)}/${current.version}/`;
  }
  root.append(h("p", { class: "lede" }, "Each card is an exported model: the same files `slm serve` would serve. Here it runs on the GPU if there is one "
    + "(`slm studio start --device cpu` keeps it off); a few seconds of generation barely disturbs a training run."),
    cards, frame);
  draw();
}
