// Verify: every runbook check, with a Run button for the read-only ones and your output next to
// the expected output. The server decides what may run (studio/verify.py); this page only names a
// check by its ID. Last results are kept in $SLM_HOME/studio/verify.json.

import { api, fmt, h } from "../lib/ui.js";

const POLL_MS = 500;
const norm = (s) => s.trim().replace(/\s+/g, " ");

async function post(path, body) {
  const r = await fetch(path, { method: "POST", body: JSON.stringify(body),
    headers: { "content-type": "application/json", "x-slm-studio": "1" } });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail ?? r.statusText);
  return data;
}

// Mark each line of the actual output: found in the expected output (after collapsing spaces), or not.
// IDs, times and speeds always differ; the runbooks say which numbers must match.
function compared(actual, expected) {
  const want = new Set(expected.split("\n").map(norm).filter(Boolean));
  const lines = actual.split("\n");
  let hits = 0;
  const pre = h("pre", { class: "diff" }, lines.map((l) => {
    const match = l.startsWith("$ ") ? "cmd" : want.has(norm(l)) ? "same" : norm(l) ? "differs" : "";
    if (match === "same") hits += 1;
    return h("span", { class: `ln ${match}` }, `${l}\n`);
  }));
  return { pre, hits, of: want.size };
}

function statusChip(result) {
  if (!result) return h("span", { class: "chip" }, "not run yet");
  const when = fmt.ago(new Date(result.time * 1000).toISOString());
  return result.exit_code === 0
    ? h("span", { class: "chip good" }, `passed · ${when} · ${result.seconds}s`)
    : h("span", { class: "chip warn" }, `exit ${result.exit_code} · ${when}`);
}

export async function render({ project, params, root }) {
  const data = await api(`/api/verify?project=${encodeURIComponent(project)}`);
  const results = data.results;
  const books = data.runbooks.filter((b) => b.checks.length);
  let current = books.find((b) => b.file === params.get("runbook")) ?? books[books.length - 1];
  let busy = false;
  const picker = h("div", { class: "cards" });
  const body = h("div", {});
  const summary = h("div", { class: "slider-row" });

  async function run(book, check, card) {
    const out = card.querySelector(".live");
    out.replaceChildren(h("p", { class: "muted small" }, "Starting…"));
    let id;
    try { ({ run_id: id } = await post("/api/verify/run", { runbook: book.file, check_id: check.id })); }
    catch (err) { out.replaceChildren(h("p", { class: "error-text" }, err.message)); return false; }
    for (;;) {
      await new Promise((ok) => setTimeout(ok, POLL_MS));
      const p = await api(`/api/verify/runs/${id}`);
      const live = h("pre", {}, p.output || "…");
      out.replaceChildren(live);
      live.scrollTop = live.scrollHeight;
      if (p.done) {
        results[check.id] = { time: Date.now() / 1000, exit_code: p.exit_code, seconds: p.seconds, output: p.output };
        card.querySelector(".status").replaceChildren(statusChip(results[check.id]));
        showResult(check, card);
        drawSummary();
        return p.exit_code === 0;
      }
    }
  }

  function showResult(check, card) {
    const r = results[check.id];
    const out = card.querySelector(".live");
    if (!r) { out.replaceChildren(); return; }
    if (check.expected) {
      const c = compared(r.output, check.expected);
      out.replaceChildren(h("div", { class: "grid two" },
        h("div", {}, h("h2", {}, "Your output"), c.pre,
          h("p", { class: "legend-note" }, `${c.hits} of ${c.of} expected lines appear exactly (spacing ignored). `
            + "Run IDs, dates, timings and speeds always differ; the runbook says which numbers must match.")),
        h("div", {}, h("h2", {}, "Expected (from the runbook)"), h("pre", {}, check.expected))));
    } else {
      out.replaceChildren(h("h2", {}, "Your output"), h("pre", {}, r.output));
    }
  }

  function card(book, check) {
    const el = h("section", { class: "panel check", id: `check-${check.id}` },
      h("div", { class: "slider-row" }, h("strong", {}, check.heading),
        h("span", { class: "status" }, statusChip(results[check.id]))),
      h("pre", { class: "cmd" }, check.text),
      h("div", { class: "slider-row" },
        check.runnable
          ? h("button", { class: "primary", onclick: async (e) => {
              if (busy) return;
              busy = true; e.target.disabled = true;
              try { await run(book, check, el); } finally { busy = false; e.target.disabled = false; }
            } }, "Run")
          : h("button", { onclick: (e) => { navigator.clipboard?.writeText(check.text); e.target.textContent = "Copied"; } }, "Copy"),
        h("span", { class: "muted small" }, check.runnable ? check.reason : `Not run from here: ${check.reason}.`),
        h("a", { class: "small", href: `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(book.file)}` }, `runbook line ${check.line} →`)),
      !results[check.id] && check.expected ? h("details", {}, h("summary", {}, "Expected output"), h("pre", {}, check.expected)) : null,
      h("div", { class: "live" }));
    showResult(check, el);
    return el;
  }

  function drawSummary() {
    const runnable = current.checks.filter((c) => c.runnable);
    const passed = runnable.filter((c) => results[c.id]?.exit_code === 0).length;
    const failed = runnable.filter((c) => results[c.id] && results[c.id].exit_code !== 0).length;
    summary.replaceChildren(...[
      h("span", { class: "chip" }, `${current.checks.length} checks`),
      h("span", { class: "chip" }, `${runnable.length} runnable here`),
      h("span", { class: "chip good" }, `${passed} passed`),
      failed ? h("span", { class: "chip warn" }, `${failed} failed`) : null,
      h("button", { class: "primary", onclick: async (e) => {
        if (busy) return;
        e.target.disabled = true;
        for (const c of runnable) {
          const el = document.getElementById(`check-${c.id}`);
          el.scrollIntoView({ block: "center", behavior: "smooth" });
          busy = true;
          try { await run(current, c, el); } finally { busy = false; }
        }
        e.target.disabled = false;
      } }, `Run all ${runnable.length} safe checks`)].filter(Boolean));
  }

  function draw() {
    picker.replaceChildren(...books.map((b) => h("button", { class: "card", "aria-pressed": String(b === current),
      onclick: () => { current = b; draw(); } },
      h("strong", {}, b.title), h("div", { class: "small muted" },
        `${b.checks.length} checks · ${b.checks.filter((c) => c.runnable).length} runnable · ${b.milestone}`))));
    let section = null;
    body.replaceChildren(...current.checks.flatMap((c) => {
      const items = [];
      if (c.section !== section) {
        section = c.section;
        if (section !== c.heading) items.push(h("h2", { class: "section-title" }, section));
      }
      items.push(card(current, c));
      return items;
    }));
    drawSummary();
  }

  root.replaceChildren(h("h1", {}, "Verify"),
    h("p", { class: "lede" }, "Every check in the runbooks for this project. Read-only ones run here and show your output against the expected output; "
      + "anything that trains, writes or uses shell syntax is shown with Copy and the reason, for your terminal."),
    picker, summary, body);
  draw();
  const target = params.get("check") && document.getElementById(`check-${params.get("check")}`);
  if (target) target.scrollIntoView({ block: "start" });
}
