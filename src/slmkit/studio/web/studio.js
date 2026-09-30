// slm studio's shell: project switcher, navigation and a hash router.
//
// URLs look like #/training?project=abc_music&run=run-49ba8968e8aa, so every view is a link.
// PAGES is the whole registry: a page is an ES module in pages/ exporting
//   render({ project, params, root, navigate })
// Phase 2 and 3 pages, and anything M3 needs, are added here and nowhere else (ADR 0009).

import { api, h } from "./lib/ui.js";

const PAGES = [
  { id: "home", title: "Your model", module: "./pages/home.js" },
  { id: "lifecycle", title: "Lifecycle", module: "./pages/lifecycle.js" },
  { id: "training", title: "Training", module: "./pages/training.js" },
  { id: "playground", title: "Playground", module: "./pages/playground.js" },
];

const root = document.getElementById("page");
const picker = document.getElementById("project");
let projects = [];
let cleanup = null;

function remembered() {
  try { return localStorage.getItem("slm-studio-project"); } catch { return null; }
}
function remember(name) {
  try { localStorage.setItem("slm-studio-project", name); } catch { /* private mode: fine */ }
}

function parse() {
  const [path, query] = location.hash.replace(/^#\/?/, "").split("?");
  const params = new URLSearchParams(query ?? "");
  const page = PAGES.find((p) => p.id === path) ?? PAGES[0];
  let project = params.get("project");
  if (!projects.some((p) => p.name === project)) {
    const last = remembered();
    project = projects.some((p) => p.name === last) ? last
      : (projects.find((p) => p.runs > 0) ?? projects[0])?.name;
  }
  return { page, project, params };
}

export function navigate(pageId, extra = {}) {
  const { project } = parse();
  const params = new URLSearchParams({ project, ...extra });
  location.hash = `#/${pageId}?${params}`;
}

function renderNav(current, project) {
  const nav = document.getElementById("nav");
  nav.replaceChildren(...PAGES.map((p) => h("a", {
    href: `#/${p.id}?project=${encodeURIComponent(project)}`,
    "aria-current": p.id === current.id ? "page" : null,
  }, p.title)));
}

async function route() {
  const { page, project, params } = parse();
  if (!project) {
    root.replaceChildren(h("p", {}, "No projects found. Train something first: see docs/runbooks/."));
    return;
  }
  remember(project);
  picker.value = project;
  renderNav(page, project);
  document.title = `${page.title} · ${project} · slm studio`;
  if (cleanup) { cleanup(); cleanup = null; }
  root.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  try {
    const mod = await import(page.module);
    const result = await mod.render({ project, params, root, navigate });
    if (typeof result === "function") cleanup = result; // pages return a cleanup (e.g. stop polling)
  } catch (err) {
    root.replaceChildren(h("p", { class: "muted" }, `This page failed to load: ${err.message}`));
    console.error(err);
  }
}

async function start() {
  projects = await api("/api/projects");
  const health = await api("/api/health");
  document.getElementById("home-path").textContent = health.slm_home;
  picker.replaceChildren(...projects.map((p) => h("option", { value: p.name },
    `${p.name}${p.runs ? ` (${p.runs} runs)` : ""}`)));
  picker.addEventListener("change", () => {
    const { page } = parse();
    location.hash = `#/${page.id}?project=${encodeURIComponent(picker.value)}`;
  });
  window.addEventListener("hashchange", route);
  await route();
}

start().catch((err) => root.replaceChildren(h("p", {}, `The studio API is not answering: ${err.message}`)));
