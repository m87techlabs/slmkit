// Learn: the repo's documentation, rendered. Concepts, runbooks, ADRs and references in reading
// order; links between documents stay in the studio; diagrams are drawn; glossary terms explain
// themselves on hover. The text is the committed Markdown, nothing rewritten.

import { api, h, showDoc } from "../lib/ui.js";

const DEFAULT_DOC = "docs/LIFECYCLE.md";
let glossaryCache = null;
let mermaidLoading = null;

function slug(text) {
  return text.toLowerCase().trim().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s+/g, "-");
}

function resolve(fromDoc, href) {
  // Relative to the document's folder, as GitHub resolves it; returns "path#anchor" without a leading /.
  const url = new URL(href, `https://repo/${fromDoc}`);
  return { path: decodeURIComponent(url.pathname.slice(1)), anchor: decodeURIComponent(url.hash.slice(1)) };
}

function loadMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  mermaidLoading ??= new Promise((ok, fail) => {
    const s = h("script", { src: "/vendor/mermaid.min.js" });
    s.onload = () => ok(window.mermaid);
    s.onerror = () => fail(new Error("Mermaid isn't downloaded (offline at first start?)"));
    document.head.append(s);
  });
  return mermaidLoading;
}

async function glossary() {
  glossaryCache ??= await api("/api/glossary");
  return glossaryCache;
}

// Wrap the first use of each glossary term in a document with a hover/focus definition.
function annotate(article, entries, project) {
  const byName = new Map();
  for (const e of entries) {
    for (const n of e.names) {
      byName.set(n, e);
      const word = !/^[A-Z0-9]{2,}/.test(n) && n.length >= 4; // acronyms match exactly; words in any case
      if (word) byName.set(n.toLowerCase(), e);
    }
  }
  const names = [...byName.keys()].sort((a, b) => b.length - a.length).map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!names.length) return;
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}_-])(${names.join("|")})(?![\\p{L}\\p{N}_])`, "u");
  const used = new Set();
  const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement.closest("pre, code, a, h1, h2, h3, h4, h5, h6, .mermaid, .term, button, th")
      ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (let node of nodes) {
    let m;
    while (node && (m = pattern.exec(node.data))) {
      const entry = byName.get(m[1]);
      if (!entry || used.has(entry.term)) {
        // Already explained in this document: look further along the same text node.
        const rest = node.splitText(m.index + m[1].length);
        node = rest;
        continue;
      }
      used.add(entry.term);
      const after = node.splitText(m.index);
      const rest = after.splitText(m[1].length);
      const span = h("span", { class: "term", tabindex: "0", "data-term": entry.term }, after.data);
      span.dataset.definition = entry.definition;
      span.dataset.link = `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent("docs/GLOSSARY.md")}&at=${slug(entry.term)}`;
      after.replaceWith(span);
      node = rest;
    }
  }
}

let tip = null;
function tooltipFor(span) {
  tip?.remove();
  tip = h("div", { class: "term-tip", role: "tooltip" },
    h("strong", {}, span.dataset.term), h("p", {}, span.dataset.definition.length > 420
      ? `${span.dataset.definition.slice(0, 420)}…` : span.dataset.definition),
    h("a", { href: span.dataset.link }, "In the glossary →"));
  document.body.append(tip);
  const r = span.getBoundingClientRect();
  const left = Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 12);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${window.scrollY + r.bottom + 6}px`;
}

async function renderDoc({ path, at, project, article, onTitle }) {
  const response = await fetch(`/api/doc?path=${encodeURIComponent(path)}`);
  if (!response.ok) { article.replaceChildren(h("p", { class: "muted" }, `Can't open ${path}.`)); return; }
  const md = await response.text();
  if (!window.marked || !window.DOMPurify) {
    article.replaceChildren(h("p", { class: "muted" }, "The Markdown renderer isn't downloaded yet; showing the raw text."), h("pre", {}, md));
    return;
  }
  article.innerHTML = window.DOMPurify.sanitize(window.marked.parse(md, { gfm: true }));
  const seen = new Map();
  for (const hd of article.querySelectorAll("h1, h2, h3, h4")) {
    const base = slug(hd.textContent);
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    hd.id = n ? `${base}-${n}` : base;
  }
  if (path === "docs/GLOSSARY.md") {
    for (const p of article.querySelectorAll("p > strong:first-child")) p.parentElement.id = slug(p.textContent.replace(/\.$/, ""));
  }
  onTitle(article.querySelector("h1")?.textContent ?? path);

  for (const a of article.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href");
    if (href.startsWith("#")) {
      a.href = `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(path)}&at=${encodeURIComponent(href.slice(1))}`;
    } else if (/^https?:/.test(href)) {
      a.target = "_blank"; a.rel = "noopener";
    } else {
      const { path: target, anchor } = resolve(path, href);
      if (target.endsWith(".md")) {
        a.href = `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(target)}${anchor ? `&at=${encodeURIComponent(anchor)}` : ""}`;
      } else if (/\.(png|gif|svg|jpe?g)$/i.test(target)) {
        a.href = `/api/image?path=${encodeURIComponent(target)}`; a.target = "_blank";
      } else if (target.endsWith("/")) {
        a.removeAttribute("href"); a.classList.add("folder"); a.title = `folder in the repo: ${target}`;
      } else {
        a.href = "#";
        a.title = `show ${target}`;
        a.addEventListener("click", async (e) => {
          e.preventDefault();
          const r = await fetch(`/api/source?path=${encodeURIComponent(target)}`);
          showDoc(target, r.ok ? await r.text() : `Not viewable here: ${target}`);
        });
      }
    }
  }
  for (const img of article.querySelectorAll("img[src]")) {
    const src = img.getAttribute("src");
    if (!/^https?:/.test(src)) img.src = `/api/image?path=${encodeURIComponent(resolve(path, src).path)}`;
    img.loading = "lazy";
  }

  const diagrams = [...article.querySelectorAll("pre > code.language-mermaid")].map((code) => {
    const div = h("div", { class: "mermaid" }, code.textContent);
    code.parentElement.replaceWith(div);
    return div;
  });
  if (diagrams.length) {
    try {
      const mermaid = await loadMermaid();
      const dark = matchMedia("(prefers-color-scheme: dark)").matches;
      mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default" });
      // One at a time, each with its own ID: rendering a page's diagrams as a batch let their
      // nodes land in the first diagram's box.
      for (const [i, div] of diagrams.entries()) {
        const { svg } = await mermaid.render(`studio-diagram-${Date.now()}-${i}`, div.textContent);
        div.innerHTML = svg; // Mermaid's own output; securityLevel "strict" sanitizes labels
      }
    } catch (err) {
      diagrams.forEach((d) => d.classList.add("mermaid-failed"));
      article.prepend(h("p", { class: "muted small" }, `Diagrams shown as text: ${err.message}`));
    }
  }

  if (path !== "docs/GLOSSARY.md") annotate(article, await glossary(), project);
  const target = at && document.getElementById(at);
  if (target) target.scrollIntoView({ block: "start" });
  else window.scrollTo(0, 0);
}

export async function render({ project, params, root }) {
  const tree = await api(`/api/docs?project=${encodeURIComponent(project)}`);
  const path = params.get("doc") ?? DEFAULT_DOC;
  const nav = h("nav", { class: "doc-nav", "aria-label": "Documents" });
  const results = h("div", { class: "search-results" });
  const search = h("input", { type: "search", placeholder: "Search the docs…", "aria-label": "Search the docs" });
  const title = h("div", { class: "muted small doc-path" });
  const article = h("article", { class: "doc" }, h("p", { class: "muted" }, "Loading…"));

  const docLink = (d, label) => h("a", { href: `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(d.path)}`,
    "aria-current": d.path === path ? "page" : null }, label ?? d.title);
  nav.append(...tree.map((s) => h("section", {}, h("h2", {}, s.title), h("ul", {}, s.docs.map((d) => h("li", {}, docLink(d)))))));

  let timer = null;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = search.value.trim();
      if (q.length < 2) { results.replaceChildren(); nav.hidden = false; return; }
      const hits = await api(`/api/search?q=${encodeURIComponent(q)}&project=${encodeURIComponent(project)}`);
      nav.hidden = true;
      results.replaceChildren(h("h2", {}, `${hits.length}${hits.length === 30 ? "+" : ""} matches`),
        h("ul", {}, hits.map((hit) => h("li", {},
          h("a", { href: `#/learn?project=${encodeURIComponent(project)}&doc=${encodeURIComponent(hit.path)}&at=${encodeURIComponent(slug(hit.heading))}` },
            hit.title, hit.heading ? ` › ${hit.heading}` : ""),
          h("div", { class: "muted small" }, hit.line)))));
    }, 200);
  });

  root.replaceChildren(h("div", { class: "learn" },
    h("aside", {}, search, results, nav),
    h("div", {}, title, article)));

  const onOver = (e) => { const t = e.target.closest?.(".term"); if (t) tooltipFor(t); };
  const onOut = (e) => {
    if (e.target.closest?.(".term") && !e.relatedTarget?.closest?.(".term-tip")) setTimeout(() => {
      if (!document.querySelector(".term-tip:hover")) { tip?.remove(); tip = null; }
    }, 300);
  };
  article.addEventListener("mouseover", onOver);
  article.addEventListener("focusin", onOver);
  article.addEventListener("mouseout", onOut);
  article.addEventListener("focusout", onOut);

  await renderDoc({ path, at: params.get("at"), project, article, onTitle: () => { title.textContent = path; } });
  return () => { tip?.remove(); tip = null; };
}
