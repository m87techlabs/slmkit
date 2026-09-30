// Shared helpers for studio pages: DOM building (always text, never HTML), formatting, the API,
// sparklines and line charts.

export async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
  return response.json();
}

// h("div", {class: "x", onclick: fn}, "text", childNode, [more]) — strings become text nodes.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  num: (n, d = 3) => (n == null ? "–" : Number(n).toFixed(d)),
  compact(n) {
    if (n == null) return "–";
    const units = [[1e9, "B"], [1e6, "M"], [1e3, "K"]];
    for (const [v, u] of units) if (Math.abs(n) >= v) return `${(n / v).toFixed(n / v < 10 ? 2 : 1)}${u}`;
    return String(Math.round(n));
  },
  date: (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "–"),
  ago(iso) {
    if (!iso) return "–";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    for (const [d, u] of [[86400, "day"], [3600, "hour"], [60, "minute"]]) {
      if (s >= d) { const n = Math.floor(s / d); return `${n} ${u}${n > 1 ? "s" : ""} ago`; }
    }
    return "just now";
  },
  gpuHours: (h) => (h == null ? "–" : h < 0.1 ? `${(h * 60).toFixed(1)} GPU-min` : `${h.toFixed(2)} GPU-h`),
};

// Axis ticks: plain numbers, but small ones (a learning rate of 0.0006) in scientific notation.
function tick(v) {
  if (v == null) return "";
  const a = Math.abs(v);
  return a !== 0 && a < 0.01 ? v.toExponential(1) : Number(v.toPrecision(4)).toString();
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Fixed categorical order: series i always gets slot i (never cycled; max 8, see dataviz rules).
export function seriesColor(i) { return cssVar(`--s${(i % 8) + 1}`); }

export function tile(label, value, sub) {
  return h("div", { class: "tile" }, h("div", { class: "value" }, value), h("div", { class: "label" }, label),
    sub ? h("div", { class: "sub" }, sub) : null);
}

// A tiny line of the last N readings, scaled to `max` (or the data's own max).
export function sparkline(values, max) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 100 32");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  const vals = values.filter((v) => v != null);
  if (vals.length < 2) return svg;
  const top = max ?? Math.max(...vals, 1);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${32 - ((v ?? 0) / top) * 30 - 1}`);
  const path = document.createElementNS(ns, "polyline");
  path.setAttribute("points", pts.join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", cssVar("--s1"));
  path.setAttribute("stroke-width", "2");
  path.setAttribute("vector-effect", "non-scaling-stroke");
  svg.append(path);
  return svg;
}

// A line chart with a hover crosshair and legend (uPlot). series: [{label, points: [[x, y], ...],
// color, dash}]. Different series may have different x values; they are merged on one x axis.
export function lineChart(container, { series, xLabel, yLabel, height = 280, logY = false }) {
  container.replaceChildren();
  if (!window.uPlot) {
    container.append(h("p", { class: "muted small" },
      "Charts need uPlot, which isn't downloaded yet (offline at first start?). Run `slm studio stop` then `slm studio start` with internet access."));
    return null;
  }
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const index = new Map(xs.map((x, i) => [x, i]));
  const ys = series.map((s) => {
    const col = new Array(xs.length).fill(null);
    for (const [x, y] of s.points) col[index.get(x)] = y;
    return col;
  });
  const axis = { stroke: cssVar("--ink-2"), grid: { stroke: cssVar("--line"), width: 1 }, ticks: { stroke: cssVar("--line") } };
  const opts = {
    width: Math.max(container.clientWidth, 320),
    height,
    scales: { x: { time: false }, y: logY ? { distr: 3 } : {} },
    axes: [{ ...axis, label: xLabel, values: (u, v) => v.map(fmt.compact) },
           { ...axis, label: yLabel, size: 64, values: (u, v) => v.map(tick) }],
    series: [{ label: xLabel, value: (u, v) => (v == null ? "–" : fmt.compact(v)) },
             ...series.map((s) => ({ label: s.label, stroke: s.color, width: 2, dash: s.dash,
               spanGaps: true, points: { show: s.points.length < 40, size: 5 },
               value: (u, v) => (v == null ? "–" : Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(4)) }))],
    cursor: { drag: { x: true, y: false } },
    legend: { live: true },
  };
  const chart = new window.uPlot(opts, [xs, ...ys], container);
  const resize = new ResizeObserver(() => chart.setSize({ width: Math.max(container.clientWidth, 320), height }));
  resize.observe(container);
  return chart;
}

export function showDoc(title, text) {
  const dialog = h("dialog", {},
    h("header", {}, h("strong", {}, title), h("button", { onclick: () => dialog.close() }, "Close")),
    h("div", { class: "body" }, h("pre", {}, text)));
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  dialog.showModal();
}
