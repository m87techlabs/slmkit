// abc_music's viewer for `slm serve`'s playground (ADR 0008). `slm export` copies this directory
// into the model as ui/; the page imports it and calls setup() and render().
//
// It draws a generated tune as sheet music and plays it in the browser with abcjs
// (https://www.abcjs.net/, MIT licence), loaded from jsDelivr with a pinned version and a
// subresource-integrity hash: the browser refuses the file if the CDN serves anything else.
// Playback samples come from abcjs's default soundfont (also fetched by the browser).

const ABCJS = {
  version: "6.7.1",
  script: "https://cdn.jsdelivr.net/npm/abcjs@6.7.1/dist/abcjs-basic-min.js",
  scriptHash: "sha384-gO9mym1Z3WJwxNm4ZpC6ZQbMyiu+72akLTHzztpwTs6KYVd3NnfkQigzPk+Oqzqy",
  css: "https://cdn.jsdelivr.net/npm/abcjs@6.7.1/abcjs-audio.css",
  cssHash: "sha384-eK1u60r2vqfWom2CLxO79qAvQKkjn5j1PjFslU5/+6gScaYuFJ92tkZ1N4z0hXbF",
};

// The forms the training corpus actually has most of (experiments.md §5: ask for what exists).
// meter and L: (unit note length) are the most common pairing for each rhythm.
const RHYTHMS = [
  { rhythm: "reel", meter: "2/2", unit: "1/8" },
  { rhythm: "jig", meter: "6/8", unit: "1/8" },
  { rhythm: "hornpipe", meter: "2/4", unit: "1/16" },
  { rhythm: "slip jig", meter: "9/8", unit: "1/8" },
  { rhythm: "strathspey", meter: "4/4", unit: "1/8" },
  { rhythm: null, meter: "3/4", unit: "1/8", label: "a tune in 3/4 (e.g. an air)" },
  { rhythm: null, meter: "6/8", unit: "1/8", label: "a tune in 6/8" },
  { rhythm: null, meter: "4/4", unit: "1/8", label: "a tune in 4/4" },
];
const KEYS = ["G", "D", "A", "C", "F", "Bb", "Em", "Am", "Dm", "Bm", "Gm", "F#m", "Amix", "Dmix", "Ador", "Edor"];
const MODES = { "": "major", m: "minor", mix: "mixolydian", dor: "dorian" };
const INSTRUMENTS = [["fiddle", 40], ["piano", 0], ["flute", 73], ["accordion", 21], ["harp", 46]];

let loading = null;
let playing = null; // the current SynthController, so a new result can stop the old one

function loadAbcjs() {
  if (window.ABCJS) return Promise.resolve();
  loading ??= new Promise((resolve, reject) => {
    const css = Object.assign(document.createElement("link"), { rel: "stylesheet", href: ABCJS.css,
      integrity: ABCJS.cssHash, crossOrigin: "anonymous" });
    const js = Object.assign(document.createElement("script"), { src: ABCJS.script,
      integrity: ABCJS.scriptHash, crossOrigin: "anonymous" });
    js.onload = resolve;
    js.onerror = () => reject(new Error(`could not load abcjs ${ABCJS.version} from jsDelivr (offline?)`));
    const style = document.createElement("style");
    style.textContent = `
      .abc-paper { background: #fff; color: #000; border-radius: 6px; padding: 8px; margin-top: 10px; }
      .abc-paper .abcjs-highlight { fill: #2a78d6; }
      .abc-tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 4px 0 8px; }
      .abc-tools label { margin: 0; display: flex; gap: 6px; align-items: center; }
      .abc-tools select { font: inherit; padding: 4px 6px; border-radius: 6px; }
      .abc-tools a { font-size: 13px; }
      .abc-warnings { font-size: 12px; margin-top: 8px; }
      .abc-builder { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }`;
    document.head.append(css, style, js);
  });
  return loading;
}

function keyWords(key) {
  const m = key.match(/^([A-G][b#]?)(m|mix|dor)?$/);
  return `${m[1]} ${MODES[m[2] ?? ""]}`;
}

// The request, in the format the model was trained on. For a fine-tuned model these are the same
// templates as the project's SFT eval prompts (project.py: EVAL_WITH_RHYTHM, EVAL_WITHOUT_RHYTHM).
function buildPrompt(stage, form, key) {
  if (stage === "sft") {
    const request = form.rhythm
      ? `${/^[aeiou]/.test(form.rhythm) ? "an" : "a"} ${form.rhythm} in ${keyWords(key)}, ${form.meter} time`
      : `a tune in ${form.meter} time in ${keyWords(key)}`;
    return `% ${request}\n`;
  }
  return (form.rhythm ? `R:${form.rhythm}\n` : "") + `M:${form.meter}\nL:${form.unit}\nK:${key}\n`;
}

function select(label, options) {
  const wrap = document.createElement("div");
  const l = Object.assign(document.createElement("label"), { textContent: label });
  const s = document.createElement("select");
  for (const [text, value] of options) s.append(new Option(text, value));
  l.append(s);
  wrap.append(l);
  return [wrap, s];
}

export async function setup({ info, container, setPrompt }) {
  loadAbcjs().catch(() => {}); // start fetching early; render() reports failures
  const [formBox, formSel] = select("Ask for", RHYTHMS.map((f, i) => [f.label ?? `a ${f.rhythm} (${f.meter})`, i]));
  const [keyBox, keySel] = select("in the key of", KEYS.map((k) => [keyWords(k), k]));
  const grid = Object.assign(document.createElement("div"), { className: "abc-builder" });
  grid.append(formBox, keyBox);
  const note = Object.assign(document.createElement("div"), {
    className: "hint",
    textContent: info.stage === "sft"
      ? "Builds a request in words, the way this model was fine-tuned. Or edit the prompt freely."
      : "This is a base model: the request is written as ABC headers. Or edit the prompt freely.",
  });
  keySel.value = "D"; // matches the model card's example prompt (a reel in D), which the page shows first
  const update = () => setPrompt(buildPrompt(info.stage, RHYTHMS[Number(formSel.value)], keySel.value));
  formSel.addEventListener("change", update);
  keySel.addEventListener("change", update);
  container.append(grid, note);
}

// A title for the sheet music: the request itself for a fine-tuned model ("% a jig in G ..."),
// or what the headers asked for, for a base model.
function titleFor(text) {
  const first = text.split("\n")[0];
  if (first.startsWith("%")) return first.replace(/^%\s*/, "").slice(0, 60);
  const rhythm = text.match(/^R:\s*(.+)$/m)?.[1]?.trim();
  const key = text.match(/^K:\s*(\S+)/m)?.[1];
  return `${rhythm ?? "a tune"}${key ? ` in ${key}` : ""}`;
}

export async function render({ result, container }) {
  await loadAbcjs();
  const ABC = window.ABCJS;
  if (playing) { playing.pause(); playing = null; }

  const title = titleFor(result.text);
  const abc = `X:1\nT:${title} (seed ${result.seed})\n${result.text}${result.text.endsWith("\n") ? "" : "\n"}`;
  const tools = Object.assign(document.createElement("div"), { className: "abc-tools" });
  const audio = document.createElement("div");
  const paper = Object.assign(document.createElement("div"), { className: "abc-paper" });
  const warnings = Object.assign(document.createElement("div"), { className: "abc-warnings" });
  container.append(tools, audio, paper, warnings);

  const tune = ABC.renderAbc(paper, abc, { responsive: "resize", add_classes: true })[0];

  // What abcjs complained about while reading the tune: the browser's view of `plays`.
  const found = (tune.warnings ?? []).map((w) => w.replace(/<[^>]*>/g, ""));
  warnings.append(Object.assign(document.createElement("div"), {
    className: found.length ? "error" : "hint",
    textContent: found.length
      ? `abcjs found ${found.length} problem(s) reading this tune:\n` + found.slice(0, 5).join("\n")
      : "abcjs read the tune without complaint.",
  }));

  const [instBox, instSel] = select("Instrument", INSTRUMENTS);
  const fileName = `${title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "")}-seed${result.seed}`;
  const abcLink = Object.assign(document.createElement("a"), { textContent: "Download .abc", download: `${fileName}.abc`,
    href: URL.createObjectURL(new Blob([abc], { type: "text/vnd.abc" })) });
  const midiLink = Object.assign(document.createElement("a"), { textContent: "Download .mid", download: `${fileName}.mid` });
  const setMidi = () => { midiLink.href = ABC.synth.getMidiFile(tune, { midiOutputType: "encoded", program: Number(instSel.value) }); };
  setMidi();
  tools.append(instBox, abcLink, midiLink);

  if (!ABC.synth.supportsAudio()) {
    audio.textContent = "This browser has no Web Audio support: use the .mid download.";
    return;
  }
  let lit = [];
  const cursor = {
    onEvent(ev) {
      lit.forEach((el) => el.classList.remove("abcjs-highlight"));
      lit = (ev.elements ?? []).flat();
      lit.forEach((el) => el.classList.add("abcjs-highlight"));
    },
    onFinished() { lit.forEach((el) => el.classList.remove("abcjs-highlight")); lit = []; },
  };
  const synth = new ABC.synth.SynthController();
  synth.load(audio, cursor, { displayLoop: true, displayRestart: true, displayPlay: true,
                              displayProgress: true, displayWarp: true });
  await synth.setTune(tune, false, { program: Number(instSel.value) });
  playing = synth;
  instSel.addEventListener("change", async () => {
    synth.pause();
    await synth.setTune(tune, false, { program: Number(instSel.value) });
    setMidi();
  });
}
