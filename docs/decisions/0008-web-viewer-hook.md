# ADR 0008 — A browser playground in `slm serve`, and the `web_viewer` hook

**Status:** accepted (M2, after Phase F)

## Context

`slm serve` answered only JSON, so trying a model meant `curl` and reading raw ABC. Seeing (and
hearing) a generated tune needs a page. The page's generic parts (a prompt, sampling settings,
statistics) belong to the engine. Drawing sheet music and playing it is specific to `abc_music`, and
the engine must not know about it (CONTRIBUTING.md rule 2). ADR 0007 also decided that the server loads only
the export, never project code.

## Decision

1. **`slm serve` serves a playground at `/`**: one static HTML file
   (`src/slmkit/serve/static/index.html`), no framework, no build step, using `/info` and `/generate`.
   URL parameters (`prompt`, `seed`, `temperature`, `max_new_tokens`, `go=1`) make a result shareable
   as a link.
2. **Project API: `web_viewer() -> Path | None`**, a directory containing `viewer.js`, an ES module that
   may export `setup({info, container, setPrompt})` and `render({info, result, container})`. Optional;
   the default is None, and the page then shows raw text only.
3. **`slm export` copies the viewer into the model as `ui/`** (export CODE_VERSION 2). The server
   mounts it at `/ui/` and reports it in `/info`. It serves the files and never executes them.
4. **`slm serve --ui DIR`** overrides the export's viewer, for developing one without re-exporting.
5. **Third-party browser code comes from a CDN, pinned, with subresource integrity.** `abc_music` uses
   abcjs 6.7.1 (MIT) from jsDelivr, recorded in the project README and STACK.md.

## Consequences

- A new project gets a working playground for free, and a tailored one by adding a directory.
- The model directory stays self-contained: the viewer travels with the weights, like the model card.
  A viewer fix needs a new model version, the same rule as every other file in an export, hence `--ui`
  for development.
- The page needs internet access to load abcjs and its soundfont; without it, generation and raw text
  still work.
- Everything the model writes is inserted as text, never HTML, so a generated `<script>` can't run.

## Alternatives considered

- **Server-side rendering with `abc2midi`** (as `--to-windows` does). Rejected: the server would need
  project code and a system package; the browser can draw and play ABC itself.
- **A separate web app** (its own server or a static site). Rejected: a second thing to run and
  version, for one page.
- **A frontend framework (React, Svelte) and a bundler.** Rejected: a build step and a Node toolchain
  for about 300 lines of plain JavaScript.
- **Vendoring abcjs into the repo.** Rejected for now: 500 KB of minified third-party code in git. The
  pinned CDN URL with an integrity hash gives the same guarantee that the code can't change. Vendoring
  becomes right if the playground must work offline.
- **A viewer chosen by name from a list in the engine** (e.g. `viewer: abc`). Rejected: the engine would
  then contain every project's viewer.
