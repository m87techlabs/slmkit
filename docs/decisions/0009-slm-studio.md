# ADR 0009 — `slm studio`: a local web app over everything slmkit has built

**Status:** accepted (Studio phase 1, before M3)

## Context

By the end of M2 the learning is spread across runbooks, concept pages, manifests, metrics logs, eval
reports and exported models. The developer wants one interactive place to see what was built and how,
play with the models, and later check each runbook step. The constraints:

- one studio for all projects, and M3's chess must fit without studio changes;
- nothing written into the repo except the studio's code;
- no Node toolchain;
- Run buttons for verification (phase 3);
- the workstation is off regularly (CONTRIBUTING.md).

## Decision

1. **A FastAPI app in `src/slmkit/studio/`**, the same stack as `slm serve`. It has read-only JSON APIs
   over `$SLM_HOME` and the repo (`data.py`), machine facts and live readings (`machine.py`: doctor.json,
   `/proc`, `nvidia-smi`), and static pages.
2. **Pages are plain ES modules, no build step.** `web/studio.js` is the shell (project switcher, hash
   router) and holds the page registry. A page is a module exporting `render({project, params, root,
   navigate})`, and adding one is one line.
3. **Everything is per project and discovered.** Projects come from `projects/*/project.py` and the
   manifests' `project` field. Grader columns come from eval reports. Runbooks declare their project with
   `<!-- slm-studio: projects=… -->`; without the marker a runbook is about the engine. No project name
   appears in studio code.
4. **The playground is `slm serve`'s.** `ModelServer` (extracted from `serve/app.py`) is used by both, and
   the page uses relative URLs so it runs at `/play/<name>/<version>/`. For exports that predate viewers,
   the studio (which runs from the repo) uses the project's current viewer.
5. **Third-party browser code is downloaded, not committed.** Each file is pinned to a version and a
   SHA-384 hash (`vendor.py`), fetched on first start into `$SLM_HOME/studio/vendor/` and refused if it
   doesn't match. Phase 1 needs one library, uPlot (MIT).
6. **`slm studio start | stop | status | run`.** A background process with its record and log in
   `$SLM_HOME/studio/`, plus a foreground variant. It binds 127.0.0.1 only.

## Consequences

- Every number on every page traces to a file, so the studio can't disagree with the pipeline. Tests
  check each API against the toy project.
- `shakespeare_char`, a project the studio was never written for, renders on every page: the M3 check,
  done early.
- The studio needs no Node, no new Python dependency, and no internet after the first start.
- The playground and `slm serve` can't drift apart: one page and one generation path.
- Pages are hand-written DOM code. Plain ES modules keep that readable at this size, and a framework
  would only pay off at many times this size.

## Alternatives considered

- **Node, with a framework and bundler (React/Svelte + Vite)**, installed into `$SLM_HOME`. Rejected at
  the developer's choice: a second toolchain, hundreds of MB of `node_modules` and a build step, for pages
  that plain modules handle.
- **A static site generated from the docs (MkDocs, Docusaurus).** Rejected: it can render runbooks, but
  not live runs, live GPU readings or a model to talk to.
- **One app per project.** Rejected: the machine, the engine and the lifecycle are shared, and one
  studio with a switcher is what makes "M3 fits without changes" testable.
- **Loading chart libraries from a CDN at page load**, as the playground does for abcjs. Rejected here:
  the studio should work offline once started, and pinned downloads with a hash check give the same
  integrity guarantee.
- **Server-sent events or WebSockets for live readings.** Rejected for now: polling every 2 s is one
  line of code and cheap. Push can replace it if phase 3 streams command output.
