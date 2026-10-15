# ADR 0010 — Publishing models to the Hugging Face Hub, and the `model_card_info` hook

**Status:** accepted (after v0.1.0). Decision 5, the demo as a Docker Space, is replaced by
[ADR 0011](0011-browser-inference.md): Docker Spaces turned out to need a paid plan, so the demo is a
static site with the model running in the browser.

## Context

An export (ADR 0007) is already a standard `LlamaForCausalLM` directory with a model card, checked
against `transformers` before it is written. Publishing one on the Hugging Face Hub, so that anyone can
load it or try it in a browser, needed three more things:

- **Metadata the Hub reads.** The Hub takes a model's licence, library, task and tags from a YAML
  block at the top of its `README.md`. Without it the page has no licence and no "text generation"
  label, and `transformers` users can't find the model by task.
- **A licence and a data statement only the project can give.** Whether the weights can be
  licensed, and on what terms, depends on the training data. For `abc_music`, ADR 0005 records an
  open question: the tune books are public domain, but the transcription files state no licence. The
  engine can't know this, and must not guess (CONTRIBUTING.md rule 2).
- **A way to try it.** A model page alone shows text; folk tunes need to be seen and heard.

## Decision

1. **The model card opens with Hub metadata.** `slm export` writes `license` (when the project states
   one), `library_name: transformers`, `pipeline_tag: text-generation`, and tags: `slmkit`, `llama`,
   `from-scratch`, the project name, then the project's own tags. Export CODE_VERSION 4.
2. **Project API: `model_card_info() -> ModelCardInfo`**, optional, returning `license` (a Hub licence
   ID or None), `tags`, and `data`: Markdown for the card's "Data and licence" section. The default
   states no licence and points to the project's README. Export records it in the manifest, so the
   card is still built from the manifest alone.
3. **The card keeps its name, `MODEL_CARD.md`.** Uploading copies it to `README.md`, the name the Hub
   shows.
4. **`abc_music` releases its weights under Apache-2.0, with the open question on the card.** The card
   says plainly that the transcriptions carry no licence of their own, and that the model is withdrawn
   and retrained without a collection if anyone objects. `shakespeare_char` states public-domain text
   and Apache-2.0.
5. **The online demo is slmkit's own `slm serve`, in a Docker Space.** The Space's Dockerfile installs
   CPU-only PyTorch and a released slmkit, downloads the model from the Hub by version tag, and runs
   `slm serve` unchanged. It lives outside the repo, in the Space's own git repository: it deploys one
   model of one project, and isn't part of the engine.

## Consequences

- Any project's export can be published with two `hf upload` commands; the card becomes the model
  page as is.
- A project that says nothing gets an honest card: no licence claimed, a pointer to its README.
- The demo is the same code and page a reader runs locally, so it can't drift from slmkit, and it
  needs no slmkit change. Its cost is a CPU-only Docker image of about 1 GB, built by the Hub.
- **Where the Space lives** (found when publishing): Hugging Face charges organisations for Docker
  Spaces, even on the free CPU tier (`402 Payment Required`). Static Spaces stay free for
  organisations, and personal accounts can still run a Docker Space for free. The Space therefore lives
  under a personal account and serves the organisation's model. If that changes, the Space moves; the
  model doesn't. A static Space would need the model to run in the browser: a JavaScript forward
  pass with its own parity test, which is possible for a model this size but is a project of its own.
- Hub versions follow slmkit's: `abc-folk:3` is the model repo's tag `v3`. The repo's `main` may move
  on; the Space pins a tag.

## Alternatives considered

- **A Gradio Space**, the Hub's default: rejected. It would be a second, different user interface to
  maintain, and Gradio would be a framework slmkit doesn't otherwise use (rule 6). slmkit already has
  a playground.
- **Renaming `MODEL_CARD.md` to `README.md` in every export:** rejected. Existing exports, runbook
  output and docs name `MODEL_CARD.md`, and inside `$SLM_HOME` "README" says less about what the file
  is. One copy at upload time costs nothing.
- **A licence option on `slm export`:** rejected. The licence follows from the data, which is the
  project's business, and an option would let one export of a model differ from the next.
- **Keeping the weights unpublished** until the transcription question is settled: rejected for now.
  The repo already names the data and publishes generated samples. Stating the risk on the card, with
  a removal path, is consistent with ADR 0005.
- **The Space's Dockerfile in this repo:** rejected for now. Containers are M4's design work. The
  Dockerfile deploys one model, and the Space's git repository is already public.
