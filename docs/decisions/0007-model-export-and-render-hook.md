# ADR 0007 — Model export format, and the `render_sample` hook in the Project API

**Status:** accepted (M2, Phase E)

## Context

Phase E makes a trained run usable outside slmkit: an exported model directory, an HTTP server, and
generated samples a person can open on Windows. Three questions needed answers:

1. What an export is: its format, its identity, and how its correctness is established.
2. What the server loads.
3. How a sample becomes something a person can use. For ABC music that is a MIDI file, for chess a
   PGN, for cricket probably a table. The engine can't know any of these (CONTRIBUTING.md rule 2), and the
   Project API had no hook for it.

## Decision

1. **Export = Hugging Face `LlamaForCausalLM` layout** under `$SLM_HOME/models/<name>/<version>/`:
   `config.json`, `generation_config.json`, fp32 `model.safetensors` (tied head stored once),
   `tokenizer.json` plus `tokenizer_config.json`, `MODEL_CARD.md`, `manifest.json`. The char tokenizer
   is written as an HF `WordLevel` model split one character at a time, so any HF loader reproduces
   slmkit's token IDs.
2. **Models are addressed as `name:version`, and versions are immutable.** Re-exporting the same run
   and step is a no-op; anything else under an existing version is refused. The manifest's inputs
   are the run and the tokenizer, so `slm lineage name:version` reaches the raw data.
3. **Parity is checked on every export, before it is published.** slmkit reloading the files must give
   identical logits; `transformers` (if installed) must match within 1e-4 from the same token IDs. The
   export is built in a `.tmp` directory and only renamed into place if both pass.
4. **`slm serve` loads the export, not a run.** It needs only the engine and the model directory, not
   the project's code.
5. **Project API: `render_sample(prompt, text) -> dict[str, bytes]`**, files keyed by extension.
   Optional; the default returns the text as `.txt`. `abc_music` returns `.abc` (with the `X:` and `T:`
   headers ABC apps expect) and `.mid` from `abc2midi` when it can play the tune. The engine owns the
   writing, so the one exception to the `/mnt/` rule stays in one place (`export/windows.py`).

## Consequences

- An export is usable by anything that reads HF checkpoints (`transformers`, vLLM, conversion
  scripts) with no slmkit code. Measured: the three `abc_music` exports match `transformers` with a
  logit difference of exactly 0.0.
- The model card is generated from the manifest, so it can't disagree with the files it describes.
  It reports the eval that matches the stage's prompt format (headers for a base model, plain
  requests for SFT).
- `render_sample` is pure apart from calling a converter, and returns bytes rather than writing, so
  it is testable without touching a filesystem path. Chess (M3) can return a PGN with no engine change.
- The server can't apply a project's `logits_processor` without loading project code. No project has
  one yet; chess (M3) will decide how it travels with the model.

## Alternatives considered

- **Ship slmkit's own char tokenizer JSON.** Rejected: nothing outside slmkit can read it, which would
  make the "standard format" claim untrue for every char model.
- **Store `lm_head.weight` as well.** Rejected: safetensors refuses aliased tensors, and saving a copy
  would make the file disagree with `tie_word_embeddings: true`.
- **bf16 weights.** Rejected for now: half the size of a 3.5 MB file saves nothing that matters, and
  fp32 keeps the round-trip check exact.
- **Content-hashed model IDs** like other artifacts. Rejected: a model is something people and clients
  refer to by name, like a container image tag. `name:version` plus immutability gives the same
  guarantee (a reference never changes meaning) and is readable. The hash-like identity is still in
  the manifest, as the source run ID and step.
- **Serve directly from a run directory.** Rejected: a run keeps changing while it trains, and would
  couple the server to training code and checkpoints that hold optimizer state.
- **Engine-side rendering (e.g. `if project == "abc_music": abc2midi`).** Rejected: breaks rule 2.
- **A `render_sample` that writes files itself.** Rejected: it would spread `/mnt/` writes across
  projects, where the path guard can't see them.
