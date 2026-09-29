# ADR 0006 — Training stages, and the SFT hooks in the Project API

**Status:** accepted (M2, Phase D)

## Context

SFT (DESIGN §6.6) reuses the trainer with different batches, starting weights and settings. The M1
trainer was pretraining-specific: it built its own packed-data sampler, run identity and eval prompts.
The Project API had `sft_examples()` but no way for a project to supply the *plain-language* prompts a
fine-tuned model should be evaluated with. CONTRIBUTING.md rule 2 requires an ADR for Project API extensions.

## Decision

1. **A `Stage` object** (`src/slmkit/train/stage.py`) supplies what differs between pretraining and SFT:
   run identity and directory, tokenizer, effective `TrainConfig`, block size, a `(split, seed) ->
   sampler` factory, eval prompts, the bits-per-character ratio (pretraining only) and optional starting
   weights. The training loop consumes a Stage and is otherwise unchanged. `Trainer(exp)` still means
   pretraining.
2. **Project API:** new optional method `sft_eval_prompts(split) -> Iterator[EvalPrompt] | None`, the
   plain-language twins of `eval_prompts`, with the same `meta`, so the same graders score adherence to
   what was asked. `sft_examples(docs)` is called per split (train, then val).
3. **SFT runs** get their own run ID: training identity + the `sft:` section + the parent run's ID. Their
   manifest records `stage: sft` and inputs `{parent, tokenizer, dataset}`. Tools that need the packed
   data (the eval baseline) follow `parent`.
4. **`slm sft <experiment>`** fine-tunes from the experiment's pretrained `ckpt/best`; `slm run` continues
   into SFT when `sft.enabled`. `slm eval --prompts headers|sft|auto` can ask a base model in words, to
   measure what SFT added.

## Consequences

- Resume, checkpoints, guards, throughput logging and tracking apply to SFT with no extra code. The M1
  resume-equivalence test passes unchanged, and an SFT resume test was added.
- The SFT stage is generic: the toy test project defines `sft_examples` in five lines, and no engine code
  names a project.
- The vocabulary is fixed at pretraining, so a project's requests must use characters (or tokens) the
  tokenizer already knows. `abc_music` tests this; other projects must too.

## Alternatives considered

- **A separate SFT trainer.** Rejected: it would duplicate resume, checkpointing, guards and logging,
  which is exactly the infrastructure that must never drift between stages.
- **A `stage` flag on `Trainer` with branches.** Rejected: branching on stage throughout the loop mixes
  concerns; a Stage object keeps the loop stage-agnostic.
- **Special tokens (`<request>`, `<answer>`) for SFT formatting.** Rejected for now: they need new,
  untrained embedding rows added after pretraining. `abc_music` uses an ABC comment line instead.
- **Deriving SFT eval prompts in the engine from `eval_prompts` meta.** Rejected: how a request is
  phrased is domain knowledge, so it belongs to the project.
