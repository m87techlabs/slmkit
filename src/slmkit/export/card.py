"""MODEL_CARD.md: the README that travels with an exported model.

Everything on the card comes from the export's manifest (itself built from the run's manifest,
status, checkpoint and latest eval report), so the card can't drift from the files it
describes. Model cards are a convention from Mitchell et al. (2019), "Model Cards for Model
Reporting": what the model is for, how it was trained, how it scored, and where it fails.
"""

from __future__ import annotations

import json
from typing import Any

STAGE_WORDS = {
    "pretrain": "a **pretrained base model**: it continues text in the style of its training data",
    "sft": "a **fine-tuned (SFT) model**: it answers requests in the format it was fine-tuned on",
}

TEMPLATE = """\
# {ref}

A {params:,}-parameter Llama-style language model trained from scratch with
[slmkit](https://github.com/m87techlabs/slmkit) for the `{project}` project. It is {stage_words}.

## How to prompt it

Generation starts from the prompt and stops at `<eos>` (token ID 1). An example prompt, exactly
as the evaluation used it:

```
{example}```

Serve it and ask over HTTP:

```bash
uv run slm serve --model {ref}
curl -s localhost:8000/generate -H 'content-type: application/json' \\
  -d '{{"prompt": {example_json}, "seed": 0}}'
```

Or load it with Hugging Face `transformers`; the directory is a standard `LlamaForCausalLM`
checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("$SLM_HOME/models/{path}")
model = AutoModelForCausalLM.from_pretrained("$SLM_HOME/models/{path}")
```

## Model

| | |
|---|---|
| architecture | `LlamaForCausalLM` (RMSNorm, RoPE, SwiGLU, tied embeddings) |
| parameters | {params:,} ({params_non_embedding:,} non-embedding) |
| vocabulary | {vocab_size} tokens |
| context | {block_size} tokens |
| weights | `model.safetensors`, fp32, {weights_mb:.1f} MB |

## Training

| | |
|---|---|
| run | `{run}` ({run_name}), checkpoint `{checkpoint}` at step {step:,} |
| tokens seen (this stage) | {tokens} |
| validation loss | {val} |
| GPU-hours (this stage) | {gpu_hours:.3f} |
| tokenizer | `{tokenizer}` |
| code | slmkit {git} |

Full lineage back to the raw data: `uv run slm lineage {ref}`.

## Evaluation

{evaluation}

## Verification

Checked before it was published, over {parity_tokens} tokens: slmkit reloads the files with max
|Δlogit| = {slmkit_diff:.1e} (identical). {hf_line}

## Data and licence

Training data, its source and its licence: `projects/{project}/README.md` in the slmkit
repository.

## Limitations

- **Context of {block_size} tokens.** slmkit's sampler keeps only the last {block_size} tokens as
  context; positions beyond that were never trained.
- **Fixed vocabulary of {vocab_size} tokens.** Text the tokenizer has no token for becomes
  `<unk>`, which the model learned nothing about. `slm serve` reports how many prompt tokens that
  affected.
- **Small and narrow.** It knows only its training domain. The scores above are averages;
  individual samples can be wrong, so check them with the project's graders.
"""


def _evaluation(ev: dict[str, Any] | None) -> str:
    if not ev:
        return "Not evaluated: run `slm eval <run_id>` before exporting to record scores here."
    rows = "\n".join(
        f"| {metric} | {stat['mean']:.3f} ± {stat['std']:.3f} |"
        for metric, stat in ev["aggregate"].items()
    )
    return (
        f"From eval report `{ev['eval_id']}`: {len(ev['seeds'])} sampling seeds × "
        f"{ev['num_samples']} samples, prompted with the project's **{ev['prompts_kind']}** "
        "prompts. Mean ± standard deviation across seeds.\n\n"
        f"| metric | score |\n|---|---|\n{rows}\n\n"
        "Metric definitions: the project's README and `docs/concepts/evaluation.md`."
    )


def model_card(manifest: dict[str, Any]) -> str:
    s, c = manifest["stats"], manifest["config"]
    parity = s.get("parity", {})
    if "hf_max_abs_diff" in parity:
        hf_line = (
            f"`transformers` {parity['transformers']} loads them with max |Δlogit| = "
            f"{parity['hf_max_abs_diff']:.1e} and identical token IDs."
        )
    else:
        hf_line = f"`transformers` check: {parity.get('transformers', 'not run')}."
    example = s["example_prompt"]
    return TEMPLATE.format(
        ref=manifest["id"],
        project=manifest["project"],
        stage_words=STAGE_WORDS.get(s["stage"], s["stage"]),
        example=example if example.endswith("\n") else example + "\n",
        # Inside a single-quoted shell string: JSON escapes the newlines; quote any ' too.
        example_json=json.dumps(example).replace("'", "'\\''"),
        path=f"{c['name']}/{c['version']}",
        params=s["params"],
        params_non_embedding=s["params_non_embedding"],
        vocab_size=s["vocab_size"],
        block_size=s["block_size"],
        weights_mb=s.get("weights_bytes", 0) / 1e6,
        run=manifest["inputs"]["run"],
        run_name=c["run_name"],
        checkpoint=s["checkpoint"],
        step=s["step"],
        tokens=f"{s['tokens_seen']:,}" if s.get("tokens_seen") else "-",
        val=f"{s['val_loss']:.4f}" if s.get("val_loss") is not None else "-",
        gpu_hours=s["gpu_hours"],
        tokenizer=manifest["inputs"]["tokenizer"],
        git=f"`{manifest.get('git_sha')}`"
        + (" (uncommitted changes)" if manifest.get("git_dirty") else ""),
        evaluation=_evaluation(s.get("eval")),
        parity_tokens=parity.get("tokens", "?"),
        slmkit_diff=parity.get("slmkit_max_abs_diff", 0.0),
        hf_line=hf_line,
    )
