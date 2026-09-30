"""Export a trained run as a Hugging Face-format model directory, and load one back.

    $SLM_HOME/models/<name>/<version>/
        config.json              LlamaForCausalLM hyperparameters
        generation_config.json   default sampling settings (from the run's `eval:` section)
        model.safetensors        the weights, fp32, tied output head stored once
        tokenizer.json           Hugging Face tokenizers format (char -> WordLevel, BPE as is)
        tokenizer_config.json    lets `AutoTokenizer.from_pretrained` find the special tokens
        MODEL_CARD.md            what it is, how to prompt it, how it was trained and scored
        manifest.json            lineage (run, tokenizer) and the parity check results

Because slmkit's parameter names already match `LlamaForCausalLM` (docs/MODEL.md), export is a
file-format change, not a conversion. Two checks prove it before the directory is published:
the export loaded back by slmkit gives *identical* logits, and loaded by `transformers` (when
installed) gives logits within float tolerance, from the same token IDs as slmkit's tokenizer.
A failed check publishes nothing.

A model version is immutable, like every other artifact: re-exporting the same checkpoint is
a no-op, and a different checkpoint needs a new version number. Serving addresses models as
`name:version`, so a version can never silently change under a client.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from slmkit import artifacts
from slmkit.config.load import ConfigError
from slmkit.eval.runner import latest_report
from slmkit.export.card import model_card
from slmkit.inference import LoadedRun, load_run
from slmkit.model import CausalLM, ModelArgs
from slmkit.model.stats import count_parameters
from slmkit.project_api import EvalPrompt
from slmkit.registry import load_project
from slmkit.sampling import generate
from slmkit.tokenizers import EOS_ID, Tokenizer, load_tokenizer
from slmkit.tokenizers.base import EOS, UNK
from slmkit.train.run import read_status

CODE_VERSION = 2  # bump when the files an export writes would change (2: the ui/ viewer)
MODELS_DIR = "models"
WEIGHTS = "model.safetensors"
CARD = "MODEL_CARD.md"
# fp32 on CPU, same weights, different kernels: agreement to ~1e-6 is typical. 1e-4 leaves
# room for summation order while still catching any real mismatch (a wrong RoPE convention
# moves logits by ~1, see test_hf_parity.py).
PARITY_ATOL = 1e-4
PARITY_TOKENS = 64  # length of the greedy continuation the logits are compared over


def models_root() -> Path:
    return artifacts.slm_home() / MODELS_DIR


def parse_ref(ref: str) -> tuple[str, int]:
    """`abc-folk:1` -> ("abc-folk", 1)."""
    name, sep, version = ref.rpartition(":")
    if not sep or not name or not version.isdigit():
        raise ConfigError(f"model reference {ref!r} is not <name>:<version>, e.g. abc-folk:1")
    return name, int(version)


def model_dir(name: str, version: int) -> Path:
    return models_root() / name / str(version)


# ---------------------------------------------------------------------------- file formats


def hf_config(args: ModelArgs) -> dict[str, Any]:
    """`config.json` for `LlamaForCausalLM`: slmkit's ModelArgs under Hugging Face's names."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": args.vocab_size,
        "hidden_size": args.d_model,
        "intermediate_size": args.ffn_hidden,
        "num_hidden_layers": args.n_layers,
        "num_attention_heads": args.n_heads,
        "num_key_value_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": args.block_size,
        "rms_norm_eps": args.norm_eps,
        # Both spellings: transformers >= 5 reads `rope_parameters`, older loaders and
        # conversion scripts (llama.cpp) read `rope_theta`.
        "rope_theta": args.rope_theta,
        "rope_parameters": {"rope_type": "default", "rope_theta": args.rope_theta},
        "tie_word_embeddings": args.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,  # dropout is a training-time regulariser; off for inference
        "bos_token_id": None,  # documents start after <eos>, there is no separate BOS token
        "eos_token_id": EOS_ID,
        "pad_token_id": None,
        "torch_dtype": "float32",
    }


def args_from_hf_config(cfg: dict[str, Any]) -> ModelArgs:
    rope = cfg.get("rope_parameters") or {}
    return ModelArgs(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["max_position_embeddings"],
        n_layers=cfg["num_hidden_layers"],
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg["num_key_value_heads"],
        ffn_hidden=cfg["intermediate_size"],
        dropout=0.0,
        tie_embeddings=cfg["tie_word_embeddings"],
        rope_theta=rope.get("rope_theta", cfg.get("rope_theta", 10_000.0)),
        norm_eps=cfg["rms_norm_eps"],
    )


def export_state_dict(model: CausalLM) -> dict[str, torch.Tensor]:
    """The weights as safetensors wants them: fp32, contiguous, and no tensor stored twice.

    With tied embeddings `lm_head.weight` *is* `model.embed_tokens.weight`. safetensors refuses
    to save two names for one storage (it can't express aliasing), and Hugging Face re-ties the
    head on load when `tie_word_embeddings` is true, so the head is simply left out.
    """
    state = {
        k: v.detach().float().cpu().contiguous().clone() for k, v in model.state_dict().items()
    }
    if model.args.tie_embeddings:
        del state["lm_head.weight"]
    return state


def generation_config(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Default sampling settings: what `slm eval` used, in Hugging Face's field names."""
    return {
        "do_sample": True,
        "temperature": eval_cfg.get("temperature", 0.8),
        # 0 and 1.0 are Hugging Face's "off" values; its own default top_k is 50, not off.
        "top_k": eval_cfg.get("top_k") or 0,
        "top_p": eval_cfg.get("top_p") or 1.0,
        "max_new_tokens": eval_cfg.get("max_new_tokens", 600),
        "eos_token_id": EOS_ID,
    }


def tokenizer_config(block_size: int) -> dict[str, Any]:
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": UNK,
        "eos_token": EOS,
        "model_max_length": block_size,
        # HF's default "clean up" removes spaces before punctuation, which would edit ABC text.
        "clean_up_tokenization_spaces": False,
    }


# ---------------------------------------------------------------------------- loading


@dataclass
class ExportedModel:
    path: Path
    name: str
    version: int
    manifest: dict[str, Any]
    tokenizer: Tokenizer
    model: CausalLM
    generation: dict[str, Any]  # generation_config.json

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


def load_model_files(path: Path, device: torch.device | None = None) -> CausalLM:
    """Rebuild slmkit's model from an export's config.json and safetensors, nothing else."""
    args = args_from_hf_config(json.loads((path / "config.json").read_text()))
    state = load_file(str(path / WEIGHTS))
    if args.tie_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model = CausalLM(args)
    model.load_state_dict(state, strict=True)
    return model.to(device or torch.device("cpu")).eval()


def load_export(ref: str, device: torch.device | None = None) -> ExportedModel:
    name, version = parse_ref(ref)
    path = model_dir(name, version)
    if not path.is_dir():
        raise ConfigError(f"no exported model {ref} (looked in {path}); see `slm models list`")
    return ExportedModel(
        path=path,
        name=name,
        version=version,
        manifest=artifacts.read_manifest(path),
        tokenizer=load_tokenizer(path),
        model=load_model_files(path, device),
        generation=json.loads((path / "generation_config.json").read_text()),
    )


def list_exports() -> list[dict[str, Any]]:
    root = models_root()
    return [
        artifacts.read_manifest(m.parent) for m in sorted(root.glob(f"*/*/{artifacts.MANIFEST}"))
    ]


# ---------------------------------------------------------------------------- parity


@torch.no_grad()
def _parity_ids(run: LoadedRun, prompt: str) -> list[int]:
    """The example prompt plus a greedy continuation: realistic text, deterministic."""
    idx = torch.tensor([run.tokenizer.encode(prompt) or [EOS_ID]])
    out = generate(run.model, idx.to(next(run.model.parameters()).device), PARITY_TOKENS,
                   temperature=0.0)  # fmt: skip
    ids: list[int] = out[0].tolist()
    return ids[: run.model.args.block_size]


@torch.no_grad()
def check_parity(path: Path, run: LoadedRun, prompt: str) -> dict[str, Any]:
    """Compare the export against the checkpoint it came from. Raises if any check fails."""
    reference = run.model.float().cpu().eval()
    ids = _parity_ids(run, prompt)
    x = torch.tensor([ids])
    want = reference(x)[0]

    # 1. slmkit reading the exported files: the same numbers, not merely close ones.
    ours = load_model_files(path)(x)[0]
    result: dict[str, Any] = {
        "tokens": len(ids),
        "slmkit_max_abs_diff": float((ours - want).abs().max()),
    }
    if result["slmkit_max_abs_diff"] != 0.0:
        raise RuntimeError(f"export round trip changed the logits: {result}")

    # 2. Hugging Face transformers, through its Auto* loaders, as a user of the export would.
    try:
        import transformers
    except ImportError:
        result["transformers"] = "not installed (uv sync --extra export): check skipped"
        return result
    text = run.tokenizer.decode(ids)
    transformers.utils.logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    hf_tok = transformers.AutoTokenizer.from_pretrained(str(path))
    hf_ids = hf_tok(text, add_special_tokens=False)["input_ids"]
    result["transformers"] = transformers.__version__
    result["tokenizer_ids_match"] = hf_ids == run.tokenizer.encode(text)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    hf_model.train(False)  # eval mode; `.eval()` is untyped in transformers' stubs
    got = hf_model(x).logits
    result["hf_max_abs_diff"] = float((got - want).abs().max())
    if not result["tokenizer_ids_match"] or not result["hf_max_abs_diff"] <= PARITY_ATOL:
        raise RuntimeError(f"transformers disagrees with slmkit on this export: {result}")
    return result


# ---------------------------------------------------------------------------- export


def example_prompt(project_prompts: list[EvalPrompt]) -> str:
    return project_prompts[0].prompt if project_prompts else "\n"


def export_run(
    run_prefix: str,
    name: str,
    version: int,
    *,
    which: str = "best",
    log: Callable[[str], None] = print,
) -> Path:
    """Write `models/<name>/<version>/` from a run's checkpoint. Returns the directory."""
    if version < 1:
        raise ConfigError("--version must be 1 or more")
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ConfigError(f"model name {name!r}: use letters, digits, '-' and '_' only")
    run = load_run(run_prefix, which, torch.device("cpu"))
    dest = model_dir(name, version)
    if dest.exists():
        old = artifacts.read_manifest(dest)
        if old["inputs"]["run"] == run.run_id and old["stats"]["step"] == run.state["step"]:
            log(f"{name}:{version} already exported from {run.run_id} step {run.state['step']}")
            return dest
        raise FileExistsError(
            f"{name}:{version} already exists (from {old['inputs']['run']} step "
            f"{old['stats']['step']}); versions are immutable, use --version {version + 1}"
        )

    cfg = run.config
    project = load_project(cfg["project"]["name"], cfg["project"]["args"], artifacts.slm_home())
    sft_prompts = project.sft_eval_prompts("val") if run.stage == "sft" else None
    prompts = list(sft_prompts if sft_prompts is not None else project.eval_prompts("val"))
    run_manifest = artifacts.read_manifest(run.run_dir)
    status = read_status(run.run_dir) or {}
    params = count_parameters(run.model)
    # The card reports the eval that matches how the model is meant to be prompted.
    report = latest_report(run.run_dir, "sft" if run.stage == "sft" else "headers")
    stats: dict[str, Any] = {
        "stage": run.stage,
        "checkpoint": run.checkpoint,
        "step": run.state["step"],
        "tokens_seen": run.state.get("tokens_seen"),
        "val_loss": run.state.get("last_val"),  # this checkpoint's validation loss
        "gpu_hours": round(status.get("gpu_seconds", 0) / 3600, 4),
        "params": params.total,
        "params_non_embedding": params.non_embedding,
        "vocab_size": run.tokenizer.vocab_size,
        "block_size": run.model.args.block_size,
        "example_prompt": example_prompt(prompts),
        "eval": (
            {
                "eval_id": report["eval_id"],
                "prompts_kind": report.get("prompts_kind", "headers"),
                "seeds": report["settings"]["seeds"],
                "num_samples": report["settings"]["num_samples"],
                "aggregate": report["model"]["aggregate"],
            }
            if report
            else None
        ),
    }
    inputs = {"run": run.run_id, "tokenizer": run_manifest["inputs"]["tokenizer"]}
    manifest = artifacts.build_manifest(
        kind="model",
        artifact=f"{name}:{version}",
        project=cfg["project"]["name"],
        code_version=CODE_VERSION,
        config={
            "name": name,
            "version": version,
            "which": which,
            "project": cfg["project"],
            "run_name": status.get("name", cfg["run"]["name"]),
        },
        inputs=inputs,
        stats=stats,
    )
    with artifacts.commit_dir(dest, manifest) as tmp:
        (tmp / "config.json").write_text(json.dumps(hf_config(run.model.args), indent=2) + "\n")
        (tmp / "generation_config.json").write_text(
            json.dumps(generation_config(cfg.get("eval", {})), indent=2) + "\n"
        )
        save_file(export_state_dict(run.model), str(tmp / WEIGHTS), metadata={"format": "pt"})
        (tmp / WEIGHTS).chmod(0o644)  # safetensors creates it owner-only; match the other files
        run.tokenizer.to_hf().save(str(tmp / "tokenizer.json"))
        (tmp / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config(run.model.args.block_size), indent=2) + "\n"
        )
        log(f"checking parity on {tmp.name} ...")
        stats["parity"] = check_parity(tmp, run, stats["example_prompt"])
        stats["weights_bytes"] = (tmp / WEIGHTS).stat().st_size
        viewer = project.web_viewer()
        if viewer is not None:  # the project's own display for `slm serve`'s page (ADR 0008)
            shutil.copytree(viewer, tmp / "ui")
            stats["viewer"] = sorted(p.name for p in (tmp / "ui").iterdir())
        (tmp / CARD).write_text(model_card(manifest))
    p = stats["parity"]
    log(
        f"  slmkit round trip: max |Δlogit| = {p['slmkit_max_abs_diff']:.1e} over {p['tokens']} tokens"
    )
    if "hf_max_abs_diff" in p:
        log(f"  transformers {p['transformers']}: max |Δlogit| = {p['hf_max_abs_diff']:.1e}, "
            f"tokenizer IDs match: {p['tokenizer_ids_match']}")  # fmt: skip
    else:
        log(f"  transformers: {p['transformers']}")
    log(f"exported {name}:{version} -> {dest}")
    return dest
