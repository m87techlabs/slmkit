"""What the Parameters page computes: a model's size, cost and memory from its architecture, and
how that compares with what this machine actually measured.

The estimates use the same functions the trainer does (`parameters_from_args`,
`flops_per_token_from_args`, docs/MODEL.md §4–5), so the page and the startup log can't disagree.
The *measured* side comes from runs on disk with the same architecture: their tokens per second, and
from that the TFLOPS and MFU this GPU achieved at that size.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from slmkit.model import ModelArgs
from slmkit.model.stats import PLANNING_TFLOPS, flops_per_token_from_args, parameters_from_args
from slmkit.studio.data import index

ARCH = ("n_layers", "d_model", "n_heads", "n_kv_heads", "ffn_hidden", "tie_embeddings")


def presets(repo: Path) -> list[dict[str, Any]]:
    out = []
    for path in (repo / "presets" / "model").glob("*.yaml"):
        text = path.read_text()
        values = yaml.safe_load(text)
        comment = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), ""
        )
        out.append(
            {"name": path.stem, "comment": comment, **{k: values[k] for k in ARCH if k in values}}
        )
    return sorted(out, key=lambda p: (p["d_model"], p["n_layers"]))


def validate(args: ModelArgs) -> None:
    """The constraints the model code assumes, as one clear message instead of a stack trace."""
    if args.d_model % args.n_heads:
        raise ValueError(
            f"d_model ({args.d_model}) must divide evenly into n_heads ({args.n_heads}) heads"
        )
    if args.n_heads % args.n_kv_heads:
        raise ValueError(
            f"n_heads ({args.n_heads}) must be a multiple of n_kv_heads ({args.n_kv_heads})"
        )
    if (args.d_model // args.n_heads) % 2:
        raise ValueError(
            "the head size (d_model / n_heads) must be even, for rotary position embeddings"
        )


def _peak_tflops(home: Path) -> float | None:
    try:
        return float(json.loads((home / "doctor.json").read_text())["bf16_tflops"])
    except (OSError, KeyError, ValueError):
        return None


def measured(home: Path, args: ModelArgs) -> list[dict[str, Any]]:
    """Complete runs whose architecture matches `args`, with their measured throughput."""
    rows = []
    for a in index(home).values():
        if a.kind != "run":
            continue
        try:
            cfg = yaml.safe_load((a.path / "config.resolved.yaml").read_text())
            st = json.loads((a.path / "status.json").read_text())
        except (OSError, ValueError):
            continue
        model = cfg.get("model", {})
        if any(model.get(k) != getattr(args, k) for k in ARCH) or not st.get("tok_per_s"):
            continue
        rows.append({"run_id": a.id, "experiment": st.get("experiment"), "stage": st.get("stage", "pretrain"),
                     "block_size": cfg["data"]["block_size"], "tok_per_s": st["tok_per_s"],
                     "tokens_seen": st.get("tokens_seen", 0), "gpu_seconds": st.get("gpu_seconds", 0)})  # fmt: skip
    return rows


def estimate(home: Path, args: ModelArgs, tokens: int) -> dict[str, Any]:
    validate(args)
    total = parameters_from_args(args)
    d, hd, v = args.d_model, args.head_dim, args.vocab_size
    attention = 2 * d * (args.n_heads * hd) + 2 * d * (args.n_kv_heads * hd)
    mlp = 3 * d * args.ffn_hidden
    norms = 2 * d
    fpt = flops_per_token_from_args(args)
    train_flops = fpt * tokens
    peak = _peak_tflops(home)
    runs = [r for r in measured(home, args) if r["stage"] == "pretrain"]
    tok_s = statistics.median(r["tok_per_s"] for r in runs) if runs else None
    achieved = tok_s * fpt / 1e12 if tok_s else None
    # Whole runs: GPU time includes evaluations, sampling and compilation, not just training steps.
    whole = [
        r["tokens_seen"] / r["gpu_seconds"] for r in runs if r["gpu_seconds"] and r["tokens_seen"]
    ]
    whole_rate = statistics.median(whole) if whole else None
    return {
        "params": {
            "total": total,
            "embedding": v * d,
            "head": 0 if args.tie_embeddings else v * d,
            "attention": args.n_layers * attention,
            "mlp": args.n_layers * mlp,
            "norms": args.n_layers * norms + d,
            "per_layer": attention + mlp + norms,
        },
        "flops_per_token": fpt,
        "training_flops": train_flops,
        "tokens": tokens,
        "tokens_per_param": tokens / total,
        "planning": {
            "tflops": PLANNING_TFLOPS,
            "gpu_hours": train_flops / (PLANNING_TFLOPS * 1e12) / 3600,
        },
        "measured": {
            "runs": len(runs),
            "tok_per_s": tok_s,
            "tflops": achieved,
            "mfu": achieved / peak if achieved and peak else None,
            "peak_tflops": peak,
            "gpu_hours": tokens / tok_s / 3600 if tok_s else None,
            "gpu_hours_whole_run": tokens / whole_rate / 3600 if whole_rate else None,
            "examples": runs[:5],
        },
        "memory_bytes": {
            "weights_fp32": 4 * total,
            "training_state": 16 * total,  # fp32 weights + grads + AdamW's two moments
            "checkpoint": 12 * total,  # weights + two moments; gradients aren't saved
            "export": 4 * total,
        },
    }
