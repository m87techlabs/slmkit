"""The `slm` command-line interface.

Each pipeline stage is a subcommand that reads artifacts and writes one new one (see
docs/DESIGN.md 6.2). Stages build any missing upstream artifacts and skip ones that already
exist, so running `slm pack` on a fresh machine does ingest -> prepare -> tokenize -> pack.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import typer

from slmkit import artifacts, pipeline
from slmkit import doctor as _doctor
from slmkit.config import ConfigError, dump_resolved, load_experiment
from slmkit.registry import ProjectNotFound
from slmkit.tokenizers import load_tokenizer

app = typer.Typer(
    add_completion=False,
    help="Build small language models from scratch on a single consumer GPU.",
)

EXPERIMENT = typer.Argument(..., help="<project>/<experiment>, e.g. shakespeare_char/ref")
SET = typer.Option(None, "--set", help="Override a config value: --set train.lr=6e-4 (repeatable)")


def _friendly_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Show configuration and path mistakes as one clear line, not a traceback."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except (ConfigError, ProjectNotFound, artifacts.ArtifactPathError) as exc:
            typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None

    return wrapper


@app.callback()
def _main() -> None:
    """Typer collapses a single-command app into the root command; this callback
    keeps subcommands addressable as `slm <command>` as the CLI grows."""


@app.command()
def doctor(
    bench: bool = typer.Option(
        False, "--bench", help="Also measure achieved bf16 TFLOPS (takes a few seconds)."
    ),
) -> None:
    """Check the environment. Must pass before any training command."""
    raise typer.Exit(code=_doctor.main(bench=bench))


@app.command()
@_friendly_errors
def config(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Print the fully resolved config: presets merged, overrides applied."""
    exp = load_experiment(experiment, set_)
    typer.echo(f"# {exp.address}  ({exp.path})")
    typer.echo(f"# config hash: {exp.config_hash()[:12]}")
    typer.echo(dump_resolved(exp.config), nl=False)


def _stage(
    experiment: str, set_: list[str] | None, fn: Callable[..., pipeline.StageResult]
) -> None:
    exp = load_experiment(experiment, set_)
    typer.echo(f"{exp.address}:")
    fn(exp, typer.echo)


@app.command()
@_friendly_errors
def ingest(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Download the project's source data into $SLM_HOME/raw/<project>/."""
    _stage(experiment, set_, pipeline.ensure_raw)


@app.command()
@_friendly_errors
def prepare(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Split documents by group into train/val (augmenting train only)."""
    _stage(experiment, set_, pipeline.ensure_dataset)


@app.command()
@_friendly_errors
def tokenize(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Fit the tokenizer on the train split."""
    _stage(experiment, set_, pipeline.ensure_tokenizer)


@app.command()
@_friendly_errors
def pack(experiment: str = EXPERIMENT, set_: list[str] | None = SET) -> None:
    """Encode both splits into flat uint16 files for training."""
    _stage(experiment, set_, pipeline.ensure_packed)


@app.command()
@_friendly_errors
def lineage(
    artifact: str = typer.Argument(..., help="An artifact ID, e.g. pk-3f9a1c2b4d5e"),
) -> None:
    """Trace an artifact back to its raw data through the manifests."""
    try:
        rows = artifacts.lineage(artifact)
    except FileNotFoundError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    for depth, via, m, seen in rows:
        pad = "    " * depth
        label = f"{via}: " if via else ""
        if seen:
            typer.echo(f"{pad}{label}{m['id']}  (shown above)")
            continue
        dirty = " (dirty)" if m.get("git_dirty") else ""
        typer.echo(
            f"{pad}{label}{m['id']}  [{m['kind']}]  "
            f"created {m['created']}  git {m.get('git_sha')}{dirty}"
        )
        for key, value in m.get("stats", {}).items():
            if isinstance(value, dict):
                value = ", ".join(value)  # e.g. raw file names
            typer.echo(f"{pad}    {key} = {value}")


@app.command()
@_friendly_errors
def model(
    experiment: str = EXPERIMENT,
    set_: list[str] | None = SET,
    batches: int = typer.Option(4, help="Validation batches for the initial-loss sanity check."),
) -> None:
    """Describe the model an experiment builds: shape, parameters, compute, memory.

    Builds the data artifacts if missing (the vocabulary size comes from the tokenizer), then
    instantiates the model on the CPU and measures its loss before any training.
    """
    import math

    import numpy as np
    import torch

    from slmkit.data.pack import open_packed
    from slmkit.data.sampler import RandomWindowSampler
    from slmkit.model import (
        CKPT_BYTES_PER_PARAM,
        PLANNING_TFLOPS,
        TRAIN_BYTES_PER_PARAM,
        CausalLM,
        ModelArgs,
        count_parameters,
        estimated_gpu_hours,
        flops_per_token,
    )

    exp = load_experiment(experiment, set_)
    cfg = exp.config
    packed = pipeline.ensure_packed(exp)
    tok = pipeline.ensure_tokenizer(exp)
    vocab = load_tokenizer(tok.path).vocab_size
    args = ModelArgs.from_config(cfg.model, vocab, cfg.data.block_size)

    torch.manual_seed(cfg.run.seed)
    net = CausalLM(args).eval()
    c = count_parameters(net)
    fpt = flops_per_token(net)
    run_flops = cfg.train.max_tokens * fpt

    sampler = RandomWindowSampler(
        open_packed(packed.path / "val.bin"), args.block_size, cfg.train.batch_size, seed=0
    )
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = (torch.from_numpy(a) for a in sampler.next_batch())
            _, loss = net(x, y)
            assert loss is not None
            losses.append(loss.item())

    mb = 1024**2
    tied = "shared with the output head" if args.tie_embeddings else "untied"
    echo = typer.echo
    echo(f"{exp.address}   model preset `{cfg.model.preset}`")
    echo(f"  vocab          {vocab} (from {tok.id})  ·  context {args.block_size} tokens")
    echo(
        f"  architecture   {args.n_layers} layers · d_model {args.d_model} · "
        f"{args.n_heads} heads x {args.head_dim} · kv heads {args.n_kv_heads} · "
        f"ffn {args.ffn_hidden} · dropout {args.dropout}"
    )
    echo(f"  parameters     {c.total:>12,} total")
    echo(
        f"                 {c.non_embedding:>12,} non-embedding   (per layer {c.per_layer:,} = "
        f"attention {c.attention_per_layer:,} + mlp {c.mlp_per_layer:,} "
        f"+ norms {c.norms_per_layer:,})"
    )
    echo(f"                 {c.embedding:>12,} embedding       ({vocab} x {args.d_model}, {tied})")
    echo(f"  compute        {fpt / 1e6:,.1f} MFLOPs per training token")
    echo(
        f"                 this run: {cfg.train.max_tokens:,} tokens = {run_flops:.2e} FLOPs "
        f"~ {estimated_gpu_hours(cfg.train.max_tokens, fpt) * 60:,.1f} GPU-min "
        f"at {PLANNING_TFLOPS:.0f} TFLOPS (small models run 2-3x slower than this)"
    )
    echo(
        f"  memory         weights {c.total * 4 / mb:,.1f} MiB fp32 · training state "
        f"~{c.total * TRAIN_BYTES_PER_PARAM / mb:,.0f} MiB + activations · "
        f"checkpoint ~{c.total * CKPT_BYTES_PER_PARAM / mb:,.0f} MiB"
    )
    echo(
        f"  sanity check   loss before training: {np.mean(losses):.3f} on {batches} val batches"
        f"   (a uniform guess scores ln {vocab} = {math.log(vocab):.3f})"
    )
