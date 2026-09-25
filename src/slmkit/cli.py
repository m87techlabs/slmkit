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
runs_app = typer.Typer(help="Inspect training runs.")
app.add_typer(runs_app, name="runs")

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
            if isinstance(value, dict):  # e.g. raw file names: list a few, count the rest
                names = list(value)
                value = ", ".join(names[:3]) + (
                    f", ... ({len(names)} total)" if len(names) > 3 else ""
                )
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
        f"at {PLANNING_TFLOPS:.0f} TFLOPS (the trainer measures the real rate)"
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


# --------------------------------------------------------------------------- training


def _train(experiment: str, set_: list[str] | None, device: str, max_minutes: float | None,
           max_steps: int | None) -> None:  # fmt: skip
    from slmkit.train.guards import TrainingDiverged
    from slmkit.train.trainer import Trainer

    exp = load_experiment(experiment, set_)
    typer.echo(f"{exp.address}:")
    pipeline.ensure_packed(exp, typer.echo)
    trainer = Trainer(exp, device=device)
    try:
        trainer.run(max_minutes=max_minutes, max_steps=max_steps)
    except TrainingDiverged as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=3) from None


DEVICE = typer.Option("auto", help="cuda, cpu, or auto (cuda when available).")
MAX_MINUTES = typer.Option(
    None, "--max-minutes", help="Checkpoint and stop after this long. Resume with the same command."
)
MAX_STEPS = typer.Option(None, "--max-steps", help="Checkpoint and stop after this many steps.")


@app.command()
@_friendly_errors
def pretrain(
    experiment: str = EXPERIMENT,
    set_: list[str] | None = SET,
    device: str = DEVICE,
    max_minutes: float | None = MAX_MINUTES,
    max_steps: int | None = MAX_STEPS,
) -> None:
    """Train from scratch, or resume automatically if the run already has a checkpoint.

    Ctrl-C finishes the current step, saves a checkpoint and exits. Stopping is normal.
    """
    _train(experiment, set_, device, max_minutes, max_steps)


@app.command()
@_friendly_errors
def run(
    experiment: str = EXPERIMENT,
    set_: list[str] | None = SET,
    device: str = DEVICE,
    max_minutes: float | None = MAX_MINUTES,
    max_steps: int | None = MAX_STEPS,
) -> None:
    """The whole pipeline: build missing data artifacts, then pretrain (resuming if possible)."""
    _train(experiment, set_, device, max_minutes, max_steps)


def _ago(iso: str | None) -> str:
    import datetime as dt

    if not iso:
        return "-"
    seconds = (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(iso)).total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds / size:.0f}{unit} ago"
    return "just now"


def _short(n: float) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(int(n))


@runs_app.command("list")
@_friendly_errors
def runs_list() -> None:
    """Every run: progress, GPU-hours used and left, best validation loss, last checkpoint."""
    from slmkit.train.run import list_runs

    rows = list_runs()
    if not rows:
        typer.echo("no runs yet under $SLM_HOME/runs")
        return
    header = (f"{'RUN':<17} {'NAME':<18} {'TOKENS':<16} {'DONE':>6} {'GPU-h':>6} {'LEFT':>6} "
              f"{'BEST VAL':>8}  {'LAST CHECKPOINT':<22} STATUS")  # fmt: skip
    typer.echo(header)
    for _, st in rows:
        # The last step can overshoot the budget by part of a step: show 100%, not 100.1%.
        done = min(1.0, st["tokens_seen"] / st["max_tokens"])
        left = st.get("gpu_hours_remaining")
        left = None if left is None else max(0.0, left)
        ck = st.get("last_checkpoint") or {}
        ckpt = f"step {ck['step']} · {_ago(ck.get('time'))}" if ck else "none yet"
        best = st.get("best_val_loss")
        status = "complete" if st["complete"] else (st.get("note") or "resumable")
        typer.echo(
            f"{st['run_id']:<17} {st['name'][:18]:<18} "
            f"{_short(st['tokens_seen']) + '/' + _short(st['max_tokens']):<16} {done:>6.1%} "
            f"{st['gpu_seconds'] / 3600:>6.2f} {'-' if left is None else f'{left:.2f}':>6} "
            f"{'-' if best is None else f'{best:.4f}':>8}  {ckpt:<22} {status}"
        )


@app.command()
@_friendly_errors
def sample(
    run_id: str = typer.Argument(..., help="A run ID (or unique prefix) from `slm runs list`."),
    prompt: str = typer.Option("\n", help="Text to continue. Default: a newline."),
    tokens: int = typer.Option(500, help="How many tokens to generate."),
    temperature: float = typer.Option(0.8, help="0 = always the most likely token."),
    top_k: int | None = typer.Option(None, help="Only sample from the k most likely tokens."),
    top_p: float | None = typer.Option(None, help="Nucleus sampling threshold."),
    which: str = typer.Option("best", help="best (lowest val loss) or latest checkpoint."),
    seed: int = typer.Option(0, help="Seed for reproducible samples."),
    device: str = DEVICE,
) -> None:
    """Generate text from a trained (or partly trained) run."""
    import torch

    from slmkit.config.schema import ModelConfig
    from slmkit.model import CausalLM, ModelArgs
    from slmkit.sampling import generate
    from slmkit.tokenizers import EOS_ID
    from slmkit.train import checkpoint
    from slmkit.train.run import load_run_config, runs_root
    from slmkit.train.trainer import pick_device

    matches = sorted(p for p in runs_root().glob(f"{run_id}*") if p.is_dir())
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise ConfigError(f"run {run_id!r} matches {len(matches)} runs ({found})")
    run_dir = matches[0]
    cfg = load_run_config(run_dir)
    manifest = artifacts.read_manifest(run_dir)
    tokenizer = load_tokenizer(artifacts.find_artifact(manifest["inputs"]["tokenizer"]))
    ckpt_dir = run_dir / checkpoint.CKPT_DIR / checkpoint.BEST
    if which == "latest" or not ckpt_dir.is_dir():
        latest = checkpoint.latest(run_dir)
        if latest is None:
            raise ConfigError(f"{run_dir.name} has no checkpoint yet")
        ckpt_dir = latest.path

    dev = pick_device(device)
    args = ModelArgs.from_config(
        ModelConfig.model_validate(cfg["model"]), tokenizer.vocab_size, cfg["data"]["block_size"]
    )
    model = CausalLM(args).to(dev)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=dev, weights_only=True))
    state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    typer.echo(f"# {run_dir.name} · {ckpt_dir.name} · step {state['step']} · "
               f"val loss {state.get('last_val') or float('nan'):.4f}")  # fmt: skip
    prompt = prompt.encode().decode("unicode_escape")  # allow "\n" on the command line
    idx = torch.tensor([tokenizer.encode(prompt)], device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = generate(model, idx, tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                       stop_token=EOS_ID, generator=gen)  # fmt: skip
    typer.echo(tokenizer.decode(out[0].tolist()))
