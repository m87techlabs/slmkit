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
models_app = typer.Typer(help="Inspect exported models.")
app.add_typer(models_app, name="models")

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
    artifact: str = typer.Argument(
        ..., help="An artifact ID (pk-3f9a1c2b4d5e) or an exported model (abc-folk:1)"
    ),
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
            elif isinstance(value, str) and "\n" in value:
                value = repr(value)  # e.g. a model's example prompt: keep it on one line
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
           max_steps: int | None, stages: tuple[str, ...] = ("pretrain",)) -> list[str] | None:  # fmt: skip
    """Run the given stages in order; a stage that stops early ends the session there.

    Returns the run IDs of the stages, or None if one stopped before completing."""
    from slmkit.train.guards import TrainingDiverged
    from slmkit.train.stage import sft_stage
    from slmkit.train.trainer import Trainer

    exp = load_experiment(experiment, set_)
    typer.echo(f"{exp.address}:")
    pipeline.ensure_packed(exp, typer.echo)
    done = []
    for kind in stages:
        stage = sft_stage(exp) if kind == "sft" else None
        trainer = Trainer(exp, stage=stage, device=device)
        try:
            outcome = trainer.run(max_minutes=max_minutes, max_steps=max_steps)
        except TrainingDiverged as exc:
            typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=3) from None
        if outcome != "complete":
            return None
        done.append(trainer.run_id)
    return done


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
    eval_: bool = typer.Option(True, "--eval/--no-eval", help="Evaluate each trained stage."),
) -> None:
    """The whole pipeline: build missing data artifacts, pretrain, SFT if `sft.enabled`, then
    evaluate each stage.

    Every stage resumes if it has a checkpoint and is skipped if already complete; an existing
    eval report is reused. So re-running the same command after any interruption continues
    where it stopped, which is what makes a sweep a plain loop over `slm run`.
    """
    exp = load_experiment(experiment, set_)
    stages = ("pretrain", "sft") if exp.config.sft.enabled else ("pretrain",)
    run_ids = _train(experiment, set_, device, max_minutes, max_steps, stages)
    if eval_ and run_ids:
        for rid in run_ids:
            typer.echo("")
            _evaluate(rid, device=device)


@app.command()
@_friendly_errors
def sft(
    experiment: str = EXPERIMENT,
    set_: list[str] | None = SET,
    device: str = DEVICE,
    max_minutes: float | None = MAX_MINUTES,
    max_steps: int | None = MAX_STEPS,
) -> None:
    """Fine-tune the experiment's pretrained model on request -> answer pairs (resumable).

    Starts from the pretrained run's best checkpoint; the loss counts only the answers.
    """
    _train(experiment, set_, device, max_minutes, max_steps, ("sft",))


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

    from slmkit.inference import load_run
    from slmkit.sampling import generate
    from slmkit.tokenizers import EOS_ID
    from slmkit.train.trainer import pick_device

    dev = pick_device(device)
    run = load_run(run_id, which, dev)
    typer.echo(f"# {run.run_id} · {run.checkpoint} · step {run.state['step']} · "
               f"val loss {run.state.get('last_val') or float('nan'):.4f}")  # fmt: skip
    prompt = prompt.encode().decode("unicode_escape")  # allow "\n" on the command line
    idx = torch.tensor([run.tokenizer.encode(prompt)], device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = generate(run.model, idx, tokens, temperature=temperature, top_k=top_k,
                       top_p=top_p, stop_token=EOS_ID, generator=gen)  # fmt: skip
    typer.echo(run.tokenizer.decode(out[0].tolist()))


@app.command("eval")
@_friendly_errors
def eval_(
    run_id: str = typer.Argument(..., help="A run ID (or unique prefix) from `slm runs list`."),
    seeds: str = typer.Option(None, help="Sampling seeds, e.g. 0,1,2. Default: eval.seeds."),
    samples: int | None = typer.Option(None, help="Samples per seed. Default: eval.num_samples."),
    which: str = typer.Option("best", help="best (lowest val loss) or latest checkpoint."),
    baseline: bool = typer.Option(True, help="Also score the no-model frequency baseline."),
    prompts: str = typer.Option(
        "auto", help="headers, sft (plain-language requests), or auto (sft for fine-tuned runs)."
    ),
    device: str = DEVICE,
) -> None:
    """Generate samples and grade them: every project grader plus novelty, mean ± spread."""
    _evaluate(run_id, seeds=seeds, samples=samples, which=which, baseline=baseline,
              prompts=prompts, device=device)  # fmt: skip


def _evaluate(run_id: str, *, seeds: str | None = None, samples: int | None = None,
              which: str = "best", baseline: bool = True, prompts: str = "auto",
              device: str = "auto") -> None:  # fmt: skip
    from slmkit.config.schema import EvalConfig
    from slmkit.eval.runner import EvalSettings, evaluate, fmt
    from slmkit.inference import load_run
    from slmkit.train.trainer import pick_device

    dev = pick_device(device)
    run = load_run(run_id, which, dev)
    ev = EvalConfig.model_validate(run.config.get("eval", {}))
    settings = EvalSettings(
        seeds=tuple(int(x) for x in seeds.split(",")) if seeds else tuple(ev.seeds),
        num_samples=samples or ev.num_samples,
        temperature=ev.temperature,
        top_k=ev.top_k,
        top_p=ev.top_p,
        max_new_tokens=ev.max_new_tokens,
    )
    typer.echo(f"{run.run_id} ({run.config['run']['name']}) · {run.checkpoint} · step "
               f"{run.state['step']} · {len(settings.seeds)} seeds × {settings.num_samples} samples")  # fmt: skip
    report, path = evaluate(
        run, settings, dev, baseline=baseline, prompts_kind=prompts, log=typer.echo
    )
    has_base = "baseline" in report
    typer.echo(
        f"\n{'metric':<16} {'model':>17}"
        + (f"  {'baseline (token frequencies)':>30}" if has_base else "")
    )
    for metric, stat in report["model"]["aggregate"].items():
        line = f"{metric:<16} {fmt(stat):>17}"
        if has_base:
            line += f"  {fmt(report['baseline']['aggregate'][metric]):>30}"
        typer.echo(line)
    typer.echo(f"\nreport: {path}")


RUN_IDS = typer.Argument(..., help="Two or more run IDs (or unique prefixes).")


@runs_app.command("compare")
@_friendly_errors
def runs_compare(
    run_ids: list[str] = RUN_IDS,
    prompts: str = typer.Option(
        "latest", help="Which eval report to show: headers, sft, or latest (any kind)."
    ),
) -> None:
    """Side by side: config, training result and latest eval report of each run."""
    import json

    from slmkit.eval.runner import fmt, latest_report
    from slmkit.inference import resolve_run
    from slmkit.train.run import load_run_config, read_status

    cols: list[dict[str, str]] = []
    metrics: list[str] = []
    for rid in run_ids:
        run_dir = resolve_run(rid)
        cfg, st = load_run_config(run_dir), read_status(run_dir) or {}
        tok = cfg["tokenizer"]
        records = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
        evals = [r for r in records if r["kind"] == "eval"]
        best = min(evals, key=lambda r: r["val_loss"]) if evals else {}
        col = {
            "run": run_dir.name[:12],
            "name": st.get("name", cfg["run"]["name"]),
            "stage": st.get("stage", "pretrain"),
            "model": cfg["model"]["preset"],
            "tokenizer": tok["type"] + (f" {tok['vocab_size']}" if tok.get("vocab_size") else ""),
            "seed": str(cfg["run"]["seed"]),
            "tokens": _short(st.get("tokens_seen", 0)),
            "GPU-h": f"{st.get('gpu_seconds', 0) / 3600:.2f}",
            "best val loss": f"{best['val_loss']:.4f}" if best else "-",
            # bpc is absent for runs trained before it existed, and None for SFT (masked loss).
            "best val bpc": f"{best['val_bpc']:.3f}" if best.get("val_bpc") is not None else "-",
        }
        report = latest_report(run_dir, None if prompts == "latest" else prompts)
        if report:
            kind = report.get("prompts_kind", "headers")
            col["eval"] = f"{report['eval_id']} ({kind}, {len(report['settings']['seeds'])} seeds)"
            for m, stat in report["model"]["aggregate"].items():
                col[m] = fmt(stat)
                if m not in metrics:
                    metrics.append(m)
        cols.append(col)
    rows = ["run", "name", "stage", "model", "tokenizer", "seed", "tokens", "GPU-h", "best val loss",
            "best val bpc", "eval", *metrics]  # fmt: skip
    width = max(18, *(len(v) for c in cols for v in c.values()))
    for row in rows:
        typer.echo(f"{row:<16}" + "".join(f"  {c.get(row, '-'):>{width}}" for c in cols))


# ---------------------------------------------------------------------------- export & serve


@app.command()
@_friendly_errors
def export(
    run_id: str = typer.Argument(..., help="A run ID (or unique prefix) from `slm runs list`."),
    name: str = typer.Option(..., help="Model name, e.g. abc-folk."),
    version: int = typer.Option(..., help="Model version: 1, 2, ... Versions are immutable."),
    which: str = typer.Option("best", help="best (lowest val loss) or latest checkpoint."),
    to_windows: bool = typer.Option(
        False, "--to-windows", help="Also write generated samples where Windows can open them."
    ),
    windows_dir: str = typer.Option(
        "/mnt/c/Users/Public/Music/slmkit", help="Where --to-windows writes (a folder per model)."
    ),
    samples: int = typer.Option(2, help="--to-windows: samples per prompt."),
    device: str = DEVICE,
) -> None:
    """Write a run's checkpoint as a Hugging Face-format model under $SLM_HOME/models/."""
    from pathlib import Path

    from slmkit.export.hf import export_run, load_export
    from slmkit.registry import load_project
    from slmkit.train.trainer import pick_device

    try:
        export_run(run_id, name, version, which=which, log=typer.echo)
    except FileExistsError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    if not to_windows:
        return
    from slmkit.export.windows import to_windows as write_windows

    dev = pick_device(device)
    exported = load_export(f"{name}:{version}", dev)
    proj = exported.manifest["config"]["project"]
    project = load_project(proj["name"], proj["args"], artifacts.slm_home())
    typer.echo(f"sampling {samples} per prompt on {dev} ...")
    dest = write_windows(exported, project, Path(windows_dir), per_prompt=samples, device=dev,
                         log=typer.echo)  # fmt: skip
    typer.echo(f"wrote {dest}")
    if dest.as_posix().startswith("/mnt/"):
        drive, rest = dest.parts[2], dest.parts[3:]
        typer.echo("on Windows: " + drive.upper() + ":\\" + "\\".join(rest))


@models_app.command("list")
def models_list() -> None:
    """Every exported model: source run, stage, size and parity result."""
    from slmkit.export.hf import list_exports

    rows = list_exports()
    if not rows:
        typer.echo("no exported models yet under $SLM_HOME/models")
        return
    typer.echo(f"{'MODEL':<20} {'STAGE':<9} {'PARAMS':>9} {'RUN':<17} {'STEP':>6} "
               f"{'HF PARITY':>10}  CREATED")  # fmt: skip
    for m in rows:
        s = m["stats"]
        hf = s.get("parity", {}).get("hf_max_abs_diff")
        typer.echo(
            f"{m['id']:<20} {s['stage']:<9} {s['params']:>9,} {m['inputs']['run']:<17} "
            f"{s['step']:>6} {'-' if hf is None else f'{hf:.1e}':>10}  {m['created']}"
        )


@app.command()
@_friendly_errors
def serve(
    model: str = typer.Option(..., "--model", help="An exported model, <name>:<version>."),
    host: str = typer.Option("127.0.0.1", help="Bind address. Keep it local; see DESIGN 6.9."),
    port: int = typer.Option(8000, help="TCP port."),
    device: str = typer.Option("cpu", help="cpu (default: these models are tiny), cuda, auto."),
    threads: int = typer.Option(
        2, min=1, help="CPU threads per generation. More than 2 gains nothing at these sizes."
    ),
    ui: str | None = typer.Option(
        None, help="Serve this viewer directory instead of the export's ui/ (viewer development)."
    ),
) -> None:
    """Serve an exported model over HTTP: a playground page at /, and /health, /info, POST /generate."""
    import torch
    import uvicorn

    from slmkit.export.hf import load_export
    from slmkit.serve.app import create_app
    from slmkit.train.trainer import pick_device

    dev = pick_device(device)
    # PyTorch's CPU default is one worker per core, and every Python thread that calls it gets its
    # own set. The server generates on worker threads, so two sets compete for the same cores:
    # measured 3.4 s instead of 0.58 s for one tune on 6 cores (serving.md §6).
    torch.set_num_threads(threads)
    exported = load_export(model, dev)
    s = exported.manifest["stats"]
    typer.echo(f"{exported.ref}: {s['params']:,} params, {s['stage']}, from "
               f"{exported.manifest['inputs']['run']} step {s['step']} · device {dev}")  # fmt: skip
    typer.echo(f"playground: http://{host}:{port}/   (API docs: http://{host}:{port}/docs)")
    from pathlib import Path

    app_ = create_app(exported, dev, Path(ui) if ui else None)
    uvicorn.run(app_, host=host, port=port, log_level="info")


@runs_app.command("summary")
@_friendly_errors
def runs_summary(
    project: str | None = typer.Option(None, help="Only this project's experiments."),
    prompts: str = typer.Option(
        "auto", help="Eval reports to use: headers, sft, or auto (sft for fine-tuned runs)."
    ),
) -> None:
    """Every experiment, averaged over its training seeds: mean ± spread across runs."""
    from slmkit.eval.runner import fmt
    from slmkit.eval.summary import summarize

    groups = summarize(prompts, project)
    if not groups:
        typer.echo("no complete runs yet under $SLM_HOME/runs")
        return
    cols: list[dict[str, str]] = []
    metrics: list[str] = []
    for g in groups:
        col = {
            "experiment": g.experiment,
            "stage": g.stage,
            "runs": f"{len(g.run_ids)} ({g.evaluated} evaluated)",
            "seeds": ",".join(str(s) for s in sorted(g.seeds)),
            "GPU-h": f"{g.gpu_hours:.2f}",
            "eval prompts": g.eval_kind or "-",
        }
        for m, stat in g.stats.items():
            # One run has no spread to show; "± 0.000" would claim a certainty it doesn't have.
            col[m] = fmt(stat) if len(g.run_ids) > 1 else f"{stat['mean']:.3f} (1 run)"
            if m not in metrics:
                metrics.append(m)
        cols.append(col)
    rows = ["experiment", "stage", "runs", "seeds", "GPU-h", "eval prompts", *metrics]
    width = max(18, *(len(v) for c in cols for v in c.values()))
    for row in rows:
        typer.echo(f"{row:<14}" + "".join(f"  {c.get(row, '-'):>{width}}" for c in cols))
