"""The training loop. See docs/concepts/the-training-loop.md for the walk-through.

One step:
    set the learning rate for `tokens_seen`  ->  forward + backward on a batch (bf16 autocast)
    ->  clip the gradient norm  ->  skip the step if anything is non-finite  ->  AdamW update

Around it:
    every `log_every_steps`  loss, lr, grad norm, tokens/s, GPU-hours remaining
    at step 50               measured throughput, MFU and peak memory, which replace the estimate
    every `eval_every_steps` validation loss **and generated samples**, best checkpoint if improved
    every 20 minutes         a full checkpoint (and on Ctrl-C, at the end, and at a session limit)

Resume is the default path: constructing a Trainer for a run directory that already holds a
checkpoint restores everything in it, including the RNG states and the sampler position.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import signal
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch

from slmkit import artifacts, pipeline
from slmkit.config.load import Experiment
from slmkit.data.pack import count_chars, open_packed
from slmkit.data.sampler import RandomWindowSampler
from slmkit.data.split import read_docs
from slmkit.model import CausalLM, ModelArgs, count_parameters, flops_per_token
from slmkit.model.stats import PLANNING_TFLOPS
from slmkit.project_api import Project
from slmkit.sampling import generate
from slmkit.tokenizers import EOS_ID, Tokenizer, load_tokenizer
from slmkit.tracking import make_tracker
from slmkit.train import checkpoint
from slmkit.train.guards import NonFiniteGuard, TrainingDiverged
from slmkit.train.optim import build_optimizer
from slmkit.train.run import create_run_dir, run_id, write_status
from slmkit.train.schedule import lr_at
from slmkit.train.seed import rng_state, seed_everything, set_rng_state

SPEC_PEAK_TFLOPS = 112.0  # used for MFU only when `slm doctor --bench` has not measured one
THROUGHPUT_WARMUP_STEPS = 10  # excluded from throughput: compilation and allocator warm-up
THROUGHPUT_REPORT_STEP = 50
# One Ctrl-C under `uv run` arrives twice within milliseconds (see _on_signal); a second
# press sooner than this after the first is treated as the same press.
SIGNAL_DEBOUNCE_S = 1.0
EVAL_SEED = 1234  # the same val batches and sample seed at every eval, so evals are comparable


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _peak_tflops() -> tuple[float, str]:
    try:
        data = json.loads((artifacts.slm_home() / "doctor.json").read_text())
        return float(data["bf16_tflops"]), "measured by slm doctor --bench"
    except (OSError, KeyError, ValueError):
        return SPEC_PEAK_TFLOPS, "spec-sheet estimate; run `slm doctor --bench` to measure"


def _fmt_hours(hours: float) -> str:
    return f"{hours * 60:.1f} GPU-min" if hours < 1 else f"{hours:.2f} GPU-h"


class Trainer:
    def __init__(
        self, exp: Experiment, *, device: str = "auto", log: Callable[[str], None] = print
    ) -> None:
        self.exp = exp
        self.cfg = cfg = exp.config
        self.device = pick_device(device)
        self._print = log

        packed = pipeline.ensure_packed(exp)
        tok = pipeline.ensure_tokenizer(exp)
        self.tokenizer: Tokenizer = load_tokenizer(tok.path)
        self.run_id = run_id(exp, packed.id, tok.id)
        self.run_dir = create_run_dir(exp, self.run_id, packed.id, tok.id)
        self._log_file = (self.run_dir / "train.log").open("a", encoding="utf-8")

        self.train_data = open_packed(packed.path / "train.bin")
        self.val_data = open_packed(packed.path / "val.bin")
        # Characters per validation token: turns per-token loss into bits per character, the
        # one number comparable across tokenizers.
        val_docs = read_docs(pipeline.ensure_dataset(exp).path / "val.jsonl")
        self.val_chars_per_token = count_chars(val_docs, append_eos=cfg.data.append_eos) / max(
            1, len(self.val_data)
        )
        self.args = ModelArgs.from_config(cfg.model, self.tokenizer.vocab_size, cfg.data.block_size)
        self.tokens_per_step = cfg.train.batch_size * cfg.data.block_size * cfg.train.grad_accum

        resume = checkpoint.latest(self.run_dir)
        if resume is None:
            seed_everything(cfg.run.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = CausalLM(self.args).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.train, self.device)
        self.sampler = RandomWindowSampler(
            self.train_data, cfg.data.block_size, cfg.train.batch_size, seed=cfg.run.seed
        )
        self.guard = NonFiniteGuard(cfg.train.max_consecutive_skips)
        self.step = 0
        self.tokens_seen = 0
        self.gpu_seconds = 0.0
        self.best_val = math.inf
        self.last_val: float | None = None
        self.measured_tok_s: float | None = None
        self._last_ckpt_info: dict[str, Any] | None = None
        self._saved_at: str | None = None
        self.resumed_from = resume
        if resume is not None:
            self._restore(resume)

        # Compile the forward pass on the GPU. Sampling uses the plain model, because its
        # growing sequence lengths would make the compiled version recompile repeatedly.
        self.forward: Callable[..., Any] = self.model
        if cfg.train.compile and self.device.type == "cuda":
            self.forward = torch.compile(self.model, mode=cfg.train.compile_mode)

        self.tracker = make_tracker(cfg.run.tracker, self.run_dir, self.step if resume else None)
        self.project: Project = pipeline.project_for(exp)
        self._stop_reason: str | None = None
        self._first_signal_at = 0.0

    # ------------------------------------------------------------------ bookkeeping

    def log(self, message: str) -> None:
        self._print(message)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def _metric(self, record: dict[str, Any]) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, "tokens": self.tokens_seen, **record}) + "\n")

    def _truncate_metrics(self) -> None:
        """Drop metrics logged after the checkpoint we resumed from; those steps re-run now."""
        path = self.run_dir / "metrics.jsonl"
        if not path.is_file():
            return
        keep = [ln for ln in path.read_text().splitlines() if json.loads(ln)["step"] <= self.step]
        path.write_text("".join(ln + "\n" for ln in keep))

    def _status(self, *, complete: bool = False, note: str = "") -> None:
        remaining = self.cfg.train.max_tokens - self.tokens_seen
        write_status(
            self.run_dir,
            {
                "run_id": self.run_id,
                "name": self.cfg.run.name,
                "experiment": self.exp.address,
                "complete": complete,
                "note": note,
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "max_tokens": self.cfg.train.max_tokens,
                "gpu_seconds": round(self.gpu_seconds, 1),
                "tok_per_s": self.measured_tok_s,
                "gpu_hours_remaining": (
                    remaining / self.measured_tok_s / 3600 if self.measured_tok_s else None
                ),
                "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
                "last_val_loss": self.last_val,
                "last_checkpoint": self._last_ckpt_info,
            },
        )

    # ------------------------------------------------------------------ checkpoints

    def _state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "gpu_seconds": self.gpu_seconds,
            "best_val": self.best_val,
            "last_val": self.last_val,
            "measured_tok_s": self.measured_tok_s,
            "sampler": self.sampler.state_dict(),
            "rng": rng_state(),
            "skipped_steps": self.guard.total_skipped,
            "config_hash": self.exp.config_hash(),
            "saved": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def save_checkpoint(self) -> Path:
        t0 = time.perf_counter()
        path = checkpoint.save(
            self.run_dir,
            self.step,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self._state(),
            keep_last_n=self.cfg.train.keep_last_n_ckpts,
        )
        self._last_ckpt_info = {"step": self.step, "tokens": self.tokens_seen, "path": path.name,
                                "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
        self.log(
            f"  checkpoint  {path.relative_to(self.run_dir)}  ({time.perf_counter() - t0:.1f}s)"
        )
        return path

    def _restore(self, ckpt: checkpoint.Checkpoint) -> None:
        model_sd, optim_sd, state = checkpoint.load(ckpt.path, self.device)
        self.model.load_state_dict(model_sd)
        self.optimizer.load_state_dict(optim_sd)
        self.sampler.load_state_dict(state["sampler"])
        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        self.gpu_seconds = state["gpu_seconds"]
        self.best_val = state["best_val"]
        self.last_val = state.get("last_val")
        self.measured_tok_s = state.get("measured_tok_s")
        self.guard.total_skipped = state.get("skipped_steps", 0)
        self._saved_at = state.get("saved")
        self._last_ckpt_info = {"step": self.step, "tokens": self.tokens_seen,
                                "path": ckpt.path.name, "time": state.get("saved")}  # fmt: skip
        set_rng_state(state["rng"])  # last: building the model above consumed random numbers
        self._truncate_metrics()

    # ------------------------------------------------------------------ evaluation

    def _autocast(self) -> AbstractContextManager[Any]:
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        )

    def _batch(self, sampler: RandomWindowSampler) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = sampler.next_batch()
        return (
            torch.from_numpy(x).to(self.device, non_blocking=True),
            torch.from_numpy(y).to(self.device, non_blocking=True),
        )

    @torch.no_grad()
    def estimate_loss(self, data: np.ndarray) -> float:
        """Mean loss over `eval_iters` batches, drawn from a fixed seed each time."""
        sampler = RandomWindowSampler(
            data, self.cfg.data.block_size, self.cfg.train.batch_size, seed=EVAL_SEED
        )
        self.model.eval()
        losses = []
        for _ in range(self.cfg.train.eval_iters):
            x, y = self._batch(sampler)
            with self._autocast():
                _, loss = self.forward(x, y)
            assert loss is not None
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    def samples(self) -> list[tuple[str, str]]:
        cfg = self.cfg
        prompts = list(self.project.eval_prompts("val"))[: cfg.train.eval_samples]
        gen = torch.Generator(device=self.device).manual_seed(EVAL_SEED)
        out = []
        for p in prompts:
            idx = torch.tensor([self.tokenizer.encode(p.prompt)], device=self.device)
            with self._autocast():
                ids = generate(
                    self.model, idx, cfg.train.sample_tokens,
                    temperature=cfg.eval.temperature, top_k=cfg.eval.top_k, top_p=cfg.eval.top_p,
                    stop_token=EOS_ID, generator=gen,
                )  # fmt: skip
            out.append((p.id, self.tokenizer.decode(ids[0].tolist())))
        return out

    def evaluate(self) -> None:
        t0 = time.perf_counter()
        val = self.estimate_loss(self.val_data)
        train = self.estimate_loss(self.train_data)
        self.last_val = val
        improved = val < self.best_val
        bpc = val / self.val_chars_per_token / math.log(2)
        self.log(
            f"eval  step {self.step:>6}  train {train:.4f}  val {val:.4f}  "
            f"(gap {val - train:+.4f}, {bpc:.3f} bpc){'  * best' if improved else ''}  "
            f"[{time.perf_counter() - t0:.1f}s]"
        )
        self.tracker.scalar("loss/val", val, self.step)
        self.tracker.scalar("bpc/val", bpc, self.step)
        self.tracker.scalar("loss/train_eval", train, self.step)
        samples = self.samples()
        for name, text in samples:
            self.log(f"── sample @ step {self.step}: {name} " + "─" * 40)
            self.log(text.rstrip())
            self.tracker.text(f"samples/{name}", text, self.step)
        self.log("─" * 72)
        self._metric({"kind": "eval", "val_loss": val, "train_loss": train, "val_bpc": bpc,
                      "samples": dict(samples)})  # fmt: skip
        if improved:
            self.best_val = val
            checkpoint.save_best(
                self.run_dir, self.model.state_dict(), self.optimizer.state_dict(), self._state()
            )

    # ------------------------------------------------------------------ training

    def train_step(self) -> tuple[float, float, float]:
        """One optimizer step. Returns (loss, grad norm before clipping, lr)."""
        cfg = self.cfg.train
        lr = lr_at(self.tokens_seen + self.tokens_per_step, lr=cfg.lr, warmup_tokens=cfg.warmup_tokens,
                   max_tokens=cfg.max_tokens, min_lr_ratio=cfg.min_lr_ratio)  # fmt: skip
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        total = 0.0
        for _ in range(cfg.grad_accum):
            x, y = self._batch(self.sampler)
            with self._autocast():
                _, loss = self.forward(x, y)
            assert loss is not None
            # Gradients add up across micro-batches; dividing makes their sum an average.
            (loss / cfg.grad_accum).backward()
            total += loss.item() / cfg.grad_accum

        # Rescale the whole gradient if its norm exceeds grad_clip: one unlucky batch cannot
        # then take a step many times larger than usual.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip))
        if self.guard.check(total, grad_norm):
            self.optimizer.step()
        else:
            self.log(f"  skipped step {self.step}: loss={total} grad_norm={grad_norm}")
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens_seen += self.tokens_per_step
        return total, grad_norm, lr

    def _banner(self) -> None:
        c = count_parameters(self.model)
        fpt = flops_per_token(self.model)
        remaining = self.cfg.train.max_tokens - self.tokens_seen
        est = remaining * fpt / (PLANNING_TFLOPS * 1e12) / 3600
        self.log(f"run {self.run_id}  ({self.exp.address}, {self.cfg.run.name})")
        self.log(f"  dir         {self.run_dir}")
        self.log(f"  device      {self.device}  bf16={self.device.type == 'cuda'}  "
                 f"compile={self.forward is not self.model}")  # fmt: skip
        self.log(f"  parameters  {c.total:,} ({c.non_embedding:,} non-embedding + "
                 f"{c.embedding:,} embedding)")  # fmt: skip
        self.log(f"  per step    {self.tokens_per_step:,} tokens = {self.cfg.train.batch_size} x "
                 f"{self.cfg.data.block_size}" + (f" x {self.cfg.train.grad_accum} accum"
                 if self.cfg.train.grad_accum > 1 else ""))  # fmt: skip
        self.log(f"  budget      {self.cfg.train.max_tokens:,} tokens "
                 f"= {self.cfg.train.max_tokens * fpt:.2e} FLOPs")  # fmt: skip
        if self.resumed_from is not None:
            done = self.tokens_seen / self.cfg.train.max_tokens
            self.log(f"  RESUMING    from {self.resumed_from.path.name} ({done:.1%} done, "
                     f"saved {self._saved_at}, {self.gpu_seconds / 3600:.2f} GPU-h so far)")  # fmt: skip
        rem = remaining / self.measured_tok_s / 3600 if self.measured_tok_s else est
        src = "measured" if self.measured_tok_s else f"estimate at {PLANNING_TFLOPS:.0f} TFLOPS"
        self.log(f"  remaining   {remaining:,} tokens ~ {_fmt_hours(rem)} ({src})")

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        now = time.monotonic()
        if self._stop_reason is None:
            self._stop_reason = signal.Signals(signum).name
            self._first_signal_at = now
            self.log(f"\n{self._stop_reason}: finishing this step, then checkpointing. "
                     "Press Ctrl-C again to abort without saving.")  # fmt: skip
        elif now - self._first_signal_at > SIGNAL_DEBOUNCE_S:
            raise KeyboardInterrupt
        # else: the duplicate `uv run` forwards. A terminal's Ctrl-C signals the whole process
        # group, so this process gets SIGINT once from the terminal and once more from uv.

    def run(self, *, max_steps: int | None = None, max_minutes: float | None = None) -> str:
        """Train until done or stopped. Returns "complete" or "stopped"."""
        cfg = self.cfg.train
        self._banner()
        if self.tokens_seen >= cfg.max_tokens:
            self.log("run already complete; nothing to do")
            return "complete"

        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in previous:
            signal.signal(s, self._on_signal)
        session_start = time.monotonic()
        gpu_base = self.gpu_seconds
        last_ckpt = time.monotonic()
        window_tokens, window_time = 0, 0.0
        steps_this_session = 0
        try:
            if self.step == 0:
                self.evaluate()  # the untrained baseline: loss ~ ln(V), samples are noise
            while self.tokens_seen < cfg.max_tokens:
                t0 = time.perf_counter()
                loss, grad_norm, lr = self.train_step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                steps_this_session += 1
                self.gpu_seconds = gpu_base + (time.monotonic() - session_start)

                if steps_this_session == THROUGHPUT_WARMUP_STEPS:
                    window_tokens, window_time = 0, 0.0
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                else:
                    window_tokens += self.tokens_per_step
                    window_time += dt

                if steps_this_session == THROUGHPUT_REPORT_STEP and window_time > 0:
                    self._report_throughput(window_tokens / window_time)

                if self.step % cfg.log_every_steps == 0:
                    # Before step 10 of a session the window still includes compilation, so
                    # its rate would be misleadingly low (25K tok/s on a resume, not 830K).
                    warm = steps_this_session > THROUGHPUT_WARMUP_STEPS and window_time > 0
                    tok_s = window_tokens / window_time if warm else 0.0
                    rate = self.measured_tok_s or tok_s
                    speed = f"{tok_s:,.0f} tok/s" if warm else "warming up"
                    eta = (
                        _fmt_hours((cfg.max_tokens - self.tokens_seen) / rate / 3600)
                        if rate
                        else "measuring"
                    )
                    self.log(f"step {self.step:>6}  loss {loss:.4f}  lr {lr:.2e}  "
                             f"gnorm {grad_norm:.2f}  {speed}  "
                             f"{self.tokens_seen / cfg.max_tokens:6.1%}  "
                             f"remaining {eta}")  # fmt: skip
                    self.tracker.scalar("loss/train", loss, self.step)
                    self.tracker.scalar("lr", lr, self.step)
                    self.tracker.scalar("grad_norm", grad_norm, self.step)
                    self.tracker.scalar("tokens_per_s", tok_s, self.step)
                    self._metric({"kind": "train", "loss": loss, "lr": lr,
                                  "grad_norm": grad_norm, "tok_per_s": tok_s})  # fmt: skip
                    self._status()

                if self.step % cfg.eval_every_steps == 0:
                    self.evaluate()

                if time.monotonic() - last_ckpt >= cfg.ckpt_every_minutes * 60:
                    self.save_checkpoint()
                    last_ckpt = time.monotonic()
                    self._status()

                if max_steps is not None and steps_this_session >= max_steps:
                    self._stop_reason = self._stop_reason or f"session limit ({max_steps} steps)"
                if max_minutes is not None and time.monotonic() - session_start >= max_minutes * 60:
                    self._stop_reason = self._stop_reason or f"session limit ({max_minutes} min)"
                if self._stop_reason:
                    break
        except TrainingDiverged:
            self._status(note="diverged")
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)

        if self.tokens_seen >= cfg.max_tokens:
            if self.step % cfg.eval_every_steps:
                self.evaluate()
            self.save_checkpoint()
            self._status(complete=True)
            self.log(f"complete: {self.tokens_seen:,} tokens, best val {self.best_val:.4f}, "
                     f"{self.gpu_seconds / 3600:.2f} GPU-h")  # fmt: skip
            self.close()
            return "complete"

        self.save_checkpoint()
        self._status(note=f"stopped: {self._stop_reason}")
        self.log(f"stopped ({self._stop_reason}) at step {self.step}, "
                 f"{self.tokens_seen / cfg.max_tokens:.1%} done. "
                 f"Run the same command to resume.")  # fmt: skip
        self.close()
        return "stopped"

    def _report_throughput(self, tok_s: float) -> None:
        self.measured_tok_s = tok_s
        fpt = flops_per_token(self.model)
        peak, source = _peak_tflops()
        mfu = tok_s * fpt / (peak * 1e12)
        remaining = (self.cfg.train.max_tokens - self.tokens_seen) / tok_s / 3600
        mem = (
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB peak"
            if self.device.type == "cuda"
            else "n/a on CPU"
        )
        self.log(f"  measured    {tok_s:,.0f} tokens/s  ·  {tok_s * fpt / 1e12:.1f} TFLOPS  ·  "
                 f"MFU {mfu:.1%} of {peak:.1f} ({source})")  # fmt: skip
        self.log(f"              memory {mem}  ·  remaining {_fmt_hours(remaining)} (measured)")
        self.tracker.scalar("mfu", mfu, self.step)

    def close(self) -> None:
        self.tracker.close()
        self._log_file.close()
