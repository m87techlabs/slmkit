"""Generate the figures in docs/images/ from real runs in $SLM_HOME.

    uv sync --extra docs
    uv run python scripts/make_figures.py

Every chart except the labelled illustration is drawn from a run's `metrics.jsonl` or from
slmkit's own code (the lr schedule), so the pictures in the docs cannot drift from what the
code actually did. Runs are found by name (`run.name` in the experiment YAML).

Style follows one small system: a light chart surface, hairline solid gridlines, 2px lines,
8px markers with a surface-coloured ring, and a fixed series order (train = slot 1 blue,
validation = slot 2 orange) so the same colour always means the same thing on every page.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from slmkit.train.schedule import lr_at

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
SLM_HOME = Path(os.environ.get("SLM_HOME", "~/slm")).expanduser()

# Palette: the reference data-viz palette, light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TRAIN = "#2a78d6"  # slot 1
VAL = "#eb6834"  # slot 2

DPI = 200
PX = 0.72  # one CSS pixel in points, for a 10-inch figure shown ~1000 px wide
LINE = 2 * PX
MARK = 8 * PX

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # ships with matplotlib: same output everywhere
        "font.size": 13 * PX,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1 * PX,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 14 * PX,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1 * PX,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)

UNIFORM = math.log(67)  # 4.205: guessing every character equally
BIGRAM = 2.4526  # best possible loss using only the previous character (the-model.md 8)


def find_run(name: str) -> Path:
    for status in (SLM_HOME / "runs").glob("*/status.json"):
        if json.loads(status.read_text()).get("name") == name:
            return status.parent
    raise SystemExit(f"no run named {name!r} under {SLM_HOME}/runs; train it first")


def evals(run_dir: Path) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["kind"] == "eval"]


def line(ax: plt.Axes, x: list[float], y: list[float], color: str, label: str) -> None:
    ax.plot(x, y, color=color, lw=LINE, solid_capstyle="round", solid_joinstyle="round",
            label=label)  # fmt: skip


def dot(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.plot([x], [y], "o", ms=MARK, color=color, mec=SURFACE, mew=2 * PX, zorder=5)


def reference(ax: plt.Axes, y: float, text: str, x_text: float) -> None:
    ax.axhline(y, color=AXIS, lw=1 * PX, zorder=1)
    ax.text(x_text, y + 0.06, text, color=MUTED, fontsize=11 * PX, ha="right", va="bottom")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote docs/images/{name}")


# --------------------------------------------------------------------------- figures


def fit_expected() -> None:
    """Illustration: what under-, well- and over-fitting look like. Not data."""
    t = [i / 100 for i in range(101)]
    shapes = {
        "Underfitting": (
            [2.6 + 1.6 * math.exp(-6 * x) for x in t],
            [2.65 + 1.6 * math.exp(-6 * x) for x in t],
            "Both losses high and close.\nThe model is too small or trained too little.",
        ),
        "A good fit": (
            [0.9 + 3.3 * math.exp(-7 * x) for x in t],
            [1.25 + 2.95 * math.exp(-7 * x) for x in t],
            "Both fall and level off together.\nA small, stable gap.",
        ),
        "Overfitting": (
            [0.4 + 3.8 * math.exp(-4.5 * x) for x in t],
            [1.25 + 2.95 * math.exp(-9 * x) + 1.1 * max(0, x - 0.3) ** 1.3 for x in t],
            "Train keeps falling, validation turns up.\nThe model is learning its training set.",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (title, (tr, va, note)) in zip(axes, shapes.items(), strict=True):
        line(ax, t, tr, TRAIN, "train loss")
        line(ax, t, va, VAL, "validation loss")
        if title == "Overfitting":
            best = min(range(len(va)), key=va.__getitem__)
            dot(ax, t[best], va[best], VAL)
            ax.annotate("stop here: lowest\nvalidation loss", (t[best], va[best]),
                        xytext=(t[best] + 0.12, va[best] + 1.0), color=INK_2,
                        fontsize=11 * PX, arrowprops={"arrowstyle": "-", "color": MUTED,
                                                      "lw": 1 * PX})  # fmt: skip
        ax.set_title(title)
        ax.text(0.02, 0.02, note, transform=ax.transAxes, color=INK_2, fontsize=11 * PX,
                va="bottom")  # fmt: skip
        ax.set_xticks([])
        ax.set_xlabel("training time →")
        ax.set_ylim(0, 4.5)
    axes[0].set_ylabel("loss (lower is better)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Illustration: the three shapes a loss curve can take", x=0.01, ha="left",
                 color=INK, fontsize=15 * PX, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fit-expected.png")


def fit_actual() -> None:
    """Three real runs on the same data and budget, differing only in model size."""
    runs = [
        ("shakespeare-underfit", "Underfit: 13K parameters", "too small to learn more"),
        ("shakespeare-nano", "Overfits: 0.85M, no dropout", "val turns up after ~20M tokens"),
        ("shakespeare-ref", "Best: 10.6M, dropout 0.2", "lower best; val rises more slowly"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
    for ax, (name, title, note) in zip(axes, runs, strict=True):
        rows = evals(find_run(name))
        x = [r["tokens"] / 1e6 for r in rows]
        line(ax, x, [r["train_loss"] for r in rows], TRAIN, "train loss")
        line(ax, x, [r["val_loss"] for r in rows], VAL, "validation loss")
        best = min(rows, key=lambda r: r["val_loss"])
        dot(ax, best["tokens"] / 1e6, best["val_loss"], VAL)
        # Labels go above the orange line, clear of the blue one beneath it.
        ax.annotate(f"best val {best['val_loss']:.2f}", (best["tokens"] / 1e6, best["val_loss"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=11 * PX)  # fmt: skip
        last = rows[-1]
        if last is not best:
            ax.text(x[-1], last["val_loss"] + 0.1, f"end {last['val_loss']:.2f}", color=INK_2,
                    fontsize=11 * PX, ha="right")  # fmt: skip
        reference(ax, BIGRAM, "bigram baseline 2.45", x[-1])
        ax.set_title(title)
        ax.text(0.98, 0.95, note, transform=ax.transAxes, color=INK_2, fontsize=11 * PX,
                ha="right", va="top")  # fmt: skip
        ax.set_xlabel("tokens seen (millions)")
        ax.set_ylim(0, 4.5)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="lower left")
    fig.suptitle("Measured: three Shakespeare models, same data, same 82M-token budget",
                 x=0.01, ha="left", color=INK, fontsize=15 * PX, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fit-actual.png")


def lr_schedule() -> None:
    kw = {"lr": 1e-3, "warmup_tokens": 1_638_400, "max_tokens": 81_920_000, "min_lr_ratio": 0.1}
    tokens = [i * 16_384 for i in range(1, 5001)]
    lrs = [lr_at(t, **kw) for t in tokens]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    line(ax, [t / 1e6 for t in tokens], lrs, TRAIN, "learning rate")
    for t, text, dy in ((1.6384, "peak 1e-3 after 1.6M-token warmup", 12),
                        (81.92, "floor 1e-4", 12)):  # fmt: skip
        dot(ax, t, lr_at(int(t * 1e6), **kw), TRAIN)
        ax.annotate(text, (t, lr_at(int(t * 1e6), **kw)), xytext=(6 if t < 10 else -6, dy),
                    textcoords="offset points", ha="left" if t < 10 else "right",
                    color=INK, fontsize=11 * PX)  # fmt: skip
    ax.set_title("Learning rate of the `ref` run: linear warmup, then cosine decay (in tokens)")
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("learning rate")
    ax.set_ylim(0, 1.15e-3)
    fig.tight_layout()
    save(fig, "lr-schedule.png")


def mfu_by_preset() -> None:
    """Measured in M1 Phase C (docs/concepts/the-training-loop.md 8)."""
    data = [
        ("nano  0.85M", 23.6),
        ("micro  4.8M", 41.5),
        ("ref  10.6M", 51.2),
        ("tiny  25.7M", 53.6),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    labels = [d[0] for d in data][::-1]
    values = [d[1] for d in data][::-1]
    bars = ax.barh(labels, values, height=0.45, color=TRAIN, zorder=2)
    for bar, v in zip(bars, values, strict=True):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center",
                color=INK, fontsize=12 * PX)  # fmt: skip
    # 43 TFLOPS planning figure / 112.1 measured peak = 38.4%. Drawn behind the bars.
    ax.axvline(38.4, color=MUTED, lw=1 * PX, zorder=1)
    ax.text(38.9, 3.62, "planning figure: 43 TFLOPS = 38.4%", color=INK_2, fontsize=11 * PX,
            ha="left", va="center")  # fmt: skip
    ax.set_ylim(-0.6, 3.95)
    ax.set_xlim(0, 65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MFU: share of the GPU's measured bf16 peak actually used for training")
    ax.set_title("Bigger models keep the GPU busier (RTX 5080, compiled, bf16, 16,384 tokens/step)")
    fig.tight_layout()
    save(fig, "mfu-by-preset.png")


def watch_it_learn() -> None:
    """The `ref` run, one frame per eval: the loss curve so far and what the model writes."""
    rows = evals(find_run("shakespeare-ref"))
    fig = plt.figure(figsize=(10, 4.8), dpi=100)
    ax = fig.add_axes((0.07, 0.13, 0.36, 0.72))
    text_ax = fig.add_axes((0.48, 0.05, 0.5, 0.8))
    text_ax.axis("off")
    xs_all = [r["tokens"] / 1e6 for r in rows]

    def draw(i: int) -> None:
        ax.clear()
        text_ax.clear()
        text_ax.axis("off")
        shown = rows[: i + 1]
        x = xs_all[: i + 1]
        line(ax, x, [r["train_loss"] for r in shown], TRAIN, "train")
        line(ax, x, [r["val_loss"] for r in shown], VAL, "validation")
        dot(ax, x[-1], shown[-1]["val_loss"], VAL)
        reference(ax, BIGRAM, "bigram 2.45", 82)
        ax.set_xlim(0, 84)
        ax.set_ylim(0, 4.6)
        ax.set_xlabel("tokens seen (millions)")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right")
        r = shown[-1]
        fig.suptitle(f"Watching `ref` learn: step {r['step']:,}   ·   train {r['train_loss']:.2f}"
                     f"   ·   validation {r['val_loss']:.2f}", x=0.01, ha="left", color=INK,
                     fontsize=16 * PX, fontweight="bold")  # fmt: skip
        sample = r["samples"].get("ROMEO", "")
        lines = [ln[:58] for ln in sample.splitlines()][:13]
        text_ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace",
                     fontsize=11, color=INK)  # fmt: skip
        text_ax.text(0, 1.04, "what it writes after the prompt \"ROMEO:\"", color=INK_2,
                     fontsize=10, va="bottom")  # fmt: skip

    frames = list(range(len(rows))) + [len(rows) - 1] * 4  # hold the last frame
    anim = FuncAnimation(fig, draw, frames=frames)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / "watch-it-learn.gif", writer=PillowWriter(fps=1.2), dpi=100)
    plt.close(fig)
    print("wrote docs/images/watch-it-learn.gif")


def abc_sweep() -> None:
    """M2 Phase F: every pretraining experiment, one dot per training seed, the mean in blue.

    Read by `slm runs summary`'s own code, so the figure and the table can't disagree. The
    baseline's mean is a vertical line through each panel: the thing every row is compared with.
    """
    from slmkit.eval.summary import summarize

    rows = [("baseline", "baseline\nnano · char · transposed"),
            ("noaug", "noaug\nno transposition"),
            ("bpe512", "bpe512\nBPE-512 tokenizer"),
            ("micro", "micro\n5.7× the parameters"),
            ("micro_noaug", "micro_noaug\nmicro, no transposition")]  # fmt: skip
    groups = {g.experiment.split("/")[1]: g for g in summarize("headers", "abc_music")
              if g.stage == "pretrain"}  # fmt: skip
    rows = [r for r in rows if r[0] in groups]
    panels = [("best val bpc", "bits per character (lower is better)"),
              ("plays", "plays: abc2midi, no errors"),
              ("bar_accuracy", "bar accuracy"),
              ("ends_on_tonic", "ends on the tonic")]  # fmt: skip
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 0.62 * len(rows) + 1.4), sharey=True)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        base = groups["baseline"].stats[metric]["mean"]
        ax.axvline(base, color=AXIS, lw=1 * PX, zorder=1)
        for i, (key, _) in enumerate(rows):
            g = groups[key]
            y = len(rows) - 1 - i
            for r in g.per_run:
                ax.plot([r[metric]], [y], "o", ms=MARK * 0.8, color=MUTED, mec=SURFACE,
                        mew=1.5 * PX, zorder=3, alpha=0.9)  # fmt: skip
            mean = g.stats[metric]["mean"]
            ax.plot([mean], [y], "D", ms=MARK * 1.05, color=TRAIN, mec=SURFACE, mew=2 * PX,
                    zorder=5)  # fmt: skip
            ax.text(mean, y + 0.28, f"{mean:.3f}", color=INK_2, fontsize=10 * PX, ha="center",
                    va="bottom")  # fmt: skip
        ax.set_title(title, loc="left", color=INK, fontsize=12 * PX)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [label for _, label in reversed(rows)])
    fig.suptitle("abc_music, Phase F: 3 training seeds per experiment, 30M tokens each",
                 x=0.01, ha="left", color=INK, fontsize=15 * PX, fontweight="bold")  # fmt: skip
    fig.text(0.01, -0.02, "grey dots: one training seed each (its eval averages 3 sampling seeds × "
             "200 samples) · blue diamond: mean · vertical line: baseline mean",
             color=MUTED, fontsize=10 * PX, ha="left")  # fmt: skip
    fig.tight_layout()
    save(fig, "abc-sweep.png")


if __name__ == "__main__":
    fit_expected()
    fit_actual()
    lr_schedule()
    mfu_by_preset()
    watch_it_learn()
    abc_sweep()
