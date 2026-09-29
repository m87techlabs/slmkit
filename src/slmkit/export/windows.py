"""`slm export --to-windows`: generated samples a person can open on the Windows side.

The one place slmkit writes under /mnt/ (CONTRIBUTING.md rule 1). It writes a handful of small
files, once, for a person to look at or listen to; nothing reads them back, so 9P's slowness
doesn't matter. This is a convenience copy, not an artifact: the model itself stays in
$SLM_HOME, and the folder is rewritten on every call.

What a sample becomes on disk is the project's decision (`Project.render_sample`, ADR 0007):
ABC music writes `.abc` and `.mid`, a text project the plain text.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from slmkit import artifacts
from slmkit.export.hf import CARD, ExportedModel
from slmkit.project_api import EvalPrompt, Project
from slmkit.sampling import generate
from slmkit.tokenizers import EOS_ID

DEFAULT_WINDOWS_DIR = Path("/mnt/c/Users/Public/Music/slmkit")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


@torch.no_grad()
def sample_prompts(
    exported: ExportedModel,
    prompts: list[EvalPrompt],
    per_prompt: int,
    device: torch.device,
) -> list[tuple[EvalPrompt, int, str, bool]]:
    """`per_prompt` samples per prompt, seeds 0..n-1, with the export's generation defaults.
    Returns (prompt, seed, prompt + completion, ended with <eos>)."""
    g = exported.generation
    model = exported.model.to(device)
    out = []
    for p in prompts:
        ids = exported.tokenizer.encode(p.prompt) or [EOS_ID]
        for seed in range(per_prompt):
            gen = torch.Generator(device=device).manual_seed(seed)
            row = generate(model, torch.tensor([ids], device=device), g["max_new_tokens"],
                           temperature=g["temperature"], top_k=g["top_k"] or None,
                           top_p=g["top_p"] if g["top_p"] < 1.0 else None,
                           stop_token=EOS_ID, generator=gen)[0, len(ids):].tolist()  # fmt: skip
            ended = EOS_ID in row
            text = p.prompt + exported.tokenizer.decode(row[: row.index(EOS_ID)] if ended else row)
            out.append((p, seed, text, ended))
    return out


def to_windows(
    exported: ExportedModel,
    project: Project,
    dest_root: Path,
    *,
    per_prompt: int,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> Path:
    stage = exported.manifest["stats"]["stage"]
    sft = project.sft_eval_prompts("val") if stage == "sft" else None
    prompts = list(sft if sft is not None else project.eval_prompts("val"))
    graders = project.graders()

    dest = artifacts.guard_path(dest_root, allow_windows=True) / _safe(
        f"{exported.name}-v{exported.version}"
    )
    if dest.exists():
        shutil.rmtree(dest)  # a convenience copy, regenerated each time (see module docstring)
    dest.mkdir(parents=True)
    shutil.copyfile(exported.path / CARD, dest / CARD)

    intro = (
        f"{per_prompt} per prompt, sampling seeds 0-{per_prompt - 1}, generation settings from "
        "`generation_config.json`. Scores from the project's graders."
    )
    index = [f"# Samples from {exported.ref}", "", intro, ""]
    header_done = False
    for prompt, seed, text, ended in sample_prompts(exported, prompts, per_prompt, device):
        scores: dict[str, float] = {"ended": float(ended)}
        for grader in graders:
            scores.update(grader(prompt, text))
        if not header_done:
            index += ["| sample | files | " + " | ".join(scores) + " |",
                      "|---|---|" + "---|" * len(scores)]  # fmt: skip
            header_done = True
        stem = _safe(f"{prompt.id}-{seed}")
        files = project.render_sample(prompt, text)
        for ext, data in files.items():
            (dest / f"{stem}{ext}").write_bytes(data)
        cells = " | ".join(f"{v:.2f}" for v in scores.values())
        index.append(f"| {stem} | {', '.join(stem + e for e in files)} | {cells} |")
        log(f"  {stem:<16} {' '.join(files):<10} " +
            " ".join(f"{k}={v:.2f}" for k, v in scores.items()))  # fmt: skip
    (dest / "SAMPLES.md").write_text("\n".join(index) + "\n")
    return dest
