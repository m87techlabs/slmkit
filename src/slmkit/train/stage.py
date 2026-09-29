"""Training stages: what the trainer trains on, and where it starts from.

The training loop (trainer.py) is the same for every stage; a Stage supplies the parts that
differ:

| | pretrain | sft |
|---|---|---|
| data | random windows from the packed stream | one prompt + answer per row, prompt masked |
| starts from | random weights | the pretrained run's best checkpoint |
| settings | `train:` in the experiment | `presets/train/sft.yaml`, then the experiment's `sft:` |
| budget | `train.max_tokens` | `epochs` passes over the SFT examples |
| eval prompts | `Project.eval_prompts` | `Project.sft_eval_prompts` (plain-language requests) |
| run ID | config + data | config + `sft:` + the parent run's ID |

Resume, checkpoints, guards, logging and throughput work identically for both, which is the
point: SFT is "the same trainer on different batches", not a second training system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from slmkit import artifacts, pipeline
from slmkit.config.load import ConfigError, Experiment, _load_preset
from slmkit.config.schema import TrainConfig
from slmkit.data.pack import count_chars, open_packed
from slmkit.data.sampler import RandomWindowSampler
from slmkit.data.sft import SFTSampler, encode_examples
from slmkit.data.split import read_docs
from slmkit.project_api import EvalPrompt
from slmkit.tokenizers import Tokenizer, load_tokenizer
from slmkit.train import checkpoint
from slmkit.train.run import RUN_CODE_VERSION, create_run_dir, run_id, runs_root, training_identity


class Sampler(Protocol):
    def next_batch(self) -> tuple[Any, Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class Stage:
    kind: str  # "pretrain" or "sft"
    name: str  # shown in logs and `slm runs list`
    run_id: str
    run_dir: Path
    tokenizer: Tokenizer
    train: TrainConfig  # the effective training settings for this stage
    block_size: int
    make_sampler: Callable[[str, int], Sampler]  # (split, seed) -> sampler
    eval_prompts: list[EvalPrompt]
    val_chars_per_token: float | None  # for bits per character; None if not meaningful
    init_weights: Path | None = None  # model.pt to start from instead of random weights
    parent: str | None = None


def pretrain_stage(exp: Experiment) -> Stage:
    cfg = exp.config
    packed = pipeline.ensure_packed(exp)
    tok = pipeline.ensure_tokenizer(exp)
    rid = run_id(exp, packed.id, tok.id)
    run_dir = create_run_dir(exp, rid, {"packed": packed.id, "tokenizer": tok.id})
    data = {s: open_packed(packed.path / f"{s}.bin") for s in ("train", "val")}
    val_docs = read_docs(pipeline.ensure_dataset(exp).path / "val.jsonl")
    chars = count_chars(val_docs, append_eos=cfg.data.append_eos)
    return Stage(
        kind="pretrain",
        name=cfg.run.name,
        run_id=rid,
        run_dir=run_dir,
        tokenizer=load_tokenizer(tok.path),
        train=cfg.train,
        block_size=cfg.data.block_size,
        make_sampler=lambda split, seed: RandomWindowSampler(
            data[split], cfg.data.block_size, cfg.train.batch_size, seed=seed
        ),
        eval_prompts=list(pipeline.project_for(exp).eval_prompts("val")),
        val_chars_per_token=chars / max(1, len(data["val"])),
    )


def sft_settings(exp: Experiment, n_train_examples: int, block_size: int) -> TrainConfig:
    """The pretraining settings, overridden by presets/train/sft.yaml, then by `sft:`."""
    preset = _load_preset("train", "sft")
    sft = exp.config.sft
    epochs = sft.epochs or preset.get("epochs", 3)
    update: dict[str, Any] = {k: v for k, v in preset.items() if k in TrainConfig.model_fields}
    update["preset"] = "sft"
    update["epochs"] = epochs
    if sft.lr is not None:
        update["lr"] = sft.lr
    batch = update.get("batch_size", exp.config.train.batch_size)
    # Budget in tokens, like pretraining: `epochs` passes over every example, each padded to a
    # full row. Examples are drawn at random, so an "epoch" is a budget, not an exact pass.
    update["max_tokens"] = max(batch * block_size, epochs * n_train_examples * block_size)
    return exp.config.train.model_copy(update=update)


def sft_stage(exp: Experiment) -> Stage:
    cfg = exp.config
    packed = pipeline.ensure_packed(exp)
    tok = pipeline.ensure_tokenizer(exp)
    dataset = pipeline.ensure_dataset(exp)
    parent = run_id(exp, packed.id, tok.id)
    best = runs_root() / parent / checkpoint.CKPT_DIR / checkpoint.BEST / "model.pt"
    if not best.is_file():
        raise ConfigError(
            f"SFT starts from the pretrained run {parent}, which has no best checkpoint yet. "
            f"Run `slm pretrain {exp.address}` first."
        )
    project = pipeline.project_for(exp)
    prompts = project.sft_eval_prompts("val")
    if prompts is None:
        raise ConfigError(f"project {exp.project!r} defines no SFT (sft_eval_prompts)")

    tokenizer = load_tokenizer(tok.path)
    block = cfg.data.block_size
    encoded = {}
    for split in ("train", "val"):
        examples = project.sft_examples(read_docs(dataset.path / f"{split}.jsonl"))
        if examples is None:
            raise ConfigError(f"project {exp.project!r} defines no SFT examples")
        encoded[split], dropped = encode_examples(examples, tokenizer, block)
        if dropped:
            print(f"  sft: {dropped} {split} examples longer than block_size were skipped")
    train_cfg = sft_settings(exp, len(encoded["train"]), block)

    identity = {**training_identity(exp), "sft": cfg.sft.model_dump(mode="json"), "stage": "sft"}
    rid = artifacts.artifact_id(
        "run",
        code_version=RUN_CODE_VERSION,
        config=identity,
        inputs={"parent": parent, "tokenizer": tok.id, "dataset": dataset.id},
    )
    run_dir = create_run_dir(
        exp, rid, {"parent": parent, "tokenizer": tok.id, "dataset": dataset.id},
        stage="sft", train=train_cfg,
    )  # fmt: skip
    return Stage(
        kind="sft",
        name=f"{cfg.run.name}-sft",
        run_id=rid,
        run_dir=run_dir,
        tokenizer=tokenizer,
        train=train_cfg,
        block_size=block,
        make_sampler=lambda split, seed: SFTSampler(
            encoded[split], block, train_cfg.batch_size, seed=seed
        ),
        eval_prompts=list(prompts),
        val_chars_per_token=None,
        init_weights=best,
        parent=parent,
    )
