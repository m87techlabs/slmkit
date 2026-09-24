"""Pydantic schemas for an experiment's resolved configuration.

Every model uses `extra="forbid"`: a typo such as `leraning_rate` fails at load time with the
exact key named, instead of being silently ignored and discovered after an hour of training.
This is the same guarantee Terraform gives for an undeclared variable.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Strict):
    name: str
    seed: int = 1337
    tracker: Literal["tensorboard", "wandb", "none"] = "tensorboard"


class ProjectRef(_Strict):
    name: str
    # Validated later against the project's own `Args` schema. The engine cannot know
    # the shape of project arguments, so it keeps them opaque here.
    args: dict[str, Any] = Field(default_factory=dict)


class TokenizerConfig(_Strict):
    type: Literal["char", "bpe", "fixed"] = "char"
    vocab_size: int | None = None  # bpe only; char and fixed derive it from the data


class DataConfig(_Strict):
    block_size: int = Field(256, gt=0)
    val_fraction: float = Field(0.05, gt=0.0, lt=1.0)
    # Changing the run seed must not re-split the data, so the split has its own salt.
    split_seed: int = 0
    # Append an end-of-document token after every document when packing. Projects whose
    # documents are independent (tunes, games) need it so the model learns where one ends.
    # A single continuous text cut into blocks (Shakespeare) does not.
    append_eos: bool = True


class ModelConfig(_Strict):
    """The architecture. See docs/MODEL.md for what each field means."""

    preset: str
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    ffn_hidden: int = Field(gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    @model_validator(mode="after")
    def _check_shapes(self) -> ModelConfig:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads={self.n_heads} is not divisible by n_kv_heads={self.n_kv_heads}"
            )
        if (self.d_model // self.n_heads) % 2:
            # RoPE rotates dimensions in pairs, so each head needs an even width.
            raise ValueError(f"head_dim={self.d_model // self.n_heads} must be even for RoPE")
        return self


class TrainConfig(_Strict):
    """Optimisation settings. Schedules are in tokens, not steps (DESIGN 6.6)."""

    preset: str
    max_tokens: int = Field(gt=0)
    optimizer: Literal["adamw", "adamw8bit"] = "adamw"
    lr: float = Field(gt=0.0)
    min_lr_ratio: float = Field(0.1, ge=0.0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(0.1, ge=0.0)
    grad_clip: float = Field(1.0, gt=0.0)
    warmup_tokens: int = Field(0, ge=0)
    batch_size: int = Field(gt=0)
    grad_accum: int = Field(1, gt=0)
    compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    eval_every_steps: int = Field(250, gt=0)
    eval_iters: int = Field(50, gt=0)  # batches averaged per val-loss estimate
    eval_samples: int = Field(8, gt=0)  # generated samples printed at every eval
    sample_tokens: int = Field(200, gt=0)  # length of each of those samples
    log_every_steps: int = Field(50, gt=0)
    ckpt_every_minutes: float = Field(20.0, gt=0.0)
    keep_last_n_ckpts: int = Field(3, gt=0)
    max_consecutive_skips: int = Field(20, gt=0)
    epochs: int | None = None  # SFT presets only


class SFTConfig(_Strict):
    enabled: bool = False
    lr: float | None = None
    epochs: int | None = None


class EvalConfig(_Strict):
    num_samples: int = Field(200, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    temperature: float = Field(0.8, gt=0.0)
    top_k: int | None = None
    top_p: float | None = None


class ExperimentConfig(_Strict):
    """The fully resolved experiment: presets merged, overrides applied."""

    run: RunConfig
    project: ProjectRef
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig
    train: TrainConfig
    sft: SFTConfig = Field(default_factory=SFTConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
