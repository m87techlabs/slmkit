"""Size, compute and memory figures for a model, used by `slm model` and the trainer's logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import nn

from slmkit.model.llama import Block, CausalLM, ModelArgs

# AdamW in mixed precision: fp32 weights (4) + fp32 gradients (4) + two fp32 moments (4 + 4).
TRAIN_BYTES_PER_PARAM = 16
# What a resume checkpoint stores per parameter: weights + two moments (no gradients).
CKPT_BYTES_PER_PARAM = 12
# Effective training throughput used for estimates before a run has measured its own:
# ~35% of the measured bf16 peak (docs/decisions/0001-stack-versions.md). The trainer replaces
# it with measured tokens/sec after 50 steps.
PLANNING_TFLOPS = 43.0


@dataclass(frozen=True)
class ParamCount:
    total: int  # unique parameters; a tied embedding/head is counted once
    embedding: int
    non_embedding: int  # the number compared across models (docs/MODEL.md section 4)
    per_layer: int
    attention_per_layer: int
    mlp_per_layer: int
    norms_per_layer: int


def _numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_parameters(model: CausalLM) -> ParamCount:
    # .parameters() de-duplicates shared tensors, so tied weights are counted once.
    total = _numel(model)
    embedding = model.model.embed_tokens.weight.numel()
    layer = cast(Block, model.model.layers[0])
    return ParamCount(
        total=total,
        embedding=embedding,
        non_embedding=total - embedding,
        per_layer=_numel(layer),
        attention_per_layer=_numel(layer.self_attn),
        mlp_per_layer=_numel(layer.mlp),
        norms_per_layer=_numel(layer.input_layernorm) + _numel(layer.post_attention_layernorm),
    )


def parameters_from_args(args: ModelArgs) -> int:
    """The same total as `count_parameters`, from the architecture alone (no model built):
    what the studio shows for any run or preset without loading weights."""
    d, hd, kv = args.d_model, args.head_dim, args.n_kv_heads
    attention = 2 * d * (args.n_heads * hd) + 2 * d * (kv * hd)  # q and o; k and v
    per_layer = attention + 3 * d * args.ffn_hidden + 2 * d  # SwiGLU's 3 matrices; 2 norms
    head = 0 if args.tie_embeddings else args.vocab_size * d
    return args.vocab_size * d + args.n_layers * per_layer + d + head


def flops_per_token(model: CausalLM, seq_len: int | None = None) -> int:
    """Training FLOPs per token: forward + backward (docs/MODEL.md section 5).

    The weight matmuls cost 6 FLOPs per parameter per token: 2 in the forward pass (a
    multiply and an add) and 4 in the backward pass (gradients for inputs and for weights).
    The parameter count here *includes* the output head, because the head is a real matmul
    over the vocabulary even when its weights are tied to the embedding; the embedding lookup
    itself is free. Attention adds 12 * n_layers * d_model * seq_len on top (scores and the
    weighted sum, which grow with context length), the same estimate nanoGPT uses for MFU.
    """
    return flops_per_token_from_args(model.args, seq_len)


def flops_per_token_from_args(args: ModelArgs, seq_len: int | None = None) -> int:
    """`flops_per_token` from the architecture alone. The matmul parameters are everything but
    the embedding lookup, plus the output head once: inside the non-embedding count when it has
    its own weights, added here when it shares the embedding's."""
    t = seq_len or args.block_size
    head = args.vocab_size * args.d_model
    non_embedding = parameters_from_args(args) - head  # the embedding table is the first V x D
    matmul_params = non_embedding + (head if args.tie_embeddings else 0)
    return 6 * matmul_params + 12 * args.n_layers * args.d_model * t


def estimated_gpu_hours(tokens: int, flops_per_tok: int, tflops: float = PLANNING_TFLOPS) -> float:
    return tokens * flops_per_tok / (tflops * 1e12) / 3600
