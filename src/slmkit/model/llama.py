"""A Llama-style decoder-only transformer, written out in full.

This is the one model family slmkit trains (CONTRIBUTING.md rule 5). docs/MODEL.md is its
specification and docs/concepts/the-model.md walks through this file. Every module and
parameter name matches Hugging Face's `LlamaForCausalLM`, so a state dict from here loads
there unchanged (tests/unit/test_hf_parity.py proves it, logit for logit).

Shapes in comments use:  B batch, T sequence length, D d_model, H heads, K kv heads,
                         Hd head_dim (= D / H), F ffn_hidden, V vocab_size.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from slmkit.config.schema import ModelConfig
from slmkit.model.init import init_weights

# Targets equal to this are ignored by the loss. SFT (M2) uses it to mask prompt tokens.
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ModelArgs:
    """Everything needed to build the network. `vocab_size` comes from the tokenizer artifact
    and `block_size` from the experiment, never from the preset (docs/MODEL.md section 3)."""

    vocab_size: int
    block_size: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_config(cls, cfg: ModelConfig, vocab_size: int, block_size: int) -> ModelArgs:
        fields = cfg.model_dump(exclude={"preset"})
        return cls(vocab_size=vocab_size, block_size=block_size, **fields)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned per-channel gain.

    Cheaper than LayerNorm (no mean subtraction, no bias) and just as good in practice. The
    arithmetic runs in fp32 even under bf16 autocast: squaring then averaging hundreds of
    values is exactly where low precision loses accuracy.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE: encode position by rotating query/key vectors, so attention scores depend on the
    *distance* between two tokens rather than on their absolute positions.

    Each pair of dimensions (i, i + Hd/2) is rotated by angle `position * theta^(-2i/Hd)`: fast
    rotation in the first dimensions, slow in the last, like the hands of a clock. There are no
    learned parameters; cos/sin tables are precomputed for every position up to block_size.

    Pairs are (i, i + Hd/2), the "rotate_half" convention Hugging Face uses, not Meta's
    original interleaved (2i, 2i + 1). Both are valid; they are not interchangeable, and using
    the other one would make exported models produce different logits.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, head_dim: int, max_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)  # (T, Hd/2)
        angles = torch.cat([angles, angles], dim=-1)  # (T, Hd)
        # Buffers move with .to(device) but are not parameters, and `persistent=False` keeps
        # them out of the state dict: they are recomputed, never learned or saved.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, heads, T, Hd) -> same shape, rotated by position."""
        t = x.shape[-2]
        cos = self.cos[:t].to(x.dtype)
        sin = self.sin[:t].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


class Attention(nn.Module):
    """Causal multi-head self-attention: each position gathers information from itself and
    earlier positions, weighted by how relevant their keys are to its query."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads, args.head_dim
        self.q_proj = nn.Linear(args.d_model, args.n_heads * args.head_dim, bias=False)
        self.k_proj = nn.Linear(args.d_model, args.n_kv_heads * args.head_dim, bias=False)
        self.v_proj = nn.Linear(args.d_model, args.n_kv_heads * args.head_dim, bias=False)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.d_model, bias=False)
        self.attn_dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        b, t, _ = x.shape
        # Project, then split the last dimension into heads: (B, T, H*Hd) -> (B, H, T, Hd).
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Position enters here and only here: on queries and keys, never on values.
        q, k = rope(q), rope(k)

        if self.n_kv_heads != self.n_heads:
            # Grouped-query attention: each key/value head serves several query heads.
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # softmax(q·kᵀ / sqrt(Hd) + causal mask) · v, as one fused kernel. `is_causal=True`
        # hides every later position from every earlier one.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)  # merge heads
        out: torch.Tensor = self.resid_dropout(self.o_proj(y))
        return out


class MLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) ).

    The gate decides, per hidden unit, how much of `up(x)` passes. Three matrices instead of
    a plain MLP's two, so the hidden width is 8/3 x D rather than 4 x D to keep the same
    parameter count.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.d_model, args.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(args.d_model, args.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(args.ffn_hidden, args.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )
        return out


class Block(nn.Module):
    """One transformer layer: attention then MLP, each pre-normed and added to the residual."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(args.d_model, args.norm_eps)
        self.self_attn = Attention(args)
        self.post_attention_layernorm = RMSNorm(args.d_model, args.norm_eps)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        # `x + f(norm(x))`: each sub-layer adds a correction to the residual stream rather
        # than replacing it, which is what lets gradients reach the first layer intact.
        x = x + self.self_attn(self.input_layernorm(x), rope)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Decoder(nn.Module):
    """Embedding, the stack of blocks, and the final norm. HF calls this `model`."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(Block(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.d_model, args.norm_eps)
        self.rope = RotaryEmbedding(args.head_dim, args.block_size, args.rope_theta)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embed_tokens(idx))  # (B, T) -> (B, T, D)
        for layer in self.layers:
            x = layer(x, self.rope)
        out: torch.Tensor = self.norm(x)
        return out


class CausalLM(nn.Module):
    """The full language model: token IDs in, next-token logits (and optionally loss) out."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model = Decoder(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        if args.tie_embeddings:
            # One matrix, two jobs: row i is token i's input vector *and* the direction the
            # output layer scores token i against.
            self.lm_head.weight = self.model.embed_tokens.weight
        init_weights(self, args.n_layers)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token IDs. targets: (B, T) next-token IDs, or IGNORE_INDEX to skip.

        Returns (logits of shape (B, T, V), mean cross-entropy loss or None).
        """
        if idx.shape[1] > self.args.block_size:
            raise ValueError(
                f"sequence of {idx.shape[1]} exceeds block_size {self.args.block_size}"
            )
        logits = self.lm_head(self.model(idx))
        loss = None
        if targets is not None:
            # Loss in fp32 regardless of autocast: log-softmax over the vocabulary is another
            # place bf16 rounding would show up directly in the number being optimised.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @staticmethod
    def expected_initial_loss(vocab_size: int) -> float:
        """A freshly initialised model should be equally unsure of every token: loss = ln(V)."""
        return math.log(vocab_size)
