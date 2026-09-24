"""slmkit's model and Hugging Face's `LlamaForCausalLM` compute the same function.

Loads slmkit's state dict into the HF implementation with `strict=True` (every name and shape
must match) and compares logits. This is the proof behind docs/MODEL.md's claim that export is
a rename, and it pins the RoPE convention: with the interleaved variant the logits would differ.

Needs the `export` extra (`transformers`); skipped without it.
"""

from __future__ import annotations

import pytest
import torch

from slmkit.model import CausalLM, ModelArgs

transformers = pytest.importorskip("transformers")


def _hf_twin(args: ModelArgs, rope_theta: float | None = None) -> torch.nn.Module:
    config = transformers.LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.d_model,
        intermediate_size=args.ffn_hidden,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        max_position_embeddings=args.block_size,
        rms_norm_eps=args.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta or args.rope_theta},
        tie_word_embeddings=args.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
    )
    return transformers.LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("n_kv_heads", [4, 2])  # plain multi-head, and grouped-query
def test_logits_match_hugging_face(n_kv_heads: int) -> None:
    torch.manual_seed(0)
    args = ModelArgs(
        vocab_size=67, block_size=64, n_layers=2, d_model=128,
        n_heads=4, n_kv_heads=n_kv_heads, ffn_hidden=384,
    )  # fmt: skip
    ours = CausalLM(args).eval()
    hf = _hf_twin(args)
    hf.load_state_dict(ours.state_dict(), strict=True)

    idx = torch.randint(0, args.vocab_size, (3, 64))
    with torch.no_grad():
        torch.testing.assert_close(ours(idx)[0], hf(idx).logits, atol=1e-5, rtol=1e-5)


def test_the_comparison_is_sensitive() -> None:
    """A deliberately different RoPE base must break the match, or the test proves nothing."""
    torch.manual_seed(0)
    args = ModelArgs(
        vocab_size=67, block_size=64, n_layers=2, d_model=128,
        n_heads=4, n_kv_heads=4, ffn_hidden=384,
    )  # fmt: skip
    ours = CausalLM(args).eval()
    hf = _hf_twin(args, rope_theta=500.0)
    hf.load_state_dict(ours.state_dict(), strict=True)
    idx = torch.randint(0, args.vocab_size, (1, 64))
    with torch.no_grad():
        assert (ours(idx)[0] - hf(idx).logits).abs().max() > 1e-3
