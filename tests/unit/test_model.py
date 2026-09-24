"""The model: shapes, exact parameter counts, causality, initialisation, and overfitting one batch."""

from __future__ import annotations

import math

import pytest
import torch

from slmkit.config import load_experiment
from slmkit.model import IGNORE_INDEX, CausalLM, ModelArgs, count_parameters, flops_per_token

V = 67  # Shakespeare's vocabulary: 65 characters + <unk> + <eos>


def tiny_args(**kw: object) -> ModelArgs:
    base: dict = {
        "vocab_size": V, "block_size": 32, "n_layers": 2, "d_model": 64,
        "n_heads": 4, "n_kv_heads": 4, "ffn_hidden": 192, "dropout": 0.0,
    }  # fmt: skip
    base.update(kw)
    return ModelArgs(**base)


def preset_model(preset: str, device: str = "cpu") -> CausalLM:
    cfg = load_experiment("shakespeare_char/ref", [f"model.preset={preset}"]).config
    with torch.device(device):
        return CausalLM(ModelArgs.from_config(cfg.model, V, 256))


# Exact non-embedding counts from docs/MODEL.md section 4. If a preset or the architecture
# changes, this test and that table must change together.
@pytest.mark.parametrize(
    ("preset", "non_embedding"),
    [
        ("nano", 853_120),
        ("micro", 4_820_224),
        ("ref", 10_621_824),
        ("tiny", 25_698_816),
        ("small", 84_953_856),
        ("medium", 303_612_928),
    ],
)
def test_preset_parameter_counts_match_the_spec(preset: str, non_embedding: int) -> None:
    # The meta device builds the module structure without allocating memory, so even
    # `medium` (1.2 GB of fp32 weights) is counted instantly.
    counts = count_parameters(preset_model(preset, device="meta"))
    assert counts.non_embedding == non_embedding
    assert counts.total == non_embedding + V * preset_model(preset, "meta").args.d_model


def test_ref_worked_example() -> None:
    c = count_parameters(preset_model("ref", "meta"))
    assert (c.attention_per_layer, c.mlp_per_layer, c.norms_per_layer) == (589_824, 1_179_648, 768)
    assert c.per_layer == 1_770_240


def test_output_shapes_and_loss() -> None:
    model = CausalLM(tiny_args())
    idx = torch.randint(0, V, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, V)
    assert loss is not None and loss.ndim == 0
    assert model(idx)[1] is None


def test_tied_embeddings_share_one_tensor() -> None:
    tied = CausalLM(tiny_args())
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    untied = CausalLM(tiny_args(tie_embeddings=False))
    assert count_parameters(untied).total == count_parameters(tied).total + V * 64


def test_parameter_names_match_hugging_face_llama() -> None:
    keys = set(CausalLM(tiny_args(n_layers=1)).state_dict())
    layer = "model.layers.0."
    assert keys == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        layer + "input_layernorm.weight",
        layer + "post_attention_layernorm.weight",
        *(layer + f"self_attn.{p}_proj.weight" for p in "qkvo"),
        *(layer + f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")),
    }  # no RoPE tables: they are recomputed, never saved


def test_causal_future_tokens_cannot_change_the_past() -> None:
    torch.manual_seed(0)
    model = CausalLM(tiny_args()).eval()
    a = torch.randint(0, V, (1, 20))
    b = a.clone()
    b[0, 12:] = (b[0, 12:] + 1) % V  # change only positions 12 onwards
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    torch.testing.assert_close(la[0, :12], lb[0, :12])  # the past is untouched
    assert not torch.allclose(la[0, 12:], lb[0, 12:])  # the future did change


def test_initial_loss_is_near_uniform() -> None:
    # A fresh model should be almost equally unsure of every token: loss ~ ln(V). Much higher
    # means the initialisation is too large; that is the first thing to check if a run starts
    # with a loss like 10 instead of ~4.2.
    torch.manual_seed(0)
    model = CausalLM(tiny_args())
    idx = torch.randint(0, V, (8, 32))
    _, loss = model(idx, torch.randint(0, V, (8, 32)))
    assert loss is not None
    assert abs(loss.item() - math.log(V)) < 0.3


def test_residual_projections_start_smaller() -> None:
    torch.manual_seed(0)
    model = CausalLM(tiny_args(n_layers=8, d_model=256, n_heads=4, n_kv_heads=4, ffn_hidden=704))
    layer = model.model.layers[0]
    q_std = layer.self_attn.q_proj.weight.std().item()
    o_std = layer.self_attn.o_proj.weight.std().item()
    assert q_std == pytest.approx(0.02, rel=0.05)
    assert o_std == pytest.approx(0.02 / math.sqrt(16), rel=0.05)


def test_ignore_index_is_excluded_from_the_loss() -> None:
    torch.manual_seed(0)
    model = CausalLM(tiny_args())
    idx = torch.randint(0, V, (2, 10))
    targets = torch.randint(0, V, (2, 10))
    masked = targets.clone()
    masked[:, :5] = IGNORE_INDEX
    logits, loss_masked = model(idx, masked)
    manual = torch.nn.functional.cross_entropy(
        logits[:, 5:].reshape(-1, V), targets[:, 5:].reshape(-1)
    )
    assert loss_masked is not None
    torch.testing.assert_close(loss_masked, manual)


def test_grouped_query_attention_runs_and_is_smaller() -> None:
    mha = CausalLM(tiny_args())
    gqa = CausalLM(tiny_args(n_kv_heads=2))
    assert gqa(torch.randint(0, V, (1, 8)))[0].shape == (1, 8, V)
    assert count_parameters(gqa).total < count_parameters(mha).total


def test_rejects_sequences_longer_than_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        CausalLM(tiny_args(block_size=8))(torch.zeros(1, 9, dtype=torch.long))


def test_flops_per_token() -> None:
    model = preset_model("ref", "meta")
    # 6 x (non-embedding + head) + 12 x layers x d_model x context
    assert flops_per_token(model) == 6 * (10_621_824 + V * 384) + 12 * 6 * 384 * 256


def test_overfits_one_batch() -> None:
    """The single most useful trainer test: a model that cannot memorise one batch is broken.

    If this fails, suspect (in order) the target shift, the causal mask, the loss, or the
    optimiser wiring, not the data or the hyperparameters.
    """
    torch.manual_seed(0)
    model = CausalLM(tiny_args())
    idx = torch.randint(0, V, (4, 32))
    targets = torch.roll(idx, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = model(idx, targets)
        assert loss is not None
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 4.0
    assert loss.item() < 0.05
