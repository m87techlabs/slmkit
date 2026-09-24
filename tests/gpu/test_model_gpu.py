"""The model on the GPU: bf16 autocast and torch.compile both work and agree with eager fp32.

Run with `make test-gpu`. Takes ~1 minute, mostly torch.compile's first compilation.
"""

from __future__ import annotations

import math

import pytest
import torch

from slmkit.model import CausalLM, ModelArgs

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

REF = ModelArgs(
    vocab_size=67, block_size=256, n_layers=6, d_model=384,
    n_heads=6, n_kv_heads=6, ffn_hidden=1024, dropout=0.0,
)  # fmt: skip


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(0, REF.vocab_size, (16, REF.block_size), device="cuda", generator=g)
    return idx, torch.roll(idx, -1, dims=1)


def test_bf16_autocast_matches_fp32() -> None:
    torch.manual_seed(0)
    model = CausalLM(REF).cuda().eval()
    idx, targets = _batch()
    with torch.no_grad():
        _, fp32 = model(idx, targets)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, bf16 = model(idx, targets)
    assert fp32 is not None and bf16 is not None
    assert abs(fp32.item() - math.log(REF.vocab_size)) < 0.3
    # bf16 keeps ~3 significant digits; the loss must agree to about that.
    assert abs(bf16.item() - fp32.item()) < 0.02


def test_backward_under_autocast_gives_finite_gradients() -> None:
    torch.manual_seed(0)
    model = CausalLM(REF).cuda()
    idx, targets = _batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # Master weights stay fp32 under autocast; only the matmuls run in bf16.
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_compiled_model_matches_eager() -> None:
    torch.manual_seed(0)
    model = CausalLM(REF).cuda().eval()
    compiled = torch.compile(model)
    idx, _ = _batch()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_logits = model(idx)[0].float()
        compiled_logits = compiled(idx)[0].float()
    torch.testing.assert_close(compiled_logits, eager_logits, atol=5e-2, rtol=5e-2)
