"""An export loads in Hugging Face `transformers` and computes the same logits (DESIGN §6.9's
acceptance test). Needs the `export` extra; skipped without it."""

from __future__ import annotations

import torch

from slmkit.export.hf import PARITY_ATOL, export_run
from slmkit.inference import load_run

import pytest  # isort: skip

transformers = pytest.importorskip("transformers")


def test_transformers_matches_slmkit_on_the_export(toy_run: str) -> None:
    path = export_run(toy_run, "toy-model", 1, log=lambda _: None)
    run = load_run(toy_run, "best", torch.device("cpu"))
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    hf.train(False)
    tok = transformers.AutoTokenizer.from_pretrained(str(path))

    text = "abc 1 2"
    assert tok(text, add_special_tokens=False)["input_ids"] == run.tokenizer.encode(text)
    torch.manual_seed(0)
    ids = torch.randint(0, run.tokenizer.vocab_size, (2, run.model.args.block_size))
    with torch.no_grad():
        torch.testing.assert_close(hf(ids).logits, run.model(ids)[0], atol=PARITY_ATOL, rtol=0)
