"""SFT: the prompt mask, the sampler, settings, and fine-tuning end to end on the toy project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from slmkit.config import ConfigError, load_experiment
from slmkit.data.sft import EncodedExample, SFTSampler, encode_examples, make_rows
from slmkit.eval.runner import EvalSettings, evaluate
from slmkit.inference import load_run
from slmkit.model import IGNORE_INDEX, CausalLM, ModelArgs
from slmkit.project_api import SFTExample
from slmkit.tokenizers import EOS_ID, CharTokenizer
from slmkit.train import checkpoint
from slmkit.train.stage import sft_settings, sft_stage
from slmkit.train.trainer import Trainer

EX = EncodedExample(ids=np.array([10, 11, 12, 20, 21, EOS_ID]), prompt_len=3)  # 3 prompt tokens


def test_rows_shift_mask_the_prompt_and_pad() -> None:
    x, y = make_rows([EX], block_size=8)
    assert x[0].tolist() == [10, 11, 12, 20, 21, EOS_ID, EOS_ID, EOS_ID]
    # Targets: position j predicts ids[j + 1]. Predicting 11 and 12 (prompt) is masked; the
    # first answer token (20) is predicted from the last prompt token, so it counts.
    assert y[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 20, 21, EOS_ID] + [IGNORE_INDEX] * 3


def test_prompt_positions_get_zero_loss_and_zero_gradient() -> None:
    """DESIGN 7: the SFT mask test. Masked positions must not teach the model anything."""
    torch.manual_seed(0)
    model = CausalLM(ModelArgs(vocab_size=30, block_size=8, n_layers=1, d_model=16,
                               n_heads=2, n_kv_heads=2, ffn_hidden=32))  # fmt: skip
    x, y = (torch.from_numpy(a) for a in make_rows([EX], block_size=8))
    logits, loss = model(x, y)
    logits.retain_grad()
    assert loss is not None
    loss.backward()
    grad = logits.grad[0]  # type: ignore[index]
    masked = (y[0] == IGNORE_INDEX).nonzero().flatten().tolist()
    assert masked == [0, 1, 5, 6, 7]
    assert all(grad[j].abs().max().item() == 0.0 for j in masked)
    assert all(grad[j].abs().max().item() > 0.0 for j in (2, 3, 4))
    manual = torch.nn.functional.cross_entropy(logits[0, 2:5].detach(), y[0, 2:5])
    torch.testing.assert_close(loss.detach(), manual)


def test_encode_drops_examples_that_do_not_fit() -> None:
    tok = CharTokenizer.train(["abcdefgh"])
    examples = [
        SFTExample(prompt="ab", completion="cd"),
        SFTExample(prompt="ab", completion="cdefgh"),
    ]
    kept, dropped = encode_examples(examples, tok, block_size=5)
    assert dropped == 1 and kept[0].prompt_len == 2
    assert kept[0].ids.tolist() == [*tok.encode("abcd"), EOS_ID]


def test_sampler_state_restores() -> None:
    a = SFTSampler([EX] * 5 + [EncodedExample(np.array([1, 2, EOS_ID]), 1)], 8, 4, seed=3)
    a.next_batch()
    state = a.state_dict()
    want = a.next_batch()[1]
    b = SFTSampler(a.examples, 8, 4, seed=99)
    b.load_state_dict(state)
    np.testing.assert_array_equal(b.next_batch()[1], want)


# ------------------------------------------------------------------------------ end to end

FAST = [
    "run.tracker=none", "train.compile=false", "train.max_tokens=640", "train.warmup_tokens=64",
    "train.eval_every_steps=10", "train.eval_iters=2", "train.eval_samples=1",
    "train.sample_tokens=4", "sft.epochs=1",
    "data.block_size=32",  # toy documents are ~13 characters: each must fit in one SFT row
]  # fmt: skip


def _exp():  # type: ignore[no-untyped-def]
    return load_experiment("toy/base", FAST)


def test_sft_needs_a_pretrained_run(toy_repo: Path, slm_home: Path) -> None:
    with pytest.raises(ConfigError, match="pretrain"):
        sft_stage(_exp())


def test_sft_settings_come_from_the_preset_and_the_experiment(toy_repo: Path) -> None:
    exp = load_experiment("toy/base", [*FAST, "sft.lr=1e-4", "sft.epochs=2"])
    s = sft_settings(exp, n_train_examples=50, block_size=8)
    assert (s.preset, s.lr, s.epochs, s.batch_size) == ("sft", 1e-4, 2, 32)
    assert s.max_tokens == 2 * 50 * 8


def test_sft_starts_from_the_pretrained_weights_and_completes(
    toy_repo: Path, slm_home: Path
) -> None:
    exp = _exp()
    base = Trainer(exp, device="cpu", log=lambda _: None)
    base.run()
    stage = sft_stage(exp)
    sft = Trainer(exp, stage=stage, device="cpu", log=lambda _: None)
    assert sft.run_dir != base.run_dir and stage.parent == base.run_id
    parent = torch.load(stage.init_weights, weights_only=True)  # type: ignore[arg-type]
    for name, tensor in sft.model.state_dict().items():
        torch.testing.assert_close(tensor, parent[name])
    assert sft.run() == "complete"
    assert checkpoint.latest(sft.run_dir) is not None


def test_sft_resumes_like_pretraining(toy_repo: Path, slm_home: Path) -> None:
    exp = load_experiment("toy/base", [*FAST, "sft.epochs=6"])  # enough steps to stop mid-way
    Trainer(exp, device="cpu", log=lambda _: None).run()
    first = Trainer(exp, stage=sft_stage(exp), device="cpu", log=lambda _: None)
    assert first.run(max_steps=2) == "stopped"
    again = Trainer(exp, stage=sft_stage(exp), device="cpu", log=lambda _: None)
    assert again.resumed_from is not None and again.step == 2


def test_eval_asks_a_fine_tuned_run_in_its_own_words(toy_repo: Path, slm_home: Path) -> None:
    exp = _exp()
    Trainer(exp, device="cpu", log=lambda _: None).run()
    sft = Trainer(exp, stage=sft_stage(exp), device="cpu", log=lambda _: None)
    sft.run()
    run = load_run(sft.run_id, "best", torch.device("cpu"))
    assert run.stage == "sft"
    settings = EvalSettings((0,), 2, 1.0, None, None, 4)
    report, _ = evaluate(run, settings, torch.device("cpu"), log=lambda _: None)
    assert report["prompts_kind"] == "sft" and report["stage"] == "sft"
