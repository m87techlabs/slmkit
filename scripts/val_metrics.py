"""Score runs' best checkpoints on their *entire* validation split.

    uv run python scripts/val_metrics.py run-a1d5 run-d020 run-f6d4

The trainer's eval samples random windows, which is quick and good enough to pick the best
checkpoint. This does one exact pass instead: the validation text cut into consecutive,
non-overlapping windows, every token scored once. It reports the metrics explained in
docs/concepts/metrics.md, plus a baseline that needs no model at all (token frequencies).
Loss, perplexity and accuracy are per token; bits per character (bpc) divides by characters
instead, so runs with different tokenizers can be compared.

A stop-gap until M2's `slm eval`, which will run project graders over generated samples.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from slmkit import artifacts
from slmkit.config.schema import ModelConfig
from slmkit.data.pack import count_chars, open_packed
from slmkit.data.split import read_docs
from slmkit.model import CausalLM, ModelArgs
from slmkit.tokenizers import load_tokenizer
from slmkit.train.run import load_run_config, runs_root


def chars_per_token(packed_dir: object, val_tokens: int, append_eos: bool) -> float:
    """Validation characters per token, to turn per-token loss into bits per character."""
    dataset = artifacts.find_artifact(artifacts.read_manifest(packed_dir)["inputs"]["dataset"])  # type: ignore[arg-type]
    return count_chars(read_docs(dataset / "val.jsonl"), append_eos=append_eos) / val_tokens


def baselines(train: np.ndarray, val: np.ndarray, vocab: int, cpt: float) -> None:
    counts = np.bincount(train, minlength=vocab)
    targets = val[1:]
    ranked = counts.argsort()[::-1]
    probs = counts / counts.sum()
    loss = float(-np.log(probs[targets] + 1e-12).mean())
    print(f"{'baseline: token frequencies':34s} loss {loss:.4f}  ppl {math.exp(loss):6.2f}  "
          f"bpc {loss / cpt / math.log(2):.3f}  top1 {(targets == ranked[0]).mean():6.1%}  "
          f"top5 {np.isin(targets, ranked[:5]).mean():6.1%}")  # fmt: skip


@torch.no_grad()
def score(run_prefix: str, device: torch.device) -> None:
    [run_dir] = [p for p in runs_root().glob(f"{run_prefix}*") if p.is_dir()]
    cfg = load_run_config(run_dir)
    inputs = artifacts.read_manifest(run_dir)["inputs"]
    tokenizer = load_tokenizer(artifacts.find_artifact(inputs["tokenizer"]))
    packed_dir = artifacts.find_artifact(inputs["packed"])
    val = np.asarray(open_packed(packed_dir / "val.bin"), np.int64)
    cpt = chars_per_token(packed_dir, len(val), cfg["data"]["append_eos"])
    block = cfg["data"]["block_size"]
    args = ModelArgs.from_config(
        ModelConfig.model_validate(cfg["model"]), tokenizer.vocab_size, block
    )
    model = CausalLM(args).to(device).eval()
    ckpt = run_dir / "ckpt" / "best" / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    n = (len(val) - 1) // block
    x = torch.from_numpy(val[: n * block].reshape(n, block)).to(device)
    y = torch.from_numpy(val[1 : n * block + 1].reshape(n, block)).to(device)
    losses, top1, top5 = [], [], []
    for i in range(0, n, 64):
        logits = model(x[i : i + 64])[0].float()
        target = y[i : i + 64]
        losses.append(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum").item())  # fmt: skip
        best5 = logits.topk(5, dim=-1).indices
        top1.append((best5[..., 0] == target).sum().item())
        top5.append((best5 == target[..., None]).any(-1).sum().item())
    total = n * block
    loss = sum(losses) / total
    print(f"{cfg['run']['name'] + ' (' + run_dir.name[:8] + ')':34s} loss {loss:.4f}  "
          f"ppl {math.exp(loss):6.2f}  bpc {loss / cpt / math.log(2):.3f}  "
          f"top1 {sum(top1) / total:6.1%}  top5 {sum(top5) / total:6.1%}  "
          f"(vocab {tokenizer.vocab_size}, {cpt:.2f} chars/token)")  # fmt: skip


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = next(p for p in runs_root().glob(f"{sys.argv[1]}*") if p.is_dir())
    inputs = artifacts.read_manifest(first)["inputs"]
    packed = artifacts.find_artifact(inputs["packed"])
    vocab = load_tokenizer(artifacts.find_artifact(inputs["tokenizer"])).vocab_size
    val = np.asarray(open_packed(packed / "val.bin"), np.int64)
    cfg = load_run_config(first)
    cpt = chars_per_token(packed, len(val), cfg["data"]["append_eos"])
    baselines(np.asarray(open_packed(packed / "train.bin")), val, vocab, cpt)
    for prefix in sys.argv[1:]:
        score(prefix, device)


if __name__ == "__main__":
    main()
