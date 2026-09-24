"""Score runs' best checkpoints on their *entire* validation split.

    uv run python scripts/val_metrics.py run-a1d5 run-d020 run-f6d4

The trainer's eval samples random windows, which is quick and good enough to pick the best
checkpoint. This does one exact pass instead: the validation text cut into consecutive,
non-overlapping windows, every token scored once. It reports the metrics explained in
docs/concepts/metrics.md, plus two baselines that need no model at all.

A stop-gap until M2's `slm eval`, which will run project graders over generated samples.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from slmkit import artifacts
from slmkit.config.schema import ModelConfig
from slmkit.data.pack import open_packed
from slmkit.model import CausalLM, ModelArgs
from slmkit.tokenizers import load_tokenizer
from slmkit.train.run import load_run_config, runs_root


def baselines(train: np.ndarray, val: np.ndarray, vocab: int) -> None:
    counts = np.bincount(train, minlength=vocab)
    targets = val[1:]
    ranked = counts.argsort()[::-1]
    probs = counts / counts.sum()
    loss = float(-np.log(probs[targets] + 1e-12).mean())
    print(f"{'baseline: letter frequencies':34s} loss {loss:.4f}  ppl {math.exp(loss):6.2f}  "
          f"bpc {loss / math.log(2):.3f}  top1 {(targets == ranked[0]).mean():6.1%}  "
          f"top5 {np.isin(targets, ranked[:5]).mean():6.1%}")  # fmt: skip


@torch.no_grad()
def score(run_prefix: str, device: torch.device) -> None:
    [run_dir] = [p for p in runs_root().glob(f"{run_prefix}*") if p.is_dir()]
    cfg = load_run_config(run_dir)
    inputs = artifacts.read_manifest(run_dir)["inputs"]
    tokenizer = load_tokenizer(artifacts.find_artifact(inputs["tokenizer"]))
    val = np.asarray(open_packed(artifacts.find_artifact(inputs["packed"]) / "val.bin"), np.int64)
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
          f"ppl {math.exp(loss):6.2f}  bpc {loss / math.log(2):.3f}  "
          f"top1 {sum(top1) / total:6.1%}  top5 {sum(top5) / total:6.1%}")  # fmt: skip


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = next(p for p in runs_root().glob(f"{sys.argv[1]}*") if p.is_dir())
    inputs = artifacts.read_manifest(first)["inputs"]
    packed = artifacts.find_artifact(inputs["packed"])
    vocab = load_tokenizer(artifacts.find_artifact(inputs["tokenizer"])).vocab_size
    baselines(np.asarray(open_packed(packed / "train.bin")),
              np.asarray(open_packed(packed / "val.bin"), np.int64), vocab)  # fmt: skip
    for prefix in sys.argv[1:]:
        score(prefix, device)


if __name__ == "__main__":
    main()
