"""Text generation: predict, pick one token, append, repeat.

The model outputs a score (logit) for every token in the vocabulary. *Sampling* is how one
token is picked from those scores, and three knobs shape the choice:

- **temperature** divides the logits before the softmax. Below 1 sharpens the distribution
  (safer, more repetitive); above 1 flattens it (more varied, more mistakes). 0 means always
  take the single most likely token (greedy).
- **top-k** keeps only the k most likely tokens and renormalises. Cuts off the long tail of
  individually unlikely tokens that together are picked surprisingly often.
- **top-p** (nucleus) keeps the smallest set of tokens whose probabilities add up to p. Adapts
  to the model's confidence: few candidates when it is sure, many when it is not.

Worked example, logits [2.0, 1.0, 0.1] for tokens A, B, C:
    temperature 1.0 -> probabilities 0.66 / 0.24 / 0.10
    temperature 0.5 -> 0.86 / 0.12 / 0.02      (sharper)
    top_k = 2       -> 0.73 / 0.27 / 0         (C can no longer be picked)

No KV cache: each new token re-runs the model over the whole (cropped) context. For models
this small that costs milliseconds, and the loop stays readable. A cache matters for serving
large models, not for printing a few samples during training.
"""

from __future__ import annotations

import torch

from slmkit.model import CausalLM


def filter_logits(
    logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Apply temperature, top-k and top-p to logits of shape (B, V). Removed tokens get -inf."""
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        # Drop a token if the tokens *before* it already reach top_p; always keep the first.
        drop_sorted = cumulative - sorted_logits.softmax(-1) >= top_p
        drop = drop_sorted.scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return logits


@torch.no_grad()
def generate(
    model: CausalLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_token: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend `idx` (B, T) by up to `max_new_tokens`. Returns (B, T + generated).

    Pass a `torch.Generator` to make samples reproducible without touching the global RNG,
    which training depends on.
    """
    was_training = model.training
    model.eval()
    block = model.args.block_size
    for _ in range(max_new_tokens):
        context = idx[:, -block:]  # the model cannot see further back than its block size
        logits = model(context)[0][:, -1, :].float()  # only the last position predicts next
        if temperature == 0.0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = filter_logits(logits, temperature, top_k, top_p).softmax(-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
        if stop_token is not None and bool((nxt == stop_token).all()):
            break
    model.train(was_training)
    return idx
