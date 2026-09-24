"""Sampling: generate(), with temperature / top-k / top-p. See generate.py."""

from slmkit.sampling.generate import filter_logits, generate

__all__ = ["filter_logits", "generate"]
