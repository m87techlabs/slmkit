"""Batch sampler: the one-token shift, bounds, and resumable state."""

from __future__ import annotations

import numpy as np
import pytest

from slmkit.data.sampler import RandomWindowSampler

DATA = np.arange(1000, dtype=np.uint16)  # token i has value i, so shifts are easy to see


def test_targets_are_inputs_shifted_by_one() -> None:
    x, y = RandomWindowSampler(DATA, block_size=16, batch_size=8, seed=0).next_batch()
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == np.int64
    np.testing.assert_array_equal(y, x + 1)  # the next token, not the same token


def test_windows_stay_in_bounds() -> None:
    sampler = RandomWindowSampler(DATA, block_size=64, batch_size=32, seed=0)
    for _ in range(50):
        x, y = sampler.next_batch()
        assert x.min() >= 0 and y.max() <= len(DATA) - 1


def test_state_restore_reproduces_batches() -> None:
    a = RandomWindowSampler(DATA, block_size=8, batch_size=4, seed=7)
    a.next_batch()
    state = a.state_dict()
    expected = [a.next_batch()[0] for _ in range(3)]

    b = RandomWindowSampler(DATA, block_size=8, batch_size=4, seed=999)
    b.load_state_dict(state)
    for want in expected:
        np.testing.assert_array_equal(b.next_batch()[0], want)


def test_rejects_split_shorter_than_a_window() -> None:
    with pytest.raises(ValueError, match="block_size"):
        RandomWindowSampler(DATA[:8], block_size=8, batch_size=1, seed=0)
