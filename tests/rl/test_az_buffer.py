"""Tests for :mod:`rl.training.az_buffer`.

Covers:

* Bounded capacity → FIFO eviction when ``len > capacity``.
* :meth:`sample` returns a stacked :class:`AZBatch` of the requested size.
* Sampling is uniform over the buffer's current contents (with
  replacement) — verified by a chi-square-free, finite-tolerance check
  on per-slot empirical frequencies.
* Defensive errors on degenerate args.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.training.az_buffer import AZBatch, AZReplayBuffer
from rl.training.self_play import SelfPlayTransition

_OBS_DIM = 8
_N_PLAYERS = 4


def _make_transition(tag: float) -> SelfPlayTransition:
    """A SelfPlayTransition with all fields encoded with ``tag`` so the
    sampling tests can identify which transition was picked."""
    return SelfPlayTransition(
        obs=np.full(_OBS_DIM, tag, dtype=np.float32),
        action_mask=np.zeros(ACTION_SPACE_SIZE, dtype=bool),
        mcts_policy=np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
        acting_seat_idx=int(tag) % _N_PLAYERS,
        value_target=np.full(_N_PLAYERS, tag, dtype=np.float32),
        vp_aux_target=np.full(_N_PLAYERS, tag, dtype=np.float32),
    )


# ----------------------------------------------------------------------
# Construction / population
# ----------------------------------------------------------------------


def test_constructor_rejects_zero_or_negative_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        AZReplayBuffer(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        AZReplayBuffer(capacity=-5)


def test_empty_buffer_reports_zero_len_and_refuses_to_sample() -> None:
    buf = AZReplayBuffer(capacity=4)
    assert len(buf) == 0
    with pytest.raises(ValueError, match="empty"):
        buf.sample(batch_size=2)


def test_extend_and_append_grow_buffer_up_to_capacity() -> None:
    buf = AZReplayBuffer(capacity=3)
    buf.extend(_make_transition(float(i)) for i in range(2))
    assert len(buf) == 2
    buf.append(_make_transition(99.0))
    assert len(buf) == 3
    assert buf.capacity == 3


def test_fifo_eviction_when_capacity_exceeded() -> None:
    """Adding ``capacity + k`` items must evict the oldest ``k``.

    Verify by tagging transitions with a known scalar and inspecting the
    underlying deque after the overflow. We sample with batch_size=1
    until we've seen every remaining tag — only the un-evicted ones
    should ever appear.
    """
    buf = AZReplayBuffer(capacity=4, rng=random.Random(0))
    for i in range(10):  # 0..9 inserted; expect last 4 retained: 6,7,8,9.
        buf.append(_make_transition(float(i)))
    assert len(buf) == 4

    # Pull a large sample; the set of unique tags observed should be exactly
    # the kept items.
    seen: set[int] = set()
    for _ in range(500):
        batch = buf.sample(batch_size=1)
        tag = int(batch.value_target[0, 0])
        seen.add(tag)
    assert seen == {6, 7, 8, 9}


# ----------------------------------------------------------------------
# Sampling shape + types
# ----------------------------------------------------------------------


def test_sample_returns_stacked_batch_of_requested_size() -> None:
    buf = AZReplayBuffer(capacity=16, rng=random.Random(7))
    for i in range(16):
        buf.append(_make_transition(float(i)))
    batch = buf.sample(batch_size=5)

    assert isinstance(batch, AZBatch)
    assert batch.obs.shape == (5, _OBS_DIM)
    assert batch.action_mask.shape == (5, ACTION_SPACE_SIZE)
    assert batch.policy_target.shape == (5, ACTION_SPACE_SIZE)
    assert batch.value_target.shape == (5, _N_PLAYERS)
    assert batch.vp_aux_target.shape == (5, _N_PLAYERS)
    assert batch.obs.dtype == np.float32
    assert batch.action_mask.dtype == bool


def test_sample_supports_batch_size_larger_than_buffer() -> None:
    """With-replacement sampling lets the trainer request a fixed B
    regardless of buffer fill; this is important early in training when
    only the first iteration's transitions exist."""
    buf = AZReplayBuffer(capacity=8, rng=random.Random(0))
    for i in range(4):
        buf.append(_make_transition(float(i)))
    batch = buf.sample(batch_size=32)
    assert batch.obs.shape == (32, _OBS_DIM)


def test_sample_rejects_zero_or_negative_batch_size() -> None:
    buf = AZReplayBuffer(capacity=2, rng=random.Random(0))
    buf.append(_make_transition(0.0))
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(batch_size=-1)


# ----------------------------------------------------------------------
# Sampling distribution
# ----------------------------------------------------------------------


def test_sample_distribution_is_approximately_uniform() -> None:
    """Over many samples, every slot in the buffer should appear with
    frequency ≈ 1/N. We don't run a real chi-square — we set a generous
    finite-sample tolerance instead so the test is deterministic under
    a fixed RNG seed."""
    n_slots = 10
    n_draws = 20_000
    expected = n_draws / n_slots  # = 2000

    buf = AZReplayBuffer(capacity=n_slots, rng=random.Random(42))
    for i in range(n_slots):
        buf.append(_make_transition(float(i)))

    counts = np.zeros(n_slots, dtype=np.int64)
    for _ in range(n_draws):
        batch = buf.sample(batch_size=1)
        counts[int(batch.value_target[0, 0])] += 1

    # Each count is Binomial(n_draws, 1/n_slots); std = sqrt(n*p*(1-p)) ≈ 42.
    # 6-sigma window (≈ ±252) is a very loose bound that catches gross
    # non-uniformities (e.g. always-picking-index-0) without flakily
    # failing on legitimate sampling noise.
    assert np.all(np.abs(counts - expected) < 6 * 42), (
        f"empirical counts {counts.tolist()} deviate from uniform "
        f"(expected {expected:.0f} per slot, 6-sigma window ±252)"
    )


def test_sample_is_deterministic_under_fixed_seed() -> None:
    """Same-seed buffer → same draw sequence. Anchors the reproducibility
    contract for tests downstream of the buffer."""
    items = [_make_transition(float(i)) for i in range(16)]

    buf1 = AZReplayBuffer(capacity=16, rng=random.Random(2026))
    buf1.extend(items)
    batch1 = buf1.sample(batch_size=8)

    buf2 = AZReplayBuffer(capacity=16, rng=random.Random(2026))
    buf2.extend(items)
    batch2 = buf2.sample(batch_size=8)

    np.testing.assert_array_equal(batch1.value_target, batch2.value_target)


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------


def test_clear_resets_buffer() -> None:
    buf = AZReplayBuffer(capacity=4, rng=random.Random(0))
    buf.extend(_make_transition(float(i)) for i in range(3))
    assert len(buf) == 3
    buf.clear()
    assert len(buf) == 0
    with pytest.raises(ValueError, match="empty"):
        buf.sample(batch_size=1)
