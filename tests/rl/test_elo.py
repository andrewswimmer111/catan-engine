"""Tests for :class:`rl.evaluation.elo.EloTracker` (rl-019)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from rl.evaluation.elo import EloConfig, EloTracker


def test_initial_rating_for_unseen_agent() -> None:
    tracker = EloTracker()
    assert tracker.rating("never_seen") == 1500.0


def test_two_player_winner_gains_loser_loses() -> None:
    tracker = EloTracker()
    tracker.update(["a", "b"], [1.0, 0.0])
    assert tracker.rating("a") > 1500.0
    assert tracker.rating("b") < 1500.0
    # Zero-sum: deltas are equal in magnitude when ratings start equal.
    assert tracker.rating("a") + tracker.rating("b") == pytest.approx(3000.0)


def test_draw_leaves_equal_ratings_unchanged() -> None:
    tracker = EloTracker()
    tracker.update(["a", "b"], [0.5, 0.5])
    assert tracker.rating("a") == pytest.approx(1500.0)
    assert tracker.rating("b") == pytest.approx(1500.0)


def test_dominant_winner_outranks_loser_after_many_games() -> None:
    """A beats B 90% of the time across 100 games → A's rating > B's."""
    tracker = EloTracker(cfg=EloConfig(k_factor=16.0))
    rng = random.Random(0)
    for _ in range(100):
        a_wins = rng.random() < 0.9
        if a_wins:
            tracker.update(["A", "B"], [1.0, 0.0])
        else:
            tracker.update(["A", "B"], [0.0, 1.0])
    assert tracker.rating("A") > tracker.rating("B")
    # 90% over 100 games should produce a sizable gap, > 100 Elo.
    assert tracker.rating("A") - tracker.rating("B") > 100.0


def test_four_player_pairwise_updates() -> None:
    """One winner among four players → winner gains, losers lose roughly
    equally, and the pairwise updates between losers cancel (draws)."""
    tracker = EloTracker()
    tracker.update(["w", "l1", "l2", "l3"], [1.0, 0.0, 0.0, 0.0])
    rw = tracker.rating("w")
    losers = [tracker.rating(x) for x in ("l1", "l2", "l3")]
    assert rw > 1500.0
    assert all(r < 1500.0 for r in losers)
    # Losers all start equal; their pairwise updates against each other are
    # 0.5 (draw) → no net change. Their loss comes purely from losing to w.
    assert losers[0] == pytest.approx(losers[1])
    assert losers[1] == pytest.approx(losers[2])


def test_pair_iteration_order_does_not_leak_into_ratings() -> None:
    """Updates use a pre-update rating snapshot so iteration order doesn't
    bias outcomes."""
    a = EloTracker()
    b = EloTracker()
    a.update(["x", "y", "z"], [1.0, 0.5, 0.0])
    b.update(["z", "y", "x"], [0.0, 0.5, 1.0])  # reversed agent order
    for k in ("x", "y", "z"):
        assert a.rating(k) == pytest.approx(b.rating(k))


def test_update_rejects_length_mismatch() -> None:
    tracker = EloTracker()
    with pytest.raises(ValueError):
        tracker.update(["a", "b", "c"], [1.0, 0.0])


def test_update_rejects_single_agent() -> None:
    tracker = EloTracker()
    with pytest.raises(ValueError):
        tracker.update(["a"], [1.0])


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    tracker = EloTracker(cfg=EloConfig(k_factor=24.0, initial_rating=1400.0))
    tracker.update(["a", "b"], [1.0, 0.0])
    path = tmp_path / "elo.json"
    tracker.save(path)

    loaded = EloTracker.load(path)
    assert loaded.cfg.k_factor == 24.0
    assert loaded.cfg.initial_rating == 1400.0
    assert loaded.rating("a") == tracker.rating("a")
    assert loaded.rating("b") == tracker.rating("b")
