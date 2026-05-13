"""Tests for :class:`rl.training.opponent_pool.OpponentPool` (rl-017)."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_LAYOUT_VERSION, OBS_SHAPE
from rl.models.mlp import MLPPolicyValue
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    ModelArch,
    save_checkpoint,
)
from rl.training.opponent_pool import OpponentPool


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
HIDDEN = (32, 32)


def _make_learner(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=HIDDEN)
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


def _write_checkpoint(path: Path, seed: int) -> None:
    agent = _make_learner(seed=seed)
    meta = CheckpointMeta(
        obs_layout_version=OBS_LAYOUT_VERSION,
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=ModelArch(
            obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, hidden=HIDDEN
        ),
        train_step=seed,
        timestamp=time.time(),
        config_hash="x" * 16,
    )
    save_checkpoint(agent, path, meta)


# ----------------------------------------------------------------------
# Bucket-eviction semantics
# ----------------------------------------------------------------------


def test_recent_buffer_evicts_oldest_at_capacity(tmp_path: Path) -> None:
    pool = OpponentPool(recent_size=3, historical_size=10)
    paths = [tmp_path / f"a{i}.pt" for i in range(5)]
    for p in paths:
        pool.add_checkpoint(p)
    # Only the last 3 paths should remain.
    assert pool.recent_paths == tuple(paths[-3:])


def test_promote_to_historical_appends_until_full(tmp_path: Path) -> None:
    pool = OpponentPool(recent_size=5, historical_size=3, rng=random.Random(0))
    for i in range(3):
        pool.add_checkpoint(tmp_path / f"ck{i}.pt")
        pool.promote_to_historical()
    assert len(pool.historical_paths) == 3


def test_promote_to_historical_noop_when_recent_empty() -> None:
    pool = OpponentPool(recent_size=4, historical_size=4)
    pool.promote_to_historical()
    assert pool.historical_paths == ()


# ----------------------------------------------------------------------
# Mix-ratio sampling
# ----------------------------------------------------------------------


def test_mix_ratios_respected_within_noise(tmp_path: Path) -> None:
    """All-current mix should always pick the live learner; all-recent mix
    should always pick from recent; etc. We use degenerate one-hot mixes to
    make the assertion crisp, and a balanced mix to confirm proportions land
    within statistical noise."""
    learner = _make_learner()
    ckpt_path = tmp_path / "opp.pt"
    _write_checkpoint(ckpt_path, seed=1)

    # All-current.
    pool = OpponentPool(
        mix=(1.0, 0.0, 0.0), rng=random.Random(0), recent_size=4, historical_size=4
    )
    pool.add_checkpoint(ckpt_path)
    samples = pool.sample_opponents(learner, n=200)
    # When the chosen bucket is "current" we get a sibling sharing the
    # learner's model — same model object identity.
    assert all(s.model is learner.model for s in samples)

    # All-recent.
    pool = OpponentPool(
        mix=(0.0, 1.0, 0.0), rng=random.Random(1), recent_size=4, historical_size=4
    )
    pool.add_checkpoint(ckpt_path)
    samples = pool.sample_opponents(learner, n=50)
    assert all(s.model is not learner.model for s in samples)


def test_balanced_mix_proportions(tmp_path: Path) -> None:
    """Over 1000 samples the empirical bucket frequencies should land near
    the configured mix. Generous tolerance keeps this from flaking."""
    learner = _make_learner()
    recent_paths = [tmp_path / f"r{i}.pt" for i in range(2)]
    historical_path = tmp_path / "h.pt"
    for p in recent_paths + [historical_path]:
        _write_checkpoint(p, seed=hash(p.name) & 0xFF)

    rng = random.Random(0)
    pool = OpponentPool(
        mix=(0.5, 0.3, 0.2), rng=rng, recent_size=4, historical_size=4
    )
    for p in recent_paths:
        pool.add_checkpoint(p)
    pool._historical.append(historical_path)  # type: ignore[attr-defined]

    # Inject one historical entry directly so the historical bucket is
    # non-empty without triggering reservoir-sampling logic.
    n = 1000
    current = 0
    recent_count = 0
    historical_count = 0
    samples = pool.sample_opponents(learner, n=n)
    for s in samples:
        if s.model is learner.model:
            current += 1
        elif s in _loaded_for(pool, historical_path):
            historical_count += 1
        else:
            recent_count += 1

    # 5σ-ish band around expected proportions for n=1000 binomial draws.
    assert abs(current / n - 0.5) < 0.08
    assert abs(recent_count / n - 0.3) < 0.08
    assert abs(historical_count / n - 0.2) < 0.08


def _loaded_for(pool: OpponentPool, path: Path) -> set[PolicyAgent]:
    cache = pool._cache  # type: ignore[attr-defined]
    return {cache[path]} if path in cache else set()


# ----------------------------------------------------------------------
# Loaded-opponent invariants
# ----------------------------------------------------------------------


def test_loaded_opponents_have_requires_grad_false(tmp_path: Path) -> None:
    learner = _make_learner()
    ckpt = tmp_path / "opp.pt"
    _write_checkpoint(ckpt, seed=1)

    pool = OpponentPool(
        mix=(0.0, 1.0, 0.0), rng=random.Random(0), recent_size=4, historical_size=4
    )
    pool.add_checkpoint(ckpt)
    opp = pool.sample_opponents(learner, n=1)[0]
    assert isinstance(opp, PolicyAgent)
    for p in opp.model.parameters():
        assert p.requires_grad is False


def test_loaded_opponents_use_stochastic_play(tmp_path: Path) -> None:
    learner = _make_learner()
    ckpt = tmp_path / "opp.pt"
    _write_checkpoint(ckpt, seed=1)

    pool = OpponentPool(
        mix=(0.0, 1.0, 0.0), rng=random.Random(0), recent_size=4, historical_size=4
    )
    pool.add_checkpoint(ckpt)
    opp = pool.sample_opponents(learner, n=1)[0]
    assert isinstance(opp, PolicyAgent)
    assert opp.stochastic_play is True


def test_current_learner_slot_uses_stochastic_play(tmp_path: Path) -> None:
    learner = _make_learner()
    pool = OpponentPool(
        mix=(1.0, 0.0, 0.0), rng=random.Random(0), recent_size=4, historical_size=4
    )
    opp = pool.sample_opponents(learner, n=1)[0]
    assert isinstance(opp, PolicyAgent)
    assert opp.stochastic_play is True
    assert opp.model is learner.model
    # Learner's own stochastic_play must remain False — we don't mutate it.
    assert learner.stochastic_play is False


def test_lru_cache_reuses_loaded_agent(tmp_path: Path) -> None:
    learner = _make_learner()
    ckpt = tmp_path / "opp.pt"
    _write_checkpoint(ckpt, seed=1)

    pool = OpponentPool(
        mix=(0.0, 1.0, 0.0),
        rng=random.Random(0),
        recent_size=4,
        historical_size=4,
        cache_size=2,
    )
    pool.add_checkpoint(ckpt)
    first = pool.sample_opponents(learner, n=1)[0]
    second = pool.sample_opponents(learner, n=1)[0]
    assert first is second  # cache hit on the same path


def test_empty_pool_falls_back_to_live_learner(tmp_path: Path) -> None:
    learner = _make_learner()
    pool = OpponentPool(
        mix=(0.0, 0.5, 0.5), rng=random.Random(0), recent_size=4, historical_size=4
    )
    # No checkpoints registered — every slot falls back to the live learner.
    opps = pool.sample_opponents(learner, n=3)
    assert all(o.model is learner.model for o in opps)


# ----------------------------------------------------------------------
# Reservoir-sampling correctness
# ----------------------------------------------------------------------


def test_reservoir_sampling_uniform_over_promoted(tmp_path: Path) -> None:
    """Over many trials with k=2 historical slots and N=10 candidates, each
    candidate should end up in the historical bucket with frequency ~k/N.
    A loose 3σ bound on a small number of trials keeps the test fast."""
    n_trials = 500
    counts: dict[Path, int] = {}
    candidate_paths = [tmp_path / f"c{i}.pt" for i in range(10)]

    for t in range(n_trials):
        pool = OpponentPool(
            recent_size=20,
            historical_size=2,
            rng=random.Random(t),
        )
        for p in candidate_paths:
            pool.add_checkpoint(p)
            pool.promote_to_historical()
        for p in pool.historical_paths:
            counts[p] = counts.get(p, 0) + 1

    expected = n_trials * 2 / 10  # 100
    for p in candidate_paths:
        assert abs(counts.get(p, 0) - expected) < 50, (
            f"candidate {p.name} appeared {counts.get(p, 0)} times, "
            f"expected ~{expected}"
        )


def test_sample_opponents_n_zero_returns_empty() -> None:
    learner = _make_learner()
    pool = OpponentPool(rng=random.Random(0))
    assert pool.sample_opponents(learner, n=0) == []


def test_sample_opponents_negative_raises() -> None:
    learner = _make_learner()
    pool = OpponentPool(rng=random.Random(0))
    with pytest.raises(ValueError):
        pool.sample_opponents(learner, n=-1)
