"""Tests for :class:`TrajectoryBuffer` (rl-012).

Two contracts to nail down:

* Single-agent GAE matches a CleanRL-style reference implementation
  bit-for-bit (within fp32 tolerance). This catches sign errors, swapped
  ``gamma``/``lam``, and stale-bootstrap bugs.
* Multi-agent GAE is per-agent: opponent rewards never leak into the
  learner's advantage, and the recursion walks each agent's subsequence
  independently. A handcrafted 12-step trajectory across four seats
  exercises this directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.ids import PlayerID
from rl.replay.buffer import TrajectoryBuffer
from rl.replay.transition import Transition


OBS_DIM = 4
ACTION_DIM = 8


def _make_transition(
    *,
    agent: PlayerID,
    reward: float,
    value: float,
    done: bool = False,
    action: int = 0,
    logp: float = 0.0,
) -> Transition:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(ACTION_DIM, dtype=np.bool_)
    return Transition(
        obs=obs,
        action=action,
        mask=mask,
        logp=logp,
        value=value,
        reward=reward,
        done=done,
        agent=agent,
    )


def _cleanrl_gae_single(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Reference GAE for a single-agent trajectory. Adapted from CleanRL ppo.py."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_adv = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0 if dones[t] else last_value
            next_adv = 0.0
        else:
            next_value = values[t + 1]
            next_adv = last_adv
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        advantages[t] = delta + gamma * lam * non_terminal * next_adv
        last_adv = advantages[t]
    return advantages


# -----------------------------------------------------------------------------
# Constructor & basic mutators
# -----------------------------------------------------------------------------


def test_add_writes_in_order_and_grows_size() -> None:
    buf = TrajectoryBuffer(capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    assert len(buf) == 0

    for i in range(3):
        buf.add(_make_transition(agent=PlayerID(1), reward=float(i), value=0.0))
    assert len(buf) == 3
    assert buf.rewards_view().tolist() == [0.0, 1.0, 2.0]


def test_add_rejects_wrong_shape() -> None:
    buf = TrajectoryBuffer(capacity=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    bad = Transition(
        obs=np.zeros(OBS_DIM + 1, dtype=np.float32),
        action=0,
        mask=np.ones(ACTION_DIM, dtype=np.bool_),
        logp=0.0,
        value=0.0,
        reward=0.0,
        done=False,
        agent=PlayerID(1),
    )
    with pytest.raises(ValueError):
        buf.add(bad)


def test_add_full_buffer_raises() -> None:
    buf = TrajectoryBuffer(capacity=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    with pytest.raises(IndexError):
        buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))


def test_clear_resets_size_but_keeps_capacity() -> None:
    buf = TrajectoryBuffer(capacity=3, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for _ in range(3):
        buf.add(_make_transition(agent=PlayerID(1), reward=1.0, value=0.5))
    buf.clear()
    assert len(buf) == 0
    assert buf.capacity == 3
    buf.add(_make_transition(agent=PlayerID(1), reward=2.0, value=0.0))
    assert len(buf) == 1


def test_add_terminal_reward_is_additive() -> None:
    buf = TrajectoryBuffer(capacity=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buf.add(_make_transition(agent=PlayerID(1), reward=0.25, value=0.0))
    buf.add_terminal_reward(0, 1.0)
    assert buf.rewards_view()[0] == pytest.approx(1.25)


def test_mark_done_sets_done_flag() -> None:
    buf = TrajectoryBuffer(capacity=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    assert not buf.dones_view()[0]
    buf.mark_done(0)
    assert buf.dones_view()[0]


# -----------------------------------------------------------------------------
# Single-agent GAE vs reference
# -----------------------------------------------------------------------------


def test_single_agent_gae_matches_cleanrl_reference() -> None:
    rng = np.random.default_rng(123)
    T = 16
    rewards = rng.standard_normal(T).astype(np.float32)
    values = rng.standard_normal(T).astype(np.float32)
    dones = np.zeros(T, dtype=bool)
    dones[5] = True
    dones[11] = True
    last_value = 0.42

    buf = TrajectoryBuffer(capacity=T, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for t in range(T):
        buf.add(
            _make_transition(
                agent=PlayerID(1),
                reward=float(rewards[t]),
                value=float(values[t]),
                done=bool(dones[t]),
            )
        )

    gamma, lam = 0.95, 0.9
    buf.compute_advantages(gamma=gamma, lam=lam, last_values={PlayerID(1): last_value})

    expected = _cleanrl_gae_single(
        rewards.astype(np.float64),
        values.astype(np.float64),
        dones,
        last_value=last_value,
        gamma=gamma,
        lam=lam,
    )
    np.testing.assert_allclose(buf.advantages_view(), expected, atol=1e-5)
    # returns = advantages + values.
    np.testing.assert_allclose(
        buf.returns_view(), expected + values, atol=1e-5
    )


def test_single_agent_terminal_done_zeroes_bootstrap() -> None:
    """Last transition is done — bootstrap from last_values must be ignored."""
    buf = TrajectoryBuffer(capacity=3, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    buf.add(_make_transition(agent=PlayerID(1), reward=1.0, value=0.0, done=True))

    buf.compute_advantages(
        gamma=0.99, lam=1.0, last_values={PlayerID(1): 999.0}  # huge bootstrap, should be ignored
    )
    # With lam=1, GAE = sum of future rewards (Monte Carlo). last done → no bootstrap.
    # adv[2] = 1.0 - 0 = 1.0
    # adv[1] = 0 + 0.99 * 0 - 0 + 0.99 * 1 * adv[2] = 0.99
    # adv[0] = 0 + 0.99 * 0 - 0 + 0.99 * 1 * adv[1] = 0.99 * 0.99 = 0.9801
    expected = np.array([0.99 * 0.99, 0.99, 1.0], dtype=np.float64)
    np.testing.assert_allclose(buf.advantages_view(), expected, atol=1e-5)


# -----------------------------------------------------------------------------
# Multi-agent: per-agent GAE attribution
# -----------------------------------------------------------------------------


def test_multi_agent_gae_handcrafted_four_seats() -> None:
    """Twelve transitions, 3 per seat, round-robin order [1,2,3,4]×3.

    Reward is +1 for seat 1 on its third move (and -1 for the other seats on
    their third move) to simulate a terminal win. GAE for seat 1 must
    propagate the +1 backwards through its own three transitions only — and
    seat 2's advantages must not see seat 1's reward at all.
    """
    seats = [PlayerID(i) for i in range(1, 5)]
    buf = TrajectoryBuffer(capacity=12, obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    per_seat_rewards = {
        PlayerID(1): [0.0, 0.0, 1.0],
        PlayerID(2): [0.0, 0.0, -1.0],
        PlayerID(3): [0.0, 0.0, -1.0],
        PlayerID(4): [0.0, 0.0, -1.0],
    }
    per_seat_values = {pid: [0.1, 0.2, 0.0] for pid in seats}
    per_seat_dones = {pid: [False, False, True] for pid in seats}

    cursor_by_seat: dict[PlayerID, int] = {pid: 0 for pid in seats}
    for _round in range(3):
        for pid in seats:
            k = cursor_by_seat[pid]
            buf.add(
                _make_transition(
                    agent=pid,
                    reward=per_seat_rewards[pid][k],
                    value=per_seat_values[pid][k],
                    done=per_seat_dones[pid][k],
                )
            )
            cursor_by_seat[pid] += 1

    gamma, lam = 0.99, 0.95
    buf.compute_advantages(
        gamma=gamma, lam=lam, last_values={pid: 0.0 for pid in seats}
    )

    advantages = buf.advantages_view().copy()
    agents = buf.agents_view().copy()

    for pid in seats:
        idxs = np.flatnonzero(agents == int(pid))
        assert idxs.size == 3
        seat_rewards = np.array(per_seat_rewards[pid], dtype=np.float64)
        seat_values = np.array(per_seat_values[pid], dtype=np.float64)
        seat_dones = np.array(per_seat_dones[pid], dtype=bool)
        expected = _cleanrl_gae_single(
            seat_rewards,
            seat_values,
            seat_dones,
            last_value=0.0,
            gamma=gamma,
            lam=lam,
        )
        np.testing.assert_allclose(advantages[idxs], expected, atol=1e-5)

    # Seat 1's advantages should be strictly positive on its trajectory
    # (terminal reward = +1, value head untrained), seat 2's strictly negative.
    seat1 = advantages[agents == int(PlayerID(1))]
    seat2 = advantages[agents == int(PlayerID(2))]
    assert (seat1 > 0).all()
    assert (seat2 < 0).all()


def test_multi_agent_with_mid_buffer_episode_boundary() -> None:
    """Two episodes concatenated for the same agent: the done flag must
    block GAE propagation across the boundary."""
    pid = PlayerID(7)
    buf = TrajectoryBuffer(capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    # Episode 1: reward=1.0 on the last step, done=True.
    buf.add(_make_transition(agent=pid, reward=0.0, value=0.0))
    buf.add(_make_transition(agent=pid, reward=1.0, value=0.0, done=True))
    # Episode 2: reward 0 throughout, non-terminal so bootstrap kicks in.
    buf.add(_make_transition(agent=pid, reward=0.0, value=0.0))
    buf.add(_make_transition(agent=pid, reward=0.0, value=0.0))

    buf.compute_advantages(
        gamma=0.99, lam=1.0, last_values={pid: 2.0}
    )

    # Episode 2 advantages: from the end, both have no per-step reward but
    # the last gets a bootstrap of 2.0 → adv[3] = 0 + 0.99*2 - 0 = 1.98,
    # adv[2] = 0 + 0.99*0 - 0 + 0.99*1*1.98 = 1.9602.
    # Episode 1: terminal so no influence from episode 2.
    # adv[1] = 1 - 0 = 1.0; adv[0] = 0 + 0.99*0 - 0 + 0.99 * 1.0 = 0.99.
    expected = np.array([0.99, 1.0, 0.99 * 0.99 * 2.0, 0.99 * 2.0])
    np.testing.assert_allclose(buf.advantages_view(), expected, atol=1e-5)


# -----------------------------------------------------------------------------
# to_batches
# -----------------------------------------------------------------------------


def test_to_batches_requires_compute_advantages_first() -> None:
    buf = TrajectoryBuffer(capacity=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buf.add(_make_transition(agent=PlayerID(1), reward=0.0, value=0.0))
    with pytest.raises(RuntimeError):
        list(buf.to_batches(batch_size=1))


def test_to_batches_filters_by_learner_agent() -> None:
    seats = [PlayerID(i) for i in range(1, 5)]
    buf = TrajectoryBuffer(capacity=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for _round in range(2):
        for pid in seats:
            buf.add(_make_transition(agent=pid, reward=0.1 * int(pid), value=0.0))
    buf.compute_advantages(gamma=0.99, lam=0.95, last_values={pid: 0.0 for pid in seats})

    learner = PlayerID(2)
    batches = list(buf.to_batches(batch_size=10, learner_agent=learner, shuffle=False))
    assert len(batches) == 1
    batch = batches[0]
    # 2 transitions for seat 2.
    assert batch.obs.shape[0] == 2


def test_to_batches_yields_all_when_learner_is_none() -> None:
    seats = [PlayerID(i) for i in range(1, 3)]
    buf = TrajectoryBuffer(capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for pid in seats:
        for _ in range(2):
            buf.add(_make_transition(agent=pid, reward=0.0, value=0.0))
    buf.compute_advantages(gamma=1.0, lam=1.0, last_values={pid: 0.0 for pid in seats})

    total = sum(len(b) for b in buf.to_batches(batch_size=10, shuffle=False))
    assert total == 4


def test_to_batches_splits_by_batch_size() -> None:
    pid = PlayerID(1)
    buf = TrajectoryBuffer(capacity=10, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for _ in range(10):
        buf.add(_make_transition(agent=pid, reward=0.0, value=0.0))
    buf.compute_advantages(gamma=0.99, lam=0.95, last_values={pid: 0.0})

    sizes = [len(b) for b in buf.to_batches(batch_size=4, shuffle=False)]
    assert sizes == [4, 4, 2]
