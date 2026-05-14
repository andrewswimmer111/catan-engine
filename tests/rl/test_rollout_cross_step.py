"""Cross-step reward integration test for :class:`RolloutWorker`.

End-to-end-from-RewardFn: drive a real :class:`CatanEnv` for one learner
transition, plant a known ``last_cross_rewards`` value on the env (the
moral equivalent of "an opponent just took Longest Road from you"), invoke
the worker's opponent-step path, and verify the learner's last stored
transition was credited.

The point of the test is *not* to reproduce an LR flip in-game (which
requires driving a long sequence of moves) — it's to lock the wiring
between :attr:`CatanEnv.last_cross_rewards` and
:meth:`TrajectoryBuffer.add_terminal_reward` so future refactors can't
silently break the per-step cross-credit path.
"""

from __future__ import annotations

import random

import pytest
import torch

from domain.actions.all_actions import EndTurnAction
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.env.rewards import ShapedReward
from rl.models.mlp import MLPPolicyValue
from rl.replay.buffer import TrajectoryBuffer
from rl.training.rollout import RolloutWorker


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _make_learner() -> PolicyAgent:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(16, 16))
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


def _make_worker(env: CatanEnv) -> tuple[RolloutWorker, TrajectoryBuffer]:
    learner = _make_learner()
    learner_seat = PLAYER_IDS[0]
    opponents = {
        pid: RandomAgent(random.Random(seed), skip_proposals=True)
        for seed, pid in enumerate(PLAYER_IDS[1:], start=100)
    }
    buffer = TrajectoryBuffer(
        capacity=64, obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE
    )
    worker = RolloutWorker(env, learner, opponents, buffer)
    return worker, buffer


def test_cross_step_credit_lands_on_learner_last_transition() -> None:
    """An opponent's action with non-zero cross_step credit must debit the
    learner's most recent stored transition.

    We patch the reward fn's ``cross_step_rewards`` to deterministically
    return a fixed credit for the learner on every step. ``env.step``
    refreshes ``_last_cross_rewards`` after each action, so writing
    directly to that cache before stepping wouldn't survive the call —
    patching the method is the stable injection point.
    """
    reward_fn = ShapedReward()
    env = CatanEnv(seed=42, reward_fn=reward_fn)
    worker, buffer = _make_worker(env)
    learner_seat = worker.learner_seat

    reward_fn.cross_step_rewards = lambda prev, action, result, acting: {  # type: ignore[method-assign]
        learner_seat: -0.25
    }

    # Drive the env until we have stored a learner transition.
    stats = worker.collect(n_steps=1, stop_at_episode=False)
    assert stats.learner_steps == 1
    assert len(buffer) == 1
    last_idx = worker._last_transition_idx
    assert last_idx == 0
    pre_reward = float(buffer._rewards[last_idx])

    # If the learner is still acting (multi-action turn), advance one more
    # transition so we're at an opponent step boundary.
    while env.current_agent == learner_seat:
        worker.collect(n_steps=1, stop_at_episode=False)
        last_idx = worker._last_transition_idx
        pre_reward = float(buffer._rewards[last_idx])

    # Run one opponent step — that's the path that triggers
    # ``_apply_cross_step_credit`` against the learner's last transition.
    acting = env.current_agent
    assert acting != learner_seat
    worker._take_opponent_step(acting)

    post_reward = float(buffer._rewards[last_idx])
    # Buffer stores rewards as float32, so compare with a tolerance.
    assert post_reward == pytest.approx(pre_reward - 0.25, abs=1e-5), (
        f"expected cross-step credit of -0.25 written to last transition; "
        f"pre={pre_reward}, post={post_reward}"
    )


def test_cross_step_credit_silent_when_no_stored_transition() -> None:
    """Opponent steps before any learner step has stored a transition are no-ops."""
    env = CatanEnv(seed=43, reward_fn=ShapedReward())
    worker, buffer = _make_worker(env)
    assert worker._last_transition_idx is None

    env._last_cross_rewards = {worker.learner_seat: -1.0}
    worker._apply_cross_step_credit()  # should not raise, should not write
    assert len(buffer) == 0


def test_cross_step_credit_skipped_on_learner_seat_absent() -> None:
    """Cross-step dicts that don't mention the learner are silently dropped."""
    env = CatanEnv(seed=44, reward_fn=ShapedReward())
    worker, buffer = _make_worker(env)

    worker.collect(n_steps=1, stop_at_episode=False)
    last_idx = worker._last_transition_idx
    assert last_idx is not None
    pre_reward = float(buffer._rewards[last_idx])

    env._last_cross_rewards = {PLAYER_IDS[2]: -0.5}  # not the learner
    worker._apply_cross_step_credit()
    assert float(buffer._rewards[last_idx]) == pre_reward
