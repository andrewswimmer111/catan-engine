"""Smoke tests for :class:`VecRolloutWorker` (rl-023)."""

from __future__ import annotations

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.models.mlp import MLPPolicyValue
from rl.replay.buffer import TrajectoryBuffer
from rl.training.vec_env import SubprocVecEnv
from rl.training.vec_rollout import VecRolloutWorker
from tests.rl.fixtures.vec_env_factory import random_opponents_factory


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _make_learner() -> PolicyAgent:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(16, 16))
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


@pytest.mark.slow
def test_vec_rollout_fills_buffer():
    learner = _make_learner()
    buffer = TrajectoryBuffer(capacity=64, obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE)
    vec = SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=0)
    try:
        worker = VecRolloutWorker(
            vec_env=vec,
            learner=learner,
            buffer=buffer,
            learner_seats=[PLAYER_IDS[0], PLAYER_IDS[0]],
        )
        stats = worker.collect(n_steps=20)
    finally:
        vec.close()
    assert stats.learner_steps >= 20
    assert len(buffer) >= 20
    # Every stored transition belongs to the learner's seat.
    assert (buffer.agents_view() == int(PLAYER_IDS[0])).all()


@pytest.mark.slow
def test_vec_rollout_bootstrap_values_match_seats():
    learner = _make_learner()
    buffer = TrajectoryBuffer(capacity=16, obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE)
    vec = SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=0)
    try:
        worker = VecRolloutWorker(
            vec_env=vec,
            learner=learner,
            buffer=buffer,
            learner_seats=[PLAYER_IDS[0], PLAYER_IDS[0]],
        )
        worker.collect(n_steps=10)
    finally:
        vec.close()
    bootstraps = worker.last_bootstrap_values
    # Single seat across envs — at most one entry, but should at least be a dict.
    assert all(isinstance(k, int) for k in bootstraps.keys())
    assert all(isinstance(v, float) for v in bootstraps.values())
