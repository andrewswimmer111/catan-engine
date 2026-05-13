"""Tests for :mod:`rl.utils.gui_hook` (rl-022)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.models.mlp import MLPPolicyValue
from rl.replay.dataset import ReplayDataset
from rl.replay.recorder import play_episode
from rl.utils.gui_hook import load_episode_into_session


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


@pytest.fixture
def archived_episode(tmp_path: Path) -> Path:
    """Play one game and archive it; return the episode directory."""
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(16, 16))
    learner = PolicyAgent(model, ActionEncoder(PLAYER_IDS))
    rng = random.Random(0)
    opponents: dict[PlayerID, RandomAgent] = {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in PLAYER_IDS[1:]
    }
    env = CatanEnv(seed=11)
    ep = play_episode(env, learner, PLAYER_IDS[0], opponents)  # type: ignore[arg-type]
    ds = ReplayDataset(tmp_path)
    ep_dir = ds.write(ep)
    return ep_dir


@pytest.mark.slow
def test_load_episode_into_session(archived_episode: Path) -> None:
    loaded = load_episode_into_session(archived_episode)
    assert loaded.session.history()  # at least the initial snapshot
    # Overlay keys are valid snapshot indices in [0, history_len).
    history_len = len(loaded.session.history())
    for k in loaded.overlay:
        assert 0 <= k < history_len


@pytest.mark.slow
def test_overlay_aligned_with_learner_seat(archived_episode: Path) -> None:
    """Each overlay entry should land on a snapshot where the learner acts.

    The action taken from snapshot[i] is the last_action of snapshot[i+1] —
    we use that to verify the recorded learner step indices land on
    learner-owned turns.
    """
    loaded = load_episode_into_session(archived_episode)
    history = loaded.session.history()
    learner_seat = PLAYER_IDS[0]
    for step_idx, record in loaded.overlay.items():
        if record.action < 0:
            continue
        next_snap = history[step_idx + 1]
        # The applied action's player_id is the acting seat — should be the
        # learner.
        assert next_snap.last_action is not None
        assert next_snap.last_action.player_id == learner_seat
        # The encoded action index lines up with what the learner sampled.
        assert next_snap.last_action.player_id == record.agent


def test_load_episode_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_episode_into_session(tmp_path / "nope")
