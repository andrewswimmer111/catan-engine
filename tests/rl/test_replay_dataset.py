"""Tests for :mod:`rl.replay.dataset` and :mod:`rl.replay.recorder` (rl-021)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.models.mlp import MLPPolicyValue
from rl.replay.dataset import (
    EpisodeRecord,
    ReplayDataset,
    StepRecord,
    is_interesting,
)
from rl.replay.recorder import play_episode
from serialization.replay import ReplayLog


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _fake_step(seed: int = 0) -> StepRecord:
    rng = np.random.default_rng(seed)
    obs = rng.random(OBS_SHAPE[0], dtype=np.float32)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    mask[0] = True
    mask[1] = True
    dist = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    dist[0] = 0.7
    dist[1] = 0.3
    return StepRecord(
        obs=obs, action=0, mask=mask, action_dist=dist,
        value=0.5, reward=0.1, agent=PLAYER_IDS[0],
    )


def _fake_episode() -> EpisodeRecord:
    from domain.game.config import GameConfig

    cfg = GameConfig(player_ids=list(PLAYER_IDS), seed=42)
    return EpisodeRecord(
        replay_log=ReplayLog(config=cfg, actions=[], events=[]),
        steps=[_fake_step(0), _fake_step(1)],
        final_vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 8, PLAYER_IDS[2]: 5, PLAYER_IDS[3]: 4},
        winner=PLAYER_IDS[0],
        metadata={"checkpoint": "test.pt", "seed": 42},
    )


def test_write_read_round_trip(tmp_path: Path) -> None:
    ds = ReplayDataset(tmp_path)
    ep = _fake_episode()
    ep_dir = ds.write(ep)
    assert ep_dir.exists()
    assert (ep_dir / "replay.json").exists()
    assert (ep_dir / "steps.npz").exists()
    assert (ep_dir / "meta.json").exists()

    ids = ds.list_episodes()
    assert len(ids) == 1
    loaded = ds.read(ids[0])
    assert loaded.winner == ep.winner
    assert loaded.final_vps == ep.final_vps
    assert loaded.metadata == ep.metadata
    assert len(loaded.steps) == len(ep.steps)
    for original, restored in zip(ep.steps, loaded.steps):
        assert original.action == restored.action
        assert original.agent == restored.agent
        assert np.allclose(original.obs, restored.obs)
        assert np.array_equal(original.mask, restored.mask)
        assert np.allclose(original.action_dist, restored.action_dist)
        assert original.value == pytest.approx(restored.value)


def test_list_episodes_sorted(tmp_path: Path) -> None:
    ds = ReplayDataset(tmp_path)
    for _ in range(3):
        ds.write(_fake_episode())
    ids = ds.list_episodes()
    assert ids == sorted(ids)
    assert len(ids) == 3
    # Counter increments across writes.
    counters = [int(i.split("_", 1)[0]) for i in ids]
    assert counters == sorted(counters)
    assert counters[-1] - counters[0] == 2


def test_list_episodes_skips_partial(tmp_path: Path) -> None:
    ds = ReplayDataset(tmp_path)
    ds.write(_fake_episode())
    # Drop one of the required files to simulate a torn write.
    full_id = ds.list_episodes()[0]
    (tmp_path / full_id / "meta.json").unlink()
    assert ds.list_episodes() == []


def test_list_episodes_skips_non_episode_dirs(tmp_path: Path) -> None:
    ds = ReplayDataset(tmp_path)
    ds.write(_fake_episode())
    (tmp_path / "scratch").mkdir()
    assert len(ds.list_episodes()) == 1


def test_empty_step_list_round_trips(tmp_path: Path) -> None:
    """Edge case: a game where the learner never acted (rare cold-start)."""
    ds = ReplayDataset(tmp_path)
    ep = _fake_episode()
    empty_ep = EpisodeRecord(
        replay_log=ep.replay_log,
        steps=[],
        final_vps=ep.final_vps,
        winner=ep.winner,
        metadata={},
    )
    ds.write(empty_ep)
    loaded = ds.read(ds.list_episodes()[0])
    assert loaded.steps == []


def test_is_interesting_close_win() -> None:
    vps = {PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 9, PLAYER_IDS[2]: 6, PLAYER_IDS[3]: 5}
    assert is_interesting(vps, winner=PLAYER_IDS[0])


def test_is_interesting_blowout_not() -> None:
    vps = {PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 4, PLAYER_IDS[2]: 3, PLAYER_IDS[3]: 2}
    assert not is_interesting(vps, winner=PLAYER_IDS[0])


def test_is_interesting_stalemate_flagged() -> None:
    vps = {PLAYER_IDS[0]: 5, PLAYER_IDS[1]: 5, PLAYER_IDS[2]: 5, PLAYER_IDS[3]: 5}
    assert is_interesting(vps, winner=None)


@pytest.mark.slow
def test_play_episode_emits_step_records(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(16, 16))
    learner = PolicyAgent(model, ActionEncoder(PLAYER_IDS))
    rng = random.Random(0)
    opponents: dict[PlayerID, RandomAgent] = {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in PLAYER_IDS[1:]
    }
    env = CatanEnv(seed=7)
    ep = play_episode(
        env=env,
        learner=learner,
        learner_seat=PLAYER_IDS[0],
        opponents=opponents,  # type: ignore[arg-type]
        metadata={"checkpoint": "x.pt"},
    )
    assert ep.final_vps.keys() == set(PLAYER_IDS)
    assert ep.metadata["checkpoint"] == "x.pt"
    assert ep.metadata["learner_seat"] == int(PLAYER_IDS[0])
    # All recorded steps are the learner's.
    assert all(s.agent == PLAYER_IDS[0] for s in ep.steps)
    # action_dist sums to ~1 on every learner step.
    for s in ep.steps:
        if s.action >= 0:
            assert s.action_dist.sum() == pytest.approx(1.0, abs=1e-4)
    # Round-trip through a dataset works for real episodes too.
    ds = ReplayDataset(tmp_path)
    ds.write(ep)
    loaded = ds.read(ds.list_episodes()[0])
    assert loaded.final_vps == ep.final_vps
