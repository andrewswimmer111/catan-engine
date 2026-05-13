"""Tests for :mod:`rl.training.checkpoint` (rl-016)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.models.mlp import MLPPolicyValue
from rl.training import checkpoint as ckpt_mod
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    IncompatibleCheckpointError,
    ModelArch,
    compute_config_hash,
    load_checkpoint,
    save_checkpoint,
)
from rl.training.config import PPOConfig, TrainConfig


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
HIDDEN = (32, 32)


def _make_agent(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=HIDDEN)
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


def _make_meta(train_step: int = 100) -> CheckpointMeta:
    return CheckpointMeta(
        obs_layout_version=ckpt_mod.OBS_LAYOUT_VERSION,
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=ModelArch(
            obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, hidden=HIDDEN
        ),
        train_step=train_step,
        timestamp=time.time(),
        config_hash=compute_config_hash(TrainConfig()),
    )


def test_save_load_round_trip_preserves_state_dict(tmp_path: Path) -> None:
    agent = _make_agent(seed=42)
    meta = _make_meta(train_step=12_345)
    path = tmp_path / "agent.pt"

    save_checkpoint(agent, path, meta)
    assert path.exists()

    loaded, loaded_meta = load_checkpoint(path)
    for k, v in agent.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"mismatch on {k}"

    assert loaded_meta.train_step == meta.train_step
    assert loaded_meta.timestamp == pytest.approx(meta.timestamp)
    assert loaded_meta.config_hash == meta.config_hash
    assert loaded_meta.obs_layout_version == meta.obs_layout_version
    assert loaded_meta.action_layout_version == meta.action_layout_version
    assert loaded_meta.model_arch == meta.model_arch


def test_load_refuses_mismatched_obs_layout_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _make_agent()
    meta = _make_meta()
    path = tmp_path / "agent.pt"
    save_checkpoint(agent, path, meta)

    # Simulate bumping the obs layout version after the save.
    monkeypatch.setattr(ckpt_mod, "OBS_LAYOUT_VERSION", meta.obs_layout_version + 1)

    with pytest.raises(IncompatibleCheckpointError) as exc:
        load_checkpoint(path)
    assert "obs_layout_version" in str(exc.value)


def test_load_refuses_mismatched_action_layout_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _make_agent()
    meta = _make_meta()
    path = tmp_path / "agent.pt"
    save_checkpoint(agent, path, meta)

    monkeypatch.setattr(
        ckpt_mod, "ACTION_LAYOUT_VERSION", meta.action_layout_version + 1
    )

    with pytest.raises(IncompatibleCheckpointError) as exc:
        load_checkpoint(path)
    assert "action_layout_version" in str(exc.value)


def test_config_hash_changes_with_lr() -> None:
    base = TrainConfig()
    bumped = TrainConfig(ppo=PPOConfig(lr=base.ppo.lr * 2))
    assert compute_config_hash(base) != compute_config_hash(bumped)


def test_config_hash_stable_across_calls() -> None:
    cfg = TrainConfig()
    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def test_load_reconstructs_agent_with_correct_arch(tmp_path: Path) -> None:
    """Confirms loaded agent's model has the same shape as the original."""
    agent = _make_agent()
    meta = _make_meta()
    path = tmp_path / "agent.pt"
    save_checkpoint(agent, path, meta)

    loaded, _ = load_checkpoint(path)
    assert loaded.model.obs_dim == OBS_SHAPE[0]
    assert loaded.model.action_dim == ACTION_SPACE_SIZE
    assert loaded.model.hidden == HIDDEN


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    agent = _make_agent()
    meta = _make_meta()
    nested = tmp_path / "sub" / "deep" / "agent.pt"
    save_checkpoint(agent, nested, meta)
    assert nested.exists()
