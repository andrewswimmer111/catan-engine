"""Tests for the GUI's checkpoint-backed AI agent factory."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from controller.session import GameSession
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from gui.ai_agent import make_ai_agent
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


PLAYER_IDS = [PlayerID(i) for i in range(4)]
HIDDEN = (32, 32)


def _write_tiny_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=HIDDEN)
    agent = PolicyAgent(model, ActionEncoder(PLAYER_IDS))
    meta = CheckpointMeta(
        obs_layout_version=OBS_LAYOUT_VERSION,
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=ModelArch(
            obs_dim=OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, hidden=HIDDEN
        ),
        train_step=0,
        timestamp=time.time(),
        config_hash="x" * 16,
    )
    save_checkpoint(agent, path, meta)


def test_make_ai_agent_picks_a_legal_action(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    _write_tiny_checkpoint(ckpt)

    agent = make_ai_agent(ckpt)
    assert agent.stochastic_play is False

    engine = GameEngine(SeededRandomizer(7))
    session = GameSession(engine, GameConfig(player_ids=PLAYER_IDS, seed=7))
    snap = session.current()
    legal = session.legal_actions()
    assert legal, "expected legal actions at game start"

    chosen = agent.choose(snap, legal)
    assert chosen in legal
