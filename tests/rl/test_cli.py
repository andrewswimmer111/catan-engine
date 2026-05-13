"""Smoke tests for :mod:`rl.cli` (rl-020)."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.cli import main
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_LAYOUT_VERSION, OBS_SHAPE
from rl.models.mlp import MLPPolicyValue
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    ModelArch,
    save_checkpoint,
)


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
HIDDEN = (16, 16)


def _write_checkpoint(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
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
        config_hash="x" * 64,
    )
    save_checkpoint(agent, path, meta)


@pytest.mark.slow
def test_evaluate_vs_random_writes_report(tmp_path: Path) -> None:
    ckpt = tmp_path / "agent.pt"
    _write_checkpoint(ckpt, seed=0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "evaluate",
            "--learner", str(ckpt),
            "--opponent", "random",
            "--games", "4",
            "--seed", "1",
            "--output-dir", str(tmp_path / "replays"),
        ])
    out = buf.getvalue()
    assert rc == 0
    assert "Eval:" in out
    assert "Per-seat learner win rate" in out
    assert "Sample games" in out
    # Output dir should hold per-game replay JSONs.
    replays = list((tmp_path / "replays").glob("*.json"))
    assert len(replays) == 4


@pytest.mark.slow
def test_evaluate_two_checkpoints_runs(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _write_checkpoint(a, seed=0)
    _write_checkpoint(b, seed=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "evaluate",
            "--learner", str(a),
            "--opponent", str(b),
            "--games", "4",
            "--seed", "0",
        ])
    assert rc == 0
    assert "Action diff" in buf.getvalue()


def test_evaluate_rejects_nonpositive_games(tmp_path: Path) -> None:
    ckpt = tmp_path / "agent.pt"
    _write_checkpoint(ckpt, seed=0)
    rc = main([
        "evaluate",
        "--learner", str(ckpt),
        "--opponent", "random",
        "--games", "0",
    ])
    assert rc == 2


def test_evaluate_unknown_opponent_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "agent.pt"
    _write_checkpoint(ckpt, seed=0)
    with pytest.raises(FileNotFoundError):
        main([
            "evaluate",
            "--learner", str(ckpt),
            "--opponent", str(tmp_path / "missing.pt"),
            "--games", "4",
        ])
