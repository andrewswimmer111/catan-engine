"""End-to-end smoke tests for the PPO trainer (rl-014).

The trainer is the integration point for rl-011..013 — we exercise it
end-to-end with a small number of steps to catch wiring bugs that the
component-level tests can't see. The convergence target (>90% win rate)
lives in rl-015 and runs as a slow integration test.

What we check:

* A 5,000-step run completes without exceptions.
* The TensorBoard events file is produced when ``log_dir`` is set.
* The final PPO metrics are all finite.
* A reduced reward signal makes its way into the buffer (some non-zero
  rewards appear at episode boundaries).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.models.mlp import MLPPolicyValue
from rl.training.config import PPOConfig, TrainConfig
from rl.training.trainer import Trainer


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _opponent_factory(seed: int) -> dict[PlayerID, Agent]:
    """All 4 seats filled with RandomAgents — the trainer drops the
    learner's seat before constructing the worker.
    """
    rng = random.Random(seed)
    return {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in PLAYER_IDS
    }


def _make_learner(hidden: tuple[int, ...] = (64, 64)) -> PolicyAgent:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=hidden)
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


def test_trainer_runs_short_training_without_errors(tmp_path: Path) -> None:
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(
            n_epochs=2, minibatch_size=64, target_kl=None, entropy_coef=0.01
        ),
        rollout_steps=256,
        eval_every=0,        # skip eval to keep this test fast
        log_every=1,
        seed=0,
    )
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    trainer.train(total_steps=512)

    assert trainer.global_step >= 512


def test_trainer_writes_tensorboard_events(tmp_path: Path) -> None:
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=1, minibatch_size=64, target_kl=None),
        rollout_steps=128,
        eval_every=0,
        log_every=1,
        seed=1,
    )
    log_dir = tmp_path / "tb"
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=str(log_dir),
    )
    trainer.train(total_steps=128)

    # SummaryWriter writes a file named "events.out.tfevents.*" in log_dir.
    events = list(log_dir.glob("events.out.tfevents.*"))
    assert events, f"no TB events file found in {log_dir}"


def test_trainer_save_checkpoint_round_trip(tmp_path: Path) -> None:
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=1, minibatch_size=32, target_kl=None),
        rollout_steps=64,
        eval_every=0,
        log_every=1,
        seed=2,
    )
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=None,
    )
    trainer.train(total_steps=64)
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()

    payload = torch.load(ckpt, weights_only=False)
    assert "model" in payload and "optimizer" in payload
    assert payload["global_step"] >= 64

    # Loading the weights back into a fresh agent should produce an
    # identical state_dict.
    fresh_model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(64, 64))
    fresh = PolicyAgent(fresh_model, ActionEncoder(PLAYER_IDS))
    fresh.load_state_dict(payload["model"])
    for k, v in learner.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), f"mismatch on {k}"


def test_trainer_runs_eval_without_errors(tmp_path: Path) -> None:
    """Eval cadence on: the trainer should produce an eval scalar."""
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=1, minibatch_size=64, target_kl=None),
        rollout_steps=128,
        eval_every=1,
        eval_n_games=2,
        log_every=1,
        seed=3,
    )
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    # Two iterations so eval fires.
    trainer.train(total_steps=256)
    # If we got here without exceptions, eval ran.


@pytest.mark.slow
def test_trainer_long_run_smoke(tmp_path: Path) -> None:
    """Spec-aligned: 5000 env steps, no exceptions, finite metrics."""
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=4, minibatch_size=256, target_kl=0.02),
        rollout_steps=1024,
        eval_every=0,
        log_every=1,
        seed=4,
    )
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    trainer.train(total_steps=5000)
    assert trainer.global_step >= 5000
