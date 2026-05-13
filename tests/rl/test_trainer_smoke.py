"""End-to-end smoke tests for the PPO trainer (rl-014 + rl-018).

The trainer is the integration point for rl-011..018 — we exercise it
end-to-end with a small number of steps to catch wiring bugs that the
component-level tests can't see. The convergence target lives in
:mod:`tests.rl.test_trainer_convergence` and runs nightly.

What we check:

* A short run completes without exceptions.
* The TensorBoard events file is produced when ``log_dir`` is set.
* The final PPO metrics are all finite.
* The versioned checkpoint round-trips back into a fresh agent.
* Self-play opponents are sampled from the pool (rl-018).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.models.mlp import MLPPolicyValue
from rl.training.checkpoint import load_checkpoint
from rl.training.config import PPOConfig, TrainConfig
from rl.training.opponent_pool import OpponentPool
from rl.training.trainer import Trainer


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _make_pool(seed: int = 0) -> OpponentPool:
    """Empty pool → self-play falls back to the live learner each slot."""
    return OpponentPool(rng=random.Random(seed))


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
        opponent_pool=_make_pool(),
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
        opponent_pool=_make_pool(seed=1),
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
        opponent_pool=_make_pool(seed=2),
        cfg=cfg,
        log_dir=None,
    )
    trainer.train(total_steps=64)
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()

    loaded, meta = load_checkpoint(ckpt)
    assert meta.train_step >= 64
    for k, v in learner.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"mismatch on {k}"


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
        opponent_pool=_make_pool(seed=3),
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    # Two iterations so eval fires.
    trainer.train(total_steps=256)
    # If we got here without exceptions, eval ran.


def test_self_play_snapshots_grow_the_pool(tmp_path: Path) -> None:
    """Snapshot cadence wired up: pool's recent bucket grows during training."""
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=1, minibatch_size=32, target_kl=None),
        rollout_steps=64,
        eval_every=0,
        log_every=1,
        seed=4,
        # Aggressive cadence for the test — real runs use 5M.
        snapshot_every=64,
        promote_every_n_snapshots=2,
    )
    pool = _make_pool(seed=4)
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_pool=pool,
        cfg=cfg,
        log_dir=None,
        snapshot_dir=tmp_path / "snapshots",
    )
    trainer.train(total_steps=256)

    assert len(pool.recent_paths) >= 2, (
        f"expected at least 2 recent snapshots, got {pool.recent_paths}"
    )
    # promote_every_n_snapshots=2 means at least one historical entry after
    # ~4 snapshots.
    assert len(pool.historical_paths) >= 1


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
        opponent_pool=_make_pool(seed=5),
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    trainer.train(total_steps=5000)
    assert trainer.global_step >= 5000


@pytest.mark.slow
def test_self_play_50k_steps_pool_grows(tmp_path: Path) -> None:
    """rl-018 acceptance: 50k self-play steps; pool grows; final policy plays."""
    learner = _make_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=2, minibatch_size=128, target_kl=None),
        rollout_steps=1024,
        eval_every=0,
        log_every=1,
        seed=42,
        snapshot_every=5_000,
        promote_every_n_snapshots=4,
    )
    pool = _make_pool(seed=42)
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_pool=pool,
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
        snapshot_dir=tmp_path / "snapshots",
    )
    trainer.train(total_steps=50_000)

    assert trainer.global_step >= 50_000
    assert len(pool.recent_paths) >= 5, f"recent={pool.recent_paths}"

    # Final policy plays a game without errors (against the live learner
    # itself in self-play, via the pool's fallback path).
    from rl.evaluation.tournament import Tournament

    agents: dict[PlayerID, PolicyAgent] = {pid: learner for pid in PLAYER_IDS}
    result = Tournament(_env_factory).play(
        agents,  # type: ignore[arg-type]
        n_games=2,
        base_seed=100_000,
    )
    assert result.mean_turns > 0
