"""Slow convergence integration test for the PPO baseline (rl-015).

Run with::

    pytest -m slow tests/rl/test_trainer_convergence.py

The test trains the default-baseline PPO config for 500k env steps
against three random opponents and asserts a > 60% win rate at the end.
The threshold is intentionally below the rl-015 ten-million-step
acceptance criterion (> 90%) — 500k is the *nightly CI* checkpoint and
catches major regressions (silent reward bugs, broken masking, etc.)
without burning hours per run.

Wall time on a CPU: ~50 minutes for 500k steps at ~170 steps/sec. Move
to GPU + vectorised envs (rl-023) to bring this under 5 minutes.
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
from rl.evaluation.tournament import Tournament
from rl.models.mlp import MLPPolicyValue
from rl.training.config import DEFAULT_BASELINE_CONFIG
from rl.training.trainer import Trainer


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]

# Tunables for this test. The full 10M-step convergence target lives outside
# the test suite — run the trainer directly with cfg.eval_every=50_000 to
# observe the climb.
TRAINING_STEPS = 500_000
EVAL_GAMES = 100
EVAL_WIN_RATE_FLOOR = 0.60


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _opponent_factory(seed: int) -> dict[PlayerID, Agent]:
    rng = random.Random(seed)
    return {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in PLAYER_IDS
    }


@pytest.mark.slow
def test_ppo_beats_random_at_500k_steps(tmp_path: Path) -> None:
    """Train the default-baseline config for 500k steps; expect > 60% win rate."""
    torch.manual_seed(0)
    cfg = DEFAULT_BASELINE_CONFIG
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=cfg.hidden_sizes)
    learner = PolicyAgent(model, ActionEncoder(PLAYER_IDS))

    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_factory=_opponent_factory,
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
    )
    trainer.train(total_steps=TRAINING_STEPS)
    trainer.save_checkpoint(tmp_path / "final.pt")

    # Final eval: fresh seeds, fixed seat for the learner. This is the
    # canonical regression metric — winning > 60% of 100 games against
    # uniform-random opponents is the floor we expect from a healthy PPO
    # baseline that has at least seen the reward signal.
    eval_seat = PLAYER_IDS[0]
    opp_rng = random.Random(99)
    agents: dict[PlayerID, object] = {}
    for pid in PLAYER_IDS:
        if pid == eval_seat:
            agents[pid] = learner
        else:
            agents[pid] = RandomAgent(
                random.Random(opp_rng.randrange(2**32)), skip_proposals=True
            )
    result = Tournament(_env_factory).play(
        agents,  # type: ignore[arg-type]
        n_games=EVAL_GAMES,
        base_seed=10_000_000,
    )
    win_rate = result.win_rates[eval_seat]
    assert win_rate >= EVAL_WIN_RATE_FLOOR, (
        f"PPO baseline win rate vs random fell below {EVAL_WIN_RATE_FLOOR:.0%}: "
        f"{win_rate:.2%} after {TRAINING_STEPS} steps"
    )
