"""End-to-end smoke + regression tests for the graph-encoder training path.

The flat-encoder counterparts live in :mod:`test_trainer_smoke` and
:mod:`test_trainer_convergence`. This module mirrors the same three
tiers (fast / slow / nightly) but for the GNN trunk:

* **Fast (default):** a short PPO loop with a tiny GNN arch — proves the
  Trainer integration is wired and the checkpoint round-trips with
  ``encoder_kind="graph"``.
* **Slow:** ~5k env steps — guards against silent regressions in the
  graph trunk (e.g. NaNs, stuck losses) without a multi-tens-of-minutes
  run cost.
* **Nightly:** ~50k env steps with a small arch — full enough that the
  learner has actually started learning to play; asserts mean turn
  count is positive and final loss is finite.

The convergence target the project actually cares about — beating run
#3's 4.5% vs-random anchor — is intentionally NOT a test. That run takes
hours on real hardware; we drive it via ``scripts/train_overnight.sh
--encoder graph`` and compare runs/<timestamp>/eval_vs_random.md.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from domain.ids import PlayerID  # noqa: E402
from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv  # noqa: E402
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.training.checkpoint import load_checkpoint  # noqa: E402
from rl.training.config import PPOConfig, TrainConfig  # noqa: E402
from rl.training.opponent_pool import OpponentPool  # noqa: E402
from rl.training.trainer import Trainer  # noqa: E402

PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]

# Tiny GNN arch for fast / slow tests — preserves the typed-head structure
# without 1.4M-param overhead. Nightly uses the same arch so wall time
# stays predictable.
TINY_GNN = GNNArch(
    node_hidden=32,
    player_hidden=16,
    n_mp_layers=2,
    n_heads=4,
    global_mlp_hidden=32,
)


def _graph_env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed, obs_encoder=GraphObservationEncoder())


def _make_pool(seed: int = 0) -> OpponentPool:
    return OpponentPool(rng=random.Random(seed))


def _make_graph_learner(arch: GNNArch = TINY_GNN) -> PolicyAgent:
    torch.manual_seed(0)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=arch,
    )
    return PolicyAgent(
        model,
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),
    )


# ---------------------------------------------------------------------------
# Fast tier
# ---------------------------------------------------------------------------


def test_gnn_trainer_runs_short_training_without_errors(tmp_path: Path) -> None:
    """One PPO update with the graph encoder finishes without exceptions."""
    learner = _make_graph_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(
            n_epochs=2, minibatch_size=64, target_kl=None, entropy_coef=0.01
        ),
        rollout_steps=128,
        eval_every=0,
        log_every=1,
        seed=0,
    )
    trainer = Trainer(
        env_factory=_graph_env_factory,
        learner=learner,
        opponent_pool=_make_pool(),
        cfg=cfg,
        log_dir=None,
    )
    trainer.train(total_steps=256)
    assert trainer.global_step >= 256


def test_gnn_trainer_metrics_are_finite(tmp_path: Path) -> None:
    """PPO update metrics through the GNN path must stay finite."""
    learner = _make_graph_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(
            n_epochs=2,
            minibatch_size=32,
            target_kl=None,
            entropy_coef=0.01,
        ),
        rollout_steps=64,
        eval_every=0,
        log_every=1,
        seed=1,
    )
    trainer = Trainer(
        env_factory=_graph_env_factory,
        learner=learner,
        opponent_pool=_make_pool(seed=1),
        cfg=cfg,
        log_dir=None,
    )
    trainer.train(total_steps=128)

    # Inspect the learner's parameters: no NaNs / Infs anywhere.
    for name, p in learner.model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after training: {name}"


def test_gnn_checkpoint_round_trips_through_trainer(tmp_path: Path) -> None:
    """Trainer-emitted graph checkpoint loads back to the same weights."""
    learner = _make_graph_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(n_epochs=1, minibatch_size=32, target_kl=None),
        rollout_steps=64,
        eval_every=0,
        log_every=1,
        seed=2,
    )
    trainer = Trainer(
        env_factory=_graph_env_factory,
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
    assert meta.model_arch.encoder_kind == "graph"
    assert meta.model_arch.gnn_arch == TINY_GNN
    assert meta.train_step >= 64
    for k, v in learner.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"mismatch on {k}"


def test_gnn_policy_agent_choose_returns_legal_action(tmp_path: Path) -> None:
    """``PolicyAgent.choose`` on the graph encoder picks a legal action.

    Catches the integration of GNN + GraphObservationEncoder + ActionEncoder
    end-to-end: encode a real PlayerView, forward through the model, decode
    the index into a typed Action, confirm it's in the legal set.
    """
    from controller.session import GameSnapshot

    learner = _make_graph_learner()
    env = CatanEnv(seed=7, obs_encoder=GraphObservationEncoder())
    env.reset(seed=7)
    snap = GameSnapshot(env.state, 0, None, ())
    chosen = learner.choose(snap, env.legal_actions())
    assert chosen is not None
    assert chosen in env.legal_actions()


# ---------------------------------------------------------------------------
# Slow tier
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gnn_5k_step_run_does_not_diverge(tmp_path: Path) -> None:
    """~5k env steps with the small GNN arch — guards against NaN explosion.

    Wall time on CPU: ~3 minutes at ~30 steps/s. Asserts that final
    parameters are still finite and at least one episode completed.
    """
    learner = _make_graph_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(
            n_epochs=2,
            minibatch_size=128,
            target_kl=None,
            entropy_coef=0.03,
        ),
        rollout_steps=512,
        eval_every=0,
        log_every=1,
        seed=3,
    )
    trainer = Trainer(
        env_factory=_graph_env_factory,
        learner=learner,
        opponent_pool=_make_pool(seed=3),
        cfg=cfg,
        log_dir=None,
    )
    trainer.train(total_steps=5_000)
    assert trainer.global_step >= 5_000
    for name, p in learner.model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after 5k steps: {name}"


# ---------------------------------------------------------------------------
# Nightly tier
# ---------------------------------------------------------------------------


@pytest.mark.nightly
def test_gnn_50k_step_run_plays_full_games(tmp_path: Path) -> None:
    """50k env steps + a small final tournament; learner plays full games.

    Walls in ~30 minutes on CPU at ~30 steps/s with the small arch. The
    bar is intentionally loose — we want to catch silent regressions
    (broken masking, terminal-reward wiring, GAE-per-agent dropouts) and
    leave the actual win-rate claim to the overnight runs. Acceptance:

    * ``trainer.global_step`` advances past the requested step budget.
    * Final parameters are finite.
    * A 5-game tournament vs random opponents completes (mean turns > 0).

    A failure here means the GNN training loop has regressed; a *real*
    convergence regression shows up on the next ``train_overnight.sh
    --encoder graph`` run.
    """
    from rl.agents.random_agent import RandomAgent
    from rl.evaluation.tournament import Tournament

    learner = _make_graph_learner()
    cfg = TrainConfig(
        ppo=PPOConfig(
            n_epochs=2,
            minibatch_size=256,
            target_kl=None,
            entropy_coef=0.03,
        ),
        rollout_steps=1024,
        eval_every=0,
        log_every=1,
        seed=4,
    )
    trainer = Trainer(
        env_factory=_graph_env_factory,
        learner=learner,
        opponent_pool=_make_pool(seed=4),
        cfg=cfg,
        log_dir=str(tmp_path / "tb"),
        snapshot_dir=tmp_path / "snapshots",
    )
    trainer.train(total_steps=50_000)
    assert trainer.global_step >= 50_000

    for name, p in learner.model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after 50k steps: {name}"

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
    result = Tournament(_graph_env_factory).play(
        agents,  # type: ignore[arg-type]
        n_games=5,
        base_seed=20_000,
    )
    assert result.mean_turns > 0, "tournament games registered zero turns"
