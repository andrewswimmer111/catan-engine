"""Tests for :mod:`rl.training.alphazero`.

Three tiers:

* **Fast.** Constructor validation + the policy/value loss against a
  hand-crafted batch (so the loss math is pinned without a real
  self-play loop).
* **Slow.** One end-to-end :meth:`AlphaZeroTrainer.train_iteration`
  with a tiny GNN and tiny rollout budget — confirms the whole
  ``self_play → buffer → update → snapshot`` loop runs cleanly and
  produces finite scalars.
* **Nightly.** A few iterations against a tiny config to verify the
  loss stays finite end-to-end without exploding; we don't assert
  monotonic decrease (too noisy at this scale).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from domain.ids import PlayerID  # noqa: E402
from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv  # noqa: E402
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.search.mcts import MCTSConfig  # noqa: E402
from rl.training.alphazero import AlphaZeroTrainer, AZTrainConfig  # noqa: E402
from rl.training.az_buffer import AZBatch  # noqa: E402
from rl.training.checkpoint import load_checkpoint  # noqa: E402
from rl.training.self_play import (  # noqa: E402
    SelfPlayConfig,
    SelfPlayTransition,
)


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
N_PLAYERS = len(PLAYER_IDS)


_TINY_GNN_VECTOR = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
)
_TINY_GNN_SCALAR = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
    value_kind="scalar",
)


def _make_vector_policy(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=_TINY_GNN_VECTOR,
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )


def _make_scalar_policy(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=_TINY_GNN_SCALAR,
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )


def _tiny_az_config(*, games_per_iter: int = 2, batches_per_iter: int = 4) -> AZTrainConfig:
    """A budget small enough for a fast / slow tier integration test."""
    return AZTrainConfig(
        self_play=SelfPlayConfig(
            mcts=MCTSConfig(
                rollouts=4, c_puct=2.0, seed=0, dirichlet_epsilon=0.25
            ),
            temperature_threshold_moves=4,
            max_moves=40,
        ),
        games_per_iter=games_per_iter,
        buffer_capacity=2000,
        batch_size=16,
        batches_per_iter=batches_per_iter,
        eval_every_iters=0,  # off in tests; enabled in dedicated test
        snapshot_every_iters=0,  # off; enabled in dedicated test
        seed=0,
    )


def _graph_env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed, obs_encoder=GraphObservationEncoder())


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_constructor_rejects_scalar_value_head() -> None:
    """AZ requires per-seat vector outputs; a scalar-head learner is a
    caller error and should be caught at construction time, not
    silently produce a shape-mismatch crash deep in the loss."""
    learner = _make_scalar_policy()
    with pytest.raises(ValueError, match="vector value head"):
        AlphaZeroTrainer(learner, _tiny_az_config())


# ----------------------------------------------------------------------
# Loss computation
# ----------------------------------------------------------------------


def _handcrafted_batch(batch_size: int = 4) -> AZBatch:
    """A small batch of zero-obs / spike-policy / zero-value transitions.

    Useful for exercising the loss computation independently of self-play.
    """
    obs = np.zeros((batch_size, GRAPH_OBS_SHAPE[0]), dtype=np.float32)
    # Mark a couple of action slots legal so the masked log-softmax has
    # somewhere to put probability mass.
    mask = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=bool)
    mask[:, 0] = True
    mask[:, 1] = True
    # Policy target: 0.7 / 0.3 split on the two legal slots.
    target = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.float32)
    target[:, 0] = 0.7
    target[:, 1] = 0.3
    value = np.zeros((batch_size, N_PLAYERS), dtype=np.float32)
    return AZBatch(
        obs=obs, action_mask=mask, policy_target=target, value_target=value
    )


def test_train_step_emits_finite_metrics() -> None:
    """One Adam step on a hand-crafted batch produces finite scalar metrics."""
    learner = _make_vector_policy(seed=5)
    trainer = AlphaZeroTrainer(learner, _tiny_az_config())
    metrics = trainer._train_step(_handcrafted_batch(batch_size=4))
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite {k}={v}"
    # Policy CE is non-negative; value MSE is non-negative.
    assert metrics["policy_loss"] >= 0.0
    assert metrics["value_loss"] >= 0.0


def test_train_step_handles_illegal_action_masking() -> None:
    """The loss must NOT produce NaN when illegal-slot logits are -1e9
    (the model's mask-fill convention). Regression guard: a naive
    ``policy_target * log_softmax(logits)`` would multiply ``0 * -inf``
    on illegal slots and emit NaN; the trainer's masked-zero path must
    handle this cleanly.
    """
    learner = _make_vector_policy(seed=6)
    trainer = AlphaZeroTrainer(learner, _tiny_az_config())
    metrics = trainer._train_step(_handcrafted_batch(batch_size=8))
    assert np.isfinite(metrics["policy_loss"])


# ----------------------------------------------------------------------
# Iteration loop (integration)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_train_iteration_runs_end_to_end(tmp_path: Path) -> None:
    """One AZ iteration completes and produces a non-empty buffer +
    finite loss metrics. Stresses the whole self-play → update path."""
    learner = _make_vector_policy(seed=11)
    trainer = AlphaZeroTrainer(
        learner,
        _tiny_az_config(games_per_iter=2, batches_per_iter=3),
        env_factory=_graph_env_factory,
        log_dir=tmp_path / "tb",
        snapshot_dir=tmp_path / "snap",
    )
    summary = trainer.train_iteration()

    # Buffer received transitions.
    assert trainer.buffer_size > 0
    # Self-play summary populated.
    assert summary["self_play/n_games"] == 2
    assert summary["self_play/n_transitions"] > 0
    # Updates produced finite losses.
    assert np.isfinite(summary["train/policy_loss"])
    assert np.isfinite(summary["train/value_loss"])
    assert np.isfinite(summary["train/total_loss"])
    # Iteration counter advanced.
    assert trainer.iteration == 1


@pytest.mark.slow
def test_train_iteration_snapshots_on_cadence(tmp_path: Path) -> None:
    """``snapshot_every_iters=1`` should produce a checkpoint after iter 1
    and the checkpoint should round-trip to the AZ-shape vector learner."""
    learner = _make_vector_policy(seed=13)
    cfg = AZTrainConfig(
        self_play=SelfPlayConfig(
            mcts=MCTSConfig(rollouts=2, c_puct=2.0, seed=0),
            max_moves=30,
        ),
        games_per_iter=1,
        buffer_capacity=200,
        batch_size=8,
        batches_per_iter=2,
        eval_every_iters=0,
        snapshot_every_iters=1,
        seed=0,
    )
    trainer = AlphaZeroTrainer(
        learner,
        cfg,
        env_factory=_graph_env_factory,
        snapshot_dir=tmp_path / "snap",
    )
    trainer.train_iteration()

    snap_path = tmp_path / "snap" / "iter_1.pt"
    assert snap_path.exists()
    loaded, meta = load_checkpoint(snap_path)
    assert meta.model_arch.gnn_arch is not None
    assert meta.model_arch.gnn_arch.value_kind == "vector"


@pytest.mark.slow
def test_train_iteration_runs_eval_on_cadence(tmp_path: Path) -> None:
    """``eval_every_iters=1`` should populate the eval/* fields in the
    iteration summary."""
    learner = _make_vector_policy(seed=15)
    cfg = AZTrainConfig(
        self_play=SelfPlayConfig(
            mcts=MCTSConfig(rollouts=2, c_puct=2.0, seed=0),
            max_moves=30,
        ),
        games_per_iter=1,
        buffer_capacity=200,
        batch_size=8,
        batches_per_iter=2,
        eval_every_iters=1,
        eval_games=2,
        snapshot_every_iters=0,
        seed=0,
    )
    trainer = AlphaZeroTrainer(
        learner, cfg, env_factory=_graph_env_factory
    )
    summary = trainer.train_iteration()

    # Both anchor evals ran.
    assert "eval/vs_random/win_rate" in summary
    assert "eval/vs_heuristic/win_rate" in summary
    # Win rates are real numbers in [0, 1].
    for key in ("eval/vs_random/win_rate", "eval/vs_heuristic/win_rate"):
        v = summary[key]
        assert 0.0 <= v <= 1.0


# ----------------------------------------------------------------------
# Stability over a few iterations
# ----------------------------------------------------------------------


@pytest.mark.nightly
def test_multi_iter_loop_keeps_losses_finite(tmp_path: Path) -> None:
    """Run several AZ iterations on a tiny config; loss never NaN/Inf
    and grad_norm stays bounded. Wall time: ~1–3 minutes on CPU."""
    learner = _make_vector_policy(seed=21)
    cfg = _tiny_az_config(games_per_iter=3, batches_per_iter=8)
    trainer = AlphaZeroTrainer(
        learner, cfg, env_factory=_graph_env_factory
    )
    for _ in range(3):
        summary = trainer.train_iteration()
        assert np.isfinite(summary["train/policy_loss"])
        assert np.isfinite(summary["train/value_loss"])
        # Reasonable grad-norm cap: max_grad_norm=5 → clipped grad_norm <= 5
        # (the metric is the *pre-clip* norm, which can be larger; cap
        # generously so this doesn't false-trigger).
        assert summary["train/grad_norm"] < 1e6, (
            f"grad_norm exploded: {summary['train/grad_norm']}"
        )
    # Buffer grew across iterations.
    assert trainer.buffer_size > 0
