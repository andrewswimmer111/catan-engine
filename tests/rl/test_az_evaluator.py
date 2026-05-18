"""Tests for :class:`rl.evaluation.az_evaluator.AZEvaluator`.

Three concerns:

* **Cadence.** ``maybe_run`` only fires on configured iterations.
* **Prior-snapshot lifecycle.** With promotion off (the default) the
  prior-snapshot refreshes every call; with promotion on it refreshes
  only when the most recent vs-prior win-rate crossed the threshold AND
  came from the same iteration.
* **Elo + ratings file.** Per-iteration learner IDs accumulate distinct
  ratings; the ratings file is written each eval when a path is set.

The slow integration test runs one real eval iteration so we know the
underlying Tournament + Elo + logger plumbing works end-to-end.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

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
from rl.evaluation.az_evaluator import AZEvalConfig, AZEvaluator  # noqa: E402
from rl.evaluation.elo import EloTracker  # noqa: E402
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.utils.logging import NoOpLogger  # noqa: E402


PLAYER_IDS = tuple(PlayerID(i) for i in range(1, 5))


_TINY_GNN = GNNArch(
    node_hidden=16, player_hidden=8, n_mp_layers=1,
    n_heads=4, global_mlp_hidden=16,
)


def _make_policy(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=_TINY_GNN,
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(list(PLAYER_IDS)),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed, obs_encoder=GraphObservationEncoder())


def _eval_cfg(**overrides) -> AZEvalConfig:
    base = dict(
        every_iters=1,
        n_games=2,
        bench_random=True,
        bench_heuristic=False,
        bench_prior_snapshot=False,
        promotion_threshold=None,
        player_ids=PLAYER_IDS,
    )
    base.update(overrides)
    return AZEvalConfig(**base)


# ----------------------------------------------------------------------
# Cadence
# ----------------------------------------------------------------------


def test_maybe_run_returns_none_when_disabled() -> None:
    """``every_iters <= 0`` disables the evaluator entirely."""
    evaluator = AZEvaluator(
        _eval_cfg(every_iters=0),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    learner = _make_policy()
    assert evaluator.maybe_run(learner, iteration=1) is None
    assert evaluator.maybe_run(learner, iteration=5) is None


def test_maybe_run_returns_none_off_cadence() -> None:
    """``every_iters=3`` fires only on multiples of 3."""
    evaluator = AZEvaluator(
        _eval_cfg(every_iters=3),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    learner = _make_policy()
    assert evaluator.maybe_run(learner, iteration=1) is None
    assert evaluator.maybe_run(learner, iteration=2) is None
    # On-cadence call returns a dict (may be empty if no benches are on).
    result = evaluator.maybe_run(learner, iteration=3)
    assert result is not None


# ----------------------------------------------------------------------
# Prior-snapshot lifecycle
# ----------------------------------------------------------------------


def test_no_prior_snapshot_means_vs_prior_bench_is_skipped() -> None:
    """Without a previous ``refresh_prior_snapshot`` call, the prior-
    snapshot benchmark must be silently skipped — otherwise iter 1
    would have nothing to compare against."""
    evaluator = AZEvaluator(
        _eval_cfg(
            bench_random=False,
            bench_heuristic=False,
            bench_prior_snapshot=True,
        ),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    learner = _make_policy()
    result = evaluator.maybe_run(learner, iteration=1)
    assert result == {}
    assert evaluator.prior_snapshot_iter is None


def test_refresh_prior_snapshot_seeds_on_first_call_regardless_of_threshold() -> None:
    """Even with a strict promotion threshold, the first refresh must
    succeed — there's nothing to compare against on iter 1, so the gate
    is bypassed for the seeding call."""
    evaluator = AZEvaluator(
        _eval_cfg(promotion_threshold=0.55),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    promoted = evaluator.refresh_prior_snapshot(_make_policy(), iteration=1)
    assert promoted is True
    assert evaluator.prior_snapshot_iter == 1


def test_continuous_mode_refreshes_prior_snapshot_every_call() -> None:
    """With ``promotion_threshold=None`` every refresh updates the prior."""
    evaluator = AZEvaluator(
        _eval_cfg(promotion_threshold=None),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    assert evaluator.refresh_prior_snapshot(_make_policy(seed=1), iteration=1)
    assert evaluator.prior_snapshot_iter == 1
    assert evaluator.refresh_prior_snapshot(_make_policy(seed=2), iteration=2)
    assert evaluator.prior_snapshot_iter == 2


def test_promotion_threshold_blocks_refresh_when_win_rate_low() -> None:
    """When the latest vs-prior win-rate is below threshold, refresh fails."""
    evaluator = AZEvaluator(
        _eval_cfg(promotion_threshold=0.55),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    # Seed the first prior-snapshot (always succeeds).
    evaluator.refresh_prior_snapshot(_make_policy(seed=1), iteration=1)
    # Simulate a vs-prior eval at iter 2 with a sub-threshold win rate.
    evaluator._last_vs_prior_win_rate = 0.30  # type: ignore[attr-defined]
    evaluator._last_vs_prior_eval_iter = 2  # type: ignore[attr-defined]
    promoted = evaluator.refresh_prior_snapshot(_make_policy(seed=2), iteration=2)
    assert promoted is False
    assert evaluator.prior_snapshot_iter == 1


def test_promotion_threshold_allows_refresh_when_win_rate_high() -> None:
    evaluator = AZEvaluator(
        _eval_cfg(promotion_threshold=0.55),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    evaluator.refresh_prior_snapshot(_make_policy(seed=1), iteration=1)
    evaluator._last_vs_prior_win_rate = 0.70  # type: ignore[attr-defined]
    evaluator._last_vs_prior_eval_iter = 2  # type: ignore[attr-defined]
    promoted = evaluator.refresh_prior_snapshot(_make_policy(seed=2), iteration=2)
    assert promoted is True
    assert evaluator.prior_snapshot_iter == 2


def test_stale_win_rate_does_not_count_for_promotion() -> None:
    """A win-rate measured at iter N must NOT gate a refresh at iter N+1.

    Misaligned eval/snapshot cadences would otherwise let an old
    favourable measurement promote a fresh (untested) learner.
    """
    evaluator = AZEvaluator(
        _eval_cfg(promotion_threshold=0.55),
        env_factory=_env_factory,
        logger=NoOpLogger(),
    )
    evaluator.refresh_prior_snapshot(_make_policy(seed=1), iteration=1)
    evaluator._last_vs_prior_win_rate = 0.99  # type: ignore[attr-defined]
    evaluator._last_vs_prior_eval_iter = 2  # type: ignore[attr-defined]
    # Attempt to refresh at iter 5 — the iter-2 win-rate is stale.
    promoted = evaluator.refresh_prior_snapshot(_make_policy(seed=3), iteration=5)
    assert promoted is False
    assert evaluator.prior_snapshot_iter == 1


# ----------------------------------------------------------------------
# Eval + Elo integration
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_maybe_run_against_random_logs_win_rate_and_elo(tmp_path: Path) -> None:
    """One real eval against random opponents: dict contains win_rate +
    elo + mean_turns, and the ratings file is written."""
    ratings_path = tmp_path / "elo.json"
    evaluator = AZEvaluator(
        _eval_cfg(bench_random=True, bench_heuristic=False),
        env_factory=_env_factory,
        logger=NoOpLogger(),
        elo=EloTracker(),
        ratings_path=ratings_path,
    )
    learner = _make_policy(seed=7)
    result = evaluator.maybe_run(learner, iteration=1)
    assert result is not None
    assert "vs_random/win_rate" in result
    assert "vs_random/elo" in result
    assert "vs_random/mean_turns" in result
    # Ratings file written + parseable JSON.
    assert ratings_path.exists()
    parsed = json.loads(ratings_path.read_text())
    assert "ratings" in parsed
    # Learner-specific iteration ID is present.
    assert any("learner@iter_" in k for k in parsed["ratings"].keys())


@pytest.mark.slow
def test_maybe_run_against_prior_snapshot(tmp_path: Path) -> None:
    """Seed a prior-snapshot, then eval; vs_prior_snapshot/* fields show."""
    evaluator = AZEvaluator(
        _eval_cfg(
            bench_random=False,
            bench_heuristic=False,
            bench_prior_snapshot=True,
        ),
        env_factory=_env_factory,
        logger=NoOpLogger(),
        elo=EloTracker(),
    )
    seed_learner = _make_policy(seed=11)
    evaluator.refresh_prior_snapshot(seed_learner, iteration=1)

    later_learner = _make_policy(seed=13)
    result = evaluator.maybe_run(later_learner, iteration=2)
    assert result is not None
    assert "vs_prior_snapshot/win_rate" in result
    assert "vs_prior_snapshot/elo" in result
    # Last-vs-prior tracking populated for the next promotion gate.
    assert evaluator.last_vs_prior_win_rate is not None


# ----------------------------------------------------------------------
# Trainer integration
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_alphazero_trainer_delegates_eval_to_evaluator(tmp_path: Path) -> None:
    """When an AZEvaluator is attached, the trainer's eval summary keys
    come from the evaluator (vs_random/vs_heuristic/win_rate, not the
    lite trainer-internal fallback)."""
    from rl.search.mcts import MCTSConfig
    from rl.training.alphazero import AlphaZeroTrainer, AZTrainConfig
    from rl.training.self_play import SelfPlayConfig

    learner = _make_policy(seed=21)
    cfg = AZTrainConfig(
        self_play=SelfPlayConfig(
            mcts=MCTSConfig(rollouts=2, c_puct=2.0, seed=0),
            max_moves=20,
        ),
        games_per_iter=1,
        buffer_capacity=100,
        batch_size=4,
        batches_per_iter=2,
        eval_every_iters=1,
        eval_games=2,
        snapshot_every_iters=1,
        seed=0,
    )
    evaluator = AZEvaluator(
        _eval_cfg(
            every_iters=1,
            n_games=2,
            bench_random=True,
            bench_heuristic=False,
            bench_prior_snapshot=True,
        ),
        env_factory=_env_factory,
        logger=NoOpLogger(),
        elo=EloTracker(),
    )
    trainer = AlphaZeroTrainer(
        learner,
        cfg,
        env_factory=_env_factory,
        snapshot_dir=tmp_path / "snap",
        evaluator=evaluator,
    )
    summary = trainer.train_iteration()
    # vs_random/* keys come from the evaluator path.
    assert "eval/vs_random/win_rate" in summary
    assert "eval/vs_random/elo" in summary
    # After the iter's snapshot, the prior-snapshot is seeded (continuous
    # mode, no threshold). Next iter's eval should include vs_prior_*.
    assert evaluator.prior_snapshot_iter == 1
    summary2 = trainer.train_iteration()
    assert "eval/vs_prior_snapshot/win_rate" in summary2
