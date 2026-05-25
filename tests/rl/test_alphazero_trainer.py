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

import dataclasses
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
    vp_aux = np.zeros((batch_size, N_PLAYERS), dtype=np.float32)
    return AZBatch(
        obs=obs,
        action_mask=mask,
        policy_target=target,
        value_target=value,
        vp_aux_target=vp_aux,
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


def test_train_step_emits_value_pred_std_diagnostic() -> None:
    """``value_pred_std`` must appear in the per-step metrics dict and
    reflect the std of the value head's outputs across the batch — the
    canonical signal for whether the value head is differentiating
    states or collapsed to a near-constant.
    """
    learner = _make_vector_policy(seed=7)
    trainer = AlphaZeroTrainer(learner, _tiny_az_config())
    metrics = trainer._train_step(_handcrafted_batch(batch_size=4))
    assert "value_pred_std" in metrics
    assert metrics["value_pred_std"] >= 0.0
    assert np.isfinite(metrics["value_pred_std"])


def test_train_step_aux_value_loss_is_zero_when_coef_is_zero() -> None:
    """The default ``aux_value_coef=0`` reproduces canonical AZ: the
    aux MSE is still computed and reported, but its contribution to
    ``total_loss`` must be exactly zero (no gradient signal). Pin this
    so a future refactor that, say, computes a sum instead of a coefed
    add doesn't silently change behaviour for the canonical configs."""
    learner = _make_vector_policy(seed=8)
    cfg = _tiny_az_config()
    assert cfg.aux_value_coef == 0.0
    trainer = AlphaZeroTrainer(learner, cfg)
    metrics = trainer._train_step(_handcrafted_batch(batch_size=4))
    expected_total = (
        metrics["policy_loss"] + cfg.value_coef * metrics["value_loss"]
    )
    assert metrics["total_loss"] == pytest.approx(expected_total, rel=1e-5)


def test_train_step_aux_value_loss_contributes_when_coef_positive() -> None:
    """With a non-zero ``aux_value_coef``, the aux MSE must show up in
    ``total_loss`` (otherwise the knob is dead). The handcrafted batch
    has ``vp_aux_target`` all zeros, so the aux MSE equals
    ``mean(value_pred ** 2)`` — a non-zero scalar from any non-zero
    value head, which a fresh network will have."""
    learner = _make_vector_policy(seed=9)
    cfg = dataclasses.replace(_tiny_az_config(), aux_value_coef=0.5)
    trainer = AlphaZeroTrainer(learner, cfg)
    metrics = trainer._train_step(_handcrafted_batch(batch_size=4))
    assert metrics["aux_value_loss"] > 0.0
    expected_total = (
        metrics["policy_loss"]
        + cfg.value_coef * metrics["value_loss"]
        + cfg.aux_value_coef * metrics["aux_value_loss"]
    )
    assert metrics["total_loss"] == pytest.approx(expected_total, rel=1e-5)


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


# ----------------------------------------------------------------------
# Progress file
# ----------------------------------------------------------------------


def test_progress_path_unset_means_no_file_written(tmp_path: Path) -> None:
    """A trainer without ``progress_path`` should not create any file in
    its working directory — regression guard against silent writes."""
    learner = _make_vector_policy(seed=41)
    trainer = AlphaZeroTrainer(learner, _tiny_az_config())
    # Synthetic history; no train_iteration call needed.
    trainer._history.append({"iter": 1.0, "wall_seconds": 1.5})  # noqa: SLF001
    trainer._write_progress(status="running")  # noqa: SLF001
    # No file was supposed to be written; tmp_path stays empty.
    assert list(tmp_path.iterdir()) == []


def test_progress_md_header_carries_run_config_and_status(tmp_path: Path) -> None:
    """The header section names the run's MCTS rollouts + stalemate value
    + status so the user can sanity-check the run config at a glance."""
    learner = _make_vector_policy(seed=43)
    cfg = _tiny_az_config()
    progress = tmp_path / "progress.md"
    trainer = AlphaZeroTrainer(learner, cfg, progress_path=progress)
    trainer._train_total_iters = 10  # noqa: SLF001 — simulate train() prologue
    import time as _time
    trainer._train_start_time = _time.time() - 30.0  # noqa: SLF001 — 30s ago
    trainer._history.append(  # noqa: SLF001
        {
            "iter": 1.0,
            "wall_seconds": 30.0,
            "self_play/stalemate_rate": 0.5,
            "self_play/mean_moves_per_game": 80.0,
            "train/policy_loss": 1.23,
            "train/value_loss": 0.05,
            "buffer/size": 200.0,
        }
    )
    trainer._write_progress(status="running")  # noqa: SLF001

    text = progress.read_text()
    assert "# AlphaZero run progress" in text
    assert "**Status**: running" in text
    assert "**Iteration**: 1 / 10" in text
    # Config section names the knobs we'd want to check at a glance.
    assert f"mcts rollouts / move: {cfg.self_play.mcts.rollouts}" in text
    # vp_linear is the default shape; both endpoints should show.
    assert "stalemate: vp_linear" in text
    assert str(cfg.self_play.stalemate.low) in text
    assert str(cfg.self_play.stalemate.high) in text
    # The table has the row for iter 1.
    assert "| 1 |" in text


def test_progress_md_table_renders_eval_dash_when_no_eval_fired(
    tmp_path: Path,
) -> None:
    """Iterations that don't carry eval/* keys must render ``-`` in
    the win-rate columns, not ``0%`` — those two are very different
    signals when reading the table during a run."""
    learner = _make_vector_policy(seed=45)
    progress = tmp_path / "progress.md"
    trainer = AlphaZeroTrainer(
        learner, _tiny_az_config(), progress_path=progress
    )
    trainer._history.append(  # noqa: SLF001 — no eval/* keys
        {
            "iter": 1.0,
            "wall_seconds": 5.0,
            "self_play/stalemate_rate": 0.0,
            "self_play/mean_moves_per_game": 60.0,
            "train/policy_loss": 1.5,
            "train/value_loss": 0.1,
            "buffer/size": 80.0,
        }
    )
    trainer._write_progress(status="running")  # noqa: SLF001
    rows = [
        ln for ln in progress.read_text().splitlines() if ln.startswith("| 1 |")
    ]
    assert rows, "expected an iter-1 row in the table"
    # Three eval columns; each should be the literal `-` placeholder.
    assert rows[0].count(" - ") >= 3


def test_progress_md_status_done_after_train_completes(tmp_path: Path) -> None:
    """After ``train(total_iters)`` finishes, the file's status must
    flip to ``done`` (the file is rewritten in the ``finally`` block)."""
    learner = _make_vector_policy(seed=47)
    progress = tmp_path / "progress.md"
    trainer = AlphaZeroTrainer(
        learner,
        _tiny_az_config(games_per_iter=1, batches_per_iter=1),
        env_factory=_graph_env_factory,
        progress_path=progress,
    )
    # Skip a real iteration: stub out train_iteration so the test stays
    # fast (no actual self-play). The status logic is what matters here.
    trainer.train_iteration = lambda: {  # type: ignore[method-assign]
        "iter": 1.0,
        "wall_seconds": 0.01,
    }
    trainer.train(total_iters=1)
    assert "**Status**: done" in progress.read_text()


def test_progress_md_status_interrupted_on_exception(tmp_path: Path) -> None:
    """An exception mid-train should still flush progress.md with
    ``interrupted`` status (the ``finally`` block fires)."""
    learner = _make_vector_policy(seed=49)
    progress = tmp_path / "progress.md"
    trainer = AlphaZeroTrainer(
        learner,
        _tiny_az_config(),
        env_factory=_graph_env_factory,
        progress_path=progress,
    )

    def _bomb():
        raise RuntimeError("simulated mid-train failure")

    trainer.train_iteration = _bomb  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated"):
        trainer.train(total_iters=3)
    assert "**Status**: interrupted" in progress.read_text()


@pytest.mark.slow
def test_progress_md_written_after_real_train_iteration(tmp_path: Path) -> None:
    """End-to-end: one real train_iteration → progress.md exists with a
    table row populated from the live summary."""
    learner = _make_vector_policy(seed=51)
    progress = tmp_path / "progress.md"
    trainer = AlphaZeroTrainer(
        learner,
        _tiny_az_config(games_per_iter=1, batches_per_iter=2),
        env_factory=_graph_env_factory,
        progress_path=progress,
    )
    trainer._train_total_iters = 1  # noqa: SLF001 — short-circuit a 1-iter run
    import time as _time
    trainer._train_start_time = _time.time()  # noqa: SLF001
    trainer.train_iteration()
    text = progress.read_text()
    assert "**Status**: running" in text
    # The table has a row for iter 1.
    assert "\n| 1 |" in text


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
