"""Tests for the ``scripts/train_alphazero.py`` driver.

Mostly **CLI shape** tests — argument parsing, config wiring,
warm-start validation, output-directory layout. The end-to-end loop is
exercised in :mod:`tests.rl.test_alphazero_trainer`; here we just
confirm the script-level glue is correct.

A single slow smoke test runs the driver against a tiny config to
verify it produces the expected on-disk artefacts (``config.json``,
``final.pt``, ``snapshots/``, ``tb/``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

# Ensure scripts/ is importable in test environment.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_alphazero  # noqa: E402


# ----------------------------------------------------------------------
# Parser / config wiring
# ----------------------------------------------------------------------


def test_parser_requires_total_iters() -> None:
    """``--total-iters`` is required; argparse should error without it."""
    parser = train_alphazero.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_propagates_defaults_to_az_train_config() -> None:
    """Calling the script with only the required flag should match the
    AZTrainConfig dataclass defaults (modulo player_ids which is a tuple
    literal, not a CLI knob)."""
    from rl.training.alphazero import AZTrainConfig
    from rl.training.self_play import SelfPlayConfig
    from rl.search.mcts import MCTSConfig

    args = train_alphazero.build_parser().parse_args(["--total-iters", "1"])
    cfg = train_alphazero._build_config(args)

    az_defaults = AZTrainConfig()
    assert cfg.games_per_iter == az_defaults.games_per_iter
    assert cfg.batches_per_iter == az_defaults.batches_per_iter
    assert cfg.batch_size == az_defaults.batch_size
    assert cfg.lr == az_defaults.lr
    assert cfg.weight_decay == az_defaults.weight_decay
    assert cfg.value_coef == az_defaults.value_coef
    assert cfg.buffer_capacity == az_defaults.buffer_capacity

    sp_defaults = SelfPlayConfig()
    assert cfg.self_play.temperature_initial == sp_defaults.temperature_initial
    assert cfg.self_play.temperature_final == sp_defaults.temperature_final
    assert (
        cfg.self_play.temperature_threshold_moves
        == sp_defaults.temperature_threshold_moves
    )
    assert cfg.self_play.stalemate_value == sp_defaults.stalemate_value
    assert cfg.self_play.max_moves == sp_defaults.max_moves

    # MCTS knobs — dirichlet_epsilon is overridden to 0.25 (self-play default).
    mcts_defaults = MCTSConfig()
    assert cfg.self_play.mcts.rollouts == mcts_defaults.rollouts
    assert cfg.self_play.mcts.c_puct == mcts_defaults.c_puct
    assert cfg.self_play.mcts.dirichlet_alpha == mcts_defaults.dirichlet_alpha
    assert cfg.self_play.mcts.dirichlet_epsilon == 0.25  # script-level default


def test_parser_overrides_propagate() -> None:
    """A non-default value on the CLI must land in the right field."""
    args = train_alphazero.build_parser().parse_args(
        [
            "--total-iters", "3",
            "--games-per-iter", "7",
            "--batches-per-iter", "9",
            "--lr", "1e-4",
            "--mcts-rollouts", "42",
            "--dirichlet-epsilon", "0.4",
            "--stalemate-value", "0.0",
            "--temperature-threshold-moves", "5",
            "--snapshot-every", "0",  # disable
        ]
    )
    cfg = train_alphazero._build_config(args)
    assert cfg.games_per_iter == 7
    assert cfg.batches_per_iter == 9
    assert cfg.lr == 1e-4
    assert cfg.self_play.mcts.rollouts == 42
    assert cfg.self_play.mcts.dirichlet_epsilon == 0.4
    assert cfg.self_play.stalemate_value == 0.0
    assert cfg.self_play.temperature_threshold_moves == 5
    assert cfg.snapshot_every_iters == 0


# ----------------------------------------------------------------------
# Warm-start validation
# ----------------------------------------------------------------------


def test_resolve_warm_start_returns_none_when_path_unset() -> None:
    assert train_alphazero._resolve_warm_start(None, torch.device("cpu")) is None


def test_resolve_warm_start_rejects_missing_path() -> None:
    with pytest.raises(SystemExit, match="--init-from"):
        train_alphazero._resolve_warm_start(
            Path("/definitely/not/here.pt"), torch.device("cpu")
        )


def test_resolve_warm_start_rejects_scalar_value_head(tmp_path: Path) -> None:
    """A scalar-head GNN checkpoint must be refused — AZ needs vector."""
    from domain.ids import PlayerID
    from rl.agents.policy_agent import PolicyAgent
    from rl.encoding._action_layout import ACTION_SPACE_SIZE
    from rl.encoding.action import ActionEncoder
    from rl.encoding.graph_observation import GRAPH_OBS_SHAPE, GraphObservationEncoder
    from rl.models.gnn import GNNArch, GNNPolicyValue
    from rl.training.checkpoint import (
        ACTION_LAYOUT_VERSION,
        CheckpointMeta,
        ModelArch,
        compute_config_hash,
        save_checkpoint,
    )
    from rl.encoding.graph_observation import GRAPH_OBS_LAYOUT_VERSION
    from rl.training.config import TrainConfig

    scalar_arch = GNNArch(
        node_hidden=16, player_hidden=8, n_mp_layers=1,
        n_heads=4, global_mlp_hidden=16, value_kind="scalar",
    )
    torch.manual_seed(0)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=scalar_arch,
    )
    agent = PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder([PlayerID(i) for i in range(1, 5)]),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )
    path = tmp_path / "scalar.pt"
    save_checkpoint(
        agent,
        path,
        CheckpointMeta(
            obs_layout_version=GRAPH_OBS_LAYOUT_VERSION,
            action_layout_version=ACTION_LAYOUT_VERSION,
            model_arch=ModelArch(
                obs_dim=GRAPH_OBS_SHAPE[0],
                action_dim=ACTION_SPACE_SIZE,
                encoder_kind="graph",
                gnn_arch=scalar_arch,
            ),
            train_step=0,
            timestamp=0.0,
            config_hash=compute_config_hash(TrainConfig()),
        ),
    )
    with pytest.raises(SystemExit, match="value_kind"):
        train_alphazero._resolve_warm_start(path, torch.device("cpu"))


# ----------------------------------------------------------------------
# End-to-end smoke
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_main_runs_a_tiny_iteration_and_writes_artefacts(tmp_path: Path) -> None:
    """One iteration end-to-end via ``main(argv)``: ``final.pt`` exists,
    ``config.json`` parseable, ``tb/`` + ``snapshots/`` directories
    populated."""
    out = tmp_path / "run"
    rc = train_alphazero.main(
        [
            "--total-iters", "1",
            "--games-per-iter", "1",
            "--batches-per-iter", "2",
            "--buffer-capacity", "100",
            "--batch-size", "8",
            "--mcts-rollouts", "2",
            "--max-moves", "20",
            "--temperature-threshold-moves", "2",
            "--snapshot-every", "1",
            "--eval-every", "0",  # skip eval to keep smoke fast
            "--device", "cpu",
            "--output-dir", str(out),
            "--seed", "0",
        ]
    )
    assert rc == 0
    assert (out / "final.pt").exists()
    assert (out / "config.json").exists()
    assert (out / "snapshots").is_dir()
    assert (out / "snapshots" / "iter_1.pt").exists()

    config_data = json.loads((out / "config.json").read_text())
    assert config_data["az"]["games_per_iter"] == 1
    assert config_data["az"]["batches_per_iter"] == 2
    assert config_data["mcts"]["rollouts"] == 2
