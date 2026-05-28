"""Tests for the ``scripts/pretrain_bc.py`` driver.

Mostly **CLI shape** tests — argument parsing, config wiring, learner
shape. A single slow smoke test runs the driver against a tiny config to
verify it produces the expected on-disk artefacts (``config.json``,
``final.pt``, ``progress.md``) and that ``final.pt`` loads as an
AZ-compatible checkpoint.
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

import pretrain_bc  # noqa: E402

from rl.training.bc_pretrain import BCConfig  # noqa: E402
from rl.training.checkpoint import load_checkpoint, model_arch_from  # noqa: E402


# ----------------------------------------------------------------------
# Parser / config wiring
# ----------------------------------------------------------------------


def test_parser_defaults_match_bcconfig() -> None:
    args = pretrain_bc.build_parser().parse_args([])
    cfg = pretrain_bc._build_config(args)
    defaults = BCConfig()
    assert cfg.n_games == defaults.n_games
    assert cfg.victory_point_target == defaults.victory_point_target
    assert cfg.max_moves == defaults.max_moves
    assert cfg.epochs == defaults.epochs
    assert cfg.batch_size == defaults.batch_size
    assert cfg.lr == defaults.lr
    assert cfg.weight_decay == defaults.weight_decay
    assert cfg.value_coef == defaults.value_coef
    assert cfg.max_grad_norm == defaults.max_grad_norm
    assert cfg.seed == defaults.seed


def test_win_vp_and_stalemate_flow_into_config() -> None:
    args = pretrain_bc.build_parser().parse_args(
        ["--win-vp", "6", "--stalemate-shape", "flat", "--stalemate-flat-value", "-0.3"]
    )
    cfg = pretrain_bc._build_config(args)
    assert cfg.victory_point_target == 6
    assert cfg.stalemate.shape == "flat"
    assert cfg.stalemate.flat_value == pytest.approx(-0.3)


def test_build_learner_is_graph_vector() -> None:
    learner = pretrain_bc._build_learner(torch.device("cpu"))
    assert learner.model.value_kind == "vector"
    arch = model_arch_from(learner.model)
    assert arch.encoder_kind == "graph"


def test_main_rejects_nonpositive_games() -> None:
    assert pretrain_bc.main(["--n-games", "0"]) == 2


# ----------------------------------------------------------------------
# Slow: end-to-end driver smoke
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_main_writes_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "bc_run"
    rc = pretrain_bc.main(
        [
            "--n-games", "2",
            "--epochs", "1",
            "--win-vp", "6",
            "--max-moves", "200",
            "--batch-size", "64",
            "--device", "cpu",
            "--output-dir", str(out),
        ]
    )
    assert rc == 0
    assert (out / "config.json").is_file()
    assert (out / "progress.md").is_file()
    assert (out / "final.pt").is_file()

    cfg_json = json.loads((out / "config.json").read_text())
    assert cfg_json["bc"]["victory_point_target"] == 6

    _agent, meta = load_checkpoint(out / "final.pt", device="cpu")
    assert meta.model_arch.encoder_kind == "graph"
    assert meta.model_arch.gnn_arch.value_kind == "vector"
