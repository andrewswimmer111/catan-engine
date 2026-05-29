"""Tests for the ``scripts/diagnose_self_play.py`` config resolution.

Fast, pure-function tests of ``_resolve_configs`` — the layering of
dataclass defaults, a present ``config.json``, and CLI overrides that
lets the diagnostic run on a checkpoint whose run dir has no AZ
``config.json`` (e.g. a behavioural-cloning run). The game-driving parts
of the diagnostic are exercised manually / in the AZ + BC workflows; here
we just pin the resolution precedence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable in test environment.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import diagnose_self_play as d  # noqa: E402

from rl.search.mcts import MCTSConfig  # noqa: E402
from rl.training.self_play import SelfPlayConfig  # noqa: E402


def _args(**kw) -> argparse.Namespace:
    base = dict(
        win_vp=None,
        max_moves=None,
        mcts_rollouts=None,
        mcts_cpuct=None,
        dirichlet_epsilon=None,
        mcts_seed=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


_AZ_CFG = {
    "self_play": {
        "victory_point_target": 6,
        "max_moves": 250,
        "temperature_initial": 1.0,
        "temperature_final": 0.0,
        "temperature_threshold_moves": 30,
    },
    "mcts": {
        "rollouts": 100,
        "c_puct": 2.0,
        "seed": 1,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
    },
    "stalemate": {"shape": "vp_linear", "flat_value": -0.25, "low": -0.5, "high": -0.1},
}


def test_defaults_match_dataclasses() -> None:
    sp, mcts = d._resolve_configs({}, _args())
    sp_def, m_def = SelfPlayConfig(), MCTSConfig()
    assert sp["victory_point_target"] == sp_def.victory_point_target
    assert sp["max_moves"] == sp_def.max_moves
    assert mcts["rollouts"] == m_def.rollouts
    assert mcts["c_puct"] == m_def.c_puct
    assert mcts["dirichlet_epsilon"] == m_def.dirichlet_epsilon
    assert sp["stalemate"]["shape"] == sp_def.stalemate.shape


def test_empty_config_uses_defaults_plus_cli() -> None:
    """The BC case: no AZ blocks, knobs supplied on the CLI."""
    sp, mcts = d._resolve_configs({}, _args(win_vp=6, mcts_rollouts=50))
    assert sp["victory_point_target"] == 6
    assert mcts["rollouts"] == 50
    # Untouched knobs fall back to dataclass defaults.
    assert sp["max_moves"] == SelfPlayConfig().max_moves
    assert mcts["c_puct"] == MCTSConfig().c_puct


def test_present_config_read_without_cli() -> None:
    sp, mcts = d._resolve_configs(_AZ_CFG, _args())
    assert sp["max_moves"] == 250
    assert mcts["seed"] == 1
    assert mcts["dirichlet_epsilon"] == 0.25
    assert sp["stalemate"]["shape"] == "vp_linear"


def test_cli_overrides_present_config() -> None:
    sp, mcts = d._resolve_configs(_AZ_CFG, _args(win_vp=8, mcts_rollouts=25))
    assert sp["victory_point_target"] == 8  # overridden
    assert mcts["rollouts"] == 25  # overridden
    assert sp["max_moves"] == 250  # config value retained
