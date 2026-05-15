"""Shape, dtype, snapshot, permutation, and hidden-info tests for the graph encoder.

Mirrors ``test_observation_encoder.py`` so the graph encoder gets the same
class of structural and invariance guarantees as the flat encoder.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from domain.engine.player_view import PlayerView, make_player_view
from domain.enums import DevCardType
from rl.encoding.graph_observation import (
    EDGE_BLOCK,
    EDGE_OFFSET,
    GLOBAL_BLOCK,
    GLOBAL_OFFSET,
    GRAPH_OBS_LAYOUT_VERSION,
    GRAPH_OBS_SHAPE,
    GRAPH_STRUCTURE,
    GraphObservationEncoder,
    N_EDGE_TYPES,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    PLAYER_BLOCK,
    PLAYER_OFFSET,
    TILE_BLOCK,
    TILE_OFFSET,
    VERTEX_BLOCK,
    VERTEX_OFFSET,
)
from rl.env.catan_env import CatanEnv

SNAPSHOT_PATH = (
    Path(__file__).parent
    / "fixtures"
    / f"graph_obs_snapshot_v{GRAPH_OBS_LAYOUT_VERSION}.npy"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rollout_view(seed: int, steps: int, viewer_seat: int = 0) -> PlayerView:
    """Drive a CatanEnv with uniform-random legal moves and return a PlayerView."""
    env = CatanEnv(seed=seed)
    env.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))
    pid = env.state.config.player_ids[viewer_seat]
    return env._engine.player_view(env.state, pid)


def _view_with_rotated_config(view: PlayerView, shift: int) -> PlayerView:
    pids = list(view.config.player_ids)
    rotated = pids[shift:] + pids[:shift]
    new_config = dataclasses.replace(view.config, player_ids=rotated)
    return dataclasses.replace(view, config=new_config)


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------


def test_shape_dtype_and_finite_for_fresh_state() -> None:
    env = CatanEnv(seed=0)
    env.reset(seed=0)
    view = env._engine.player_view(env.state, env.state.current_player)
    obs = GraphObservationEncoder().encode(view)
    assert obs.shape == GRAPH_OBS_SHAPE
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_shape_and_finite_after_random_rollout(seed: int) -> None:
    view = _rollout_view(seed=seed, steps=30)
    obs = GraphObservationEncoder().encode(view)
    assert obs.shape == GRAPH_OBS_SHAPE
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    # Per-feature normalisation keeps everything in a tight band; the dev
    # hand counter is the loosest at count / 25 ≈ 0.4 for a 10-card hand.
    assert obs.max() <= 5.0
    assert obs.min() >= -1e-6


def test_layout_offsets_are_strictly_increasing_and_total_matches_shape() -> None:
    offsets = [
        TILE_OFFSET,
        VERTEX_OFFSET,
        EDGE_OFFSET,
        PLAYER_OFFSET,
        GLOBAL_OFFSET,
    ]
    assert offsets == sorted(offsets)
    total = TILE_BLOCK + VERTEX_BLOCK + EDGE_BLOCK + PLAYER_BLOCK + GLOBAL_BLOCK
    assert total == GRAPH_OBS_SHAPE[0]
    assert GLOBAL_OFFSET + GLOBAL_BLOCK == GRAPH_OBS_SHAPE[0]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_byte_for_byte() -> None:
    """Encoded fixture view must match the committed ``graph_obs_snapshot_v1.npy``.

    To regenerate (after intentional layout change + version bump), run:

        PYTHONPATH=src venv/bin/python tests/rl/fixtures/build_graph_obs_snapshot.py
    """
    from tests.rl.fixtures.build_obs_snapshot import _build_snapshot_view

    expected = np.load(SNAPSHOT_PATH)
    actual = GraphObservationEncoder().encode(_build_snapshot_view())
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Permutation invariance — seat rotation must cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_obs_invariant_to_cyclic_seat_rotation(shift: int) -> None:
    """Rotating ``config.player_ids`` must not change the encoded obs.

    The encoder remaps every player-keyed slot into viewer-perspective seat
    order, so the encoded vector should be a function of the game state, not
    of the seat-cycle starting point.
    """
    view = _rollout_view(seed=5, steps=25, viewer_seat=0)
    enc = GraphObservationEncoder()
    base = enc.encode(view)
    rotated = enc.encode(_view_with_rotated_config(view, shift=shift))
    np.testing.assert_array_equal(base, rotated)


def test_obs_changes_when_viewer_changes() -> None:
    """Sanity: encoding from a different seat *does* change the obs.

    Guards against an encoder that ignores the viewer entirely (which would
    pass the rotation test trivially).
    """
    env = CatanEnv(seed=11)
    env.reset(seed=11)
    rng = random.Random(11)
    for _ in range(40):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))
    pids = list(env.state.config.player_ids)
    enc = GraphObservationEncoder()
    obs_a = enc.encode(env._engine.player_view(env.state, pids[0]))
    obs_b = enc.encode(env._engine.player_view(env.state, pids[1]))
    assert not np.array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Hidden information — opponent dev cards must not leak
# ---------------------------------------------------------------------------


def test_hidden_opponent_dev_hand_does_not_leak() -> None:
    """Mutating opponent dev cards (count fixed) leaves the obs unchanged."""
    env = CatanEnv(seed=3)
    env.reset(seed=3)
    rng = random.Random(3)
    for _ in range(30):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))

    pids = list(env.state.config.player_ids)
    viewer = pids[0]
    opponent = pids[1]

    opp_state = env.state.players[opponent]
    opp_state.dev_cards_in_hand = [
        (DevCardType.KNIGHT, 0),
        (DevCardType.MONOPOLY, 0),
        (DevCardType.YEAR_OF_PLENTY, 0),
    ]
    obs_a = GraphObservationEncoder().encode(make_player_view(env.state, viewer))

    opp_state.dev_cards_in_hand = [
        (DevCardType.ROAD_BUILDING, 0),
        (DevCardType.VICTORY_POINT, 0),
        (DevCardType.KNIGHT, 0),
    ]
    obs_b = GraphObservationEncoder().encode(make_player_view(env.state, viewer))

    np.testing.assert_array_equal(obs_a, obs_b)


def test_hidden_info_test_actually_observes_count_changes() -> None:
    """Sanity: if the opponent's dev-hand *count* changes, the obs does too."""
    env = CatanEnv(seed=3)
    env.reset(seed=3)
    rng = random.Random(3)
    for _ in range(30):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))

    pids = list(env.state.config.player_ids)
    viewer = pids[0]
    opponent = pids[1]

    opp_state = env.state.players[opponent]
    opp_state.dev_cards_in_hand = [(DevCardType.KNIGHT, 0)]
    obs_a = GraphObservationEncoder().encode(make_player_view(env.state, viewer))

    opp_state.dev_cards_in_hand = [
        (DevCardType.KNIGHT, 0),
        (DevCardType.KNIGHT, 0),
    ]
    obs_b = GraphObservationEncoder().encode(make_player_view(env.state, viewer))

    assert not np.array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Graph structure invariants
# ---------------------------------------------------------------------------


def test_graph_structure_node_count() -> None:
    assert GRAPH_STRUCTURE.n_nodes == N_TILES + N_VERTICES + N_EDGES


def test_graph_structure_edge_index_shape() -> None:
    assert GRAPH_STRUCTURE.edge_index.dtype == np.int64
    assert GRAPH_STRUCTURE.edge_index.shape[0] == 2
    assert GRAPH_STRUCTURE.edge_index.shape[1] > 0
    assert GRAPH_STRUCTURE.edge_index.shape[1] == GRAPH_STRUCTURE.edge_type.shape[0]


def test_graph_structure_edges_are_bidirectional() -> None:
    """Every directed (u, v) must have a partner (v, u)."""
    ei = GRAPH_STRUCTURE.edge_index
    pairs = {tuple(int(x) for x in ei[:, i]) for i in range(ei.shape[1])}
    for u, v in pairs:
        assert (v, u) in pairs, f"missing reverse edge for ({u}, {v})"


def test_graph_structure_edge_types_in_range() -> None:
    et = GRAPH_STRUCTURE.edge_type
    assert et.dtype == np.int64
    assert int(et.min()) >= 0
    assert int(et.max()) < N_EDGE_TYPES


def test_graph_structure_endpoints_in_node_range() -> None:
    ei = GRAPH_STRUCTURE.edge_index
    n = GRAPH_STRUCTURE.n_nodes
    assert int(ei.min()) >= 0
    assert int(ei.max()) < n


def test_graph_structure_no_self_loops() -> None:
    ei = GRAPH_STRUCTURE.edge_index
    assert not (ei[0] == ei[1]).any()


def test_graph_structure_tile_vertex_edges_count() -> None:
    """Each tile has 6 vertex corners; 19 × 6 = 114 directed edges per direction
    (228 total bidirectional)."""
    et = GRAPH_STRUCTURE.edge_type
    from rl.encoding.graph_observation import EDGE_TYPE_TILE_VERTEX

    n_tv = int((et == EDGE_TYPE_TILE_VERTEX).sum())
    assert n_tv == 19 * 6 * 2  # 228
