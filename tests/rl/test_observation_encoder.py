"""Shape, dtype, snapshot, permutation, and hidden-info tests for the obs encoder."""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from domain.engine.game_engine import GameEngine
from domain.engine.player_view import (
    PlayerPerspectiveState,
    PlayerView,
    make_player_view,
)
from domain.engine.randomizer import SeededRandomizer
from domain.enums import DevCardType
from domain.game.config import GameConfig
from domain.ids import PlayerID
from rl.encoding.observation import (
    FlatObservationEncoder,
    OBS_LAYOUT_VERSION,
    OBS_SHAPE,
)
from rl.env.catan_env import CatanEnv

SNAPSHOT_PATH = (
    Path(__file__).parent / "fixtures" / f"obs_snapshot_v{OBS_LAYOUT_VERSION}.npy"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rollout_view(seed: int, steps: int, viewer_seat: int = 0) -> PlayerView:
    """Drive ``CatanEnv`` with uniform-random legal moves and return a PlayerView.

    The viewer is the player at ``viewer_seat`` in ``config.player_ids``,
    independent of whose turn it is.
    """
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
    """Return a copy of ``view`` whose ``config.player_ids`` is cyclically rotated.

    All other game data — board, buildings, players dict — is unchanged. The
    encoded observation should be invariant to ``shift`` because the encoder
    remaps every player-keyed slot into viewer-perspective seat order.
    """
    pids = list(view.config.player_ids)
    rotated = pids[shift:] + pids[:shift]
    new_config = dataclasses.replace(view.config, player_ids=rotated)
    return dataclasses.replace(view, config=new_config)


# ---------------------------------------------------------------------------
# Shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_shape_dtype_and_finite_for_fresh_state() -> None:
    env = CatanEnv(seed=0)
    env.reset(seed=0)
    view = env._engine.player_view(env.state, env.state.current_player)
    obs = FlatObservationEncoder().encode(view)
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_shape_and_finite_after_random_rollout(seed: int) -> None:
    view = _rollout_view(seed=seed, steps=30)
    obs = FlatObservationEncoder().encode(view)
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    # All normalisation should keep values in a reasonable band; the bank
    # bytes can reach 1.0 but nothing should explode.
    assert obs.max() <= 5.0
    assert obs.min() >= -1e-6


def test_layout_offsets_are_strictly_increasing_and_total_matches_shape() -> None:
    from rl.encoding.observation import (
        AWARDS_OFFSET,
        EDGES_OFFSET,
        OPPONENTS_OFFSET,
        PHASE_PENDING_OFFSET,
        PORTS_OFFSET,
        SELF_HAND_OFFSET,
        TILES_OFFSET,
        VERTICES_OFFSET,
    )
    from rl.encoding.features import (
        AWARDS_WIDTH,
        EDGES_BLOCK,
        OPPONENTS_BLOCK,
        PHASE_PENDING_WIDTH,
        PORTS_BLOCK,
        SELF_HAND_WIDTH,
        TILES_BLOCK,
        VERTICES_BLOCK,
    )

    offsets = [
        TILES_OFFSET,
        VERTICES_OFFSET,
        EDGES_OFFSET,
        PORTS_OFFSET,
        SELF_HAND_OFFSET,
        OPPONENTS_OFFSET,
        AWARDS_OFFSET,
        PHASE_PENDING_OFFSET,
    ]
    assert offsets == sorted(offsets)
    total = (
        TILES_BLOCK
        + VERTICES_BLOCK
        + EDGES_BLOCK
        + PORTS_BLOCK
        + SELF_HAND_WIDTH
        + OPPONENTS_BLOCK
        + AWARDS_WIDTH
        + PHASE_PENDING_WIDTH
    )
    assert total == OBS_SHAPE[0]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_byte_for_byte() -> None:
    """Encoded fixture view must match the committed ``obs_snapshot_v1.npy`` exactly.

    To regenerate (after intentional layout change + version bump), run:

        PYTHONPATH=src .venv/bin/python tests/rl/fixtures/build_obs_snapshot.py
    """
    from tests.rl.fixtures.build_obs_snapshot import _build_snapshot_view

    expected = np.load(SNAPSHOT_PATH)
    actual = FlatObservationEncoder().encode(_build_snapshot_view())
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Permutation invariance — viewer-perspective remap cancels seat rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 2, 3])
def test_obs_invariant_to_cyclic_seat_rotation(shift: int) -> None:
    """Rotating ``config.player_ids`` must not change the encoded obs.

    Why: viewer-perspective remapping should cancel rotations of the seat
    cycle. If the encoder mistakenly reads a player slot by absolute index
    instead of by seat-rotation distance, this test fails.
    """
    view = _rollout_view(seed=5, steps=25, viewer_seat=0)
    enc = FlatObservationEncoder()
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
    enc = FlatObservationEncoder()
    obs_a = enc.encode(env._engine.player_view(env.state, pids[0]))
    obs_b = enc.encode(env._engine.player_view(env.state, pids[1]))
    assert not np.array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Hidden information — opponent dev cards must not leak
# ---------------------------------------------------------------------------


def test_hidden_opponent_dev_hand_does_not_leak() -> None:
    """Mutating opponent dev cards (count fixed) leaves the obs unchanged.

    PlayerView reduces opponent ``dev_cards_in_hand`` to an integer count, so
    the encoder physically cannot leak which cards are held — but we still
    test the seam by mutating the underlying ``GameState`` directly. The
    opponent's hidden cards change identity; the count stays the same; the
    encoded obs must be identical.
    """
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

    # Give the opponent a known fixed hand of 3 dev cards.
    opp_state = env.state.players[opponent]
    opp_state.dev_cards_in_hand = [
        (DevCardType.KNIGHT, 0),
        (DevCardType.MONOPOLY, 0),
        (DevCardType.YEAR_OF_PLENTY, 0),
    ]
    view_a = make_player_view(env.state, viewer)
    obs_a = FlatObservationEncoder().encode(view_a)

    # Swap to a different set of 3 cards (same count).
    opp_state.dev_cards_in_hand = [
        (DevCardType.ROAD_BUILDING, 0),
        (DevCardType.VICTORY_POINT, 0),
        (DevCardType.KNIGHT, 0),
    ]
    view_b = make_player_view(env.state, viewer)
    obs_b = FlatObservationEncoder().encode(view_b)

    np.testing.assert_array_equal(obs_a, obs_b)


def test_hidden_info_test_actually_observes_count_changes() -> None:
    """Sanity: if the opponent's dev-hand *count* changes, the obs differs.

    Without this guard, the no-leak test could pass for a buggy encoder that
    simply never reads opponent dev-hand info at all.
    """
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
    obs_a = FlatObservationEncoder().encode(make_player_view(env.state, viewer))

    opp_state.dev_cards_in_hand = [
        (DevCardType.KNIGHT, 0),
        (DevCardType.KNIGHT, 0),
    ]
    obs_b = FlatObservationEncoder().encode(make_player_view(env.state, viewer))

    assert not np.array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Misc validation
# ---------------------------------------------------------------------------


def test_self_hand_writer_rejects_opponent_row() -> None:
    """Defensive: passing an opponent perspective row to write_self_hand fails loud."""
    from rl.encoding.features import write_self_hand

    opp_row = PlayerPerspectiveState(
        player_id=PlayerID(0),
        resources={},
        dev_cards_in_hand=3,  # int → opponent row
        dev_cards_played=[],
        roads_built=0,
        settlements_built=0,
        cities_built=0,
        knights_played=0,
        has_played_dev_card_this_turn=False,
        victory_points_public=0,
    )
    out = np.zeros(20, dtype=np.float32)
    with pytest.raises(ValueError):
        write_self_hand(out, 0, opp_row)
