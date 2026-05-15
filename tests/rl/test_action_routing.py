"""Tests for the action routing table used by the typed GNN policy head."""

from __future__ import annotations

import numpy as np
import pytest

from rl.encoding._action_layout import (
    _ACTION_LAYOUT_VERSION,
    ACTION_SPACE_SIZE,
    BUY_DEV_INDEX,
    CITY_START,
    DISCARD_INDEX,
    END_TURN_INDEX,
    KNIGHT_INDEX,
    MARITIME_TRADE_START,
    MONOPOLY_START,
    N_EDGES,
    N_MARITIME_TRADES,
    N_RESOURCES,
    N_STEAL_SLOTS,
    N_TILES,
    N_VERTICES,
    N_YEAR_OF_PLENTY,
    ROAD_BUILDING_INDEX,
    ROAD_START,
    ROBBER_MOVE_START,
    ROLL_INDEX,
    SETTLEMENT_START,
    STEAL_START,
    YEAR_OF_PLENTY_START,
)
from rl.encoding.action_routing import (
    ACTION_ROUTING,
    ACTION_ROUTING_LAYOUT_VERSION,
    ActionHead,
    N_GLOBAL_ACTIONS,
)


def test_layout_version_pinned_to_action_layout() -> None:
    """A drift between the routing table and the underlying layout must fail."""
    assert ACTION_ROUTING_LAYOUT_VERSION == _ACTION_LAYOUT_VERSION


def test_per_head_sizes_match_layout_widths() -> None:
    """Each head's output count must match the layout block it covers."""
    assert ACTION_ROUTING.head_size(ActionHead.ROAD_EDGE) == N_EDGES
    assert ACTION_ROUTING.head_size(ActionHead.SETTLE_VERTEX) == N_VERTICES
    assert ACTION_ROUTING.head_size(ActionHead.CITY_VERTEX) == N_VERTICES
    assert ACTION_ROUTING.head_size(ActionHead.MOVE_ROBBER_TILE) == N_TILES
    assert ACTION_ROUTING.head_size(ActionHead.STEAL_SEAT) == N_STEAL_SLOTS

    n_globals_expected = (
        N_MARITIME_TRADES
        + 5  # roll, end turn, buy dev, knight, road building
        + N_RESOURCES  # monopoly per resource
        + N_YEAR_OF_PLENTY
        + 1  # discard
    )
    assert N_GLOBAL_ACTIONS == n_globals_expected
    assert ACTION_ROUTING.head_size(ActionHead.GLOBAL) == n_globals_expected


def test_total_partition_covers_action_space_exactly() -> None:
    """Per-head arrays together partition ``[0, ACTION_SPACE_SIZE)`` exactly."""
    all_indices = np.concatenate(
        [
            ACTION_ROUTING.indices_for(h)
            for h in (
                ActionHead.ROAD_EDGE,
                ActionHead.SETTLE_VERTEX,
                ActionHead.CITY_VERTEX,
                ActionHead.MOVE_ROBBER_TILE,
                ActionHead.STEAL_SEAT,
                ActionHead.GLOBAL,
            )
        ]
    )
    assert all_indices.shape[0] == ACTION_SPACE_SIZE
    assert np.unique(all_indices).shape[0] == ACTION_SPACE_SIZE
    np.testing.assert_array_equal(
        np.sort(all_indices), np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    )


def test_spatial_heads_target_contiguous_layout_ranges() -> None:
    """Spatial heads hit the contiguous index ranges defined in ``_action_layout``."""
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.ROAD_EDGE),
        np.arange(ROAD_START, ROAD_START + N_EDGES),
    )
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.SETTLE_VERTEX),
        np.arange(SETTLEMENT_START, SETTLEMENT_START + N_VERTICES),
    )
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.CITY_VERTEX),
        np.arange(CITY_START, CITY_START + N_VERTICES),
    )
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.MOVE_ROBBER_TILE),
        np.arange(ROBBER_MOVE_START, ROBBER_MOVE_START + N_TILES),
    )
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.STEAL_SEAT),
        np.arange(STEAL_START, STEAL_START + N_STEAL_SLOTS),
    )


def test_global_head_targets_all_non_spatial_indices_in_canonical_order() -> None:
    """Global head output index k maps to the k-th non-spatial action in layout order."""
    expected = np.concatenate(
        [
            np.arange(
                MARITIME_TRADE_START,
                MARITIME_TRADE_START + N_MARITIME_TRADES,
            ),
            np.array(
                [
                    ROLL_INDEX,
                    END_TURN_INDEX,
                    BUY_DEV_INDEX,
                    KNIGHT_INDEX,
                    ROAD_BUILDING_INDEX,
                ]
            ),
            np.arange(MONOPOLY_START, MONOPOLY_START + N_RESOURCES),
            np.arange(
                YEAR_OF_PLENTY_START,
                YEAR_OF_PLENTY_START + N_YEAR_OF_PLENTY,
            ),
            np.array([DISCARD_INDEX]),
        ]
    )
    np.testing.assert_array_equal(
        ACTION_ROUTING.indices_for(ActionHead.GLOBAL), expected
    )


def test_indices_view_is_read_only() -> None:
    """``indices_for`` exposes a view that can't be mutated by callers."""
    view = ACTION_ROUTING.indices_for(ActionHead.ROAD_EDGE)
    with pytest.raises(ValueError):
        view[0] = -1


def test_indices_arrays_are_int64() -> None:
    """Index dtype is int64 so the model can use them directly as gather/scatter indices."""
    for head in ActionHead:
        assert ACTION_ROUTING.indices_for(head).dtype == np.int64
