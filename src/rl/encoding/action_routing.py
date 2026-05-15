"""Static mapping between action indices and the graph element each one names.

The flat action layout in :mod:`._action_layout` mixes board-element actions
(build a road on EdgeID, settle a VertexID, move the robber to TileID),
player-element actions (steal from a seat slot), and pure global actions
(roll, end turn, buy dev, play dev cards, maritime trade, discard). A flat
MLP head treats all 249 indices the same; a typed graph head produces
spatial logits from per-node embeddings, player logits from per-player
embeddings, and global logits from a pooled MLP, then scatters all of them
back into one ``(B, ACTION_SPACE_SIZE)`` tensor.

This module owns the *scatter table* — the per-action-type sequence of
action-space indices the model writes its head outputs into. The layout
mirrors :mod:`._action_layout` exactly; bumping ``_ACTION_LAYOUT_VERSION``
must be paired with a regeneration of this table (the
:data:`ACTION_ROUTING_LAYOUT_VERSION` constant is mechanically tied to it
so a drift between the two raises at import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

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

__all__ = [
    "ACTION_ROUTING_LAYOUT_VERSION",
    "ActionHead",
    "ActionRouting",
    "ACTION_ROUTING",
    "N_GLOBAL_ACTIONS",
]


# Pinned to the layout version this table was derived against. The unit
# tests assert equality with ``_ACTION_LAYOUT_VERSION``; bumping the layout
# without re-deriving this module is a layout error and must fail loud.
ACTION_ROUTING_LAYOUT_VERSION: Final[int] = 1


class ActionHead(Enum):
    """Which sub-head in the typed policy produces an action's logit.

    Spatial heads consume per-node embeddings: an edge embedding for
    :data:`ROAD_EDGE`, a vertex embedding for :data:`SETTLE_VERTEX` and
    :data:`CITY_VERTEX`, a tile embedding for :data:`MOVE_ROBBER_TILE`.
    :data:`STEAL_SEAT` consumes a per-player embedding (seat-rotation order,
    viewer at slot 0). :data:`GLOBAL` consumes a pooled board+player
    embedding via an MLP.
    """

    ROAD_EDGE = "road_edge"
    SETTLE_VERTEX = "settle_vertex"
    CITY_VERTEX = "city_vertex"
    MOVE_ROBBER_TILE = "move_robber_tile"
    STEAL_SEAT = "steal_seat"
    GLOBAL = "global"


@dataclass(frozen=True)
class ActionRouting:
    """Per-head index tables for scattering head outputs back into the
    flat ``(ACTION_SPACE_SIZE,)`` action vector.

    Each ``*_indices`` array gives, in head-output order, the action-space
    index the head's k-th output lands at. The model assembles its final
    logits with ``out[:, indices] = head_logits``.
    """

    road_edge_indices: np.ndarray  # (N_EDGES,)
    settle_vertex_indices: np.ndarray  # (N_VERTICES,)
    city_vertex_indices: np.ndarray  # (N_VERTICES,)
    move_robber_tile_indices: np.ndarray  # (N_TILES,)
    steal_seat_indices: np.ndarray  # (N_STEAL_SLOTS,)
    global_indices: np.ndarray  # (N_GLOBAL_ACTIONS,)

    def head_size(self, head: ActionHead) -> int:
        """Number of outputs the head produces."""
        return int(self._array_for(head).shape[0])

    def indices_for(self, head: ActionHead) -> np.ndarray:
        """Read-only view of the scatter indices for ``head``."""
        arr = self._array_for(head)
        view = arr.view()
        view.flags.writeable = False
        return view

    def _array_for(self, head: ActionHead) -> np.ndarray:
        if head is ActionHead.ROAD_EDGE:
            return self.road_edge_indices
        if head is ActionHead.SETTLE_VERTEX:
            return self.settle_vertex_indices
        if head is ActionHead.CITY_VERTEX:
            return self.city_vertex_indices
        if head is ActionHead.MOVE_ROBBER_TILE:
            return self.move_robber_tile_indices
        if head is ActionHead.STEAL_SEAT:
            return self.steal_seat_indices
        if head is ActionHead.GLOBAL:
            return self.global_indices
        raise AssertionError(f"unhandled ActionHead: {head}")


def _build_routing() -> ActionRouting:
    """Construct the scatter tables from ``_action_layout`` constants.

    Built once at module import. Verifies the partition is exact (every
    action-space index is owned by exactly one head) and that the heads
    cover ``ACTION_SPACE_SIZE`` slots in total.
    """
    road_edge = np.arange(ROAD_START, ROAD_START + N_EDGES, dtype=np.int64)
    settle = np.arange(
        SETTLEMENT_START, SETTLEMENT_START + N_VERTICES, dtype=np.int64
    )
    city = np.arange(CITY_START, CITY_START + N_VERTICES, dtype=np.int64)
    robber = np.arange(
        ROBBER_MOVE_START, ROBBER_MOVE_START + N_TILES, dtype=np.int64
    )
    steal = np.arange(STEAL_START, STEAL_START + N_STEAL_SLOTS, dtype=np.int64)

    # Global actions: every remaining action-space index, in canonical
    # action-layout order (maritime trades, roll, end turn, buy dev, knight,
    # road building, monopoly per resource, year-of-plenty pairs, discard).
    global_indices = np.concatenate(
        [
            np.arange(
                MARITIME_TRADE_START,
                MARITIME_TRADE_START + N_MARITIME_TRADES,
                dtype=np.int64,
            ),
            np.array(
                [
                    ROLL_INDEX,
                    END_TURN_INDEX,
                    BUY_DEV_INDEX,
                    KNIGHT_INDEX,
                    ROAD_BUILDING_INDEX,
                ],
                dtype=np.int64,
            ),
            np.arange(
                MONOPOLY_START, MONOPOLY_START + N_RESOURCES, dtype=np.int64
            ),
            np.arange(
                YEAR_OF_PLENTY_START,
                YEAR_OF_PLENTY_START + N_YEAR_OF_PLENTY,
                dtype=np.int64,
            ),
            np.array([DISCARD_INDEX], dtype=np.int64),
        ]
    )

    _validate_partition(
        [road_edge, settle, city, robber, steal, global_indices]
    )

    return ActionRouting(
        road_edge_indices=road_edge,
        settle_vertex_indices=settle,
        city_vertex_indices=city,
        move_robber_tile_indices=robber,
        steal_seat_indices=steal,
        global_indices=global_indices,
    )


def _validate_partition(arrays: list[np.ndarray]) -> None:
    """Confirm the per-head arrays partition ``[0, ACTION_SPACE_SIZE)`` exactly."""
    concat = np.concatenate(arrays)
    if concat.shape[0] != ACTION_SPACE_SIZE:
        raise AssertionError(
            f"action routing total size {concat.shape[0]} != "
            f"ACTION_SPACE_SIZE {ACTION_SPACE_SIZE}"
        )
    if np.unique(concat).shape[0] != ACTION_SPACE_SIZE:
        raise AssertionError(
            "action routing has duplicate action indices; partition is not exact"
        )
    expected = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    if not np.array_equal(np.sort(concat), expected):
        raise AssertionError(
            "action routing does not cover [0, ACTION_SPACE_SIZE) exactly"
        )


def _verify_layout_version_match() -> None:
    """Cross-check that this module is in sync with ``_action_layout``.

    Mismatch indicates someone bumped the action layout without updating
    the routing table — failing here at import is much cheaper than
    discovering it during training.
    """
    if ACTION_ROUTING_LAYOUT_VERSION != _ACTION_LAYOUT_VERSION:
        raise AssertionError(
            f"ACTION_ROUTING_LAYOUT_VERSION ({ACTION_ROUTING_LAYOUT_VERSION}) "
            f"!= _ACTION_LAYOUT_VERSION ({_ACTION_LAYOUT_VERSION}); "
            "regenerate the routing table or bump this constant deliberately."
        )


_verify_layout_version_match()

ACTION_ROUTING: Final[ActionRouting] = _build_routing()

# Exposed as a top-level constant so model code can size its global head
# without reaching into ``ACTION_ROUTING``.
N_GLOBAL_ACTIONS: Final[int] = int(ACTION_ROUTING.global_indices.shape[0])
