"""Per-block feature writers for :class:`FlatObservationEncoder`.

Each writer fills a fixed slice of a preallocated float32 array. Keeping
writers small and composable means the layout documented in
``observation.py`` stays the single source of truth for offsets; this module
only knows the *shape* of its own block.

All features are normalised to roughly [0, 1]. Hidden information (opponent
hand contents) is never read here — opponent rows expose only counts.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from domain.board.occupancy import BoardOccupancy
from domain.board.topology import BoardTopology
from domain.engine.player_view import PlayerPerspectiveState, PlayerView
from domain.enums import BuildingType, DevCardType, PortType, Resource, TurnPhase
from domain.game.bank import Bank
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from domain.turn.pending import (
    DiscardPending,
    DomesticTradePending,
    MonopolyPending,
    PendingEffect,
    RoadBuildingPending,
    RobberMovePending,
    StealPending,
    YearOfPlentyPending,
)

# ---------------------------------------------------------------------------
# Block-width constants — exposed for offset arithmetic in observation.py
# ---------------------------------------------------------------------------

N_TILES: Final[int] = 19
N_VERTICES: Final[int] = 54
N_EDGES: Final[int] = 72
N_PORT_SLOTS: Final[int] = 9
N_PLAYER_SLOTS: Final[int] = 4  # self + up to 3 opponents

_TILE_RESOURCES: Final[tuple[Resource, ...]] = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
    Resource.DESERT,
)
_RESOURCE_TO_INDEX: Final[dict[Resource, int]] = {r: i for i, r in enumerate(_TILE_RESOURCES)}

_HAND_RESOURCES: Final[tuple[Resource, ...]] = _TILE_RESOURCES[:5]  # no desert
_HAND_RESOURCE_TO_INDEX: Final[dict[Resource, int]] = {r: i for i, r in enumerate(_HAND_RESOURCES)}

_DEV_CARDS: Final[tuple[DevCardType, ...]] = (
    DevCardType.KNIGHT,
    DevCardType.ROAD_BUILDING,
    DevCardType.YEAR_OF_PLENTY,
    DevCardType.MONOPOLY,
    DevCardType.VICTORY_POINT,
)
_DEV_CARD_TO_INDEX: Final[dict[DevCardType, int]] = {c: i for i, c in enumerate(_DEV_CARDS)}

_PORT_TYPES: Final[tuple[PortType, ...]] = (
    PortType.THREE_TO_ONE,
    PortType.WOOD_TWO,
    PortType.BRICK_TWO,
    PortType.SHEEP_TWO,
    PortType.WHEAT_TWO,
    PortType.ORE_TWO,
)
_PORT_TYPE_TO_INDEX: Final[dict[PortType, int]] = {p: i for i, p in enumerate(_PORT_TYPES)}

_PHASES: Final[tuple[TurnPhase, ...]] = (
    TurnPhase.INITIAL_SETTLEMENT,
    TurnPhase.INITIAL_ROAD,
    TurnPhase.ROLL,
    TurnPhase.DISCARD,
    TurnPhase.MOVE_ROBBER,
    TurnPhase.STEAL,
    TurnPhase.MAIN,
    TurnPhase.BUILD_ROADS,
    TurnPhase.YEAR_OF_PLENTY_SELECT,
    TurnPhase.MONOPOLY_SELECT,
    TurnPhase.STALEMATE,
    TurnPhase.GAME_OVER,
)
_PHASE_TO_INDEX: Final[dict[TurnPhase, int]] = {p: i for i, p in enumerate(_PHASES)}

# Pending one-hot: 7 concrete types + 1 "no pending" → 8 slots.
_PENDING_TYPES: Final[tuple[type, ...]] = (
    DiscardPending,
    RobberMovePending,
    StealPending,
    DomesticTradePending,
    RoadBuildingPending,
    YearOfPlentyPending,
    MonopolyPending,
)
_PENDING_TYPE_TO_INDEX: Final[dict[type, int]] = {t: i for i, t in enumerate(_PENDING_TYPES)}

# Per-element widths
TILE_WIDTH: Final[int] = len(_TILE_RESOURCES) + 11 + 1  # 6 + 11 + 1 = 18
VERTEX_WIDTH: Final[int] = 1 + N_PLAYER_SLOTS * 2  # empty + settlement[4] + city[4] = 9
EDGE_WIDTH: Final[int] = 1 + N_PLAYER_SLOTS  # empty + owner[4] = 5
PORT_SLOT_WIDTH: Final[int] = len(_PORT_TYPES)  # 6
OPPONENT_WIDTH: Final[int] = 9
N_OPPONENT_SLOTS: Final[int] = N_PLAYER_SLOTS - 1  # 3
SELF_HAND_WIDTH: Final[int] = 5 + 5 + 5  # resources, dev-in-hand by type, dev-played by type
AWARDS_WIDTH: Final[int] = 20
PHASE_PENDING_WIDTH: Final[int] = len(_PHASES) + len(_PENDING_TYPES) + 1  # 12 + 8

# Block widths
TILES_BLOCK: Final[int] = N_TILES * TILE_WIDTH  # 342
VERTICES_BLOCK: Final[int] = N_VERTICES * VERTEX_WIDTH  # 486
EDGES_BLOCK: Final[int] = N_EDGES * EDGE_WIDTH  # 360
PORTS_BLOCK: Final[int] = N_PORT_SLOTS * PORT_SLOT_WIDTH  # 54
OPPONENTS_BLOCK: Final[int] = N_OPPONENT_SLOTS * OPPONENT_WIDTH  # 27

# Normalisation constants — single source of truth. Bumping any of these
# requires regenerating the snapshot fixture.
_RESOURCE_NORM: Final[float] = 30.0
_TURN_NORM: Final[float] = 200.0
_CHIT_MAX: Final[int] = 12  # 2..12 → 11 slots
_BANK_NORM: Final[float] = 19.0
_DEV_DECK_NORM: Final[float] = 25.0
_DEV_HAND_NORM: Final[float] = 25.0
_ROADS_NORM: Final[float] = 15.0
_SETTLEMENTS_NORM: Final[float] = 5.0
_CITIES_NORM: Final[float] = 4.0
_KNIGHTS_NORM: Final[float] = 14.0
_VP_NORM: Final[float] = 10.0


def _chit_index(dice_number: int) -> int:
    """Map dice number 2..12 to slot 0..10."""
    return dice_number - 2


# ---------------------------------------------------------------------------
# Tiles block
# ---------------------------------------------------------------------------


def write_tiles(
    out: np.ndarray, offset: int, topology: BoardTopology, occupancy: BoardOccupancy
) -> None:
    """Write the tiles block, iterating in TileID order."""
    for tid_int in range(N_TILES):
        tid = TileID(tid_int)
        base = offset + tid_int * TILE_WIDTH
        tile = topology.tiles[tid]
        if tile.resource is not None:
            out[base + _RESOURCE_TO_INDEX[tile.resource]] = 1.0
        chit_base = base + len(_TILE_RESOURCES)
        if tile.dice_number is not None and 2 <= tile.dice_number <= _CHIT_MAX:
            out[chit_base + _chit_index(tile.dice_number)] = 1.0
        if occupancy.robber_tile == tid:
            out[base + TILE_WIDTH - 1] = 1.0


# ---------------------------------------------------------------------------
# Vertices block
# ---------------------------------------------------------------------------


def write_vertices(
    out: np.ndarray,
    offset: int,
    occupancy: BoardOccupancy,
    seat_of_player: dict[PlayerID, int],
) -> None:
    """Write the vertices block.

    ``seat_of_player`` is the viewer-perspective seat lookup (self → 0, others
    by rotation distance). Vertices with a building owned by an unknown player
    leave the row's owner slots zero (the empty bit is also zero — caller can
    inspect the building slots to determine occupancy).
    """
    for vid_int in range(N_VERTICES):
        base = offset + vid_int * VERTEX_WIDTH
        b = occupancy.buildings.get(VertexID(vid_int))
        if b is None:
            out[base] = 1.0
            continue
        owner, btype = b
        seat = seat_of_player.get(owner)
        if seat is None:
            continue
        if btype is BuildingType.SETTLEMENT:
            out[base + 1 + seat] = 1.0
        else:  # CITY
            out[base + 1 + N_PLAYER_SLOTS + seat] = 1.0


# ---------------------------------------------------------------------------
# Edges block
# ---------------------------------------------------------------------------


def write_edges(
    out: np.ndarray,
    offset: int,
    occupancy: BoardOccupancy,
    seat_of_player: dict[PlayerID, int],
) -> None:
    """Write the edges block (roads), iterating in EdgeID order."""
    for eid_int in range(N_EDGES):
        base = offset + eid_int * EDGE_WIDTH
        owner = occupancy.roads.get(EdgeID(eid_int))
        if owner is None:
            out[base] = 1.0
            continue
        seat = seat_of_player.get(owner)
        if seat is None:
            continue
        out[base + 1 + seat] = 1.0


# ---------------------------------------------------------------------------
# Ports block
# ---------------------------------------------------------------------------


def write_ports(out: np.ndarray, offset: int, topology: BoardTopology) -> None:
    """Per-port one-hot of port type, iterating in topology.ports order."""
    for i, port in enumerate(topology.ports):
        if i >= N_PORT_SLOTS:
            break
        if port.port_type is None:
            continue
        base = offset + i * PORT_SLOT_WIDTH
        out[base + _PORT_TYPE_TO_INDEX[port.port_type]] = 1.0


# ---------------------------------------------------------------------------
# Self hand
# ---------------------------------------------------------------------------


def write_self_hand(
    out: np.ndarray, offset: int, self_state: PlayerPerspectiveState
) -> None:
    """Resources, dev-in-hand by type, and dev-played by type for the viewer.

    Fails loud if ``dev_cards_in_hand`` is an int (would indicate the caller
    passed an opponent's perspective row).
    """
    if isinstance(self_state.dev_cards_in_hand, int):
        raise ValueError(
            "write_self_hand received an opponent row "
            "(dev_cards_in_hand is an int); pass the viewer's own row"
        )

    for r, c in self_state.resources.items():
        idx = _HAND_RESOURCE_TO_INDEX.get(r)
        if idx is None:
            continue
        out[offset + idx] = c / _RESOURCE_NORM

    hand_offset = offset + 5
    for card, _bought_turn in self_state.dev_cards_in_hand:
        out[hand_offset + _DEV_CARD_TO_INDEX[card]] += 1.0 / _DEV_HAND_NORM

    played_offset = offset + 10
    for card in self_state.dev_cards_played:
        out[played_offset + _DEV_CARD_TO_INDEX[card]] += 1.0 / _DEV_HAND_NORM


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------


def write_opponents(
    out: np.ndarray,
    offset: int,
    opponents_in_seat_order: list[PlayerPerspectiveState | None],
) -> None:
    """Write up to ``N_OPPONENT_SLOTS`` opponent rows in seat-rotation order.

    Missing slots (3-player game) stay zero.
    """
    for i, opp in enumerate(opponents_in_seat_order):
        if i >= N_OPPONENT_SLOTS:
            break
        if opp is None:
            continue
        base = offset + i * OPPONENT_WIDTH
        total_resources = sum(opp.resources.values())
        out[base + 0] = total_resources / _RESOURCE_NORM
        # dev_cards_in_hand is an int for opponents — confirm and use as-is.
        dev_in_hand = (
            opp.dev_cards_in_hand
            if isinstance(opp.dev_cards_in_hand, int)
            else len(opp.dev_cards_in_hand)
        )
        out[base + 1] = dev_in_hand / _DEV_HAND_NORM
        out[base + 2] = opp.roads_built / _ROADS_NORM
        out[base + 3] = opp.settlements_built / _SETTLEMENTS_NORM
        out[base + 4] = opp.cities_built / _CITIES_NORM
        out[base + 5] = opp.knights_played / _KNIGHTS_NORM
        out[base + 6] = len(opp.dev_cards_played) / _DEV_HAND_NORM
        out[base + 7] = opp.victory_points_public / _VP_NORM
        out[base + 8] = 1.0 if opp.has_played_dev_card_this_turn else 0.0


# ---------------------------------------------------------------------------
# Awards & counters
# ---------------------------------------------------------------------------


def write_awards(
    out: np.ndarray,
    offset: int,
    view: PlayerView,
    self_state: PlayerPerspectiveState,
    bank: Bank,
    seat_of_player: dict[PlayerID, int],
) -> None:
    """Game-global stats and viewer counters not duplicated elsewhere."""
    out[offset + 0] = min(view.turn_number, _TURN_NORM) / _TURN_NORM

    bank_off = offset + 1
    for r, c in bank.resources.items():
        idx = _HAND_RESOURCE_TO_INDEX.get(r)
        if idx is None:
            continue
        out[bank_off + idx] = c / _BANK_NORM

    out[offset + 6] = view.dev_deck_remaining / _DEV_DECK_NORM

    # longest road / largest army: (held by self?, anyone holds?)
    out[offset + 7] = 1.0 if view.longest_road_holder == self_state.player_id else 0.0
    out[offset + 8] = 1.0 if view.longest_road_holder is not None else 0.0
    out[offset + 9] = 1.0 if view.largest_army_holder == self_state.player_id else 0.0
    out[offset + 10] = 1.0 if view.largest_army_holder is not None else 0.0

    out[offset + 11] = 1.0 if view.current_player == self_state.player_id else 0.0

    out[offset + 12] = self_state.victory_points_public / _VP_NORM
    out[offset + 13] = self_state.roads_built / _ROADS_NORM
    out[offset + 14] = self_state.settlements_built / _SETTLEMENTS_NORM
    out[offset + 15] = self_state.cities_built / _CITIES_NORM
    out[offset + 16] = self_state.knights_played / _KNIGHTS_NORM
    out[offset + 17] = 1.0 if self_state.has_played_dev_card_this_turn else 0.0

    out[offset + 18] = 1.0 if view.winner == self_state.player_id else 0.0
    out[offset + 19] = 1.0 if view.winner is not None else 0.0


# ---------------------------------------------------------------------------
# Phase + pending
# ---------------------------------------------------------------------------


def write_phase_pending(
    out: np.ndarray, offset: int, phase: TurnPhase, pending: PendingEffect | None
) -> None:
    out[offset + _PHASE_TO_INDEX[phase]] = 1.0
    pending_off = offset + len(_PHASES)
    if pending is None:
        out[pending_off + len(_PENDING_TYPES)] = 1.0
        return
    idx = _PENDING_TYPE_TO_INDEX.get(type(pending))
    if idx is None:
        # Defensive: future pending types should bump the layout version.
        return
    out[pending_off + idx] = 1.0
