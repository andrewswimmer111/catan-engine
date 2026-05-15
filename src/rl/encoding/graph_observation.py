"""Graph observation encoding of a :class:`PlayerView`.

The encoder produces a flat ``float32`` vector laid out in five contiguous
blocks (tiles, vertices, edges, players, globals). The model unpacks the
vector into per-type feature tensors and runs message passing over a
fixed graph structure (``edge_index`` / ``edge_type``) cached at module
import; the topology of a standard 4-player Catan board is fixed across
games, so the adjacency never has to flow through the observation.

Layout
------

Per-element widths::

    tile feature  (TILE_FEAT_DIM   = 18) — resource6 + chit11 + robber1
    vertex feature(VERTEX_FEAT_DIM = 16) — port7 + empty1 + settle_owner4 + city_owner4
    edge feature  (EDGE_FEAT_DIM   =  5) — empty1 + road_owner4
    player row    (PLAYER_FEAT_DIM = 27) — full per-seat row (see _PLAYER_*_OFFSET)
    global vector (GLOBAL_FEAT_DIM = 30) — turn + bank5 + dev_deck1 + phase12 + pending8 + 3 award bits

Blocks (cumulative, ``GRAPH_OBS_SHAPE = (1704,)``)::

    [0       .. 342)  tiles    — 19 × 18
    [342     .. 1206) vertices — 54 × 16
    [1206    .. 1566) edges    — 72 × 5
    [1566    .. 1674) players  — 4  × 27
    [1674    .. 1704) globals  — 30

Perspective
-----------

All player-keyed slots (owner one-hots on vertices/edges, the four-row
player block) use viewer-perspective seat order: the viewer occupies seat
slot 0, other seats follow in seat-rotation distance through
``view.config.player_ids``. Identical to the flat encoder's convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from domain.board.layout import build_standard_board
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

__all__ = [
    "GRAPH_OBS_LAYOUT_VERSION",
    "GRAPH_OBS_SHAPE",
    "GraphObservationEncoder",
    "GraphStructure",
    "GRAPH_STRUCTURE",
    "TILE_FEAT_DIM",
    "VERTEX_FEAT_DIM",
    "EDGE_FEAT_DIM",
    "PLAYER_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "TILE_BLOCK",
    "VERTEX_BLOCK",
    "EDGE_BLOCK",
    "PLAYER_BLOCK",
    "GLOBAL_BLOCK",
    "TILE_OFFSET",
    "VERTEX_OFFSET",
    "EDGE_OFFSET",
    "PLAYER_OFFSET",
    "GLOBAL_OFFSET",
    "N_TILES",
    "N_VERTICES",
    "N_EDGES",
    "N_PLAYERS",
    "N_EDGE_TYPES",
    "VERTEX_SETTLE_SLOT_OFFSET",
    "VERTEX_CITY_SLOT_OFFSET",
    "EDGE_ROAD_SLOT_OFFSET",
]


GRAPH_OBS_LAYOUT_VERSION: Final[int] = 1

# ---------------------------------------------------------------------------
# Counts (mirror the standard-board topology)
# ---------------------------------------------------------------------------

N_TILES: Final[int] = 19
N_VERTICES: Final[int] = 54
N_EDGES: Final[int] = 72
N_PLAYERS: Final[int] = 4

# Adjacency relation types. Stable order — bumping requires bumping the
# layout version and rebuilding the cached structure.
EDGE_TYPE_TILE_VERTEX: Final[int] = 0
EDGE_TYPE_VERTEX_EDGE: Final[int] = 1
EDGE_TYPE_VERTEX_VERTEX: Final[int] = 2
EDGE_TYPE_TILE_TILE: Final[int] = 3
N_EDGE_TYPES: Final[int] = 4

# ---------------------------------------------------------------------------
# Feature widths
# ---------------------------------------------------------------------------

_TILE_RESOURCES: Final[tuple[Resource, ...]] = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
    Resource.DESERT,
)
_RESOURCE_TO_INDEX: Final[dict[Resource, int]] = {
    r: i for i, r in enumerate(_TILE_RESOURCES)
}

_HAND_RESOURCES: Final[tuple[Resource, ...]] = _TILE_RESOURCES[:5]
_HAND_RESOURCE_TO_INDEX: Final[dict[Resource, int]] = {
    r: i for i, r in enumerate(_HAND_RESOURCES)
}

_DEV_CARDS: Final[tuple[DevCardType, ...]] = (
    DevCardType.KNIGHT,
    DevCardType.ROAD_BUILDING,
    DevCardType.YEAR_OF_PLENTY,
    DevCardType.MONOPOLY,
    DevCardType.VICTORY_POINT,
)
_DEV_CARD_TO_INDEX: Final[dict[DevCardType, int]] = {
    c: i for i, c in enumerate(_DEV_CARDS)
}

_PORT_TYPES: Final[tuple[PortType, ...]] = (
    PortType.THREE_TO_ONE,
    PortType.WOOD_TWO,
    PortType.BRICK_TWO,
    PortType.SHEEP_TWO,
    PortType.WHEAT_TWO,
    PortType.ORE_TWO,
)
_PORT_TYPE_TO_INDEX: Final[dict[PortType, int]] = {
    p: i for i, p in enumerate(_PORT_TYPES)
}
_PORT_NONE_INDEX: Final[int] = len(_PORT_TYPES)  # 6 → "no port at this vertex"

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

_PENDING_TYPES: Final[tuple[type, ...]] = (
    DiscardPending,
    RobberMovePending,
    StealPending,
    DomesticTradePending,
    RoadBuildingPending,
    YearOfPlentyPending,
    MonopolyPending,
)
_PENDING_TYPE_TO_INDEX: Final[dict[type, int]] = {
    t: i for i, t in enumerate(_PENDING_TYPES)
}
_PENDING_NONE_INDEX: Final[int] = len(_PENDING_TYPES)  # 7 → "no pending"

# Tile feature: [resource(6), chit(11), robber(1)]
TILE_FEAT_DIM: Final[int] = len(_TILE_RESOURCES) + 11 + 1  # 18
_TILE_CHIT_OFFSET: Final[int] = len(_TILE_RESOURCES)
_TILE_ROBBER_OFFSET: Final[int] = _TILE_CHIT_OFFSET + 11

# Vertex feature: [port(7), empty(1), settle_owner(4), city_owner(4)]
VERTEX_FEAT_DIM: Final[int] = (len(_PORT_TYPES) + 1) + 1 + N_PLAYERS + N_PLAYERS  # 16
_VERTEX_EMPTY_OFFSET: Final[int] = len(_PORT_TYPES) + 1
VERTEX_SETTLE_SLOT_OFFSET: Final[int] = _VERTEX_EMPTY_OFFSET + 1
VERTEX_CITY_SLOT_OFFSET: Final[int] = VERTEX_SETTLE_SLOT_OFFSET + N_PLAYERS

# Edge feature: [empty(1), road_owner(4)]
EDGE_FEAT_DIM: Final[int] = 1 + N_PLAYERS  # 5
_EDGE_EMPTY_OFFSET: Final[int] = 0
EDGE_ROAD_SLOT_OFFSET: Final[int] = 1

# Player feature row (per seat, 27 slots total):
#   resources(5), total_res(1), dev_in_hand_by_type(5), dev_in_hand_count(1),
#   dev_played_by_type(5), roads(1), settlements(1), cities(1), knights(1),
#   vp(1), has_played_dev(1), holds_lr(1), holds_la(1), is_curr(1), is_winner(1)
# An ``is_viewer`` slot would be redundant — the viewer always lives at
# seat slot 0, so the model can infer it from the row's position.
PLAYER_FEAT_DIM: Final[int] = 5 + 1 + 5 + 1 + 5 + 10  # 27

_PLAYER_RESOURCES_OFFSET: Final[int] = 0
_PLAYER_TOTAL_RES_OFFSET: Final[int] = 5
_PLAYER_DEV_HAND_TYPE_OFFSET: Final[int] = 6
_PLAYER_DEV_HAND_COUNT_OFFSET: Final[int] = 11
_PLAYER_DEV_PLAYED_OFFSET: Final[int] = 12
_PLAYER_ROADS_OFFSET: Final[int] = 17
_PLAYER_SETTLEMENTS_OFFSET: Final[int] = 18
_PLAYER_CITIES_OFFSET: Final[int] = 19
_PLAYER_KNIGHTS_OFFSET: Final[int] = 20
_PLAYER_VP_OFFSET: Final[int] = 21
_PLAYER_PLAYED_DEV_OFFSET: Final[int] = 22
_PLAYER_HOLDS_LR_OFFSET: Final[int] = 23
_PLAYER_HOLDS_LA_OFFSET: Final[int] = 24
_PLAYER_IS_CURR_OFFSET: Final[int] = 25
_PLAYER_IS_WINNER_OFFSET: Final[int] = 26

# Global feature:
#   [turn(1), bank(5), dev_deck(1), phase(12), pending(8), lr_held(1), la_held(1), winner_exists(1)]
GLOBAL_FEAT_DIM: Final[int] = 1 + 5 + 1 + len(_PHASES) + (len(_PENDING_TYPES) + 1) + 1 + 1 + 1
assert GLOBAL_FEAT_DIM == 30

_GLOBAL_TURN_OFFSET: Final[int] = 0
_GLOBAL_BANK_OFFSET: Final[int] = 1
_GLOBAL_DEV_DECK_OFFSET: Final[int] = 6
_GLOBAL_PHASE_OFFSET: Final[int] = 7
_GLOBAL_PENDING_OFFSET: Final[int] = _GLOBAL_PHASE_OFFSET + len(_PHASES)  # 19
_GLOBAL_LR_HELD_OFFSET: Final[int] = (
    _GLOBAL_PENDING_OFFSET + len(_PENDING_TYPES) + 1
)  # 27
_GLOBAL_LA_HELD_OFFSET: Final[int] = _GLOBAL_LR_HELD_OFFSET + 1  # 28
_GLOBAL_WINNER_EXISTS_OFFSET: Final[int] = _GLOBAL_LA_HELD_OFFSET + 1  # 29

# Block widths
TILE_BLOCK: Final[int] = N_TILES * TILE_FEAT_DIM  # 342
VERTEX_BLOCK: Final[int] = N_VERTICES * VERTEX_FEAT_DIM  # 864
EDGE_BLOCK: Final[int] = N_EDGES * EDGE_FEAT_DIM  # 360
PLAYER_BLOCK: Final[int] = N_PLAYERS * PLAYER_FEAT_DIM  # 108
GLOBAL_BLOCK: Final[int] = GLOBAL_FEAT_DIM  # 30

# Block offsets (cumulative)
TILE_OFFSET: Final[int] = 0
VERTEX_OFFSET: Final[int] = TILE_OFFSET + TILE_BLOCK
EDGE_OFFSET: Final[int] = VERTEX_OFFSET + VERTEX_BLOCK
PLAYER_OFFSET: Final[int] = EDGE_OFFSET + EDGE_BLOCK
GLOBAL_OFFSET: Final[int] = PLAYER_OFFSET + PLAYER_BLOCK

_GRAPH_OBS_LEN: Final[int] = GLOBAL_OFFSET + GLOBAL_BLOCK  # 1704
GRAPH_OBS_SHAPE: Final[tuple[int, ...]] = (_GRAPH_OBS_LEN,)

# Normalisation constants (mirror the flat encoder so the two share scale).
_RESOURCE_NORM: Final[float] = 30.0
_TURN_NORM: Final[float] = 200.0
_BANK_NORM: Final[float] = 19.0
_DEV_DECK_NORM: Final[float] = 25.0
_DEV_HAND_NORM: Final[float] = 25.0
_ROADS_NORM: Final[float] = 15.0
_SETTLEMENTS_NORM: Final[float] = 5.0
_CITIES_NORM: Final[float] = 4.0
_KNIGHTS_NORM: Final[float] = 14.0
_VP_NORM: Final[float] = 10.0
_CHIT_MAX: Final[int] = 12


# ---------------------------------------------------------------------------
# Static graph structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphStructure:
    """Fixed connectivity of the standard 4-player Catan board graph.

    Node indices are laid out tiles-first, then vertices, then edges:
    ``[0, N_TILES)`` are tiles, ``[N_TILES, N_TILES+N_VERTICES)`` are
    vertices, ``[N_TILES+N_VERTICES, N_TILES+N_VERTICES+N_EDGES)`` are
    edges. ``edge_index`` is the bidirectional adjacency in PyG's
    ``(2, E)`` source-target layout; ``edge_type`` carries the integer
    relation tag (one of the ``EDGE_TYPE_*`` constants) per directed edge.
    """

    edge_index: np.ndarray  # int64, shape (2, E)
    edge_type: np.ndarray  # int64, shape (E,)
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _tile_node_index(tile_id: int) -> int:
    return tile_id


def _vertex_node_index(vertex_id: int) -> int:
    return N_TILES + vertex_id


def _edge_node_index(edge_id: int) -> int:
    return N_TILES + N_VERTICES + edge_id


def _build_graph_structure(topology: BoardTopology) -> GraphStructure:
    """Derive the directed adjacency tensors from a canonical board topology.

    The topology of the standard 4-player Catan board is fixed across all
    games (only the resource/pip/port assignments vary), so this runs once
    at module import and the result is cached as :data:`GRAPH_STRUCTURE`.
    """
    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []

    def add_undirected(a: int, b: int, t: int) -> None:
        src.append(a)
        dst.append(b)
        etype.append(t)
        src.append(b)
        dst.append(a)
        etype.append(t)

    # vertex↔tile (incidence) — each vertex's adjacent tiles.
    for vid_int in range(N_VERTICES):
        v = topology.vertices[VertexID(vid_int)]
        v_node = _vertex_node_index(vid_int)
        for tid in v.adjacent_tiles:
            add_undirected(v_node, _tile_node_index(int(tid)), EDGE_TYPE_TILE_VERTEX)

    # vertex↔edge (incidence) — each vertex's adjacent edges.
    for vid_int in range(N_VERTICES):
        v = topology.vertices[VertexID(vid_int)]
        v_node = _vertex_node_index(vid_int)
        for eid in v.adjacent_edges:
            add_undirected(v_node, _edge_node_index(int(eid)), EDGE_TYPE_VERTEX_EDGE)

    # vertex↔vertex (proximity) — degree-2/3 adjacency.
    seen_vv: set[tuple[int, int]] = set()
    for vid_int in range(N_VERTICES):
        v = topology.vertices[VertexID(vid_int)]
        for other in v.adjacent_vertices:
            a, b = vid_int, int(other)
            key = (min(a, b), max(a, b))
            if key in seen_vv:
                continue
            seen_vv.add(key)
            add_undirected(
                _vertex_node_index(a),
                _vertex_node_index(b),
                EDGE_TYPE_VERTEX_VERTEX,
            )

    # tile↔tile (share an edge — i.e. two common vertices).
    tile_vertices: dict[int, frozenset[int]] = {}
    for tid_int in range(N_TILES):
        v_ids = {
            int(vid)
            for vid in range(N_VERTICES)
            if TileID(tid_int) in topology.vertices[VertexID(vid)].adjacent_tiles
        }
        tile_vertices[tid_int] = frozenset(v_ids)
    for a in range(N_TILES):
        for b in range(a + 1, N_TILES):
            if len(tile_vertices[a] & tile_vertices[b]) >= 2:
                add_undirected(
                    _tile_node_index(a),
                    _tile_node_index(b),
                    EDGE_TYPE_TILE_TILE,
                )

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_type = np.array(etype, dtype=np.int64)
    return GraphStructure(
        edge_index=edge_index,
        edge_type=edge_type,
        n_nodes=N_TILES + N_VERTICES + N_EDGES,
    )


# Built once at import. The standard-board topology is the single source of
# truth for connectivity; randomised resource/chit/port assignments don't
# touch adjacency.
GRAPH_STRUCTURE: Final[GraphStructure] = _build_graph_structure(build_standard_board())


# ---------------------------------------------------------------------------
# Per-state feature writers
# ---------------------------------------------------------------------------


def _seat_order(player_ids: list[PlayerID], viewer: PlayerID) -> list[PlayerID]:
    """Rotate ``player_ids`` so ``viewer`` is at index 0."""
    try:
        i = player_ids.index(viewer)
    except ValueError as e:
        raise ValueError(
            f"viewer {viewer} is not in player_ids {player_ids}"
        ) from e
    return player_ids[i:] + player_ids[:i]


def _write_tile_block(
    out: np.ndarray,
    topology: BoardTopology,
    occupancy: BoardOccupancy,
) -> None:
    for tid_int in range(N_TILES):
        base = TILE_OFFSET + tid_int * TILE_FEAT_DIM
        tile = topology.tiles[TileID(tid_int)]
        if tile.resource is not None:
            out[base + _RESOURCE_TO_INDEX[tile.resource]] = 1.0
        if tile.dice_number is not None and 2 <= tile.dice_number <= _CHIT_MAX:
            out[base + _TILE_CHIT_OFFSET + (tile.dice_number - 2)] = 1.0
        if occupancy.robber_tile == TileID(tid_int):
            out[base + _TILE_ROBBER_OFFSET] = 1.0


def _write_vertex_block(
    out: np.ndarray,
    topology: BoardTopology,
    occupancy: BoardOccupancy,
    seat_of_player: dict[PlayerID, int],
) -> None:
    for vid_int in range(N_VERTICES):
        base = VERTEX_OFFSET + vid_int * VERTEX_FEAT_DIM
        vertex = topology.vertices[VertexID(vid_int)]

        # Port one-hot. Vertices with no port slot get the "no port" bit.
        if vertex.port is None:
            out[base + _PORT_NONE_INDEX] = 1.0
        else:
            out[base + _PORT_TYPE_TO_INDEX[vertex.port]] = 1.0

        building = occupancy.buildings.get(VertexID(vid_int))
        if building is None:
            out[base + _VERTEX_EMPTY_OFFSET] = 1.0
            continue
        owner, btype = building
        seat = seat_of_player.get(owner)
        if seat is None:
            # Unrecognised owner — leave both empty bit and owner bits zero
            # so the model can spot the inconsistency. Mirrors the flat
            # encoder's behaviour for the same case.
            continue
        if btype is BuildingType.SETTLEMENT:
            out[base + VERTEX_SETTLE_SLOT_OFFSET + seat] = 1.0
        else:
            out[base + VERTEX_CITY_SLOT_OFFSET + seat] = 1.0


def _write_edge_block(
    out: np.ndarray,
    occupancy: BoardOccupancy,
    seat_of_player: dict[PlayerID, int],
) -> None:
    for eid_int in range(N_EDGES):
        base = EDGE_OFFSET + eid_int * EDGE_FEAT_DIM
        owner = occupancy.roads.get(EdgeID(eid_int))
        if owner is None:
            out[base + _EDGE_EMPTY_OFFSET] = 1.0
            continue
        seat = seat_of_player.get(owner)
        if seat is None:
            continue
        out[base + EDGE_ROAD_SLOT_OFFSET + seat] = 1.0


def _write_player_block(
    out: np.ndarray,
    view: PlayerView,
    seat_order: list[PlayerID],
    viewer: PlayerID,
) -> None:
    """Write the per-seat player rows in viewer-perspective seat order."""
    for slot, pid in enumerate(seat_order):
        row = view.players.get(pid)
        if row is None:
            continue
        base = PLAYER_OFFSET + slot * PLAYER_FEAT_DIM
        _write_player_row(out, base, row, view, viewer, pid)


def _write_player_row(
    out: np.ndarray,
    base: int,
    row: PlayerPerspectiveState,
    view: PlayerView,
    viewer: PlayerID,
    pid: PlayerID,
) -> None:
    """Fill the 27-slot row for one seat.

    Resource and dev-in-hand-by-type slots are only written for the viewer's
    own row; opponent rows leave those zero and report aggregate counts.
    """
    is_viewer = pid == viewer

    if is_viewer:
        for r, c in row.resources.items():
            idx = _HAND_RESOURCE_TO_INDEX.get(r)
            if idx is None:
                continue
            out[base + _PLAYER_RESOURCES_OFFSET + idx] = c / _RESOURCE_NORM

    total_resources = sum(row.resources.values())
    out[base + _PLAYER_TOTAL_RES_OFFSET] = total_resources / _RESOURCE_NORM

    if is_viewer and isinstance(row.dev_cards_in_hand, list):
        for card, _bought_turn in row.dev_cards_in_hand:
            out[base + _PLAYER_DEV_HAND_TYPE_OFFSET + _DEV_CARD_TO_INDEX[card]] += (
                1.0 / _DEV_HAND_NORM
            )
        dev_in_hand_count = len(row.dev_cards_in_hand)
    elif isinstance(row.dev_cards_in_hand, int):
        dev_in_hand_count = row.dev_cards_in_hand
    else:
        dev_in_hand_count = len(row.dev_cards_in_hand)

    out[base + _PLAYER_DEV_HAND_COUNT_OFFSET] = dev_in_hand_count / _DEV_HAND_NORM

    for card in row.dev_cards_played:
        out[base + _PLAYER_DEV_PLAYED_OFFSET + _DEV_CARD_TO_INDEX[card]] += (
            1.0 / _DEV_HAND_NORM
        )

    out[base + _PLAYER_ROADS_OFFSET] = row.roads_built / _ROADS_NORM
    out[base + _PLAYER_SETTLEMENTS_OFFSET] = row.settlements_built / _SETTLEMENTS_NORM
    out[base + _PLAYER_CITIES_OFFSET] = row.cities_built / _CITIES_NORM
    out[base + _PLAYER_KNIGHTS_OFFSET] = row.knights_played / _KNIGHTS_NORM
    out[base + _PLAYER_VP_OFFSET] = row.victory_points_public / _VP_NORM
    out[base + _PLAYER_PLAYED_DEV_OFFSET] = (
        1.0 if row.has_played_dev_card_this_turn else 0.0
    )
    out[base + _PLAYER_HOLDS_LR_OFFSET] = 1.0 if view.longest_road_holder == pid else 0.0
    out[base + _PLAYER_HOLDS_LA_OFFSET] = 1.0 if view.largest_army_holder == pid else 0.0
    out[base + _PLAYER_IS_CURR_OFFSET] = 1.0 if view.current_player == pid else 0.0
    out[base + _PLAYER_IS_WINNER_OFFSET] = 1.0 if view.winner == pid else 0.0


def _write_global_block(out: np.ndarray, view: PlayerView, bank: Bank) -> None:
    base = GLOBAL_OFFSET
    out[base + _GLOBAL_TURN_OFFSET] = min(view.turn_number, _TURN_NORM) / _TURN_NORM
    for r, c in bank.resources.items():
        idx = _HAND_RESOURCE_TO_INDEX.get(r)
        if idx is None:
            continue
        out[base + _GLOBAL_BANK_OFFSET + idx] = c / _BANK_NORM
    out[base + _GLOBAL_DEV_DECK_OFFSET] = view.dev_deck_remaining / _DEV_DECK_NORM
    out[base + _GLOBAL_PHASE_OFFSET + _PHASE_TO_INDEX[view.phase]] = 1.0
    if view.pending is None:
        out[base + _GLOBAL_PENDING_OFFSET + _PENDING_NONE_INDEX] = 1.0
    else:
        idx = _PENDING_TYPE_TO_INDEX.get(type(view.pending))
        if idx is not None:
            out[base + _GLOBAL_PENDING_OFFSET + idx] = 1.0
    out[base + _GLOBAL_LR_HELD_OFFSET] = (
        1.0 if view.longest_road_holder is not None else 0.0
    )
    out[base + _GLOBAL_LA_HELD_OFFSET] = (
        1.0 if view.largest_army_holder is not None else 0.0
    )
    out[base + _GLOBAL_WINNER_EXISTS_OFFSET] = 1.0 if view.winner is not None else 0.0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class GraphObservationEncoder:
    """Encode a :class:`PlayerView` as a flat float32 vector packing five blocks.

    The encoder is stateless; instances are cheap to construct. ``out_shape``
    is a class attribute so the trainer can size buffers without
    instantiating. The fixed graph adjacency is owned by
    :data:`GRAPH_STRUCTURE` and is never copied into the observation —
    consumers (the GNN model) hold it as a module buffer.
    """

    out_shape: Final[tuple[int, ...]] = GRAPH_OBS_SHAPE
    layout_version: Final[int] = GRAPH_OBS_LAYOUT_VERSION

    def encode(self, view: PlayerView) -> np.ndarray:
        """Return the flat graph observation for ``view`` (dtype ``float32``).

        The returned array is freshly allocated; callers may freely mutate it.
        """
        out = np.zeros(GRAPH_OBS_SHAPE, dtype=np.float32)

        viewer = self._viewer_id(view)
        seat_order = _seat_order(list(view.config.player_ids), viewer)
        seat_of_player: dict[PlayerID, int] = {
            pid: i for i, pid in enumerate(seat_order)
        }

        _write_tile_block(out, view.topology, view.occupancy)
        _write_vertex_block(out, view.topology, view.occupancy, seat_of_player)
        _write_edge_block(out, view.occupancy, seat_of_player)
        _write_player_block(out, view, seat_order, viewer)
        _write_global_block(out, view, view.bank)

        return out

    @staticmethod
    def _viewer_id(view: PlayerView) -> PlayerID:
        """The viewer is the player whose row exposes the full dev hand list."""
        for pid, row in view.players.items():
            if not isinstance(row.dev_cards_in_hand, int):
                return pid
        raise ValueError(
            "no viewer found in PlayerView — every row has integer dev_cards_in_hand"
        )
