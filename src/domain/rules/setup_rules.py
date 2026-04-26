"""
Initial placement rules: legal settlement sites, legal first roads, setup order,
and starting resources from the second settlement.
"""

from __future__ import annotations

from domain.actions.all_actions import PlaceRoadAction, PlaceSettlementAction
from domain.enums import Resource, TurnPhase
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, VertexID


def _n_players(state: GameState) -> int:
    return len(state.config.player_ids)


def legal_setup_settlements(state: GameState) -> list[PlaceSettlementAction]:
    """
    Legal settlement vertices: empty, and no building on any vertex in
    :meth:`BoardTopology.vertices_within_distance_two` (per engine contract).
    """
    if state.phase is not TurnPhase.INITIAL_SETTLEMENT:
        return []
    occ = state.occupancy
    topo = state.topology
    pid = state.current_player
    out: list[PlaceSettlementAction] = []
    for vid, _v in topo.vertices.items():
        if vid in occ.buildings:
            continue
        blocked = topo.vertices_within_distance_two(vid)
        if any(bv in occ.buildings for bv in blocked):
            continue
        out.append(PlaceSettlementAction(player_id=pid, vertex_id=vid))
    return out


def legal_setup_roads(state: GameState) -> list[PlaceRoadAction]:
    """Roads on free edges incident to the settlement just placed in this setup turn."""
    if state.phase is not TurnPhase.INITIAL_ROAD:
        return []
    if state.last_settlement_vertex is None:
        return []
    pid = state.current_player
    base = state.last_settlement_vertex
    out: list[PlaceRoadAction] = []
    for edge in state.topology.edges_adjacent_to_vertex(base):
        if edge.edge_id in state.occupancy.roads:
            continue
        out.append(PlaceRoadAction(player_id=pid, edge_id=edge.edge_id))
    return out


def next_setup_player(state: GameState) -> PlayerID:
    """
    The player who places the next settlement: ``setup_order[setup_index]``.

    Call only while setup is unfinished (``setup_index < len(setup_order)``).
    """
    if state.setup_index >= len(state.setup_order):
        raise ValueError("setup is already complete")
    return state.setup_order[state.setup_index]


def second_settlement_resources(state: GameState, vertex_id: VertexID) -> dict[Resource, int]:
    """
    Starting hand from the second settlement: one card per adjacent *production* tile,
    skipping desert and unassigned tiles.
    """
    gain: dict[Resource, int] = {}
    for tile in state.topology.tiles_adjacent_to_vertex(vertex_id):
        if tile.is_desert() or tile.resource is None:
            continue
        r = tile.resource
        gain[r] = gain.get(r, 0) + 1
    return gain


def is_second_settlement_turn(state: GameState) -> bool:
    """Second round of the snake (each player's second settlement) grants resources."""
    n = _n_players(state)
    return state.setup_index >= n
