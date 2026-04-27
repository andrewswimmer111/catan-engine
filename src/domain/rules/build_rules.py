"""
Main-phase build actions: costs, piece limits, connectivity, and dev-card purchase.
"""

from __future__ import annotations

from domain.actions.all_actions import (
    BuildCityAction,
    BuildRoadAction,
    BuildSettlementAction,
    BuyDevCardAction,
    EndTurnAction,
)
from domain.board.topology import BoardTopology
from domain.enums import BuildingType, Resource, TurnPhase
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, VertexID
from domain.turn.pending import RoadBuildingPending

# --- Costs (Catan 4e-style) ---

ROAD_COST: dict[Resource, int] = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
}
SETTLEMENT_COST: dict[Resource, int] = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
}
CITY_COST: dict[Resource, int] = {
    Resource.ORE: 3,
    Resource.WHEAT: 2,
}
DEV_CARD_COST: dict[Resource, int] = {
    Resource.ORE: 1,
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
}

MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


def _settlement_distance_ok(state: GameState, vertex_id: VertexID) -> bool:
    occ = state.occupancy
    if vertex_id in occ.buildings:
        return False
    blocked = state.topology.vertices_within_distance_two(vertex_id)
    return not any(bv in occ.buildings for bv in blocked)


def _road_reaches_network(
    state: GameState, topology: BoardTopology, player_id: PlayerID, new_edge: EdgeID
) -> bool:
    """True if the new road touches your building or another of your roads (other than this slot)."""
    e = topology.edges[new_edge]
    for v in e.vertices:
        b = state.occupancy.buildings.get(v)
        if b is not None and b[0] == player_id:
            return True
    for v in e.vertices:
        for inc in topology.edges_adjacent_to_vertex(v):
            if inc.edge_id == new_edge:
                continue
            if state.occupancy.roads.get(inc.edge_id) == player_id:
                return True
    return False


def _can_afford_road(
    state: GameState, player_id: PlayerID, free: bool
) -> bool:
    if free:
        return True
    return state.players[player_id].can_afford(ROAD_COST)


def legal_build_roads(state: GameState) -> list[BuildRoadAction]:
    """``MAIN`` (paid) or ``BUILD_ROADS`` (unpaid) road placement; connectivity applies."""
    if state.phase is TurnPhase.MAIN:
        if state.pending is not None:
            return []
    elif state.phase is TurnPhase.BUILD_ROADS:
        if not isinstance(state.pending, RoadBuildingPending):
            return []
    else:
        return []

    pid = state.current_player
    if state.players[pid].roads_built >= MAX_ROADS:
        return []
    free = state.phase is TurnPhase.BUILD_ROADS
    if not _can_afford_road(state, pid, free):
        return []
    out: list[BuildRoadAction] = []
    topo = state.topology
    for eid, edge in topo.edges.items():
        if eid in state.occupancy.roads:
            continue
        if not _road_reaches_network(state, topo, pid, eid):
            continue
        out.append(BuildRoadAction(player_id=pid, edge_id=eid))
    return out


def _settlement_touches_my_road(state: GameState, vertex_id: VertexID, player_id: PlayerID) -> bool:
    for e in state.topology.edges_adjacent_to_vertex(vertex_id):
        if state.occupancy.roads.get(e.edge_id) == player_id:
            return True
    return False


def legal_build_settlements(state: GameState) -> list[BuildSettlementAction]:
    if state.phase is not TurnPhase.MAIN or state.pending is not None:
        return []
    pid = state.current_player
    if state.players[pid].settlements_built >= MAX_SETTLEMENTS:
        return []
    if not state.players[pid].can_afford(SETTLEMENT_COST):
        return []
    out: list[BuildSettlementAction] = []
    for vid, _v in state.topology.vertices.items():
        if not _settlement_distance_ok(state, vid):
            continue
        if not _settlement_touches_my_road(state, vid, pid):
            continue
        out.append(BuildSettlementAction(player_id=pid, vertex_id=vid))
    return out


def legal_build_cities(state: GameState) -> list[BuildCityAction]:
    if state.phase is not TurnPhase.MAIN or state.pending is not None:
        return []
    pid = state.current_player
    if state.players[pid].cities_built >= MAX_CITIES:
        return []
    if not state.players[pid].can_afford(CITY_COST):
        return []
    out: list[BuildCityAction] = []
    for vid, (owner, btype) in state.occupancy.buildings.items():
        if owner != pid or btype is not BuildingType.SETTLEMENT:
            continue
        out.append(BuildCityAction(player_id=pid, vertex_id=vid))
    return out


def legal_buy_dev_card(state: GameState) -> list[BuyDevCardAction]:
    if state.phase is not TurnPhase.MAIN or state.pending is not None:
        return []
    pid = state.current_player
    if not state.players[pid].can_afford(DEV_CARD_COST):
        return []
    if state.dev_deck.remaining() == 0:
        return []
    return [BuyDevCardAction(player_id=pid)]


def legal_end_turn(state: GameState) -> list[EndTurnAction]:
    if state.phase is not TurnPhase.MAIN or state.pending is not None:
        return []
    return [EndTurnAction(player_id=state.current_player)]
