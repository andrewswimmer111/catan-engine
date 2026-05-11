"""Rule helpers for HeuristicAgent: setup, discard, robber, build, trade.

Split out so :mod:`heuristic_agent` stays focused on phase dispatch. This
module is the baseline strategy itself — pip-count placement, abundance-
weighted discards, leader-targeting robber, build-priority main turns.
"""

from __future__ import annotations

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import DomesticTradeState, Resource
from domain.game.state import GameState
from domain.ids import PlayerID, TileID, VertexID
from domain.rules import build_rules, victory
from domain.turn.pending import DiscardPending

__all__ = [
    "PIP_BY_NUMBER",
    "RESOURCE_VALUE",
    "vertex_pip_count",
    "choose_setup_settlement",
    "choose_setup_road",
    "choose_discard",
    "choose_robber_move",
    "choose_steal_target",
    "choose_main",
    "best_road",
    "domestic_trade_fallback",
]

# Standard Catan pip-count (dot value) for each production roll.
PIP_BY_NUMBER: dict[int, int] = {
    2: 1, 12: 1, 3: 2, 11: 2, 4: 3, 10: 3, 5: 4, 9: 4, 6: 5, 8: 5,
}

# Higher = more valuable; tie-breaker for discards + YOP/monopoly targets.
RESOURCE_VALUE: dict[Resource, int] = {
    Resource.WHEAT: 5, Resource.ORE: 4, Resource.BRICK: 3,
    Resource.WOOD: 3, Resource.SHEEP: 2,
}

# Build cost priority order used by the planner and maritime trade gating.
_BUILD_PRIORITY: tuple[dict[Resource, int], ...] = (
    build_rules.CITY_COST,
    build_rules.SETTLEMENT_COST,
    build_rules.ROAD_COST,
    build_rules.DEV_CARD_COST,
)


def vertex_pip_count(state: GameState, vertex_id: VertexID) -> int:
    total = 0
    for tile in state.topology.tiles_adjacent_to_vertex(vertex_id):
        if tile.dice_number is None:
            continue
        total += PIP_BY_NUMBER.get(tile.dice_number, 0)
    return total


def choose_setup_settlement(
    legal: list[A.PlaceSettlementAction], state: GameState
) -> A.PlaceSettlementAction:
    return max(
        legal,
        key=lambda a: (vertex_pip_count(state, a.vertex_id), -int(a.vertex_id)),
    )


def choose_setup_road(
    legal: list[A.PlaceRoadAction], state: GameState
) -> A.PlaceRoadAction:
    base = state.last_settlement_vertex
    if base is None:
        return legal[0]
    topo = state.topology

    def score(a: A.PlaceRoadAction) -> tuple[int, int]:
        v1, v2 = topo.edges[a.edge_id].vertices
        other = v2 if v1 == base else v1
        return (vertex_pip_count(state, other), -int(a.edge_id))

    return max(legal, key=score)


def choose_discard(
    state: GameState, player_id: PlayerID
) -> A.DiscardResourcesAction:
    pend = state.pending
    if not isinstance(pend, DiscardPending):
        raise ValueError("choose_discard called outside DISCARD phase")
    need = pend.cards_to_discard.get(player_id, 0)
    hand = dict(state.players[player_id].resources)
    discard: dict[Resource, int] = {}
    for _ in range(need):
        best: Resource | None = None
        for r, c in hand.items():
            if c <= 0:
                continue
            if best is None or (hand[r], -RESOURCE_VALUE[r]) > (
                hand[best], -RESOURCE_VALUE[best]
            ):
                best = r
        if best is None:
            raise ValueError("not enough cards to discard")
        discard[best] = discard.get(best, 0) + 1
        hand[best] -= 1
    return A.DiscardResourcesAction(player_id=player_id, resources=discard)


def _leader(state: GameState) -> PlayerID:
    return max(
        state.config.player_ids,
        key=lambda pid: (
            victory.compute_victory_points(state, pid),
            state.players[pid].resource_count(),
        ),
    )


def _players_on_tile(state: GameState, tile_id: TileID) -> dict[PlayerID, int]:
    seats: set[PlayerID] = set()
    for vid, (owner, _bt) in state.occupancy.buildings.items():
        if tile_id in state.topology.vertices[vid].adjacent_tiles:
            seats.add(owner)
    return {p: state.players[p].resource_count() for p in seats}


def choose_robber_move(
    legal: list[A.MoveRobberAction], state: GameState, me: PlayerID
) -> A.MoveRobberAction:
    leader = _leader(state)

    def score(a: A.MoveRobberAction) -> tuple[int, int, int]:
        seats = _players_on_tile(state, a.tile_id)
        not_own = 0 if me in seats else 1
        on_leader = 1 if leader in seats and leader != me else 0
        victim_hand = sum(c for pid, c in seats.items() if pid != me)
        return (on_leader, not_own, victim_hand)

    return max(legal, key=score)


def choose_steal_target(
    legal: list[A.StealResourceAction], state: GameState
) -> A.StealResourceAction:
    return max(
        legal, key=lambda a: state.players[a.target_player_id].resource_count()
    )


def _settle_able(state: GameState, v: VertexID) -> bool:
    occ = state.occupancy
    if v in occ.buildings:
        return False
    return not any(
        adj in occ.buildings for adj in state.topology.vertices[v].adjacent_vertices
    )


def best_road(
    state: GameState, legal: list[A.BuildRoadAction]
) -> A.BuildRoadAction | None:
    """Score by best settlement potential unlocked. Prefer roads whose far end
    is itself settle-able after placement; fall back to distance-2 potential.
    """
    if not legal:
        return None
    topo = state.topology

    def score(a: A.BuildRoadAction) -> tuple[int, int]:
        v1, v2 = topo.edges[a.edge_id].vertices
        direct, indirect = 0, 0
        for v in (v1, v2):
            if _settle_able(state, v):
                direct = max(direct, vertex_pip_count(state, v))
            for adj in topo.vertices[v].adjacent_vertices:
                if _settle_able(state, adj):
                    indirect = max(indirect, vertex_pip_count(state, adj))
        return (direct, indirect)

    return max(legal, key=score)


def _best_building_pick(legal: list[Action], state: GameState) -> Action | None:
    if not legal:
        return None
    return max(legal, key=lambda a: vertex_pip_count(state, a.vertex_id))


def _planned_resource(state: GameState, me: PlayerID) -> Resource:
    """Resource most needed to fund the next build (city > settle > road > dev)."""
    p = state.players[me]
    for cost in _BUILD_PRIORITY:
        for r, c in cost.items():
            if p.resources.get(r, 0) < c:
                return r
    return Resource.WHEAT


def _useful_maritime(
    state: GameState, me: PlayerID, legal: list[A.MaritimeTradeAction]
) -> A.MaritimeTradeAction | None:
    """Trade a surplus card for one we need toward the next planned build."""
    if not legal:
        return None
    hand = dict(state.players[me].resources)
    for cost in _BUILD_PRIORITY:
        deficit = {r: c - hand.get(r, 0) for r, c in cost.items() if c - hand.get(r, 0) > 0}
        if not deficit:
            continue
        for a in legal:
            if a.receive not in deficit:
                continue
            spare = hand.get(a.give, 0) - cost.get(a.give, 0)
            if spare >= a.give_count:
                return a
    return None


def choose_main(
    state: GameState, legal: list[Action], me: PlayerID
) -> Action | None:
    cities = [a for a in legal if isinstance(a, A.BuildCityAction)]
    settlements = [a for a in legal if isinstance(a, A.BuildSettlementAction)]
    roads = [a for a in legal if isinstance(a, A.BuildRoadAction)]
    buys = [a for a in legal if isinstance(a, A.BuyDevCardAction)]
    maritimes = [a for a in legal if isinstance(a, A.MaritimeTradeAction)]
    knights = [a for a in legal if isinstance(a, A.PlayKnightAction)]
    yops = [a for a in legal if isinstance(a, A.PlayYearOfPlentyAction)]
    monos = [a for a in legal if isinstance(a, A.PlayMonopolyAction)]
    rbs = [a for a in legal if isinstance(a, A.PlayRoadBuildingAction)]
    end = [a for a in legal if isinstance(a, A.EndTurnAction)]

    # Play knights aggressively — needed to chase largest army (+2 VP).
    if knights:
        return knights[0]

    if yops:
        want = _planned_resource(state, me)
        for a in yops:
            if a.resource1 is want and a.resource2 is want:
                return a
        for a in yops:
            if want in (a.resource1, a.resource2):
                return a
        return yops[0]

    if monos:
        want = _planned_resource(state, me)
        for a in monos:
            if a.resource is want:
                return a
        return monos[0]

    if rbs and roads:
        return rbs[0]

    pick = _best_building_pick(cities, state)
    if pick is not None:
        return pick
    pick = _best_building_pick(settlements, state)
    if pick is not None:
        return pick
    rd = best_road(state, roads)
    if rd is not None:
        return rd
    if buys:
        return buys[0]
    mt = _useful_maritime(state, me, maritimes)
    if mt is not None:
        return mt
    if end:
        return end[0]
    return None


def domestic_trade_fallback(legal: list[Action]) -> Action | None:
    """Reject / cancel domestic trades — the heuristic never participates."""
    for a in legal:
        if isinstance(a, A.CancelDomesticTradeAction):
            return a
    for a in legal:
        if (
            isinstance(a, A.RespondDomesticTradeAction)
            and a.response is DomesticTradeState.REJECTED
        ):
            return a
    return None
