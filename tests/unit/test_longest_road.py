"""Unit tests for :mod:`domain.rules.longest_road`."""

from __future__ import annotations

import pytest

from domain.board.layout import build_standard_board
from domain.board.occupancy import BoardOccupancy
from domain.enums import BuildingType, TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck, standard_dev_deck_composition
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from domain.rules import longest_road as lr


def _minimal_state() -> GameState:
    topo = build_standard_board()
    pids = [PlayerID(0), PlayerID(1), PlayerID(2), PlayerID(3)]
    return GameState(
        config=GameConfig(player_ids=pids, seed=0),
        topology=topo,
        occupancy=BoardOccupancy(robber_tile=TileID(0)),
        players={p: PlayerState(player_id=p) for p in pids},
        bank=Bank(),
        dev_deck=DevelopmentDeck(cards=standard_dev_deck_composition()),
        current_player=PlayerID(0),
        phase=TurnPhase.MAIN,
        turn_number=1,
    )


def _other_v(topo, eid: EdgeID, at: VertexID) -> VertexID:
    v1, v2 = topo.edges[eid].vertices
    if at == v1:
        return v2
    if at == v2:
        return v1
    raise ValueError(f"vertex {int(at)} not incident to edge {int(eid)}")


def _path_of_length(
    s: GameState, start: EdgeID, length: int, pid: PlayerID
) -> list[EdgeID] | None:
    """DFS: first simple path of ``length`` edges from ``start``."""
    topo = s.topology
    found: list[EdgeID] | None = None

    def walk(cur_v: VertexID, visited: frozenset[EdgeID], chain: list[EdgeID]) -> None:
        nonlocal found
        if found is not None:
            return
        if len(chain) == length:
            found = list(chain)
            return
        b = s.occupancy.buildings.get(cur_v)
        if b is not None and b[0] != pid:
            return
        for nxt in topo.vertices[cur_v].adjacent_edges:
            if nxt in visited:
                continue
            walk(_other_v(topo, nxt, cur_v), visited | {nxt}, chain + [nxt])

    v1, v2 = topo.edges[start].vertices
    walk(v1, frozenset({start}), [start])
    walk(v2, frozenset({start}), [start])
    return found


def _find_any_path(s: GameState, length: int, pid: PlayerID) -> list[EdgeID] | None:
    for e0 in s.topology.edges:
        p = _path_of_length(s, e0, length, pid)
        if p is not None:
            return p
    return None


def test_linear_road_of_five_counts_as_five() -> None:
    s = _minimal_state()
    pid = PlayerID(0)
    p = _find_any_path(s, 5, pid)
    assert p is not None, "no 5-edge path on standard map"
    for e in p:
        s.occupancy.roads[e] = pid
    assert lr.compute_longest_road(s, pid) == 5


def test_two_roads_from_one_intersection_forms_path_of_at_most_two_edges() -> None:
    """A fork with two incident roads cannot score more than two connected road edges on that site."""
    s = _minimal_state()
    pid = PlayerID(0)
    e0 = min(s.topology.edges.keys())
    n1 = min(s.topology.edges[e0].adjacent_edges)
    s.occupancy.roads[e0] = pid
    s.occupancy.roads[n1] = pid
    assert lr.compute_longest_road(s, pid) == 2


def test_three_spokes_at_one_vertex_counts_as_two_not_three() -> None:
    """A 3-spoke intersection (T/Y shape) has longest simple road length 2."""
    s = _minimal_state()
    pid = PlayerID(0)

    spoke_edges: list[EdgeID] | None = None
    for _, vertex in sorted(s.topology.vertices.items(), key=lambda item: int(item[0])):
        if len(vertex.adjacent_edges) >= 3:
            spoke_edges = sorted(vertex.adjacent_edges, key=int)[:3]
            break

    assert spoke_edges is not None, "expected at least one interior degree-3 vertex"
    for eid in spoke_edges:
        s.occupancy.roads[eid] = pid

    # In a T/Y shape, any simple path can take at most two spokes.
    assert lr.compute_longest_road(s, pid) == 2


def test_opponent_settlement_on_path_vertex_reduces_computed_length() -> None:
    s = _minimal_state()
    p0, p1 = PlayerID(0), PlayerID(1)
    p = _find_any_path(s, 5, p0)
    assert p is not None
    for e in p:
        s.occupancy.roads[e] = p0
    assert lr.compute_longest_road(s, p0) == 5
    a, b = s.topology.edges[p[0]].vertices
    c, d = s.topology.edges[p[1]].vertices
    if a in (c, d):
        internal = a
    else:
        internal = b
    assert internal in s.topology.edges[p[0]].vertices and internal in s.topology.edges[
        p[1]
    ].vertices
    s.occupancy.buildings[internal] = (p1, BuildingType.SETTLEMENT)
    assert lr.compute_longest_road(s, p0) < 5


def test_award_transfers_to_player_with_strictly_greater_road_length() -> None:
    s = _minimal_state()
    p0, p1 = PlayerID(0), PlayerID(1)
    path4 = _find_any_path(s, 4, p0)
    assert path4 is not None
    for e in path4:
        s.occupancy.roads[e] = p0
    lr.update_longest_road_award(s)
    assert s.longest_road_holder is None
    s2 = _minimal_state()
    p6 = _find_any_path(s2, 6, p1)
    assert p6 is not None
    for e in p6:
        s2.occupancy.roads[e] = p1
    lr.update_longest_road_award(s2)
    assert s2.longest_road_holder == p1
