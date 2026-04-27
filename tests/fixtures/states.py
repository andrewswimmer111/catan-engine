"""Constructors for :class:`GameState` used across unit and integration tests."""

from __future__ import annotations

import copy

import pytest

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import BuildingType, DevCardType, TurnPhase
from domain.game.config import GameConfig
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, VertexID
from domain.rules import longest_road as lr
from domain.rules import victory


def _shared_v(top, e1: EdgeID, e2: EdgeID) -> VertexID | None:
    a1, a2 = top.edges[e1].vertices
    b1, b2 = top.edges[e2].vertices
    for x in (a1, a2):
        if x == b1 or x == b2:
            return x
    return None


def _path_of_len(
    s: GameState, start: EdgeID, length: int, pid: PlayerID
) -> list[EdgeID] | None:
    top = s.topology
    found: list[EdgeID] | None = None

    def walk(
        cur: EdgeID, visited: frozenset[EdgeID], chain: list[EdgeID]
    ) -> None:
        nonlocal found
        if found is not None:
            return
        if len(chain) == length:
            found = list(chain)
            return
        e = top.edges[cur]
        for nxt in e.adjacent_edges:
            if nxt in visited:
                continue
            v = _shared_v(top, cur, nxt)
            if v is None:
                continue
            b = s.occupancy.buildings.get(v)
            if b is not None and b[0] != pid:
                continue
            walk(nxt, visited | {nxt}, chain + [nxt])

    walk(start, frozenset({start}), [start])
    return found


def _find_road_path(s: GameState, length: int, pid: PlayerID) -> list[EdgeID] | None:
    for e0 in s.topology.edges:
        p = _path_of_len(s, e0, length, pid)
        if p is not None:
            return p
    return None


def _settlement_sort_key(a: A.PlaceSettlementAction) -> int:
    return a.vertex_id


def _road_sort_key(a: A.PlaceRoadAction) -> int:
    return a.edge_id


def _first_setup_action(acts: list[Action]) -> Action:
    """Pick a fixed legal action for deterministic setup (min vertex, then min edge)."""
    settlements = [a for a in acts if isinstance(a, A.PlaceSettlementAction)]
    if settlements:
        return min(settlements, key=_settlement_sort_key)
    roads = [a for a in acts if isinstance(a, A.PlaceRoadAction)]
    if roads:
        return min(roads, key=_road_sort_key)
    raise AssertionError(f"expected setup action, got {acts!r}")


def _complete_initial_setup(state: GameState, engine: GameEngine) -> GameState:
    s = state
    while s.phase in (TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD):
        acts = engine.legal_actions(s)
        a = _first_setup_action(acts)
        s = engine.apply_action(s, a).state
    assert s.phase is TurnPhase.ROLL, "setup should end in ROLL"
    return s


def fresh_game_state(
    n_players: int = 4, seed: int = 0, board_variant: str = "standard"
) -> GameState:
    if n_players not in (3, 4):
        raise ValueError("fixtures only support 3 or 4 players (engine sprint scope)")
    pids = [PlayerID(i) for i in range(n_players)]
    cfg = GameConfig(player_ids=pids, seed=seed, board_variant=board_variant)
    engine = GameEngine(SeededRandomizer(seed))
    return engine.new_game(cfg)


def post_setup_state(seed: int = 0, n_players: int = 4) -> GameState:
    """Initial settlement and road phase finished; the next action is a dice roll."""
    s0 = fresh_game_state(n_players=n_players, seed=seed)
    engine = GameEngine(SeededRandomizer(seed))
    return _complete_initial_setup(s0, engine)


def near_win_state(
    player_id: PlayerID = PlayerID(0), points: int = 9, seed: int = 0
) -> GameState:
    """
    A main-phase state where ``player_id`` has ``points`` total VP
    (including hidden VP and special awards), one action away from 10 in typical builds.
    """
    s = post_setup_state(seed=seed, n_players=4)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.current_player = player_id
    s.turn_number = 2
    p = s.players[player_id]
    p.resources.clear()
    p.settlements_built = 0
    p.cities_built = 0
    p.victory_points_public = 0
    p.dev_cards_in_hand = []
    p.knights_played = 0
    s.occupancy.roads = {}
    s.occupancy.buildings = {}
    s.longest_road_holder = None
    s.largest_army_holder = None
    s.winner = None

    if points == 9:
        p.settlements_built = 1
        p.victory_points_public = 1
        vid = min(s.topology.vertices.keys())
        s.occupancy.buildings[vid] = (player_id, BuildingType.SETTLEMENT)
        p.knights_played = 3
        s.largest_army_holder = player_id
        path5 = _find_road_path(s, 5, player_id)
        if path5 is None:
            raise RuntimeError("no 5-edge road path on standard map (fixture error)")
        for e in path5:
            s.occupancy.roads[e] = player_id
        p.roads_built = 5
        lr.update_longest_road_award(s)
        if s.longest_road_holder != player_id:
            raise RuntimeError("expected longest road to remain after 5 road segments")
        for _ in range(4):
            p.dev_cards_in_hand.append((DevCardType.VICTORY_POINT, 0))
    else:
        raise NotImplementedError(f"near_win_state only implements points=9, got {points!r}")
    assert victory.compute_victory_points(s, player_id) == points
    return s


@pytest.fixture
def post_setup() -> GameState:
    return post_setup_state()


@pytest.fixture
def four_player_fresh() -> GameState:
    return fresh_game_state(4, seed=0)
