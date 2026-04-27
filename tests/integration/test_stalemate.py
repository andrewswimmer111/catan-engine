"""Stalemate and VP-stall termination."""

from __future__ import annotations

import copy

import domain.events.all_events as E
from domain.actions import all_actions as A
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import BuildingType, DevCardType, EndReason, Resource, TurnPhase
from domain.board.layout import build_standard_board
from domain.board.occupancy import BoardOccupancy
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck, standard_dev_deck_composition
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import PlayerID, TileID
from domain.rules import robber_rules, transitions, victory
from tests.fixtures.states import post_setup_state
from tests.unit.test_longest_road import _find_any_path, _minimal_state


def test_building_cap_and_empty_deck_emits_stalemate_no_progress() -> None:
    s = post_setup_state(0, 4)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.dev_deck.cards.clear()
    for pid in s.config.player_ids:
        p = s.players[pid]
        p.settlements_built = 1
        p.cities_built = 4
    s.current_player = PlayerID(0)
    s.turn_number = 3
    out = transitions.apply(
        SeededRandomizer(0), s, A.EndTurnAction(player_id=PlayerID(0))
    )
    assert out.state.phase is TurnPhase.STALEMATE
    assert out.state.end_reason is EndReason.STALEMATE_NO_PROGRESS
    assert any(
        isinstance(e, E.GameStalled)
        and e.reason is EndReason.STALEMATE_NO_PROGRESS
        for e in out.events
    )


def test_vp_stall_end_after_threshold_turns() -> None:
    s = post_setup_state(0, 4)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turns_since_vp_change = transitions.VP_STALL_TURN_THRESHOLD
    s.current_player = PlayerID(0)
    out = transitions.apply(
        SeededRandomizer(0), s, A.EndTurnAction(player_id=PlayerID(0))
    )
    assert out.state.phase is TurnPhase.STALEMATE
    assert out.state.end_reason is EndReason.STALEMATE_VP_STALL


def test_vp_stall_resets_when_longest_road_award_changes_vp() -> None:
    s = _minimal_state()
    pid0 = PlayerID(0)
    chain = _find_any_path(s, 4, pid0)
    assert chain is not None
    e0 = chain[0]
    v0, _v1 = s.topology.edges[e0].vertices
    s.occupancy.buildings[v0] = (pid0, BuildingType.SETTLEMENT)
    for e in chain:
        s.occupancy.roads[e] = pid0
    s.players[pid0].settlements_built = 1
    s.players[pid0].roads_built = 4
    s.players[pid0].resources.update({Resource.WOOD: 2, Resource.BRICK: 2})
    s.turns_since_vp_change = 50
    s.current_player = pid0
    from domain.rules import build_rules

    road_acts = build_rules.legal_build_roads(s)
    assert road_acts, "expected a fifth road placement for longest-road VP"
    out = transitions.apply(SeededRandomizer(0), s, road_acts[0])
    assert out.state.players[pid0].roads_built == 5
    assert out.state.longest_road_holder == pid0
    assert out.state.turns_since_vp_change == 0


def test_vp_stall_resets_when_largest_army_award_changes_vp() -> None:
    topo = build_standard_board()
    pids = [PlayerID(i) for i in range(4)]
    s = GameState(
        config=GameConfig(player_ids=pids, seed=1),
        topology=topo,
        occupancy=BoardOccupancy(robber_tile=TileID(0)),
        players={pid: PlayerState(player_id=pid) for pid in pids},
        bank=Bank(),
        dev_deck=DevelopmentDeck(cards=standard_dev_deck_composition()),
        current_player=pids[0],
        phase=TurnPhase.MAIN,
        turn_number=1,
    )
    s.turns_since_vp_change = 44
    p0 = s.players[PlayerID(0)]
    p1 = s.players[PlayerID(1)]
    p0.dev_cards_in_hand = [(DevCardType.KNIGHT, 0)]
    p0.knights_played = 3
    p1.knights_played = 3
    s.largest_army_holder = PlayerID(1)

    rng = SeededRandomizer(1)
    out_k = transitions.apply(
        rng, copy.deepcopy(s), A.PlayKnightAction(player_id=PlayerID(0))
    )
    assert out_k.state.phase is TurnPhase.MOVE_ROBBER
    out_m = None
    for mva in robber_rules.legal_robber_moves(out_k.state):
        out_try = transitions.apply(rng, copy.deepcopy(out_k.state), mva)
        if out_try.state.phase is TurnPhase.MAIN and out_try.state.pending is None:
            out_m = out_try
            break
    assert out_m is not None
    assert out_m.state.largest_army_holder == PlayerID(0)
    assert victory.compute_victory_points(out_m.state, PlayerID(0)) >= 2
    assert out_m.state.turns_since_vp_change == 0


def test_hidden_vp_total_still_wins_on_roll_post_action() -> None:
    s = post_setup_state(0, 4)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.ROLL
    s.pending = None
    s.current_player = PlayerID(0)
    p0 = s.players[PlayerID(0)]
    p0.settlements_built = 0
    p0.cities_built = 0
    p0.dev_cards_in_hand = [(DevCardType.VICTORY_POINT, 0)] * 10
    s.longest_road_holder = None
    s.largest_army_holder = None
    p0.knights_played = 0
    assert victory.compute_victory_points(s, PlayerID(0)) == 10
    out = None
    for seed in range(500):
        out = transitions.apply(
            SeededRandomizer(seed), s, A.RollDiceAction(player_id=PlayerID(0))
        )
        if out.state.winner == PlayerID(0):
            break
    else:
        raise AssertionError("expected a dice seed that avoids robber-seven for this fixture")
    assert out is not None
    assert out.state.winner == PlayerID(0)
    assert out.state.end_reason is EndReason.WINNER
    assert out.state.phase is TurnPhase.GAME_OVER


def test_setup_road_deadlock_resolves_via_engine() -> None:
    s = _minimal_state()
    s.phase = TurnPhase.INITIAL_ROAD
    s.setup_order = list(s.config.player_ids) * 2
    s.setup_index = 0
    s.turn_number = 0
    pid = PlayerID(0)
    s.current_player = pid
    topo = s.topology
    edge_ids = list(topo.edges.keys())
    e_first = edge_ids[0]
    v_a, _v_b = topo.edges[e_first].vertices
    s.occupancy.buildings[v_a] = (pid, BuildingType.SETTLEMENT)
    s.last_settlement_vertex = v_a
    for i, e in enumerate(topo.edges_adjacent_to_vertex(v_a)):
        owner = PlayerID((i % 3) + 1)
        s.occupancy.roads[e.edge_id] = owner
        s.players[owner].roads_built += 1
    eng = GameEngine(SeededRandomizer(0))
    from domain.rules.legal_actions import legal_actions

    assert not legal_actions(s)
    s2 = eng.resolve_if_no_legal_actions(s)
    assert s2.is_terminal()
    assert s2.phase is TurnPhase.STALEMATE
    assert s2.end_reason is EndReason.STALEMATE_NO_PROGRESS
