"""Unit tests for :func:`domain.rules.legal_actions.legal_actions`."""

from __future__ import annotations

import copy

from domain.actions import all_actions as A
from domain.enums import BuildingType, Resource, TurnPhase
from domain.ids import PlayerID
from domain.rules.legal_actions import legal_actions
from tests.fixtures.states import fresh_game_state, post_setup_state


def test_initial_settlement_empty_board_all_fifty_four_vertices_legal() -> None:
    s = fresh_game_state(4, seed=1)
    assert s.phase is TurnPhase.INITIAL_SETTLEMENT
    leg = legal_actions(s)
    verts = {a.vertex_id for a in leg if isinstance(a, A.PlaceSettlementAction)}
    assert verts == set(s.topology.vertices.keys())


def test_after_one_settlement_distance_rule_excludes_banned_vertices() -> None:
    s = fresh_game_state(4, seed=0)
    v_place = min(s.topology.vertices.keys())
    s = copy.deepcopy(s)
    s.occupancy.buildings[v_place] = (PlayerID(0), BuildingType.SETTLEMENT)
    s.current_player = PlayerID(1)
    adjacent = s.topology.vertices[v_place].adjacent_vertices
    blocked = {v_place} | set(adjacent)
    leg = legal_actions(s)
    allowed = {a.vertex_id for a in leg if isinstance(a, A.PlaceSettlementAction)}
    assert blocked.isdisjoint(allowed)
    for vid in s.topology.vertices:
        if vid in s.occupancy.buildings:
            continue
        if vid in blocked:
            assert vid not in allowed
        else:
            assert vid in allowed


def test_roll_phase_only_roll_dice_when_no_playable_dev_cards() -> None:
    s = post_setup_state(seed=0)
    assert s.phase is TurnPhase.ROLL
    s = copy.deepcopy(s)
    s.players[s.current_player].dev_cards_in_hand.clear()
    leg = legal_actions(s)
    assert leg == [A.RollDiceAction(player_id=s.current_player)]


def test_main_phase_no_resources_has_no_build_or_buy_dev_actions() -> None:
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    p = s.players[s.current_player]
    p.resources.clear()
    p.dev_cards_in_hand.clear()
    p.has_played_dev_card_this_turn = False
    leg = legal_actions(s)
    types = {type(a) for a in leg}
    assert A.BuildRoadAction not in types
    assert A.BuildSettlementAction not in types
    assert A.BuildCityAction not in types
    assert A.BuyDevCardAction not in types


def test_main_phase_with_road_cost_resources_offers_road_on_adjacent_free_edge() -> None:
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    pid = s.current_player
    p = s.players[pid]
    p.resources = {Resource.WOOD: 1, Resource.BRICK: 1}
    p.roads_built = 0
    eid = min(s.topology.edges.keys())
    v1, _ = s.topology.edges[eid].vertices
    s.occupancy.buildings = {v1: (pid, BuildingType.SETTLEMENT)}
    s.occupancy.roads = {}
    leg = legal_actions(s)
    assert any(
        isinstance(a, A.BuildRoadAction) and a.edge_id == eid for a in leg
    ), f"expected BuildRoad on {eid!r} in {leg!r}"


def test_main_phase_is_not_merged_into_roll_legals() -> None:
    s = post_setup_state(0)
    assert TurnPhase.ROLL is s.phase
    s2 = copy.deepcopy(s)
    s2.phase = TurnPhase.MAIN
    s2.pending = None
    assert {type(a) for a in legal_actions(s)} != {type(a) for a in legal_actions(s2)}
