"""Regression tests for bugs fixed during refactor."""

from __future__ import annotations

import copy

from domain.actions import all_actions as A
from domain.engine.randomizer import SeededRandomizer
from domain.enums import BuildingType, DevCardType, EndReason, TurnPhase
from domain.rules import build_rules, transitions, victory
from tests.fixtures.states import post_setup_state


def test_paid_settlement_increments_victory_points_public() -> None:
    """Bug: ``victory_points_public`` was only updated for *initial* settlements."""
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    pid = s.current_player

    # Wipe buildings and roads so we have a clean slate for a paid placement.
    s.occupancy.buildings = {}
    s.occupancy.roads = {}
    for p in s.players.values():
        p.settlements_built = 0
        p.cities_built = 0
        p.roads_built = 0
        p.victory_points_public = 0

    # Place a road for `pid` and target one of its endpoints as the new site.
    eid = min(s.topology.edges.keys())
    target_vid, _ = s.topology.edges[eid].vertices
    s.occupancy.roads[eid] = pid
    s.players[pid].roads_built = 1

    p = s.players[pid]
    before_vp = p.victory_points_public
    p.resources = dict(build_rules.SETTLEMENT_COST)
    s2 = transitions.apply(
        SeededRandomizer(0),
        s,
        A.BuildSettlementAction(player_id=pid, vertex_id=target_vid),
    ).state
    assert s2.players[pid].victory_points_public == before_vp + 1
    assert s2.players[pid].settlements_built == 1
    assert target_vid in s2.occupancy.buildings


def test_paid_city_increments_victory_points_public() -> None:
    """A settlement upgraded to a city is +1 VP (1 → 2)."""
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    pid = s.current_player
    p = s.players[pid]
    p.resources = dict(build_rules.CITY_COST)

    target_vid = next(
        v for v, (own, bt) in s.occupancy.buildings.items()
        if own == pid and bt is BuildingType.SETTLEMENT
    )
    before_vp = p.victory_points_public
    s2 = transitions.apply(
        SeededRandomizer(0),
        s,
        A.BuildCityAction(player_id=pid, vertex_id=target_vid),
    ).state
    assert s2.players[pid].victory_points_public == before_vp + 1


def test_check_winner_only_considers_current_player() -> None:
    """A non-current player at 10 VP should not win during another's turn."""
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 5

    other = next(
        pid for pid in s.config.player_ids if pid != s.current_player
    )
    s.players[other].dev_cards_in_hand = [(DevCardType.VICTORY_POINT, 0)] * 10
    # The non-current player has 10+ VP but cannot win on someone else's turn.
    assert victory.check_winner(s) is None

    s.current_player = other
    assert victory.check_winner(s) == other


def test_road_building_card_with_no_legal_placements_returns_to_main() -> None:
    """Bug: BUILD_ROADS phase with no legal moves left previously deadlocked."""
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 5
    pid = s.current_player
    p = s.players[pid]
    p.dev_cards_in_hand = [(DevCardType.ROAD_BUILDING, 0)]
    # Pretend the player has placed 15 roads — all road tokens used.
    p.roads_built = build_rules.MAX_ROADS

    out = transitions.apply(
        SeededRandomizer(0),
        s,
        A.PlayRoadBuildingAction(player_id=pid),
    )
    assert out.state.phase is TurnPhase.MAIN
    assert out.state.pending is None


def test_knight_setting_largest_army_then_winning_keeps_game_over() -> None:
    """If the knight grants Largest Army and pushes to 10 VP, the game ends and stays ended."""
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 8
    pid = s.current_player
    p = s.players[pid]
    # Set the player up at 8 public VP with 2 knights already played, holding a
    # third knight in hand. Playing it puts knights_played=3 -> Largest Army (+2 VP)
    # and pushes them to >= 10 VP.
    p.victory_points_public = 8
    p.settlements_built = 2  # already counted toward VP
    p.cities_built = 3
    p.knights_played = 2
    p.dev_cards_in_hand = [(DevCardType.KNIGHT, 0)]

    out = transitions.apply(
        SeededRandomizer(0),
        s,
        A.PlayKnightAction(player_id=pid),
    )
    assert out.state.is_terminal()
    assert out.state.winner == pid
    assert out.state.end_reason is EndReason.WINNER
    # GAME_OVER must persist — not be overwritten by MOVE_ROBBER.
    assert out.state.phase is TurnPhase.GAME_OVER
