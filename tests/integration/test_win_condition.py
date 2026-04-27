"""Victory: immediate end at 10 VP, hidden dev VP, and special-award points."""

from __future__ import annotations

import copy

import pytest

from domain.actions import all_actions as A
from domain.engine.randomizer import SeededRandomizer
from domain.enums import DevCardType, Resource, TurnPhase
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.rules import victory
from domain.rules import transitions
from tests.fixtures.states import near_win_state, post_setup_state


def test_game_ends_with_winner_on_reaching_ten_victory_points() -> None:
    s0 = near_win_state(PlayerID(0), points=9, seed=0)
    assert victory.compute_victory_points(s0, PlayerID(0)) == 9
    s0 = copy.deepcopy(s0)
    s0.phase = TurnPhase.MAIN
    s0.pending = None
    pid = PlayerID(0)
    p = s0.players[pid]
    p.resources[Resource.ORE] = 3
    p.resources[Resource.WHEAT] = 2
    vid, _b = min(s0.occupancy.buildings.items())
    s1 = transitions.apply(
        SeededRandomizer(0), s0, A.BuildCityAction(player_id=pid, vertex_id=vid)
    ).state
    assert s1.winner == pid
    assert s1.phase is TurnPhase.GAME_OVER
    assert s1.is_terminal()


def test_victory_point_dev_cards_in_hand_add_to_stated_victory_total() -> None:
    s = post_setup_state(0, 4)
    s = copy.deepcopy(s)
    s.current_player = PlayerID(0)
    p0 = s.players[PlayerID(0)]
    p0.settlements_built = 0
    p0.cities_built = 0
    p0.dev_cards_in_hand = [(DevCardType.VICTORY_POINT, 0)] * 5
    assert victory.compute_victory_points(s, PlayerID(0)) == 5


def test_largest_army_and_longest_road_add_two_each_to_victory() -> None:
    s: GameState = post_setup_state(0, 4)
    s = copy.deepcopy(s)
    s.current_player = PlayerID(0)
    s.largest_army_holder = PlayerID(0)
    s.longest_road_holder = PlayerID(0)
    p0 = s.players[PlayerID(0)]
    p0.settlements_built = 0
    p0.cities_built = 0
    assert victory.compute_victory_points(s, PlayerID(0)) == 4
