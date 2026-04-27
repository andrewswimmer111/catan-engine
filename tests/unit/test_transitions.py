"""Unit tests for :mod:`domain.rules.transitions` (apply action effects)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from domain.actions import all_actions as A
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import BuildingType, Resource, TurnPhase
from domain.rules import transitions
from tests.fixtures.states import _complete_initial_setup, fresh_game_state, post_setup_state


@dataclass
class _FixedDice:
    """Test double: fixed die results and trivial RNG ops."""

    a: int
    b: int

    def roll_dice(self) -> tuple[int, int]:
        return (self.a, self.b)

    def shuffle_dev_deck(self, cards: list) -> list:
        return list(cards)

    def choose_stolen_resource(self, resources: list) -> Any:
        return resources[0]

    def shuffled(self, items: list) -> list:
        return list(items)

    def choice(self, items: list) -> Any:
        return items[0]


def test_place_settlement_increments_settlements_built_and_records_building() -> None:
    s0 = fresh_game_state(4, seed=0)
    pid = s0.current_player
    a = A.PlaceSettlementAction(
        player_id=pid, vertex_id=min(s0.topology.vertices.keys())
    )
    s1 = transitions.apply(SeededRandomizer(0), s0, a).state
    assert s1.players[pid].settlements_built == 1
    assert a.vertex_id in s1.occupancy.buildings
    assert s1.occupancy.buildings[a.vertex_id][0] is pid
    assert s1.occupancy.buildings[a.vertex_id][1] is BuildingType.SETTLEMENT


def test_roll_non_seven_distribute_then_enters_main() -> None:
    s0 = fresh_game_state(4, seed=0)
    s0 = _complete_initial_setup(s0, GameEngine(SeededRandomizer(0)))
    s0 = copy.deepcopy(s0)
    assert s0.phase is TurnPhase.ROLL
    r = _FixedDice(2, 3)
    s1 = transitions.apply(r, s0, A.RollDiceAction(player_id=s0.current_player)).state
    assert s1.phase is TurnPhase.MAIN


def test_roll_seven_with_all_hands_seven_or_fewer_enters_move_robber() -> None:
    s0 = fresh_game_state(4, seed=0)
    s0 = _complete_initial_setup(s0, GameEngine(SeededRandomizer(0)))
    r = _FixedDice(3, 4)
    s1 = transitions.apply(r, s0, A.RollDiceAction(player_id=s0.current_player)).state
    assert s1.phase is TurnPhase.MOVE_ROBBER


def test_end_turn_advances_to_next_seat() -> None:
    s0 = post_setup_state(0, 4)
    s0 = copy.deepcopy(s0)
    s0.phase = TurnPhase.MAIN
    s0.pending = None
    s0.turn_number = 1
    cur = s0.current_player
    order = s0.config.player_ids
    expected = order[(order.index(cur) + 1) % len(order)]
    s1 = transitions.apply(
        SeededRandomizer(0),
        s0,
        A.EndTurnAction(player_id=cur),
    ).state
    assert s1.current_player == expected
    assert s1.phase is TurnPhase.ROLL
