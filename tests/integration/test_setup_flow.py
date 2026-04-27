"""End-to-end initial placement and setup completion."""

from __future__ import annotations

import pytest

from domain.actions import all_actions as A
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.rules import setup_rules
from tests.fixtures.states import (
    _complete_initial_setup,
    _first_setup_action,
    fresh_game_state,
    post_setup_state,
)


def test_four_player_snake_draft_ends_in_roll_with_turn_counter_one() -> None:
    s = _complete_initial_setup(
        fresh_game_state(4, seed=3), GameEngine(SeededRandomizer(3))
    )
    assert s.phase is TurnPhase.ROLL
    assert s.turn_number == 1
    assert s.current_player == s.setup_order[0]


def test_three_player_snake_draft_completes_to_roll() -> None:
    s = _complete_initial_setup(
        fresh_game_state(3, seed=2), GameEngine(SeededRandomizer(2))
    )
    assert s.phase is TurnPhase.ROLL
    assert s.turn_number == 1
    assert len(s.setup_order) == 6


def test_last_setup_road_triggers_transition_to_roll() -> None:
    s0 = post_setup_state(seed=0)
    assert s0.phase is TurnPhase.ROLL


def test_second_settlement_grants_resource_cards_matching_adjacent_tiles() -> None:
    """First second-round settlement (snake) grants cards per adjacent production hex."""
    s = fresh_game_state(4, seed=0)
    eng = GameEngine(SeededRandomizer(0))
    for _ in range(64):
        if s.phase is TurnPhase.INITIAL_SETTLEMENT and setup_rules.is_second_settlement_turn(
            s
        ):
            a = _first_setup_action(eng.legal_actions(s))
            assert isinstance(a, A.PlaceSettlementAction)
            before = s.players[a.player_id].resource_count()
            want = sum(setup_rules.second_settlement_resources(s, a.vertex_id).values())
            s = eng.apply_action(s, a).state
            after = s.players[a.player_id].resource_count()
            assert after - before == want
            return
        a = _first_setup_action(eng.legal_actions(s))
        s = eng.apply_action(s, a).state
    pytest.fail("second settlement not reached in bounded steps")
