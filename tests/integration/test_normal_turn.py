"""Full-turn flows after initial setup (roll, main, dev restrictions)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from domain.actions import all_actions as A
from domain.engine.randomizer import SeededRandomizer
from domain.enums import DevCardType, TurnPhase
from domain.rules import transitions
from domain.rules.legal_actions import legal_actions
from tests.fixtures.states import post_setup_state


@dataclass
class _Dice:
    a: int
    b: int
    d1: int = 0
    d2: int = 0

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


def test_roll_distribute_and_end_advances_to_next_seat() -> None:
    s0 = post_setup_state(seed=0)
    s0 = copy.deepcopy(s0)
    assert s0.phase is TurnPhase.ROLL
    pid = s0.current_player
    r = _Dice(2, 2)
    s1 = transitions.apply(r, s0, A.RollDiceAction(player_id=pid)).state
    assert s1.phase is TurnPhase.MAIN
    s2 = transitions.apply(SeededRandomizer(0), s1, A.EndTurnAction(player_id=pid)).state
    nxt = s2.config.player_ids[(s2.config.player_ids.index(pid) + 1) % 4]
    assert s2.current_player == nxt
    assert s2.phase is TurnPhase.ROLL


def test_dev_card_bought_this_same_turn_cannot_yet_be_played() -> None:
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 1
    pid = s.current_player
    p = s.players[pid]
    p.dev_cards_in_hand = [(DevCardType.KNIGHT, s.turn_number)]
    assert [] == [
        a
        for a in legal_actions(s)
        if isinstance(
            a,
            (A.PlayKnightAction, A.PlayRoadBuildingAction, A.PlayMonopolyAction),
        )
    ]


def test_at_most_one_non_knight_dev_played_per_main_turn() -> None:
    s = post_setup_state(seed=0)
    s = copy.deepcopy(s)
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    pid = s.current_player
    p = s.players[pid]
    p.dev_cards_in_hand = [
        (DevCardType.KNIGHT, 0),
        (DevCardType.KNIGHT, 0),
    ]
    p.has_played_dev_card_this_turn = True
    assert A.PlayKnightAction not in {type(x) for x in legal_actions(s)}
