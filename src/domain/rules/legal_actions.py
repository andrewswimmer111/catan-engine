"""
Legal move generation. The :class:`GameEngine` calls :func:`legal_actions` as
its only way to list valid inputs for the current state.
"""

from __future__ import annotations

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.rules import build_rules, dev_card_rules, setup_rules, trade_rules
from domain.turn.pending import DomesticTradePending


def legal_actions(state: GameState) -> list[Action]:
    """Return every :class:`Action` that is legal in ``state`` for implemented phases."""
    p = state.phase
    if p is TurnPhase.INITIAL_SETTLEMENT:
        return list(setup_rules.legal_setup_settlements(state))
    if p is TurnPhase.INITIAL_ROAD:
        return list(setup_rules.legal_setup_roads(state))
    if isinstance(state.pending, DomesticTradePending):
        return list(trade_rules.legal_domestic_turn(state))
    if p is TurnPhase.BUILD_ROADS:
        return list(build_rules.legal_build_roads(state))
    if p is TurnPhase.ROLL:
        return _roll_legals(state)
    if p is TurnPhase.MAIN:
        return _main_legals(state)
    return []


def _roll_legals(state: GameState) -> list[Action]:
    out: list[Action] = [A.RollDiceAction(player_id=state.current_player)]
    for a in dev_card_rules.legal_dev_card_plays(state):
        out.append(a)
    return out


def _main_legals(state: GameState) -> list[Action]:
    if state.pending is not None:
        return []
    out: list[Action] = []
    out += build_rules.legal_build_roads(state)
    out += build_rules.legal_build_settlements(state)
    out += build_rules.legal_build_cities(state)
    out += build_rules.legal_buy_dev_card(state)
    out += trade_rules.legal_maritime_trades(state)
    out += trade_rules.legal_propose_domestic_1_1(state)
    out += build_rules.legal_end_turn(state)
    out += dev_card_rules.legal_dev_card_plays(state)
    return out
