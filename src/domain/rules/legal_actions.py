"""
Legal move generation. The :class:`GameEngine` calls :func:`legal_actions` as
its only way to list valid inputs for the current state.

Each turn phase is handled by a small generator function; phases are dispatched
through :data:`_PHASE_HANDLERS` plus a couple of pending-effect overrides that
short-circuit the phase (in-flight domestic trade, discard).
"""

from __future__ import annotations

from typing import Callable

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.rules import build_rules, dev_card_rules, robber_rules, setup_rules, trade_rules
from domain.turn.pending import DiscardPending, DomesticTradePending


def _initial_settlement(state: GameState) -> list[Action]:
    return list(setup_rules.legal_setup_settlements(state))


def _initial_road(state: GameState) -> list[Action]:
    return list(setup_rules.legal_setup_roads(state))


def _discard(state: GameState) -> list[Action]:
    return list(robber_rules.legal_discard_actions(state))


def _move_robber(state: GameState) -> list[Action]:
    return list(robber_rules.legal_robber_moves(state))


def _steal(state: GameState) -> list[Action]:
    return list(robber_rules.legal_steal_actions(state))


def _build_roads_phase(state: GameState) -> list[Action]:
    return list(build_rules.legal_build_roads(state))


def _roll(state: GameState) -> list[Action]:
    out: list[Action] = [A.RollDiceAction(player_id=state.current_player)]
    out.extend(dev_card_rules.legal_dev_card_plays(state))
    return out


def _main(state: GameState) -> list[Action]:
    if state.pending is not None:
        return []
    out: list[Action] = []
    out += build_rules.legal_build_roads(state)
    out += build_rules.legal_build_settlements(state)
    out += build_rules.legal_build_cities(state)
    out += build_rules.legal_buy_dev_card(state)
    out += trade_rules.legal_maritime_trades(state)
    out += trade_rules.legal_propose_domestic_single_resource(state)
    out += build_rules.legal_end_turn(state)
    out += dev_card_rules.legal_dev_card_plays(state)
    return out


_PHASE_HANDLERS: dict[TurnPhase, Callable[[GameState], list[Action]]] = {
    TurnPhase.INITIAL_SETTLEMENT: _initial_settlement,
    TurnPhase.INITIAL_ROAD: _initial_road,
    TurnPhase.DISCARD: _discard,
    TurnPhase.MOVE_ROBBER: _move_robber,
    TurnPhase.STEAL: _steal,
    TurnPhase.BUILD_ROADS: _build_roads_phase,
    TurnPhase.ROLL: _roll,
    TurnPhase.MAIN: _main,
}


def legal_actions(state: GameState) -> list[Action]:
    """Return every :class:`Action` that is legal in ``state`` for implemented phases."""
    # In-flight pending effects can override the nominal phase.
    if isinstance(state.pending, DomesticTradePending):
        return list(trade_rules.legal_domestic_turn(state))
    if state.phase is TurnPhase.DISCARD and isinstance(state.pending, DiscardPending):
        return _discard(state)

    handler = _PHASE_HANDLERS.get(state.phase)
    if handler is None:
        return []
    return handler(state)
