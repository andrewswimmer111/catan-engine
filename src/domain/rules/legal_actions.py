"""
Legal move generation. The :class:`GameEngine` calls :func:`legal_actions` as
its only way to list valid inputs for the current state.
"""

from __future__ import annotations

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.rules import setup_rules


def legal_actions(state: GameState) -> list[Action]:
    """Return every :class:`Action` that is legal in ``state`` for implemented phases."""
    p = state.phase
    if p is TurnPhase.INITIAL_SETTLEMENT:
        return list(setup_rules.legal_setup_settlements(state))
    if p is TurnPhase.INITIAL_ROAD:
        return list(setup_rules.legal_setup_roads(state))
    if p is TurnPhase.ROLL:
        return [A.RollDiceAction(player_id=state.current_player)]
    return []
