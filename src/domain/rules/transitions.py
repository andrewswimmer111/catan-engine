"""
State transitions: the only place :class:`GameState` mutates for a single action.

The facade calls :func:`apply` only after the action is confirmed legal. This
module owns event emission and StepResult construction.
"""

from __future__ import annotations

import copy

from domain.actions.base import Action
from domain.engine.randomizer import Randomizer
from domain.engine.step_result import StepResult
from domain.events.base import GameEvent
from domain.game.state import GameState


def apply(rng: Randomizer, state: GameState, action: Action) -> StepResult:
    """
    Apply a *verified-legal* action and return the resulting snapshot.

    Until full rules land, this returns a deep-copied state with no events and
    no rule-side mutations (a no-op placeholder).
    """
    _ = rng  # reserved for rules that need randomness
    new_state = copy.deepcopy(state)
    events: list[GameEvent] = []
    return StepResult(
        state=new_state,
        events=events,
        is_terminal=new_state.is_terminal(),
        winner=new_state.winner,
        action=action,
    )
