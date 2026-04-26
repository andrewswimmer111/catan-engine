"""
One atomic step of the game loop: new state, emitted events, and replay metadata.

``transitions`` produces this; the facade returns it from :meth:`GameEngine.apply_action`.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.actions.base import Action
from domain.events.base import GameEvent
from domain.game.state import GameState
from domain.ids import PlayerID


@dataclass(frozen=True)
class StepResult:
    """Outcome of applying a single legal action."""

    state: GameState
    events: list[GameEvent]
    is_terminal: bool
    winner: PlayerID | None
    action: Action
