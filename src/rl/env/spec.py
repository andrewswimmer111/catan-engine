from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from domain.engine.player_view import PlayerView
from domain.enums import TurnPhase
from domain.ids import PlayerID

if TYPE_CHECKING:
    from domain.actions.base import Action
    from domain.events.base import GameEvent

# The raw observation type returned by CatanEnv (no tensor encoding yet).
Observation = PlayerView

__all__ = ["Observation", "Info"]


class Info(TypedDict):
    """Step metadata returned alongside each observation."""

    current_agent: PlayerID
    legal_actions: list[Action]
    last_events: list[GameEvent]
    current_phase: TurnPhase
