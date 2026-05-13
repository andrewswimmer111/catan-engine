from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from domain.enums import TurnPhase
from domain.ids import PlayerID

if TYPE_CHECKING:
    from domain.actions.base import Action
    from domain.events.base import GameEvent

# The encoded observation type returned by CatanEnv
Observation = np.ndarray

__all__ = ["Observation", "Info"]


class Info(TypedDict):
    """Step metadata returned alongside each observation."""

    current_agent: PlayerID
    action_mask: np.ndarray
    legal_actions: list[Action]
    last_events: list[GameEvent]
    phase: TurnPhase
