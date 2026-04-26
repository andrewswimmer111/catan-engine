"""
Legal move generation. The :class:`GameEngine` calls :func:`legal_actions` as
its only way to list valid inputs for the current state.

Rule implementations will fill this out; the sprint-5 contract is the signature
and import path, not a complete ruleset.
"""

from __future__ import annotations

from domain.actions.base import Action
from domain.game.state import GameState


def legal_actions(state: GameState) -> list[Action]:
    """Return every action that is legal in ``state`` (empty until rules exist)."""
    return []
