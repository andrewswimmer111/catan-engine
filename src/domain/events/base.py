"""
Observable facts emitted when the game state advances.

Events are the output side of a transition: pure data records suitable for
logging, UI updates, and replay. They are not commands and carry no
player-intent beyond what already happened in the last step.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GameEvent(Protocol):
    """
    Every event is stamped with the turn in which it occurred.

    The engine may emit multiple events per transition; the turn number
    allows consumers to order and filter a stream without re-deriving it from
    the game state.
    """

    turn_number: int
