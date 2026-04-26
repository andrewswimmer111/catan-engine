"""
Turn phase helpers. The canonical enum is :class:`domain.enums.TurnPhase` — re-exported here
so callers may import either location consistently.
"""

from __future__ import annotations

from domain.enums import TurnPhase

__all__ = [
    "TurnPhase",
    "is_setup_phase",
    "requires_active_player_only",
]


def is_setup_phase(phase: TurnPhase) -> bool:
    return phase in (TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD)


def requires_active_player_only(phase: TurnPhase) -> bool:
    """
    If False, the rules layer may return legal actions for multiple seats
    (e.g. DISCARD: every player with too many cards).
    """
    return phase != TurnPhase.DISCARD
