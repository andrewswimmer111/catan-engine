"""Single source of truth for "who must act next" given a :class:`GameState`.

Most phases drive off ``state.current_player``, but during ``DISCARD`` the
engine emits legal actions for every seat that still owes a discard, not the
seat that rolled the 7. Every RL-side dispatcher needs the same rule, so it
lives here once instead of being recopied into the env, the encoder, and
each agent.
"""

from __future__ import annotations

from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending

__all__ = ["acting_player"]


def acting_player(state: GameState) -> PlayerID:
    """Player expected to act next in ``state``.

    Returns the first seat still listed in :class:`DiscardPending` during the
    DISCARD phase (multiple seats may owe a discard after a 7), and
    ``state.current_player`` otherwise.
    """
    if state.phase is TurnPhase.DISCARD and isinstance(state.pending, DiscardPending):
        return next(iter(state.pending.cards_to_discard))
    return state.current_player
