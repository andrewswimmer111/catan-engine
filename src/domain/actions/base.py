"""
Player-intent data submitted to the game engine.

Every action carries the :attr:`player_id` of the seat issuing it. The
rule engine and legality check layer accept these as inputs; they contain no
behavior—only the payload needed to describe what a player asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import PlayerID


@dataclass(frozen=True)
class Action:
    """Base action: every concrete action is tagged with the acting player."""

    player_id: PlayerID
