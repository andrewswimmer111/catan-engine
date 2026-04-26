"""
Static game parameters fixed at construction time (player list, RNG seed, board
variant) plus sprint-scoped validation of player count.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import PlayerID

SPRINT1_PLAYER_COUNTS: frozenset[int] = frozenset((3, 4))


@dataclass(frozen=True)
class GameConfig:
    """
    Immutable parameters chosen before a game instance exists.

    For sprint 1 only three- and four-player games are valid.
    """

    player_ids: list[PlayerID]
    seed: int
    board_variant: str = "standard"

    def __post_init__(self) -> None:
        n = len(self.player_ids)
        if n not in SPRINT1_PLAYER_COUNTS:
            raise ValueError(f"sprint 1 only supports 3 or 4 players; got {n}")
