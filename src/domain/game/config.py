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

    ``victory_point_target`` is the VP count required to win (default
    ``10`` — standard Catan rules). Lower values are supported as a
    training curriculum knob: with the AZ chicken-and-egg failure
    mode where no agent can reach 10 VP in a reasonable game length,
    starting curriculum training at, say, ``6`` lets the value head
    actually observe terminal wins, bootstrap the policy iteration,
    then graduate to higher targets. The threshold is read at every
    on-turn check (see :func:`domain.rules.victory.check_winner`); a
    mid-game change is not supported.
    """

    player_ids: list[PlayerID]
    seed: int
    board_variant: str = "standard"
    victory_point_target: int = 10

    def __post_init__(self) -> None:
        n = len(self.player_ids)
        if n not in SPRINT1_PLAYER_COUNTS:
            raise ValueError(f"sprint 1 only supports 3 or 4 players; got {n}")
        if self.victory_point_target < 2:
            # A 2-seat-settlement initial-placement already yields 2 VP,
            # so any target below 3 would award the win at the first
            # check. 2 is the lowest non-trivial value; ban anything
            # below it to surface programmer errors.
            raise ValueError(
                f"victory_point_target must be >= 2; got {self.victory_point_target}"
            )
