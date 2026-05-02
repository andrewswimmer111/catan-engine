from __future__ import annotations

from dataclasses import dataclass

from domain.enums import EndReason
from domain.ids import PlayerID

__all__ = ["GameStats", "TournamentResult"]


@dataclass(frozen=True)
class GameStats:
    winner: PlayerID | None
    final_vps: dict[PlayerID, int]
    turn_count: int
    end_reason: EndReason
    action_histogram: dict[str, int]


@dataclass(frozen=True)
class TournamentResult:
    games: list[GameStats]
    win_rates: dict[PlayerID, float]
    mean_vp: dict[PlayerID, float]
    mean_turns: float
