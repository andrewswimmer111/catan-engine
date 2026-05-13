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
    # Per-seat counts keyed by acting PlayerID. Lets downstream callers
    # (e.g. the rl-020 eval CLI) compute per-policy action diffs when seats
    # are split into "learner role" vs "opponent role". The unkeyed
    # ``action_histogram`` above remains the aggregate sum for backwards
    # compat with existing benchmark tests.
    per_seat_action_histogram: dict[PlayerID, dict[str, int]]


@dataclass(frozen=True)
class TournamentResult:
    games: list[GameStats]
    win_rates: dict[PlayerID, float]
    mean_vp: dict[PlayerID, float]
    mean_turns: float
