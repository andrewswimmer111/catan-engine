"""Markdown / plain-text report formatting for the rl-020 eval CLI.

The eval CLI runs a "learner vs opponent" tournament rotated through the four
seats, then hands the per-rotation :class:`TournamentResult` objects here.
This module is split into two layers:

* :func:`aggregate_evaluation` — pure aggregation over the per-rotation
  results. Produces an :class:`EvalComparison` summary (win rates by seat,
  mean VPs by role, per-policy action diffs, and three sample-game picks).
* :func:`format_evaluation_report` — turns an :class:`EvalComparison` into a
  one-page Markdown blob suitable for stdout.

Splitting the layers keeps the formatter free of game-playing logic so tests
can build canned comparisons without spinning up the engine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from domain.enums import EndReason
from domain.ids import PlayerID
from rl.evaluation.metrics import GameStats, TournamentResult

__all__ = [
    "EvalComparison",
    "GamePick",
    "RotationResult",
    "aggregate_evaluation",
    "format_evaluation_report",
]


@dataclass(frozen=True)
class RotationResult:
    """One seat-rotation of the eval tournament.

    ``learner_seat`` is the single seat the learner occupied for the games in
    ``result``; every other seat in ``result.win_rates`` was the opponent.
    ``replay_paths`` is parallel to ``result.games`` — entry ``i`` is the
    on-disk replay path for game ``i`` if the CLI archived it, ``None``
    otherwise.
    """

    learner_seat: PlayerID
    result: TournamentResult
    replay_paths: list[Path | None]


@dataclass(frozen=True)
class GamePick:
    """Reference to one notable game in the rotation set."""

    label: str  # "close_win", "blowout", "stalemate"
    learner_seat: PlayerID
    winner: PlayerID | None
    learner_vp: int
    top_opponent_vp: int
    turn_count: int
    end_reason: EndReason
    replay_path: Path | None


@dataclass(frozen=True)
class EvalComparison:
    """Aggregated learner-vs-opponent stats across all rotations."""

    learner_label: str
    opponent_label: str
    n_games: int
    learner_win_rate: float
    opponent_win_rate: float
    stalemate_rate: float
    per_seat_learner_win_rate: dict[PlayerID, float]
    mean_vp_learner: float
    mean_vp_opponent: float
    mean_turns: float
    # Aggregate counts, summed across games where the seat was the learner
    # (learner) or one of the three opponent seats (opponent).
    learner_action_counts: dict[str, int]
    opponent_action_counts: dict[str, int]
    sample_games: list[GamePick]


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def aggregate_evaluation(
    learner_label: str,
    opponent_label: str,
    rotations: list[RotationResult],
) -> EvalComparison:
    """Reduce the per-rotation results into a single :class:`EvalComparison`.

    ``rotations`` should cover every seat the learner occupied; the function
    does no normalisation of game counts across rotations — uneven n_games
    per rotation feed straight through to the win-rate averages.
    """
    if not rotations:
        raise ValueError("aggregate_evaluation requires at least one rotation")

    all_games: list[tuple[PlayerID, GameStats, Path | None]] = []
    for rot in rotations:
        if len(rot.replay_paths) != len(rot.result.games):
            raise ValueError(
                "replay_paths length must match result.games length"
            )
        for g, rp in zip(rot.result.games, rot.replay_paths):
            all_games.append((rot.learner_seat, g, rp))

    n_games = len(all_games)
    learner_wins = 0
    opponent_wins = 0
    stalemates = 0
    vp_learner_total = 0
    vp_opponent_total = 0
    turn_total = 0
    learner_actions: dict[str, int] = defaultdict(int)
    opponent_actions: dict[str, int] = defaultdict(int)
    per_seat_learner_games: dict[PlayerID, int] = defaultdict(int)
    per_seat_learner_wins: dict[PlayerID, int] = defaultdict(int)

    for learner_seat, g, _ in all_games:
        per_seat_learner_games[learner_seat] += 1
        if g.winner is None:
            stalemates += 1
        elif g.winner == learner_seat:
            learner_wins += 1
            per_seat_learner_wins[learner_seat] += 1
        else:
            opponent_wins += 1

        vp_learner_total += g.final_vps.get(learner_seat, 0)
        # "Opponent" VP is the best-performing opponent that game — gives a
        # tighter comparison than averaging all three (where 4-player Catan's
        # zero-sum nature drags the mean toward the learner).
        opp_vps = [vp for pid, vp in g.final_vps.items() if pid != learner_seat]
        vp_opponent_total += max(opp_vps) if opp_vps else 0
        turn_total += g.turn_count

        for pid, counts in g.per_seat_action_histogram.items():
            target = learner_actions if pid == learner_seat else opponent_actions
            for name, c in counts.items():
                target[name] += c

    per_seat_win_rate = {
        seat: (
            per_seat_learner_wins[seat] / per_seat_learner_games[seat]
            if per_seat_learner_games[seat] > 0
            else 0.0
        )
        for seat in per_seat_learner_games
    }

    return EvalComparison(
        learner_label=learner_label,
        opponent_label=opponent_label,
        n_games=n_games,
        learner_win_rate=learner_wins / n_games,
        opponent_win_rate=opponent_wins / n_games,
        stalemate_rate=stalemates / n_games,
        per_seat_learner_win_rate=per_seat_win_rate,
        mean_vp_learner=vp_learner_total / n_games,
        mean_vp_opponent=vp_opponent_total / n_games,
        mean_turns=turn_total / n_games,
        learner_action_counts=dict(learner_actions),
        opponent_action_counts=dict(opponent_actions),
        sample_games=_pick_sample_games(all_games),
    )


def _pick_sample_games(
    all_games: list[tuple[PlayerID, GameStats, Path | None]]
) -> list[GamePick]:
    """Pick one close win, one blowout, and one stalemate (when available).

    Selection prefers games the learner won for "close_win" / "blowout" — the
    learner perspective is what the report is about. Falls back to "best
    available" if no perfect match exists. Up to five total picks are
    produced (two close wins, two blowouts, one stalemate).
    """
    learner_wins: list[tuple[int, tuple[PlayerID, GameStats, Path | None]]] = []
    stalemate_picks: list[tuple[PlayerID, GameStats, Path | None]] = []

    for seat, g, rp in all_games:
        if g.winner is None:
            stalemate_picks.append((seat, g, rp))
        elif g.winner == seat:
            opp_vps = [vp for pid, vp in g.final_vps.items() if pid != seat]
            margin = g.final_vps.get(seat, 0) - max(opp_vps) if opp_vps else 0
            learner_wins.append((margin, (seat, g, rp)))

    learner_wins.sort(key=lambda t: t[0])
    close = [w for w in learner_wins if w[0] <= 2][:2]
    blowouts = list(reversed(learner_wins))[:2]
    if not close and learner_wins:
        close = learner_wins[:1]
    if not blowouts and learner_wins:
        blowouts = list(reversed(learner_wins))[:1]

    picks: list[GamePick] = []
    for margin, (seat, g, rp) in close:
        picks.append(_to_pick("close_win", seat, g, rp))
    for margin, (seat, g, rp) in blowouts:
        picks.append(_to_pick("blowout", seat, g, rp))
    if stalemate_picks:
        seat, g, rp = stalemate_picks[0]
        picks.append(_to_pick("stalemate", seat, g, rp))

    return picks


def _to_pick(
    label: str,
    seat: PlayerID,
    g: GameStats,
    rp: Path | None,
) -> GamePick:
    opp_vps = [vp for pid, vp in g.final_vps.items() if pid != seat]
    return GamePick(
        label=label,
        learner_seat=seat,
        winner=g.winner,
        learner_vp=g.final_vps.get(seat, 0),
        top_opponent_vp=max(opp_vps) if opp_vps else 0,
        turn_count=g.turn_count,
        end_reason=g.end_reason,
        replay_path=rp,
    )


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def format_evaluation_report(comp: EvalComparison) -> str:
    """Render an :class:`EvalComparison` as a Markdown one-pager.

    The output is plain Markdown — terminals render it as text and the same
    blob can be pasted into a PR description without tweaking.
    """
    lines: list[str] = []
    lines.append(f"# Eval: {comp.learner_label} vs {comp.opponent_label}")
    lines.append("")
    lines.append(f"- Games: {comp.n_games}")
    lines.append(f"- Mean turns/game: {comp.mean_turns:.1f}")
    lines.append("")
    lines.append("## Outcomes")
    lines.append("")
    lines.append(
        f"| outcome | rate |\n| --- | --- |\n"
        f"| learner win | {comp.learner_win_rate:.3f} |\n"
        f"| opponent win | {comp.opponent_win_rate:.3f} |\n"
        f"| stalemate | {comp.stalemate_rate:.3f} |"
    )
    lines.append("")
    lines.append("## Per-seat learner win rate")
    lines.append("")
    lines.append("| seat | win rate |\n| --- | --- |")
    for seat in sorted(comp.per_seat_learner_win_rate.keys(), key=int):
        lines.append(
            f"| P{int(seat)} | {comp.per_seat_learner_win_rate[seat]:.3f} |"
        )
    lines.append("")
    lines.append("## VP")
    lines.append("")
    lines.append(
        f"- Mean learner VP: {comp.mean_vp_learner:.2f}\n"
        f"- Mean top-opponent VP: {comp.mean_vp_opponent:.2f}"
    )
    lines.append("")
    lines.append("## Action diff (per-game rate; +learner / −opponent)")
    lines.append("")
    lines.append(_format_action_diff(
        comp.learner_action_counts,
        comp.opponent_action_counts,
        n_games=comp.n_games,
    ))
    lines.append("")
    lines.append("## Sample games")
    lines.append("")
    if comp.sample_games:
        lines.append(_format_sample_games(comp.sample_games))
    else:
        lines.append("_(no notable games to highlight)_")
    return "\n".join(lines)


def _format_action_diff(
    learner: dict[str, int],
    opponent: dict[str, int],
    *,
    n_games: int,
) -> str:
    """Per-game-rate diff between learner and opponent action distributions.

    Opponent counts are summed over the three opponent seats per game, so we
    divide by ``3 * n_games`` to bring them to the same per-seat-per-game
    scale as the learner. Sort by absolute diff descending; show the top 10.
    """
    if n_games == 0:
        return "_(no games)_"
    if not learner and not opponent:
        return "_(no actions recorded)_"

    learner_per_game = {k: v / n_games for k, v in learner.items()}
    opp_per_game = {k: v / (3 * n_games) for k, v in opponent.items()}
    all_keys = set(learner_per_game) | set(opp_per_game)

    rows: list[tuple[str, float, float, float]] = []
    for k in all_keys:
        a = learner_per_game.get(k, 0.0)
        b = opp_per_game.get(k, 0.0)
        rows.append((k, a, b, a - b))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    out = ["| action | learner | opponent | diff |", "| --- | --- | --- | --- |"]
    for k, a, b, d in rows[:10]:
        out.append(f"| {k} | {a:.2f} | {b:.2f} | {d:+.2f} |")
    return "\n".join(out)


def _format_sample_games(picks: Iterable[GamePick]) -> str:
    out: list[str] = []
    for p in picks:
        winner_str = "stalemate" if p.winner is None else f"P{int(p.winner)}"
        path_str = str(p.replay_path) if p.replay_path is not None else "(no replay archived)"
        out.append(
            f"- **{p.label}** — learner P{int(p.learner_seat)} "
            f"(VP {p.learner_vp} vs top-opp {p.top_opponent_vp}), "
            f"winner {winner_str}, turns={p.turn_count}, "
            f"end={p.end_reason.name} — `{path_str}`"
        )
    return "\n".join(out)
