"""Tests for :mod:`rl.evaluation.report` (rl-020)."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.enums import EndReason
from domain.ids import PlayerID
from rl.evaluation.metrics import GameStats, TournamentResult
from rl.evaluation.report import (
    EvalComparison,
    GamePick,
    RotationResult,
    aggregate_evaluation,
    format_evaluation_report,
)
from rl.evaluation.tournament import aggregate_games


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _game(
    winner: PlayerID | None,
    vps: dict[PlayerID, int],
    turn_count: int = 30,
    end: EndReason = EndReason.WINNER,
    per_seat: dict[PlayerID, dict[str, int]] | None = None,
) -> GameStats:
    return GameStats(
        winner=winner,
        final_vps=vps,
        turn_count=turn_count,
        end_reason=end,
        action_histogram={},
        per_seat_action_histogram=per_seat or {pid: {} for pid in PLAYER_IDS},
    )


def _rotation(
    learner_seat: PlayerID,
    games: list[GameStats],
    paths: list[Path | None] | None = None,
) -> RotationResult:
    if paths is None:
        paths = [None] * len(games)
    return RotationResult(
        learner_seat=learner_seat,
        result=aggregate_games(games, PLAYER_IDS),
        replay_paths=paths,
    )


def test_aggregate_basic_counts():
    # 2 games, learner at seat 1 in both. Wins one, loses one.
    games = [
        _game(winner=PLAYER_IDS[0], vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 6, PLAYER_IDS[2]: 5, PLAYER_IDS[3]: 4}),
        _game(winner=PLAYER_IDS[1], vps={PLAYER_IDS[0]: 7, PLAYER_IDS[1]: 10, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 3}),
    ]
    comp = aggregate_evaluation("A", "B", [_rotation(PLAYER_IDS[0], games)])
    assert comp.n_games == 2
    assert comp.learner_win_rate == 0.5
    assert comp.opponent_win_rate == 0.5
    assert comp.stalemate_rate == 0.0
    assert comp.per_seat_learner_win_rate[PLAYER_IDS[0]] == 0.5


def test_aggregate_stalemate_credit():
    g = _game(winner=None, vps={pid: 5 for pid in PLAYER_IDS}, end=EndReason.STALEMATE_NO_PROGRESS)
    comp = aggregate_evaluation("A", "B", [_rotation(PLAYER_IDS[0], [g])])
    assert comp.stalemate_rate == 1.0
    assert comp.learner_win_rate == 0.0
    assert comp.opponent_win_rate == 0.0


def test_aggregate_per_seat_win_rate_across_rotations():
    # learner wins seat 1, loses seat 2
    rot1 = _rotation(
        PLAYER_IDS[0],
        [_game(winner=PLAYER_IDS[0], vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 5, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4})],
    )
    rot2 = _rotation(
        PLAYER_IDS[1],
        [_game(winner=PLAYER_IDS[0], vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 5, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4})],
    )
    comp = aggregate_evaluation("A", "B", [rot1, rot2])
    assert comp.per_seat_learner_win_rate[PLAYER_IDS[0]] == 1.0
    assert comp.per_seat_learner_win_rate[PLAYER_IDS[1]] == 0.0
    assert comp.learner_win_rate == 0.5


def test_aggregate_action_diff():
    # Learner builds more roads than the opponents. Per_seat counts:
    # seat 1 (learner): {RoadBuilt: 4}; seats 2,3,4 (opp): each {BuyDev: 1}
    per_seat = {
        PLAYER_IDS[0]: {"BuildRoadAction": 4},
        PLAYER_IDS[1]: {"BuyDevCardAction": 1},
        PLAYER_IDS[2]: {"BuyDevCardAction": 1},
        PLAYER_IDS[3]: {"BuyDevCardAction": 1},
    }
    g = _game(
        winner=PLAYER_IDS[0],
        vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 5, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4},
        per_seat=per_seat,
    )
    comp = aggregate_evaluation("A", "B", [_rotation(PLAYER_IDS[0], [g])])
    assert comp.learner_action_counts == {"BuildRoadAction": 4}
    assert comp.opponent_action_counts == {"BuyDevCardAction": 3}


def test_pick_sample_games_close_win_and_blowout():
    close = _game(
        winner=PLAYER_IDS[0],
        vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 9, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4},
    )
    blowout = _game(
        winner=PLAYER_IDS[0],
        vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 3, PLAYER_IDS[2]: 2, PLAYER_IDS[3]: 1},
    )
    comp = aggregate_evaluation(
        "A", "B", [_rotation(PLAYER_IDS[0], [close, blowout])]
    )
    labels = {p.label for p in comp.sample_games}
    assert "close_win" in labels
    assert "blowout" in labels


def test_format_includes_required_sections():
    g = _game(winner=PLAYER_IDS[0], vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 5, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4})
    comp = aggregate_evaluation("learner", "random", [_rotation(PLAYER_IDS[0], [g])])
    text = format_evaluation_report(comp)
    assert "Eval: learner vs random" in text
    assert "Outcomes" in text
    assert "Per-seat learner win rate" in text
    assert "VP" in text
    assert "Action diff" in text
    assert "Sample games" in text


def test_aggregate_rejects_empty_rotations():
    with pytest.raises(ValueError):
        aggregate_evaluation("A", "B", [])


def test_replay_path_propagates_to_pick():
    g = _game(winner=PLAYER_IDS[0], vps={PLAYER_IDS[0]: 10, PLAYER_IDS[1]: 9, PLAYER_IDS[2]: 4, PLAYER_IDS[3]: 4})
    rot = _rotation(PLAYER_IDS[0], [g], paths=[Path("/tmp/g0.json")])
    comp = aggregate_evaluation("A", "B", [rot])
    assert any(p.replay_path == Path("/tmp/g0.json") for p in comp.sample_games)
