from __future__ import annotations

import pytest

from rl.evaluation.benchmarks import (
    DEFAULT_PLAYER_IDS,
    bench_heuristic_vs_heuristic,
    bench_heuristic_vs_random,
    bench_random_vs_random,
)
from rl.evaluation.metrics import TournamentResult

HEUR_SEAT = DEFAULT_PLAYER_IDS[0]
RANDOM_SEATS = DEFAULT_PLAYER_IDS[1:]


# ----------------------------------------------------------------------
# Fast (default suite): shape + reproducibility checks
# ----------------------------------------------------------------------


def test_benchmark_callables_return_tournament_result():
    """Each benchmark runs to completion and returns the expected shape."""
    for result in (
        bench_random_vs_random(n_games=2, base_seed=0),
        bench_heuristic_vs_random(n_games=2, base_seed=0),
        bench_heuristic_vs_heuristic(n_games=2, base_seed=0),
    ):
        assert isinstance(result, TournamentResult)
        assert len(result.games) == 2
        assert set(result.win_rates) == set(DEFAULT_PLAYER_IDS)
        assert set(result.mean_vp) == set(DEFAULT_PLAYER_IDS)


def test_same_seed_reproduces_same_games():
    """A fixed ``base_seed`` produces identical per-game stats across runs."""
    a = bench_heuristic_vs_random(n_games=2, base_seed=7)
    b = bench_heuristic_vs_random(n_games=2, base_seed=7)
    for ga, gb in zip(a.games, b.games):
        assert ga.winner == gb.winner
        assert ga.final_vps == gb.final_vps
        assert ga.turn_count == gb.turn_count
        assert ga.action_histogram == gb.action_histogram


# ----------------------------------------------------------------------
# Slow / nightly (regression checks — ``nightly`` if routinely ~30s+ wall time)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_bench_random_vs_random_terminates_cleanly():
    """Sanity: 30 random-vs-random games all terminate with valid stats."""
    result = bench_random_vs_random(n_games=30, base_seed=0)

    assert len(result.games) == 30
    for g in result.games:
        assert g.turn_count > 0
    assert 0.0 <= sum(result.win_rates.values()) <= 1.0 + 1e-9


@pytest.mark.slow
def test_bench_heuristic_vs_random_dominance():
    """Heuristic dominates random both in wins and in mean VP.

    Threshold uses VP-margin because the engine's 50-turn VP-stall rule ends
    many games before anyone hits 10 VP, suppressing absolute win rate. The
    heuristic's per-game mean VP (~7-8) still towers over random's (~3) and
    the heuristic captures every completed game.
    """
    result = bench_heuristic_vs_random(n_games=30, base_seed=200)

    heur_wins = result.win_rates[HEUR_SEAT]
    random_wins_total = sum(result.win_rates[pid] for pid in RANDOM_SEATS)
    assert heur_wins > random_wins_total, result.win_rates

    heur_vp = result.mean_vp[HEUR_SEAT]
    best_random_vp = max(result.mean_vp[pid] for pid in RANDOM_SEATS)
    assert heur_vp > 2 * best_random_vp, result.mean_vp


@pytest.mark.nightly
def test_bench_heuristic_vs_heuristic_seat_symmetry():
    """4 heuristics play symmetric games: per-seat mean VP stays within 2.0.

    With identical agents this measures only the engine's seat-rotation
    fairness; the 2.0-VP slack absorbs the ~0.3-VP standard error from 100
    games. (Win-rate spread is trivially 0 — all games stalemate under the
    50-turn no-VP-change rule — so it's not a useful symmetry signal.)
    """
    result = bench_heuristic_vs_heuristic(n_games=100, base_seed=500)

    vps = [result.mean_vp[pid] for pid in DEFAULT_PLAYER_IDS]
    spread = max(vps) - min(vps)
    assert spread < 2.0, result.mean_vp
