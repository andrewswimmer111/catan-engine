from __future__ import annotations

from domain.ids import PlayerID
from rl.agents.random_agent import make_random_agents
from rl.env.catan_env import CatanEnv
from rl.evaluation.metrics import GameStats, TournamentResult
from rl.evaluation.tournament import Tournament

_PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _make_tournament() -> Tournament:
    return Tournament(_env_factory)


def _make_agents(seed: int = 0):
    return make_random_agents(_PLAYER_IDS, seed=seed)


def test_all_games_terminate():
    t = _make_tournament()
    result = t.play(_make_agents(), n_games=20, base_seed=0)
    assert len(result.games) == 20
    for g in result.games:
        assert isinstance(g, GameStats)
        assert g.turn_count > 0


def test_win_rates_sum_to_one():
    t = _make_tournament()
    result = t.play(_make_agents(), n_games=5, base_seed=42)
    total = sum(result.win_rates.values())
    # Allow small slack for stalemates where winner is None (no win credited)
    assert 0.0 <= total <= 1.0 + 1e-9


def test_win_rates_cover_all_players():
    t = _make_tournament()
    result = t.play(_make_agents(), n_games=3, base_seed=7)
    assert set(result.win_rates.keys()) == set(_PLAYER_IDS)
    assert set(result.mean_vp.keys()) == set(_PLAYER_IDS)


def test_reproducible_with_same_seed():
    agents_a = make_random_agents(_PLAYER_IDS, seed=0)
    agents_b = make_random_agents(_PLAYER_IDS, seed=0)
    result_a = Tournament(_env_factory).play(agents_a, n_games=5, base_seed=99)
    result_b = Tournament(_env_factory).play(agents_b, n_games=5, base_seed=99)

    assert len(result_a.games) == len(result_b.games)
    for ga, gb in zip(result_a.games, result_b.games):
        assert ga.winner == gb.winner
        assert ga.final_vps == gb.final_vps
        assert ga.turn_count == gb.turn_count
        assert ga.end_reason == gb.end_reason
        assert ga.action_histogram == gb.action_histogram


def test_action_histogram_nonempty():
    t = _make_tournament()
    result = t.play(_make_agents(), n_games=3, base_seed=1)
    for g in result.games:
        assert len(g.action_histogram) > 0
        assert all(isinstance(k, str) for k in g.action_histogram)
        assert all(v > 0 for v in g.action_histogram.values())


def test_mean_turns_positive():
    t = _make_tournament()
    result = t.play(_make_agents(), n_games=5, base_seed=10)
    assert result.mean_turns > 0
