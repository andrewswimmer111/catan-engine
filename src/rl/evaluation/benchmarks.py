"""Reproducible benchmark callables — heuristic baseline regression checks.

Each benchmark fixes a seed list (``base_seed`` + ``range(n_games)``) and
returns a :class:`TournamentResult`. The random opponents skip domestic-trade
proposals by default so games don't spend most of their wall-clock in
unconverted trade flows (~20× speedup); the rest of random's behavior is
unchanged. Tests in :mod:`tests.rl.test_benchmarks` assert win-rate and VP
bounds with generous margins so they don't flake on per-seed variance.
"""

from __future__ import annotations

import random
from typing import Callable

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.metrics import TournamentResult
from rl.evaluation.tournament import Tournament

__all__ = [
    "DEFAULT_PLAYER_IDS",
    "bench_random_vs_random",
    "bench_heuristic_vs_random",
    "bench_heuristic_vs_heuristic",
]

DEFAULT_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _split_rng(seed: int) -> Callable[[], random.Random]:
    """Build a deterministic stream of sub-rngs keyed off a single seed."""
    parent = random.Random(seed)
    return lambda: random.Random(parent.randrange(2**32))


def bench_random_vs_random(
    n_games: int = 30, base_seed: int = 0, *, skip_proposals: bool = True
) -> TournamentResult:
    next_rng = _split_rng(base_seed)
    agents: dict[PlayerID, Agent] = {
        pid: RandomAgent(next_rng(), skip_proposals=skip_proposals)
        for pid in DEFAULT_PLAYER_IDS
    }
    return Tournament(_env_factory).play(agents, n_games=n_games, base_seed=base_seed)


def bench_heuristic_vs_random(
    n_games: int = 30, base_seed: int = 0, *, skip_proposals: bool = True
) -> TournamentResult:
    next_rng = _split_rng(base_seed)
    heur_seat = DEFAULT_PLAYER_IDS[0]
    agents: dict[PlayerID, Agent] = {}
    for pid in DEFAULT_PLAYER_IDS:
        if pid == heur_seat:
            agents[pid] = HeuristicAgent()
        else:
            agents[pid] = RandomAgent(next_rng(), skip_proposals=skip_proposals)
    return Tournament(_env_factory).play(agents, n_games=n_games, base_seed=base_seed)


def bench_heuristic_vs_heuristic(
    n_games: int = 50, base_seed: int = 0
) -> TournamentResult:
    agents: dict[PlayerID, Agent] = {pid: HeuristicAgent() for pid in DEFAULT_PLAYER_IDS}
    return Tournament(_env_factory).play(agents, n_games=n_games, base_seed=base_seed)
