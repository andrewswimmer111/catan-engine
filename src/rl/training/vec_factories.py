"""Picklable :class:`VecEnvFactory` implementations for the trainer's vec path.

:class:`SubprocVecEnv` spawns each worker via ``multiprocessing.spawn``, which
pickles the factory by import path. Closures over local state (an
``OpponentPool``, a learner reference) don't survive that round-trip, so the
factories that the trainer uses must be plain module-level callables. The
helpers here cover the two baseline opponent setups (random / heuristic) used
during early-bring-up training; richer setups (e.g. pool-sampled opponents)
need their own scheme since pool checkpoints live in the main process.

Seat rotation
-------------

The factory derives the learner's seat from its incoming seed:
``learner_seat = PlayerID(seed % 4 + 1)``. :class:`SubprocVecEnv` passes
``base_seed + env_index`` to each subprocess, so with ``num_envs >= 4`` and a
``base_seed`` aligned to 0 the four learner seats are exercised in parallel.
Per-subprocess seats stay fixed for the run — opponent diversity (the other
half of seat-rotation's motivation) comes from re-sampling the SubprocVecEnv,
which the trainer doesn't do mid-run today.
"""

from __future__ import annotations

import random

from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.training.vec_env import WorkerBundle

__all__ = [
    "random_opponents_factory",
    "heuristic_opponents_factory",
]


_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]


def _learner_seat_for(seed: int) -> PlayerID:
    return _PLAYER_IDS[seed % len(_PLAYER_IDS)]


def random_opponents_factory(seed: int) -> WorkerBundle:
    """Bundle: fresh env at ``seed``, seat-rotated learner, three random opponents.

    Opponents skip domestic-trade proposals (``skip_proposals=True``) for the
    same reason the benchmarks do — un-converted proposals dominate wall-clock
    in random play and the model doesn't learn from them yet.
    """
    env = CatanEnv(seed=seed)
    rng = random.Random(seed)
    learner_seat = _learner_seat_for(seed)
    opponents = {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in _PLAYER_IDS
        if pid != learner_seat
    }
    return WorkerBundle(env=env, learner_seat=learner_seat, opponents=opponents)


def heuristic_opponents_factory(seed: int) -> WorkerBundle:
    """Bundle: fresh env at ``seed``, seat-rotated learner, three heuristic opponents.

    Heuristic opponents are deterministic given the engine seed — they don't
    take an rng — so the per-subprocess trajectory is fully reproducible from
    ``seed`` alone.
    """
    env = CatanEnv(seed=seed)
    learner_seat = _learner_seat_for(seed)
    opponents = {
        pid: HeuristicAgent() for pid in _PLAYER_IDS if pid != learner_seat
    }
    return WorkerBundle(env=env, learner_seat=learner_seat, opponents=opponents)
