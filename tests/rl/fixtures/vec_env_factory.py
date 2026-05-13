"""Picklable :class:`SubprocVecEnv` factories for tests.

The tests need to spawn worker subprocesses, which requires that the factory
function be importable by name. Test-local closures over fixtures don't
survive ``mp.spawn``, so the factories live here at module level.
"""

from __future__ import annotations

import random

from domain.ids import PlayerID
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.training.vec_env import WorkerBundle


_PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def random_opponents_factory(seed: int) -> WorkerBundle:
    """Bundle: fresh env at ``seed``, learner at seat 1, three random opponents.

    Each opponent gets its own ``random.Random`` seeded from ``seed`` so the
    per-env trajectory is fully reproducible given the same input seed.
    """
    env = CatanEnv(seed=seed)
    rng = random.Random(seed)
    learner_seat = _PLAYER_IDS[0]
    opponents = {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
        for pid in _PLAYER_IDS[1:]
    }
    return WorkerBundle(env=env, learner_seat=learner_seat, opponents=opponents)
