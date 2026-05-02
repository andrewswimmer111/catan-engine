from __future__ import annotations

import random

from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.ids import PlayerID

__all__ = ["RandomAgent", "make_random_agents"]


class RandomAgent:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        return self._rng.choice(legal) if legal else None


def make_random_agents(
    player_ids: list[PlayerID], seed: int
) -> dict[PlayerID, RandomAgent]:
    rng = random.Random(seed)
    return {pid: RandomAgent(random.Random(rng.randrange(2**32))) for pid in player_ids}
