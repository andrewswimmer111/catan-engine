from __future__ import annotations

import random

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.ids import PlayerID

__all__ = ["RandomAgent", "make_random_agents"]


class RandomAgent:
    """Uniform-random legal action picker.

    ``skip_proposals=True`` filters out :class:`ProposeDomesticTradeAction` so
    games don't spend most of their wall-clock in 4-step trade flows that
    rarely complete. Used by benchmarks where wall-time matters; left off by
    default so the agent stays a true uniform-random baseline.
    """

    def __init__(self, rng: random.Random, *, skip_proposals: bool = False) -> None:
        self._rng = rng
        self._skip_proposals = skip_proposals

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        if not legal:
            return None
        pool = legal
        if self._skip_proposals:
            filtered = [a for a in legal if not isinstance(a, A.ProposeDomesticTradeAction)]
            if filtered:
                pool = filtered
        return self._rng.choice(pool)


def make_random_agents(
    player_ids: list[PlayerID], seed: int, *, skip_proposals: bool = False,
) -> dict[PlayerID, RandomAgent]:
    rng = random.Random(seed)
    return {
        pid: RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=skip_proposals)
        for pid in player_ids
    }
