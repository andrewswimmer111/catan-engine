"""
Pluggable randomness for dice, deck shuffles, and robber steals.

The engine accepts a :class:`Randomizer` so simulations, tests, and training
runs can swap in a fixed :class:`SeededRandomizer` for full reproducibility.
"""

from __future__ import annotations

import random
from typing import Protocol

from domain.enums import DevCardType, Resource


class Randomizer(Protocol):
    """Abstract source of alea used by rules and transitions."""

    def roll_dice(self) -> tuple[int, int]:
        """Two independent fair die results, each in ``1..6`` inclusive."""
        ...

    def shuffle_dev_deck(self, cards: list[DevCardType]) -> list[DevCardType]:
        """
        Return a permutation of ``cards``.

        Must not rely on mutating the argument; callers may reuse the input list.
        """
        ...

    def choose_stolen_resource(self, resources: list[Resource]) -> Resource:
        """Pick one element from a non-empty multiset of resource cards."""
        ...


class SeededRandomizer:
    """
    Deterministic :class:`Randomizer` backed by :class:`random.Random`.

    Same ``seed`` yields the same sequence of ``roll_dice`` outcomes, deck
    shuffles (for identical inputs), and steal choices (for identical inputs).
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def roll_dice(self) -> tuple[int, int]:
        return self._rng.randint(1, 6), self._rng.randint(1, 6)

    def shuffle_dev_deck(self, cards: list[DevCardType]) -> list[DevCardType]:
        out = list(cards)
        self._rng.shuffle(out)
        return out

    def choose_stolen_resource(self, resources: list[Resource]) -> Resource:
        if not resources:
            raise ValueError("choose_stolen_resource requires a non-empty list")
        return self._rng.choice(resources)
