"""
Ordered development card deck: draw order, :class:`EmptyDeckError`, and the
static 25-card factory composition used before shuffling at game start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.enums import DevCardType


class EmptyDeckError(LookupError):
    """Raised when :meth:`DevelopmentDeck.draw` is called on an empty deck."""


@dataclass
class DevelopmentDeck:
    """
    Ordered dev cards; front of the list is the next card drawn
    (``pop(0)`` in :meth:`draw`).
    """

    cards: list[DevCardType] = field(default_factory=list)

    def draw(self) -> DevCardType:
        if not self.cards:
            raise EmptyDeckError("no development cards remain")
        return self.cards.pop(0)

    def remaining(self) -> int:
        return len(self.cards)


def standard_dev_deck_composition() -> list[DevCardType]:
    """The 25-card deck before shuffling: 14 knights, 5 VP, 2 RB, 2 YoP, 2 monopoly."""
    return (
        [DevCardType.KNIGHT] * 14
        + [DevCardType.VICTORY_POINT] * 5
        + [DevCardType.ROAD_BUILDING] * 2
        + [DevCardType.YEAR_OF_PLENTY] * 2
        + [DevCardType.MONOPOLY] * 2
    )
