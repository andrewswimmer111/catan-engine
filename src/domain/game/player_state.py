"""
Per-player mutable snapshot: hand, dev cards, build counts, and public VP.
Holds no global game fields; that belongs in :mod:`domain.game.state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.enums import DevCardType, Resource
from domain.ids import PlayerID


@dataclass
class PlayerState:
    player_id: PlayerID
    resources: dict[Resource, int] = field(default_factory=dict)
    dev_cards_in_hand: list[tuple[DevCardType, int]] = field(default_factory=list)
    dev_cards_played: list[DevCardType] = field(default_factory=list)
    roads_built: int = 0
    settlements_built: int = 0
    cities_built: int = 0
    knights_played: int = 0
    has_played_dev_card_this_turn: bool = False
    """Victory points visible from the table (buildings + awards, not hidden VP cards)."""
    victory_points_public: int = 0

    def resource_count(self) -> int:
        return sum(self.resources.values())

    def can_afford(self, cost: dict[Resource, int]) -> bool:
        return all(self.resources.get(r, 0) >= c for r, c in cost.items())
