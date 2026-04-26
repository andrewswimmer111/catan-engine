"""
Data-only descriptions of in-progress turn interrupts (what the engine is
waiting on before the turn can continue).  Rule logic lives elsewhere; this
module is the union of all ``PendingEffect`` types and their fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums import DomesticTradeState, Resource
from domain.ids import PlayerID

__all__ = [
    "DiscardPending",
    "RobberMovePending",
    "StealPending",
    "DomesticTradePending",
    "RoadBuildingPending",
    "YearOfPlentyPending",
    "MonopolyPending",
    "PendingEffect",
]


@dataclass(frozen=True)
class DiscardPending:
    players_who_must_discard: frozenset[PlayerID]


@dataclass(frozen=True)
class RobberMovePending:
    pass


@dataclass(frozen=True)
class StealPending:
    valid_targets: frozenset[PlayerID]


@dataclass(frozen=True)
class DomesticTradePending:
    offer: dict[Resource, int]
    request: dict[Resource, int]
    responses: dict[PlayerID, DomesticTradeState]


@dataclass(frozen=True)
class RoadBuildingPending:
    roads_remaining: int


@dataclass(frozen=True)
class YearOfPlentyPending:
    resources_remaining: int


@dataclass(frozen=True)
class MonopolyPending:
    pass


PendingEffect = (
    DiscardPending
    | RobberMovePending
    | StealPending
    | DomesticTradePending
    | RoadBuildingPending
    | YearOfPlentyPending
    | MonopolyPending
)
