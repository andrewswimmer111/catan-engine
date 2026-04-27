"""
Data-only descriptions of in-progress turn interrupts (what the engine is
waiting on before the turn can continue).  Rule logic lives elsewhere; this
module is the union of all ``PendingEffect`` types and their fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums import DomesticTradeState, Resource, TurnPhase
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
    """
    After a 7, each listed player must discard that many resource cards
    in one :class:`DiscardResourcesAction` (``sum ==`` their count, ``floor`` of
    half their pre-discard hand was recorded when the 7 was rolled).
    """

    cards_to_discard: dict[PlayerID, int]


@dataclass(frozen=True)
class RobberMovePending:
    """
    After the robber is moved and any steal is resolved, restore ``return_phase``
    (``MAIN`` after a 7 was rolled, or the phase in effect when a knight was
    played, so a knight before rolling returns to :data:`~domain.enums.TurnPhase.ROLL`).
    """

    return_phase: TurnPhase


@dataclass(frozen=True)
class StealPending:
    valid_targets: frozenset[PlayerID]
    return_phase: TurnPhase


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
