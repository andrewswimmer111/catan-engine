"""
Exhaustive set of player actions the engine recognizes.

This module is the single registry of *what* can be requested; legality and
state updates live elsewhere. Add new :class:`Action` subclasses here when the
ruleset grows so Task 9 serialization and ``legal_actions`` stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from domain.actions.base import Action
from domain.enums import DomesticTradeState, Resource
from domain.ids import EdgeID, PlayerID, TileID, VertexID

# --- Setup ---


@dataclass(frozen=True)
class PlaceSettlementAction(Action):
    vertex_id: VertexID


@dataclass(frozen=True)
class PlaceRoadAction(Action):
    edge_id: EdgeID


# --- Build ---


@dataclass(frozen=True)
class BuildSettlementAction(Action):
    vertex_id: VertexID


@dataclass(frozen=True)
class BuildRoadAction(Action):
    edge_id: EdgeID


@dataclass(frozen=True)
class BuildCityAction(Action):
    vertex_id: VertexID


# --- Turn ---


@dataclass(frozen=True)
class RollDiceAction(Action):
    pass


@dataclass(frozen=True)
class EndTurnAction(Action):
    pass


# --- Robber ---


@dataclass(frozen=True)
class DiscardResourcesAction(Action):
    resources: dict[Resource, int]


@dataclass(frozen=True)
class MoveRobberAction(Action):
    tile_id: TileID


@dataclass(frozen=True)
class StealResourceAction(Action):
    target_player_id: PlayerID


# --- Dev cards ---


@dataclass(frozen=True)
class BuyDevCardAction(Action):
    pass


@dataclass(frozen=True)
class PlayKnightAction(Action):
    pass


@dataclass(frozen=True)
class PlayRoadBuildingAction(Action):
    pass


@dataclass(frozen=True)
class PlayYearOfPlentyAction(Action):
    resource1: Resource
    resource2: Resource


@dataclass(frozen=True)
class PlayMonopolyAction(Action):
    resource: Resource


# --- Trade ---


@dataclass(frozen=True)
class MaritimeTradeAction(Action):
    give: Resource
    give_count: int
    receive: Resource


@dataclass(frozen=True)
class ProposeDomesticTradeAction(Action):
    offer: dict[Resource, int]
    request: dict[Resource, int]


@dataclass(frozen=True)
class RespondDomesticTradeAction(Action):
    response: DomesticTradeState


@dataclass(frozen=True)
class ConfirmDomesticTradeAction(Action):
    trade_with: PlayerID


@dataclass(frozen=True)
class CancelDomesticTradeAction(Action):
    pass


AnyAction = Union[
    PlaceSettlementAction,
    PlaceRoadAction,
    BuildSettlementAction,
    BuildRoadAction,
    BuildCityAction,
    RollDiceAction,
    EndTurnAction,
    DiscardResourcesAction,
    MoveRobberAction,
    StealResourceAction,
    BuyDevCardAction,
    PlayKnightAction,
    PlayRoadBuildingAction,
    PlayYearOfPlentyAction,
    PlayMonopolyAction,
    MaritimeTradeAction,
    ProposeDomesticTradeAction,
    RespondDomesticTradeAction,
    ConfirmDomesticTradeAction,
    CancelDomesticTradeAction,
]
