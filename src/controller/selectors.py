"""
Pure functions for filtering and ranking legal actions.

Takes the full set of actions returned by the engine and narrows it down
according to phase, player, or heuristic criteria, providing building blocks
that both agent policies and UI helpers can compose without duplicating logic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, TypeVar

import domain.actions.all_actions as A
from domain.actions.base import Action
from domain.ids import EdgeID, PlayerID, TileID, VertexID

__all__ = [
    "vertex_targets",
    "edge_targets",
    "tile_targets",
    "player_steal_targets",
    "ACTION_GROUPS",
    "grouped",
]

_VERTEX_ACTION_TYPES = (
    A.PlaceSettlementAction,
    A.BuildSettlementAction,
    A.BuildCityAction,
)

_EDGE_ACTION_TYPES = (
    A.PlaceRoadAction,
    A.BuildRoadAction,
)

ACTION_GROUPS: dict[str, tuple[type[Action], ...]] = {
    "Build": (A.BuildSettlementAction, A.BuildRoadAction, A.BuildCityAction),
    "Setup": (A.PlaceSettlementAction, A.PlaceRoadAction),
    "Turn": (A.RollDiceAction, A.EndTurnAction),
    "Robber": (A.MoveRobberAction, A.StealResourceAction, A.DiscardResourcesAction),
    "DevCard": (
        A.BuyDevCardAction,
        A.PlayKnightAction,
        A.PlayRoadBuildingAction,
        A.PlayYearOfPlentyAction,
        A.PlayMonopolyAction,
    ),
    "Trade": (
        A.MaritimeTradeAction,
        A.ProposeDomesticTradeAction,
        A.RespondDomesticTradeAction,
        A.ConfirmDomesticTradeAction,
        A.CancelDomesticTradeAction,
    ),
}

T = TypeVar("T")

def _targets_by_type(
    legal: list[Action],
    action_types: tuple[type[Action], ...],
    get_id: Callable[[Action], T],) -> dict[type[Action], set[T]]:
    """For each _ action class, the set of legal target _ IDs."""

    result: dict[type[Action], set[T]] = defaultdict(set)
    for action in legal:
        if isinstance(action, action_types):
            result[type(action)].add(get_id(action))
    return dict(result)

def vertex_targets(legal: list[Action]) -> dict[type[Action], set[VertexID]]:
    return _targets_by_type(legal, _VERTEX_ACTION_TYPES, lambda a: a.vertex_id)


def edge_targets(legal: list[Action]) -> dict[type[Action], set[EdgeID]]:
    return _targets_by_type(legal, _EDGE_ACTION_TYPES, lambda a: a.edge_id)


def tile_targets(legal: list[Action]) -> set[TileID]:
    """Tiles legal as MoveRobberAction targets."""
    return {action.tile_id for action in legal if isinstance(action, A.MoveRobberAction)}


def player_steal_targets(legal: list[Action]) -> set[PlayerID]:
    """Players legal as StealResourceAction targets."""
    return {action.target_player_id for action in legal if isinstance(action, A.StealResourceAction)}


def grouped(legal: list[Action]) -> dict[str, list[Action]]:
    """Partition legal actions into named groups defined by ACTION_GROUPS."""
    result: dict[str, list[Action]] = {}
    for group, types in ACTION_GROUPS.items():
        matching = [a for a in legal if isinstance(a, types)]
        if matching:
            result[group] = matching
    return result
