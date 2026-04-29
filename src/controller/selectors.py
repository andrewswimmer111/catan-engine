"""
Pure functions for filtering and ranking legal actions.

Takes the full set of actions returned by the engine and narrows it down
according to phase, player, or heuristic criteria, providing building blocks
that both agent policies and UI helpers can compose without duplicating logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def vertex_targets(legal: list[Action]) -> dict[type[Action], set[VertexID]]:
    """For each vertex-targeted action class, the set of legal target vertex IDs."""
    result: dict[type[Action], set[VertexID]] = {}
    for action in legal:
        if isinstance(action, _VERTEX_ACTION_TYPES):
            cls = type(action)
            if cls not in result:
                result[cls] = set()
            result[cls].add(action.vertex_id)
    return result


def edge_targets(legal: list[Action]) -> dict[type[Action], set[EdgeID]]:
    """For each edge-targeted action class, the set of legal target edge IDs."""
    result: dict[type[Action], set[EdgeID]] = {}
    for action in legal:
        if isinstance(action, _EDGE_ACTION_TYPES):
            cls = type(action)
            if cls not in result:
                result[cls] = set()
            result[cls].add(action.edge_id)
    return result


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
