"""Bidirectional mapping between :class:`Action` instances and discrete indices.

Index ranges are defined in :mod:`rl.encoding._action_layout`. The encoder
covers every action class listed in the layout table; domestic-trade actions
(propose / respond / confirm / cancel) and idle pending-effect placeholders
are intentionally not represented yet — encoding them raises ``ValueError``.

The discard action carries a combinatorial payload (which cards to drop) that
is not enumerated. ``encode`` collapses every ``DiscardResourcesAction`` to
:data:`_action_layout.DISCARD_INDEX`; ``decode`` returns a
:class:`DiscardSentinel`. The environment is responsible for replacing the
sentinel with a concrete discard chosen by a heuristic — the engine never
sees the sentinel.

Steal encoding is keyed by *seat position* in the configured player list, not
by raw ``PlayerID`` value, because PIDs are not assumed to follow a fixed
indexing convention (the codebase has both 0-indexed and 1-indexed
constructions). The encoder takes ``player_ids`` in its constructor so the
seat lookup is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Union

import numpy as np

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import PortType, Resource, TurnPhase
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from domain.turn.pending import DiscardPending
from rl.encoding._action_layout import (
    ACTION_SPACE_SIZE,
    BUY_DEV_INDEX,
    CITY_START,
    DISCARD_INDEX,
    END_TURN_INDEX,
    KNIGHT_INDEX,
    MARITIME_TRADE_PAIRS,
    MARITIME_TRADE_PAIR_TO_OFFSET,
    MARITIME_TRADE_START,
    MONOPOLY_START,
    N_STEAL_SLOTS,
    RESOURCES,
    RESOURCE_TO_INDEX,
    ROAD_BUILDING_INDEX,
    ROAD_START,
    ROBBER_MOVE_START,
    ROLL_INDEX,
    SETTLEMENT_START,
    STEAL_START,
    YEAR_OF_PLENTY_START,
    YOP_PAIRS,
    yop_pair_offset,
)

__all__ = ["ActionEncoder", "DiscardSentinel", "DecodedAction", "ACTION_SPACE_SIZE"]


@dataclass(frozen=True)
class DiscardSentinel:
    """Returned by :meth:`ActionEncoder.decode` for the discard trigger.

    Carries the player who owes the discard so the environment can dispatch
    to the right hand when generating the concrete :class:`DiscardResourcesAction`.
    """

    player_id: PlayerID


DecodedAction = Union[Action, DiscardSentinel]


_TWO_ONE_FOR_RESOURCE: Final[dict[PortType, Resource]] = {
    PortType.WOOD_TWO: Resource.WOOD,
    PortType.BRICK_TWO: Resource.BRICK,
    PortType.SHEEP_TWO: Resource.SHEEP,
    PortType.WHEAT_TWO: Resource.WHEAT,
    PortType.ORE_TWO: Resource.ORE,
}


def _maritime_ratio(state: GameState, player_id: PlayerID, give: Resource) -> int:
    """Mirror of ``trade_rules._best_maritime_ratio`` so ``decode`` is self-contained."""
    r = 4
    for vid, (owner, _bt) in state.occupancy.buildings.items():
        if owner != player_id:
            continue
        pt = state.topology.vertices[vid].port
        if pt is None:
            continue
        if pt is PortType.THREE_TO_ONE:
            r = min(r, 3)
        elif pt in _TWO_ONE_FOR_RESOURCE:
            if _TWO_ONE_FOR_RESOURCE[pt] is give:
                r = min(r, 2)
    return r


def _discard_player_id(state: GameState) -> PlayerID:
    """Player owed a discard. Falls back to ``current_player`` outside DISCARD."""
    if state.phase is TurnPhase.DISCARD and isinstance(state.pending, DiscardPending):
        return next(iter(state.pending.cards_to_discard))
    return state.current_player


class ActionEncoder:
    """Maps :class:`Action` ↔ discrete index in ``[0, ACTION_SPACE_SIZE)``.

    A new encoder must be constructed per game (or per ``GameConfig``) so the
    seat-keyed steal indices line up with the configured player list.
    """

    action_space_size: Final[int] = ACTION_SPACE_SIZE

    def __init__(self, player_ids: list[PlayerID]) -> None:
        if len(player_ids) > N_STEAL_SLOTS:
            raise ValueError(
                f"layout supports at most {N_STEAL_SLOTS} seats; got {len(player_ids)}"
            )
        self._player_ids: list[PlayerID] = list(player_ids)
        self._seat_of: dict[PlayerID, int] = {pid: i for i, pid in enumerate(player_ids)}

    @classmethod
    def for_state(cls, state: GameState) -> "ActionEncoder":
        """Convenience constructor — pulls ``player_ids`` from the state config."""
        return cls(list(state.config.player_ids))

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, action: Action) -> int:
        """Return the discrete index for ``action``.

        Raises ``ValueError`` for actions outside the layout (currently the
        domestic-trade family).
        """
        if isinstance(action, (A.PlaceRoadAction, A.BuildRoadAction)):
            return ROAD_START + int(action.edge_id)
        if isinstance(action, (A.PlaceSettlementAction, A.BuildSettlementAction)):
            return SETTLEMENT_START + int(action.vertex_id)
        if isinstance(action, A.BuildCityAction):
            return CITY_START + int(action.vertex_id)
        if isinstance(action, A.MoveRobberAction):
            return ROBBER_MOVE_START + int(action.tile_id)
        if isinstance(action, A.StealResourceAction):
            seat = self._seat_of.get(action.target_player_id)
            if seat is None:
                raise ValueError(
                    f"steal target {action.target_player_id} is not in the "
                    f"configured player list {self._player_ids}"
                )
            return STEAL_START + seat
        if isinstance(action, A.MaritimeTradeAction):
            return MARITIME_TRADE_START + MARITIME_TRADE_PAIR_TO_OFFSET[
                (action.give, action.receive)
            ]
        if isinstance(action, A.RollDiceAction):
            return ROLL_INDEX
        if isinstance(action, A.EndTurnAction):
            return END_TURN_INDEX
        if isinstance(action, A.BuyDevCardAction):
            return BUY_DEV_INDEX
        if isinstance(action, A.PlayKnightAction):
            return KNIGHT_INDEX
        if isinstance(action, A.PlayRoadBuildingAction):
            return ROAD_BUILDING_INDEX
        if isinstance(action, A.PlayMonopolyAction):
            return MONOPOLY_START + RESOURCE_TO_INDEX[action.resource]
        if isinstance(action, A.PlayYearOfPlentyAction):
            return YEAR_OF_PLENTY_START + yop_pair_offset(
                action.resource1, action.resource2
            )
        if isinstance(action, A.DiscardResourcesAction):
            return DISCARD_INDEX
        raise ValueError(
            f"action {type(action).__name__} is not representable in the "
            f"current discrete layout (version 1)"
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, idx: int, state: GameState) -> DecodedAction:
        """Reconstruct a typed action (or :class:`DiscardSentinel`) from ``idx``.

        ``state`` is consulted only to fill in fields that are not part of the
        index (player_id, phase-dependent action class, maritime trade ratio).
        Decoding does NOT validate legality — applying an illegal decoded
        action raises :class:`IllegalActionError` at the engine boundary.
        """
        if not 0 <= idx < ACTION_SPACE_SIZE:
            raise ValueError(f"index {idx} out of range [0, {ACTION_SPACE_SIZE})")

        pid = state.current_player

        if idx < SETTLEMENT_START:
            edge_id = EdgeID(idx - ROAD_START)
            if state.phase is TurnPhase.INITIAL_ROAD:
                return A.PlaceRoadAction(player_id=pid, edge_id=edge_id)
            return A.BuildRoadAction(player_id=pid, edge_id=edge_id)

        if idx < CITY_START:
            vertex_id = VertexID(idx - SETTLEMENT_START)
            if state.phase is TurnPhase.INITIAL_SETTLEMENT:
                return A.PlaceSettlementAction(player_id=pid, vertex_id=vertex_id)
            return A.BuildSettlementAction(player_id=pid, vertex_id=vertex_id)

        if idx < ROBBER_MOVE_START:
            vertex_id = VertexID(idx - CITY_START)
            return A.BuildCityAction(player_id=pid, vertex_id=vertex_id)

        if idx < STEAL_START:
            tile_id = TileID(idx - ROBBER_MOVE_START)
            return A.MoveRobberAction(player_id=pid, tile_id=tile_id)

        if idx < MARITIME_TRADE_START:
            seat = idx - STEAL_START
            if seat >= len(self._player_ids):
                raise ValueError(
                    f"steal seat index {seat} exceeds player count "
                    f"{len(self._player_ids)}"
                )
            return A.StealResourceAction(
                player_id=pid, target_player_id=self._player_ids[seat]
            )

        if idx < ROLL_INDEX:
            give, receive = MARITIME_TRADE_PAIRS[idx - MARITIME_TRADE_START]
            return A.MaritimeTradeAction(
                player_id=pid,
                give=give,
                give_count=_maritime_ratio(state, pid, give),
                receive=receive,
            )

        if idx == ROLL_INDEX:
            return A.RollDiceAction(player_id=pid)
        if idx == END_TURN_INDEX:
            return A.EndTurnAction(player_id=pid)
        if idx == BUY_DEV_INDEX:
            return A.BuyDevCardAction(player_id=pid)
        if idx == KNIGHT_INDEX:
            return A.PlayKnightAction(player_id=pid)
        if idx == ROAD_BUILDING_INDEX:
            return A.PlayRoadBuildingAction(player_id=pid)

        if idx < YEAR_OF_PLENTY_START:
            return A.PlayMonopolyAction(
                player_id=pid, resource=RESOURCES[idx - MONOPOLY_START]
            )

        if idx < DISCARD_INDEX:
            r1, r2 = YOP_PAIRS[idx - YEAR_OF_PLENTY_START]
            return A.PlayYearOfPlentyAction(
                player_id=pid, resource1=r1, resource2=r2
            )

        # idx == DISCARD_INDEX
        return DiscardSentinel(player_id=_discard_player_id(state))

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def mask(self, legal: list[Action]) -> np.ndarray:
        """Boolean mask of shape ``(ACTION_SPACE_SIZE,)``.

        Entry ``i`` is True iff some action in ``legal`` encodes to ``i``.
        Domestic-trade and other unrepresented actions are silently skipped
        — they cannot be selected through the discrete head until the layout
        grows to cover them.
        """
        out = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        for a in legal:
            try:
                idx = self.encode(a)
            except ValueError:
                continue
            out[idx] = True
        return out
