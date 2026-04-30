"""
Maritime and domestic (player-to-player) trade legals.
"""

from __future__ import annotations

from typing import Final

from domain.actions.all_actions import (
    CancelDomesticTradeAction,
    ConfirmDomesticTradeAction,
    MaritimeTradeAction,
    ProposeDomesticTradeAction,
    RespondDomesticTradeAction,
)
from domain.enums import DomesticTradeState, PortType, Resource, TurnPhase
from domain.enums import tradeable_resources
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DomesticTradePending

# Single-resource domestic-trade ratios: 1:1, N:1, and 1:N with N capped at 3.
# Larger or asymmetric multi-resource bundles are intentionally omitted to keep
# the action space finite; extend when the agent needs richer offers.
DOMESTIC_SINGLE_RESOURCE_RATIOS: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (2, 1),
    (3, 1),
    (1, 2),
    (1, 3),
)

_TWO_ONE_FOR_RESOURCE: dict[PortType, Resource] = {
    PortType.WOOD_TWO: Resource.WOOD,
    PortType.BRICK_TWO: Resource.BRICK,
    PortType.SHEEP_TWO: Resource.SHEEP,
    PortType.WHEAT_TWO: Resource.WHEAT,
    PortType.ORE_TWO: Resource.ORE,
}


def _best_maritime_ratio(
    state: GameState, player_id: PlayerID, give: Resource
) -> int:
    """Minimum number of ``give`` cards to return 1 from the bank (4:1, 3:1, or 2:1)."""
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


def legal_maritime_trades(state: GameState) -> list[MaritimeTradeAction]:
    if state.phase is not TurnPhase.MAIN or state.pending is not None:
        return []
    pid = state.current_player
    hand = state.players[pid].resources
    out: list[MaritimeTradeAction] = []
    for give in tradeable_resources():
        ratio = _best_maritime_ratio(state, pid, give)
        if hand.get(give, 0) < ratio:
            continue
        for receive in tradeable_resources():
            if receive is give:
                continue
            if state.bank.resources.get(receive, 0) < 1:
                continue
            out.append(
                MaritimeTradeAction(
                    player_id=pid, give=give, give_count=ratio, receive=receive
                )
            )
    return out


def _can_propose(state: GameState) -> bool:
    return state.phase is TurnPhase.MAIN and state.pending is None


def legal_domestic_turn(state: GameState) -> list[
    ProposeDomesticTradeAction
    | RespondDomesticTradeAction
    | ConfirmDomesticTradeAction
    | CancelDomesticTradeAction
]:
    """In-flight trade handling or empty when no offer is on the table."""
    p = state.pending
    pids = state.config.player_ids
    if not isinstance(p, DomesticTradePending):
        return []
    out: list = []
    for pid in pids:
        if pid == state.current_player:
            for oth in pids:
                if oth == state.current_player:
                    continue
                if (
                    p.responses.get(oth) is DomesticTradeState.ACCEPTED
                    and state.players[oth].can_afford(p.request)
                ):
                    out.append(
                        ConfirmDomesticTradeAction(
                            player_id=state.current_player, trade_with=oth
                        )
                    )
            out.append(
                CancelDomesticTradeAction(player_id=state.current_player)
            )
        else:
            if pid not in p.responses:
                out.append(
                    RespondDomesticTradeAction(
                        player_id=pid, response=DomesticTradeState.REJECTED
                    )
                )
                if state.players[pid].can_afford(p.request):
                    out.append(
                        RespondDomesticTradeAction(
                            player_id=pid, response=DomesticTradeState.ACCEPTED
                        )
                    )
    return out


def legal_propose_domestic_single_resource(
    state: GameState,
) -> list[ProposeDomesticTradeAction]:
    """
    Enumerate single-resource domestic openers across distinct resource pairs.

    Covered ratios (give_count : receive_count): 1:1, 2:1, 3:1, 1:2, 1:3.
    ``give_count`` is gated by the proposer's hand; the request side is not
    filtered by the responder's hand (the responder simply rejects if short).
    Multi-resource bundle offers are not enumerated here.
    """
    if not _can_propose(state):
        return []
    pid = state.current_player
    hand = state.players[pid].resources
    out: list[ProposeDomesticTradeAction] = []
    for a in tradeable_resources():
        held = hand.get(a, 0)
        for b in tradeable_resources():
            if a == b:
                continue
            for give_count, receive_count in DOMESTIC_SINGLE_RESOURCE_RATIOS:
                if held < give_count:
                    continue
                out.append(
                    ProposeDomesticTradeAction(
                        player_id=pid,
                        offer={a: give_count},
                        request={b: receive_count},
                    )
                )
    return out