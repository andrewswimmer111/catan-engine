"""
Robber, 7s, discards, and stealing.

Discards use :class:`DiscardPending.cards_to_discard` for how many cards each
player must lose in a single :class:`DiscardResourcesAction`.
"""

from __future__ import annotations

from domain.actions.all_actions import (
    DiscardResourcesAction,
    MoveRobberAction,
    StealResourceAction,
)
from domain.enums import Resource, TurnPhase
from domain.enums import tradeable_resources
from domain.game.state import GameState
from domain.ids import PlayerID, TileID
from domain.turn.pending import DiscardPending, RobberMovePending, StealPending


def _floor_discard_count(hand_size: int) -> int:
    return hand_size // 2


def cards_to_discard_on_seven(state: GameState) -> dict[PlayerID, int]:
    """``player_id -> must discard this many`` for every player with a hand > 7."""
    out: dict[PlayerID, int] = {}
    for pid in state.config.player_ids:
        n = state.players[pid].resource_count()
        if n > 7:
            out[pid] = _floor_discard_count(n)
    return out


def _enumerate_discard_options(
    hand: dict[Resource, int], need: int
) -> list[dict[Resource, int]]:
    """All ``dict`` subsets of ``hand`` with total ``need`` cards."""
    if need == 0:
        return [{}]
    if need < 0:
        return []
    types = [r for r in tradeable_resources() if hand.get(r, 0) > 0]
    out: list[dict[Resource, int]] = []

    def rec(i: int, rem: int, cur: dict[Resource, int]) -> None:
        if rem == 0:
            out.append(dict(cur))
            return
        if i >= len(types) or rem < 0:
            return
        r = types[i]
        mx = min(hand.get(r, 0), rem)
        for k in range(mx + 1):
            if k:
                cur[r] = k
            rec(i + 1, rem - k, cur)
            if k:
                del cur[r]

    rec(0, need, {})
    return out


def legal_discard_actions(state: GameState) -> list[DiscardResourcesAction]:
    if state.phase is not TurnPhase.DISCARD:
        return []
    if not isinstance(state.pending, DiscardPending):
        return []
    pend = state.pending
    out: list[DiscardResourcesAction] = []
    for pid, need in pend.cards_to_discard.items():
        hand = state.players[pid].resources
        for opt in _enumerate_discard_options(hand, need):
            out.append(DiscardResourcesAction(player_id=pid, resources=dict(opt)))
    return out


def legal_robber_moves(state: GameState) -> list[MoveRobberAction]:
    if state.phase is not TurnPhase.MOVE_ROBBER:
        return []
    if not isinstance(state.pending, RobberMovePending):
        return []
    pid = state.current_player
    cur = state.occupancy.robber_tile
    return [
        MoveRobberAction(player_id=pid, tile_id=tid)
        for tid in state.topology.tiles
        if tid != cur
    ]


def players_adjacent_to_tile(
    state: GameState, tile_id: TileID, exclude: PlayerID
) -> frozenset[PlayerID]:
    """Players with a building on a vertex touching ``tile_id``, except ``exclude``."""
    found: set[PlayerID] = set()
    for vid, (owner, _bt) in state.occupancy.buildings.items():
        if owner == exclude:
            continue
        if tile_id in state.topology.vertices[vid].adjacent_tiles:
            if state.players[owner].resource_count() > 0:
                found.add(owner)
    return frozenset(found)


def legal_steal_actions(state: GameState) -> list[StealResourceAction]:
    if state.phase is not TurnPhase.STEAL:
        return []
    if not isinstance(state.pending, StealPending):
        return []
    pid = state.current_player
    return [
        StealResourceAction(player_id=pid, target_player_id=t)
        for t in state.pending.valid_targets
    ]
