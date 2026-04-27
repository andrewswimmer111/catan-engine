"""
Development card play legals (draw timing, one play per main turn, VP never played).
"""

from __future__ import annotations

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import DevCardType, Resource, TurnPhase, tradeable_resources
from domain.game.state import GameState


def _turn_playable(turn_bought: int, state: GameState) -> bool:
    """Cards bought on turn T may be played from turn T+1 onward."""
    return state.turn_number > turn_bought


def _year_of_plenty_legal(
    state: GameState, r1: Resource, r2: Resource
) -> bool:
    b = state.bank
    t = sum(b.resources.get(r, 0) for r in tradeable_resources())
    if t == 0:
        return False
    if t == 1:
        for r in tradeable_resources():
            if b.resources.get(r, 0) == 1:
                return r1 is r2 and r1 is r
        return False
    if r1 is r2:
        return b.resources.get(r1, 0) >= 2
    return b.resources.get(r1, 0) >= 1 and b.resources.get(r2, 0) >= 1


def legal_dev_card_plays(state: GameState) -> list[Action]:
    """
    In-hand, timing, and at-most-once-per-turn (except VP, which is never
    *played* as an action; it counts silently).
    """
    pid = state.current_player
    p = state.players[pid]
    if p.has_played_dev_card_this_turn:
        return []
    if state.phase in (TurnPhase.ROLL, TurnPhase.MAIN, TurnPhase.BUILD_ROADS):
        pass
    else:
        return []
    if state.pending is not None:
        return []

    out: list[Action] = []
    have_knight = False
    have_rb = False
    have_yop = False
    have_m = False
    for card, turn_bought in p.dev_cards_in_hand:
        if card is DevCardType.VICTORY_POINT:
            continue
        if not _turn_playable(turn_bought, state):
            continue
        if card is DevCardType.KNIGHT:
            if state.phase in (TurnPhase.ROLL, TurnPhase.MAIN) and not have_knight:
                have_knight = True
        elif card is DevCardType.ROAD_BUILDING and state.phase is TurnPhase.MAIN:
            if not have_rb:
                have_rb = True
        elif card is DevCardType.YEAR_OF_PLENTY and state.phase is TurnPhase.MAIN:
            if not have_yop:
                have_yop = True
        elif card is DevCardType.MONOPOLY and state.phase is TurnPhase.MAIN:
            if not have_m:
                have_m = True
    if have_knight:
        out.append(A.PlayKnightAction(player_id=pid))
    if have_rb:
        out.append(A.PlayRoadBuildingAction(player_id=pid))
    if have_yop:
        seen: set[tuple[Resource, Resource]] = set()
        for r1 in tradeable_resources():
            for r2 in tradeable_resources():
                if not _year_of_plenty_legal(state, r1, r2):
                    continue
                if (r1, r2) in seen:
                    continue
                seen.add((r1, r2))
                out.append(
                    A.PlayYearOfPlentyAction(
                        player_id=pid, resource1=r1, resource2=r2
                    )
                )
    if have_m:
        for r in tradeable_resources():
            out.append(A.PlayMonopolyAction(player_id=pid, resource=r))
    return out
