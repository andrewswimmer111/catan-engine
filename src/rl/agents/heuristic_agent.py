"""Rule-based Catan agent — a strong baseline opponent for trained policies.

Phase dispatch lives here; the per-phase strategy lives in
:mod:`rl.agents._heuristic_rules`. :func:`heuristic_discard` is exported
separately so :class:`CatanEnv` can resolve the action encoder's discard
sentinel without instantiating an agent.
"""

from __future__ import annotations

import random

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending
from rl.agents._heuristic_rules import (
    best_road,
    choose_discard,
    choose_main,
    choose_robber_move,
    choose_setup_road,
    choose_setup_settlement,
    choose_steal_target,
    domestic_trade_fallback,
)

__all__ = ["HeuristicAgent", "heuristic_discard"]


def heuristic_discard(
    state: GameState, player_id: PlayerID
) -> A.DiscardResourcesAction:
    """Resolve a discard for ``player_id`` — delegated by ``CatanEnv``."""
    return choose_discard(state, player_id)


class HeuristicAgent:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        if not legal:
            return None
        state = snap.state
        phase = state.phase
        me = _whoami(state)

        if phase is TurnPhase.INITIAL_SETTLEMENT:
            settlements = [a for a in legal if isinstance(a, A.PlaceSettlementAction)]
            if settlements:
                return choose_setup_settlement(settlements, state)
        elif phase is TurnPhase.INITIAL_ROAD:
            roads = [a for a in legal if isinstance(a, A.PlaceRoadAction)]
            if roads:
                return choose_setup_road(roads, state)
        elif phase is TurnPhase.DISCARD:
            return heuristic_discard(state, me)
        elif phase is TurnPhase.MOVE_ROBBER:
            moves = [a for a in legal if isinstance(a, A.MoveRobberAction)]
            if moves:
                return choose_robber_move(moves, state, me)
        elif phase is TurnPhase.STEAL:
            steals = [a for a in legal if isinstance(a, A.StealResourceAction)]
            if steals:
                return choose_steal_target(steals, state)
        elif phase is TurnPhase.ROLL:
            knights = [a for a in legal if isinstance(a, A.PlayKnightAction)]
            if knights:
                return knights[0]
            for a in legal:
                if isinstance(a, A.RollDiceAction):
                    return a
        elif phase is TurnPhase.BUILD_ROADS:
            roads = [a for a in legal if isinstance(a, A.BuildRoadAction)]
            pick = best_road(state, roads)
            if pick is not None:
                return pick
        elif phase is TurnPhase.MAIN:
            choice = choose_main(state, legal, me)
            if choice is not None:
                return choice

        fb = domestic_trade_fallback(legal)
        if fb is not None:
            return fb
        return legal[0]


def _whoami(state: GameState) -> PlayerID:
    if state.phase is TurnPhase.DISCARD and isinstance(state.pending, DiscardPending):
        return next(iter(state.pending.cards_to_discard))
    return state.current_player
