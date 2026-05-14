"""Rule-based Catan agent — a strong baseline opponent for trained policies.

Phase dispatch lives here; the per-phase strategy lives in
:mod:`rl.agents._heuristic_rules`. :func:`heuristic_discard` is exported
separately so :class:`CatanEnv` can resolve the action encoder's discard
sentinel without instantiating an agent.
"""

from __future__ import annotations

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.ids import PlayerID
from rl.acting_player import acting_player
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
    """Deterministic rule-based agent.

    Takes no constructor arguments — every tie is broken by a stable rule
    (lowest id, highest pip, etc.), so an RNG would be unused state.
    """

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        if not legal:
            return None
        state = snap.state
        phase = state.phase
        me = acting_player(state)

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
