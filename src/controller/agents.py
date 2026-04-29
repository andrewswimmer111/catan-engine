"""
Agent dispatch and action selection for AI-controlled players.

Maps each AI player seat to a policy (random, heuristic, or model-backed) and
delegates to that policy to choose a legal action on each turn, keeping
strategy logic decoupled from session management.
"""

from __future__ import annotations

import random
from typing import Protocol

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.actions.base import Action

__all__ = ["Agent", "HumanAgent", "ScriptedAgent"]


class Agent(Protocol):
    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None: ...


class HumanAgent:
    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        return None


class ScriptedAgent:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        end_turn = [a for a in legal if isinstance(a, A.EndTurnAction)]
        proposals = {A.ProposeDomesticTradeAction, A.MaritimeTradeAction}
        non_trade = [a for a in legal if type(a) not in proposals]
        if end_turn and self._rng.random() < 0.35:
            return end_turn[0]
        if non_trade:
            return self._rng.choice(non_trade)
        return self._rng.choice(legal)
