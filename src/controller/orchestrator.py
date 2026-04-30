"""
Drives non-human turns on demand.

Sits between the GUI (which steps on user gestures or wall-clock) and the
session: looks up the active player's :class:`~controller.agents.Agent`, asks
it to pick a legal action, and applies it. Human seats cause ``step_once`` to
return ``False`` so the GUI can fall back to direct user input.
"""

from __future__ import annotations

from controller.agents import Agent
from controller.session import GameSession
from domain.ids import PlayerID


class Orchestrator:
    def __init__(self, session: GameSession, agents: dict[PlayerID, Agent]) -> None:
        self._session = session
        self._agents = agents

    def set_agent(self, player_id: PlayerID, agent: Agent) -> None:
        self._agents[player_id] = agent

    def set_session(self, session: GameSession) -> None:
        self._session = session

    def step_once(self) -> bool:
        """Apply one action for the current player if their agent is non-human.

        Returns True if an action was applied, False if the active agent is
        human (choose() returned None) or the game is already terminal.
        """
        snap = self._session.current()
        if snap.state.is_terminal():
            return False
        pid = snap.state.current_player
        agent = self._agents.get(pid)
        if agent is None:
            return False
        legal = self._session.legal_actions()
        action = agent.choose(snap, legal)
        if action is None:
            return False
        self._session.apply(action)
        return True

    def run_until_human(self, max_steps: int = 1000) -> int:
        """Repeatedly call step_once() until a human turn or terminal state.

        Returns the number of steps taken.
        """
        steps = 0
        while steps < max_steps:
            if not self.step_once():
                break
            steps += 1
        return steps
