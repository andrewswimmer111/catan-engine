"""Composite Agent that splits placement decisions from in-game play.

A :class:`PlacementOverrideAgent` delegates moves during the two opening-
placement phases (:attr:`TurnPhase.INITIAL_SETTLEMENT` and
:attr:`TurnPhase.INITIAL_ROAD`) to a dedicated ``placement_agent`` and
routes every other phase to the wrapped ``main_agent``. Both delegates
satisfy the engine-facing :class:`Agent` protocol; the composite is itself
an :class:`Agent`, so it slots into :class:`rl.evaluation.tournament.Tournament`
and the CLI eval path with no changes to existing call sites.

The wrapper is a *diagnostic tool*: it isolates "how much of the learner's
performance gap is opening-quality?" by running the same trained policy
with a known-good opening. It does not touch the training-side
``act(obs, mask)`` API — RL rollouts drive the learner through ``act``
directly, so wrapping a :class:`PolicyAgent` with this composite affects
only ``choose``-driven evaluation / play paths.
"""

from __future__ import annotations

from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.enums import TurnPhase
from rl.agents.base import Agent

__all__ = ["PLACEMENT_PHASES", "PlacementOverrideAgent"]


PLACEMENT_PHASES: frozenset[TurnPhase] = frozenset(
    {TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD}
)


class PlacementOverrideAgent:
    """Agent that delegates opening placements to a separate strategy.

    During :data:`PLACEMENT_PHASES` the call routes to ``placement_agent``;
    every other phase falls through to ``main_agent``. The wrapper holds no
    state of its own.
    """

    def __init__(self, main_agent: Agent, placement_agent: Agent) -> None:
        self._main = main_agent
        self._placement = placement_agent

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        if snap.state.phase in PLACEMENT_PHASES:
            return self._placement.choose(snap, legal)
        return self._main.choose(snap, legal)

    @property
    def main_agent(self) -> Agent:
        return self._main

    @property
    def placement_agent(self) -> Agent:
        return self._placement
