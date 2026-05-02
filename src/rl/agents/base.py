from __future__ import annotations

from typing import Protocol

from controller.agents import Agent as Agent
from controller.session import GameSnapshot
from domain.actions.base import Action

__all__ = ["Agent", "RLAgent"]


class RLAgent(Protocol):
    """Placeholder for the trainable agent interface (fleshed out in rl-011)."""

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None: ...
