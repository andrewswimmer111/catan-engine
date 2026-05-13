"""Agent protocols for the RL layer.

Two protocols live here. :class:`Agent` is re-exported from
:mod:`controller.agents` and represents the engine-facing contract — anything
that exposes ``choose(snap, legal) -> Action | None`` plays in tournaments and
the GUI. :class:`RLAgent` is the training-side contract: a policy that, given
a flat observation and boolean action mask, samples an integer action index
and reports the diagnostics PPO needs (log-prob, value, entropy).

The trainable :class:`PolicyAgent` (rl-011) satisfies both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from controller.agents import Agent as Agent
from controller.session import GameSnapshot
from domain.actions.base import Action

__all__ = ["Agent", "ActStep", "RLAgent"]


@dataclass(frozen=True)
class ActStep:
    """Single-step output from :meth:`RLAgent.act`.

    Fields are plain Python scalars so the buffer can store them without
    dragging device-resident tensors into rollout-collection code paths.
    """

    action_idx: int
    logp: float
    value: float
    entropy: float


class RLAgent(Protocol):
    """Training-side contract for a stochastic policy with a value head.

    Implementations sample (or argmax, when ``deterministic``) over the legal
    action set defined by ``mask`` and return the diagnostics PPO needs to
    bookkeep the trajectory. The engine-facing ``choose`` method is provided
    by adapters (see :class:`rl.agents.policy_agent.PolicyAgent`) so the same
    object can drive both training and play.
    """

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> ActStep: ...

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None: ...
