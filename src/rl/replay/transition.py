"""Single rollout step record stored in :class:`TrajectoryBuffer`.

A transition records everything PPO needs to reconstruct the policy
distribution and recompute advantages: the encoded observation, the action
the policy picked, the mask used to constrain it, the log-probability and
value the policy emitted, the scalar reward, the done flag, and the seat
that acted. Per-agent identity is essential because PPO's advantage
computation walks each agent's subsequence independently when the trajectory
interleaves moves from multiple seats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.ids import PlayerID

__all__ = ["Transition"]


@dataclass
class Transition:
    """One (s, a, r, ...) tuple emitted by :class:`RolloutWorker`.

    ``done`` is the *agent-level* done flag: True iff the game ended before
    this agent's next move. The rollout worker is responsible for setting
    this retroactively when a game terminates (see rl-014).
    """

    obs: np.ndarray
    action: int
    mask: np.ndarray
    logp: float
    value: float
    reward: float
    done: bool
    agent: PlayerID
