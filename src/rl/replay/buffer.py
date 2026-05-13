"""Rollout storage with per-agent Generalized Advantage Estimation.

In single-agent PPO, GAE walks the trajectory backwards once. Catan
interleaves moves from four seats, so the standard recursion would mix
opponent rewards into the learner's advantages. We instead compute GAE
*per agent*: each seat's transitions form an in-order subsequence, the
recursion runs over that subsequence, and "next state" for agent X means
"X's next stored transition" — opponents' moves between X's turns are
skipped.

The buffer pre-allocates fixed-shape numpy arrays for the row data and
keeps obs/mask in 2-D arrays for fast slicing. After ``compute_advantages``,
``to_batches`` materialises torch tensors on demand.

Mutators ``add_terminal_reward`` and ``mark_done`` exist so the rollout
worker can retroactively credit the multi-agent terminal reward to every
seat's last transition (only the seat that *acted* at termination is hit
by the per-step reward function — see rl-014's design notes).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from domain.ids import PlayerID
from rl.replay.transition import Transition

__all__ = ["Transition", "TrajectoryBatch", "TrajectoryBuffer"]


@dataclass
class TrajectoryBatch:
    """One PPO minibatch sliced from a finalised :class:`TrajectoryBuffer`.

    All fields are float32 torch tensors (boolean for ``masks``, long for
    ``actions``). Advantages have already been per-agent-GAE'd; returns are
    ``advantages + values`` and serve as the value-head target.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    old_logps: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.obs.shape[0]


class TrajectoryBuffer:
    """Fixed-capacity rollout buffer with per-agent GAE.

    ``compute_advantages`` must be called once after the rollout finishes
    and before ``to_batches``; calling ``to_batches`` without computing
    advantages raises ``RuntimeError``.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._size = 0
        self._advantages_ready = False

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._masks = np.zeros((capacity, action_dim), dtype=np.bool_)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._logps = np.zeros(capacity, dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._agents = np.zeros(capacity, dtype=np.int64)
        self._advantages = np.zeros(capacity, dtype=np.float32)
        self._returns = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    # Capacity / introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, t: Transition) -> None:
        """Append a transition. Invalidates any previous advantage results."""
        if self._size >= self._capacity:
            raise IndexError(
                f"TrajectoryBuffer is full (capacity={self._capacity})"
            )
        i = self._size
        if t.obs.shape != (self._obs_dim,):
            raise ValueError(
                f"transition obs shape {t.obs.shape} != ({self._obs_dim},)"
            )
        if t.mask.shape != (self._action_dim,):
            raise ValueError(
                f"transition mask shape {t.mask.shape} != ({self._action_dim},)"
            )
        self._obs[i] = t.obs
        self._masks[i] = t.mask
        self._actions[i] = t.action
        self._logps[i] = t.logp
        self._values[i] = t.value
        self._rewards[i] = t.reward
        self._dones[i] = t.done
        self._agents[i] = int(t.agent)
        self._size += 1
        self._advantages_ready = False

    def add_terminal_reward(self, idx: int, reward: float) -> None:
        """Add ``reward`` to the transition at ``idx``.

        Used by the rollout worker at game termination to retroactively
        credit the multi-agent terminal reward to every seat's last
        transition. Additive (not overwriting) so an existing per-step
        reward on the terminal-step transition is preserved.
        """
        self._require_index(idx)
        self._rewards[idx] += float(reward)
        self._advantages_ready = False

    def mark_done(self, idx: int) -> None:
        """Mark the transition at ``idx`` as the agent's last in this episode."""
        self._require_index(idx)
        self._dones[idx] = True
        self._advantages_ready = False

    def clear(self) -> None:
        """Reset write cursor and invalidate advantages. Storage is reused."""
        self._size = 0
        self._advantages_ready = False

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        gamma: float,
        lam: float,
        last_values: dict[PlayerID, float],
    ) -> None:
        """Compute per-agent GAE in place over the live portion of the buffer.

        ``last_values`` provides V(s) bootstraps for each agent whose final
        in-buffer transition is non-terminal (i.e. the rollout was cut by
        capacity, not by episode end). Agents whose last transition is
        already ``done=True`` ignore this dict.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        indices_by_agent: dict[int, list[int]] = {}
        for i in range(self._size):
            indices_by_agent.setdefault(int(self._agents[i]), []).append(i)

        for agent_int, indices in indices_by_agent.items():
            agent_pid = PlayerID(agent_int)
            self._gae_one_agent(
                indices,
                gamma=gamma,
                lam=lam,
                bootstrap=float(last_values.get(agent_pid, 0.0)),
            )

        self._returns[: self._size] = (
            self._advantages[: self._size] + self._values[: self._size]
        )
        self._advantages_ready = True

    def _gae_one_agent(
        self,
        indices: list[int],
        gamma: float,
        lam: float,
        bootstrap: float,
    ) -> None:
        """Backward GAE recursion over one agent's subsequence of indices."""
        last_adv = 0.0
        # Walk in reverse over this agent's transitions.
        for k in range(len(indices) - 1, -1, -1):
            i = indices[k]
            done_i = bool(self._dones[i])
            if k == len(indices) - 1:
                next_value = 0.0 if done_i else bootstrap
                next_adv = 0.0
            else:
                i_next = indices[k + 1]
                next_value = float(self._values[i_next])
                next_adv = last_adv
            non_terminal = 0.0 if done_i else 1.0
            delta = (
                float(self._rewards[i])
                + gamma * next_value * non_terminal
                - float(self._values[i])
            )
            adv = delta + gamma * lam * non_terminal * next_adv
            self._advantages[i] = adv
            last_adv = adv

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def to_batches(
        self,
        batch_size: int,
        learner_agent: PlayerID | None = None,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Iterator[TrajectoryBatch]:
        """Yield :class:`TrajectoryBatch` minibatches.

        If ``learner_agent`` is given, only that agent's transitions are
        yielded — opponents in self-play are frozen and contribute no
        gradients. Advantages are normalised per emitted batch
        ``(adv - mean) / (std + 1e-8)`` happens in the PPO update, not
        here, so callers can run multiple epochs over the same buffer.
        """
        if not self._advantages_ready:
            raise RuntimeError(
                "compute_advantages must be called before to_batches"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if learner_agent is None:
            indices = np.arange(self._size, dtype=np.int64)
        else:
            indices = np.flatnonzero(
                self._agents[: self._size] == int(learner_agent)
            ).astype(np.int64)

        if indices.size == 0:
            return

        if shuffle:
            g = rng if rng is not None else np.random.default_rng()
            g.shuffle(indices)

        for start in range(0, indices.size, batch_size):
            chunk = indices[start : start + batch_size]
            yield TrajectoryBatch(
                obs=torch.from_numpy(self._obs[chunk]),
                actions=torch.from_numpy(self._actions[chunk]),
                masks=torch.from_numpy(self._masks[chunk]),
                old_logps=torch.from_numpy(self._logps[chunk]),
                advantages=torch.from_numpy(self._advantages[chunk]),
                returns=torch.from_numpy(self._returns[chunk]),
            )

    # ------------------------------------------------------------------
    # Read-only views (for tests and debugging)
    # ------------------------------------------------------------------

    def advantages_view(self) -> np.ndarray:
        """Return a view of the live advantages slice (read-only intent)."""
        return self._advantages[: self._size]

    def returns_view(self) -> np.ndarray:
        return self._returns[: self._size]

    def agents_view(self) -> np.ndarray:
        return self._agents[: self._size]

    def rewards_view(self) -> np.ndarray:
        return self._rewards[: self._size]

    def dones_view(self) -> np.ndarray:
        return self._dones[: self._size]

    def values_view(self) -> np.ndarray:
        return self._values[: self._size]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_index(self, idx: int) -> None:
        if not 0 <= idx < self._size:
            raise IndexError(
                f"index {idx} out of bounds for buffer of size {self._size}"
            )
