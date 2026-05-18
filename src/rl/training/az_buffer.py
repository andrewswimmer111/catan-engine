"""Bounded replay buffer for AlphaZero self-play transitions.

A single :class:`AZReplayBuffer` instance holds the most recent
``capacity`` :class:`rl.training.self_play.SelfPlayTransition` records
across self-play games, evicting the oldest in FIFO order when the
bound is exceeded. :meth:`sample` draws a mini-batch *with replacement*
uniformly over the buffer's current contents — the AlphaZero
convention; sampling without replacement would systematically
under-represent recent games once the buffer fills up.

:meth:`sample` returns a :class:`AZBatch` of pre-stacked numpy arrays
ready for ``torch.as_tensor(..., device=...)``. The buffer itself stays
device-agnostic and CPU-resident: a 100k-transition buffer at
``GRAPH_OBS_SHAPE[0]=1704`` floats is ~680 MB on CPU; pushing it onto
an accelerator would crowd out the model.

v1 warts (documented per project convention):

* **Not thread- or process-safe.** :func:`extend` and :meth:`sample`
  rely on the underlying ``collections.deque`` being mutated from one
  thread. The (deferred) parallel-self-play worker setup in az-009 will
  need a queue or lock around this.
* **No persistence.** A warm-start (``--init-from``) starts with an
  empty buffer; the first iteration's training step sees only the
  current iteration's self-play data. This is the standard AZ choice;
  buffer persistence is a future ergonomic, not a correctness issue.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rl.training.self_play import SelfPlayTransition

__all__ = ["AZReplayBuffer", "AZBatch"]


@dataclass(frozen=True)
class AZBatch:
    """One stacked mini-batch ready for the trainer's forward pass.

    Shapes:

    * ``obs`` — ``(B, obs_dim)``
    * ``action_mask`` — ``(B, ACTION_SPACE_SIZE)``, boolean
    * ``policy_target`` — ``(B, ACTION_SPACE_SIZE)``, sums to 1 per row
    * ``value_target`` — ``(B, n_players)``, viewer-as-slot-0 (the same
      convention :class:`rl.models.gnn.GNNPolicyValue` outputs in
      ``value_kind="vector"`` mode, so the value loss is direct MSE).
    """

    obs: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    value_target: np.ndarray


class AZReplayBuffer:
    """Bounded FIFO replay buffer of :class:`SelfPlayTransition` records."""

    def __init__(
        self,
        capacity: int,
        rng: random.Random | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive (got {capacity})")
        self._capacity = capacity
        # ``deque(maxlen=N)`` gives O(1) append + automatic FIFO eviction.
        # Random access for sampling is O(k) per draw (deque indexing is
        # O(n) worst-case, but ``random.choices`` only does ``k`` indexings
        # so for ``k=128`` the cost is dominated by tensor stacking, not
        # buffer access).
        self._items: deque[SelfPlayTransition] = deque(maxlen=capacity)
        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def extend(self, transitions: Iterable[SelfPlayTransition]) -> None:
        """Append ``transitions`` to the buffer, evicting oldest first."""
        self._items.extend(transitions)

    def append(self, transition: SelfPlayTransition) -> None:
        """Append a single transition."""
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> AZBatch:
        """Draw ``batch_size`` transitions uniformly at random with replacement.

        With-replacement sampling matches the AlphaZero convention — the
        buffer is a stand-in for an infinite stream, and each gradient
        step should be (approximately) an i.i.d. sample of recent
        experience. Drawing without replacement systematically
        under-weights the most recent games whenever the buffer is
        large.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size})")
        if not self._items:
            raise ValueError("cannot sample from an empty buffer")

        picked = self._rng.choices(self._items, k=batch_size)
        return _stack_transitions(picked)


def _stack_transitions(transitions: list[SelfPlayTransition]) -> AZBatch:
    """Stack a list of transitions into the per-field numpy arrays of an :class:`AZBatch`."""
    obs = np.stack([t.obs for t in transitions])
    masks = np.stack([t.action_mask for t in transitions])
    policy = np.stack([t.mcts_policy for t in transitions])
    value = np.stack([t.value_target for t in transitions])
    return AZBatch(
        obs=obs,
        action_mask=masks,
        policy_target=policy,
        value_target=value,
    )
