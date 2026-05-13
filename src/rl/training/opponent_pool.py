"""Self-play opponent pool — current learner + recent + historical checkpoints.

The pool exists so the learner doesn't always play the same opponent. Each
episode draws three opponents from a mix of three buckets:

- **current**: the live learner itself, used in *stochastic-play* mode and
  under ``torch.no_grad`` so opponent forward passes don't accumulate
  autograd graph. The underlying parameters stay trainable — only the
  inference side is grad-free.
- **recent**: a bounded FIFO deque of the last ``recent_size`` checkpoint
  paths. New snapshots evict the oldest entry.
- **historical**: a reservoir-sampled list of up to ``historical_size``
  paths. :meth:`promote_to_historical` is called by the trainer every K
  snapshots; it considers the most recent checkpoint as a candidate and
  applies classic reservoir sampling so the historical bucket remains a
  uniform random sample over all candidates ever considered.

Loaded opponents are :class:`PolicyAgent` instances with ``requires_grad=False``
on every parameter and ``stochastic_play=True`` so they sample from the
policy rather than playing greedily. They live in a small LRU cache keyed by
path so a checkpoint loaded once doesn't get reloaded every episode it's
drawn.
"""

from __future__ import annotations

import random
from collections import OrderedDict, deque
from pathlib import Path

from controller.agents import Agent
from rl.agents.policy_agent import PolicyAgent
from rl.training.checkpoint import load_checkpoint

__all__ = ["OpponentPool"]


_DEFAULT_LRU_SIZE: int = 4


class OpponentPool:
    """Holds opponent checkpoint paths and samples opponents per a configured mix."""

    def __init__(
        self,
        recent_size: int = 8,
        historical_size: int = 32,
        mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
        rng: random.Random | None = None,
        *,
        cache_size: int = _DEFAULT_LRU_SIZE,
    ) -> None:
        if recent_size <= 0:
            raise ValueError(f"recent_size must be positive, got {recent_size}")
        if historical_size <= 0:
            raise ValueError(f"historical_size must be positive, got {historical_size}")
        if any(w < 0 for w in mix):
            raise ValueError(f"mix weights must be non-negative, got {mix}")
        total = sum(mix)
        if total <= 0:
            raise ValueError(f"mix weights must sum to a positive value, got {mix}")
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")

        self._mix = (mix[0] / total, mix[1] / total, mix[2] / total)
        self._recent: deque[Path] = deque(maxlen=recent_size)
        self._historical: list[Path] = []
        self._historical_size = historical_size
        self._n_promoted = 0
        self._rng = rng if rng is not None else random.Random()
        self._cache: OrderedDict[Path, PolicyAgent] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._recent) + len(self._historical)

    @property
    def recent_paths(self) -> tuple[Path, ...]:
        return tuple(self._recent)

    @property
    def historical_paths(self) -> tuple[Path, ...]:
        return tuple(self._historical)

    @property
    def mix(self) -> tuple[float, float, float]:
        return self._mix

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_checkpoint(self, path: Path) -> None:
        """Register a freshly written checkpoint in the recent bucket.

        Eviction is FIFO at ``recent_size``. The historical bucket is only
        modified by :meth:`promote_to_historical`.
        """
        self._recent.append(Path(path))

    def promote_to_historical(self) -> None:
        """Consider the most recent checkpoint for the historical reservoir.

        Implements classic reservoir sampling: every call counts as one new
        candidate. If historical is below capacity the candidate is appended;
        otherwise it replaces a random slot with probability
        ``historical_size / n_promoted_so_far``. The result is a uniformly
        random sample over all checkpoints ever promoted.
        """
        if not self._recent:
            return
        candidate = self._recent[-1]
        self._n_promoted += 1
        if len(self._historical) < self._historical_size:
            self._historical.append(candidate)
            return
        j = self._rng.randrange(self._n_promoted)
        if j < self._historical_size:
            self._historical[j] = candidate

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_opponents(self, learner: PolicyAgent, n: int = 3) -> list[Agent]:
        """Draw ``n`` opponents independently per the configured mix.

        Each slot independently picks a bucket from ``mix`` and a checkpoint
        from that bucket. If the chosen bucket is empty, falls back across the
        other buckets in priority order recent → historical → current so the
        caller always gets ``n`` playable agents.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return [self._sample_one(learner) for _ in range(n)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_one(self, learner: PolicyAgent) -> Agent:
        bucket = self._choose_bucket()
        if bucket == "current":
            return _live_learner_opponent(learner)
        if bucket == "recent" and self._recent:
            return self._load_frozen(self._rng.choice(list(self._recent)))
        if bucket == "historical" and self._historical:
            return self._load_frozen(self._rng.choice(self._historical))
        # Fallback chain: recent → historical → current. Ensures the caller
        # always gets a playable agent even when the rolled bucket is empty.
        if self._recent:
            return self._load_frozen(self._rng.choice(list(self._recent)))
        if self._historical:
            return self._load_frozen(self._rng.choice(self._historical))
        return _live_learner_opponent(learner)

    def _choose_bucket(self) -> str:
        r = self._rng.random()
        if r < self._mix[0]:
            return "current"
        if r < self._mix[0] + self._mix[1]:
            return "recent"
        return "historical"

    def _load_frozen(self, path: Path) -> PolicyAgent:
        key = Path(path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        agent, _meta = load_checkpoint(key)
        _freeze(agent)
        agent.stochastic_play = True
        self._cache[key] = agent
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return agent


def _freeze(agent: PolicyAgent) -> None:
    """Disable gradients on every parameter — loaded opponents are read-only."""
    for p in agent.model.parameters():
        p.requires_grad_(False)


def _live_learner_opponent(learner: PolicyAgent) -> PolicyAgent:
    """Stochastic-mode sibling sharing the learner's model.

    The new agent reuses the same ``MLPPolicyValue`` and encoders — opponent
    forward passes already run under ``torch.no_grad`` inside
    :meth:`PolicyAgent.act`, so no graph accumulates even though the
    parameters remain trainable on the live learner.
    """
    return PolicyAgent(
        model=learner.model,
        action_encoder=learner.action_encoder,
        obs_encoder=learner.obs_encoder,
        device=learner.device,
        stochastic_play=True,
    )
