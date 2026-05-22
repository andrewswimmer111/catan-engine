"""Per-seat value target at stalemate-terminated game ends.

Used by both the AlphaZero self-play pipeline (to produce the value
target labels for the buffer) and by :mod:`rl.search.mcts` (to back up
the same value when search reaches a terminal stalemate leaf). The
two must use the same formula — if MCTS backs up a different value
than the network is trained to predict, the resulting Q estimates
fight the value head's own outputs and PUCT degrades.

Why this exists
---------------

The locked-in Phase 3 design used a single scalar (``-0.25``) for
every seat at every stalemate. Run #AZ-1 falsified that choice: the
constant target has zero variance, so the value head collapsed onto
the constant (``value_loss → 0.000`` by iter 3), MCTS leaves all
evaluated to the same number, PUCT lost its tactical signal, and the
policy bootstrap broke. The fix is to give the target *variance* —
stalemates with different VP distributions need to produce different
training targets so the value head has something to learn from.

Shapes
------

* ``"flat"`` — every seat gets ``flat_value``. The legacy AZ scalar
  target; kept for back-compat and as a sanity-check baseline so we
  can A/B against the constant-target regime if needed.
* ``"vp_linear"`` (default) — each seat lands in ``[low, high]``
  (default ``[-0.5, -0.1]``) by averaging:

  - **Rank score**: competitive rank, where the leader's score is 1
    and last place is 0. Ties on VP share the same score.
  - **VP score**: ``clip(vp_i / 10, 0, 1)``. Normalised to the
    winning-threshold VP count.

  Combined score (``0.5 * rank + 0.5 * vp``) is mapped linearly into
  ``[low, high]``. The leader of a 9-VP stalemate lands near ``high``;
  a 0-VP last-place seat lands at ``low``. **Actual wins still pay
  +1.0**, well above the band — preserving the
  ``max(stalemate_target) < min(win_target)`` invariant that keeps
  the policy from converging on "stalemate-as-leader" as a substitute
  for winning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from domain.game.state import GameState
from domain.rules.victory import compute_victory_points

__all__ = ["StalemateShape", "StalemateValueConfig"]


StalemateShape = Literal["flat", "vp_linear"]


@dataclass(frozen=True)
class StalemateValueConfig:
    """How to compute the per-seat value vector at a stalemate.

    Both :class:`rl.training.self_play.SelfPlayConfig` and
    :class:`rl.search.mcts.MCTSConfig` carry one of these. Construct
    it once at the top of a training run and pass the same instance to
    both, so the network's training target and MCTS's terminal backup
    are identical by construction.

    The MCTS defaults (constructed inside :class:`MCTSConfig`) pin
    ``shape="flat"`` with ``flat_value=0.0`` so inference-time MCTS
    keeps its historical zero-draw behaviour for callers that don't
    explicitly opt into a different target.
    """

    shape: StalemateShape = "vp_linear"
    flat_value: float = -0.25
    """Constant target used when ``shape == 'flat'``. Ignored otherwise."""

    low: float = -0.5
    """Bottom of the ``vp_linear`` band (worst-stalemate-seat target)."""

    high: float = -0.1
    """Top of the ``vp_linear`` band (best-stalemate-seat target)."""

    def __post_init__(self) -> None:
        if self.shape not in ("flat", "vp_linear"):
            raise ValueError(f"unknown stalemate shape: {self.shape!r}")
        if self.shape == "vp_linear" and self.high < self.low:
            raise ValueError(
                f"stalemate high ({self.high}) must be >= low ({self.low})"
            )

    def compute(self, state: GameState, n_players: int) -> np.ndarray:
        """Return the per-seat stalemate target as a ``float32`` vector.

        Index order matches ``state.config.player_ids``; the caller is
        responsible for any seat rotation (e.g. AZ rotates the acting
        seat to slot 0 for training targets).
        """
        if self.shape == "flat":
            return np.full(n_players, self.flat_value, dtype=np.float32)
        vps = np.fromiter(
            (
                compute_victory_points(state, pid)
                for pid in state.config.player_ids
            ),
            dtype=np.float64,
            count=n_players,
        )
        return self._vp_linear(vps).astype(np.float32, copy=False)

    def _vp_linear(self, vps: np.ndarray) -> np.ndarray:
        """Map ``vps`` (shape ``(n,)``) into ``[low, high]`` per the docstring formula."""
        n = vps.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        # Competitive rank: count of seats with strictly greater VP.
        # Ties (same VP) → same count → identical target. Stable and
        # cheap; the O(n^2) is fine for n=4.
        ranks = np.array(
            [int((vps > v).sum()) for v in vps], dtype=np.float64
        )
        denom = max(n - 1, 1)
        rank_score = 1.0 - ranks / denom  # in [0, 1]
        # Cap VP score at the winning threshold; a 12 VP late-game
        # stalemate (e.g. via dev cards then truncation) shouldn't
        # score higher than a 10 VP one.
        vp_score = np.clip(vps / 10.0, 0.0, 1.0)
        combined = 0.5 * rank_score + 0.5 * vp_score
        return self.low + (self.high - self.low) * combined
