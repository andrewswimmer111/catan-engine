"""Pairwise Elo tracker for multi-player Catan tournaments.

Elo is built for two-player zero-sum games; Catan is 4-player. The standard
conversion — used here — is to convert each 4-player game into ``C(4, 2) = 6``
pairwise outcomes (per-pair winner / loser / draw) and apply a standard Elo
update to each pair. Concretely: for a game with scores ``s_a > s_b``, the
pair ``(a, b)`` counts as a win for ``a``; if scores tie, it counts as a draw.

The wart
--------

This double-counts information from a single underlying observation — the
six pairwise updates from one game aren't independent. That understates
rating *variance*, so:

- Relative ordering ("is the new checkpoint better than the old one?")
  remains trustworthy.
- Absolute ratings (e.g. "this agent is at 1632 Elo") are not directly
  comparable to chess-style ratings.

This is good enough for the rl-019 evaluation loop; if absolute numbers
ever matter we'd swap in TrueSkill (which handles N-player natively).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["EloTracker", "EloConfig"]


@dataclass(frozen=True)
class EloConfig:
    """Elo update hyperparameters.

    ``k_factor`` is the standard chess-style K (rating change per game when
    scoring 100% against an equal-rated opponent). 32 is the FIDE value for
    blitz; we keep it here because the noise in 4-player Catan is large
    enough that smaller K's update too slowly to track real shifts.
    """

    k_factor: float = 32.0
    initial_rating: float = 1500.0


@dataclass
class EloTracker:
    """In-memory pairwise Elo ratings keyed by agent ID."""

    cfg: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, agents: list[str], scores: list[float]) -> None:
        """Run pairwise Elo updates across every (i, j) pair in this game.

        ``scores`` is per-agent in the same order as ``agents``. Pair
        outcomes are derived by comparing scores: higher score wins, equal
        is a draw. Typical 4-player game: one winner with score 1.0, three
        losers with score 0.0 — yields three wins (winner vs each loser)
        and three draws (loser vs loser).
        """
        if len(agents) != len(scores):
            raise ValueError(
                f"agents/scores length mismatch: {len(agents)} vs {len(scores)}"
            )
        if len(agents) < 2:
            raise ValueError(
                f"need at least 2 agents per update, got {len(agents)}"
            )
        # Snapshot pre-update ratings so all pair updates reference the same
        # baseline — otherwise pair-iteration order leaks into the result.
        # Deltas are accumulated and applied at the end so a single agent's
        # three pair updates compound rather than overwriting each other.
        snapshot = {a: self.rating(a) for a in agents}
        deltas: dict[str, float] = defaultdict(float)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ra = snapshot[agents[i]]
                rb = snapshot[agents[j]]
                outcome_a = _pair_outcome(scores[i], scores[j])
                expected_a = _expected_score(ra, rb)
                delta_a = self.cfg.k_factor * (outcome_a - expected_a)
                deltas[agents[i]] += delta_a
                # Zero-sum: a's gain is b's loss against equal-rated opponents,
                # and more generally pair updates conserve sum across the pair.
                deltas[agents[j]] -= delta_a
        for a in agents:
            self.ratings[a] = snapshot[a] + deltas[a]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def rating(self, agent: str) -> float:
        """Current rating for ``agent``, defaulting to the initial rating."""
        return self.ratings.get(agent, self.cfg.initial_rating)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k_factor": self.cfg.k_factor,
            "initial_rating": self.cfg.initial_rating,
            "ratings": dict(self.ratings),
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "EloTracker":
        data = json.loads(Path(path).read_text())
        cfg = EloConfig(
            k_factor=float(data.get("k_factor", 32.0)),
            initial_rating=float(data.get("initial_rating", 1500.0)),
        )
        return cls(cfg=cfg, ratings=dict(data.get("ratings", {})))


def _expected_score(ra: float, rb: float) -> float:
    """Standard Elo expected-score formula: 1 / (1 + 10**((rb-ra)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _pair_outcome(sa: float, sb: float) -> float:
    """1.0 if a beat b, 0.0 if a lost, 0.5 on equal scores."""
    if sa > sb:
        return 1.0
    if sa < sb:
        return 0.0
    return 0.5
