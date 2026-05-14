"""Periodic evaluation scheduler for the self-play trainer (rl-019).

The scheduler is a *passive* helper — the trainer (or a test harness) calls
:meth:`EvalScheduler.maybe_run` after each iteration, and the scheduler
decides whether the configured ``every_steps`` interval has elapsed. When
it has, the scheduler runs every benchmark callable, updates the Elo
tracker from per-game outcomes, and logs win-rate / Elo scalars through
the trainer's logger.

Benchmarks
----------

Each benchmark is a ``Callable[[PolicyAgent], TournamentResult]`` —
typically a closure capturing an env factory and the opponent setup. The
defaults shipped here cover the three benchmark families listed in the
spec: vs random, vs heuristic, vs sampled-from-pool.

Elo identity
------------

Per-eval, the learner is registered under a checkpoint-id-like string
``f"learner@step_{global_step}"`` and the opposing baseline under a static
ID (``"random"`` / ``"heuristic"`` / ``"pool"``). This means new learner
ids appear with each eval, which is exactly what we want — Elo trajectory
across training shows up as a chronological list of learner@step entries
in the persisted ratings file.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.elo import EloTracker
from rl.evaluation.metrics import GameStats, TournamentResult
from rl.evaluation.tournament import Tournament
from rl.replay.dataset import ReplayDataset, is_interesting
from rl.replay.recorder import play_episode
from rl.training.opponent_pool import OpponentPool
from rl.training.trainer import Trainer

__all__ = [
    "BenchmarkCallable",
    "EvalScheduler",
    "make_bench_vs_random",
    "make_bench_vs_heuristic",
    "make_bench_vs_pool",
]


BenchmarkCallable = Callable[[PolicyAgent], TournamentResult]


class EvalScheduler:
    """Fires periodic benchmarks and tracks Elo across training.

    The scheduler is intentionally passive — the trainer drives it via
    :meth:`maybe_run`. This keeps the eval loop decoupled from the training
    loop and lets external harnesses (notebooks, ad-hoc scripts) reuse the
    same scheduler against any trainer.
    """

    def __init__(
        self,
        trainer: Trainer,
        every_steps: int,
        benchmarks: dict[str, BenchmarkCallable],
        elo: EloTracker | None = None,
        ratings_path: Path | None = None,
        archive_root: Path | None = None,
        archive_n_games: int = 5,
        archive_env_factory: Callable[[int], CatanEnv] | None = None,
    ) -> None:
        if every_steps <= 0:
            raise ValueError(f"every_steps must be positive, got {every_steps}")
        self._trainer = trainer
        self._every_steps = every_steps
        self._benchmarks = dict(benchmarks)
        self._elo = elo if elo is not None else EloTracker()
        self._ratings_path = Path(ratings_path) if ratings_path else None
        # Negative sentinel so the first ``maybe_run`` after construction
        # fires when ``current_step >= 0``.
        self._last_eval_step: int = -every_steps

        self._archive: ReplayDataset | None = (
            ReplayDataset(archive_root) if archive_root is not None else None
        )
        self._archive_n_games = archive_n_games
        # The archiver needs to construct its own envs (the benchmarks own
        # their factories internally and don't expose them). Caller can
        # override with whatever factory matches the active game config.
        self._archive_env_factory = archive_env_factory or (
            lambda seed: CatanEnv(seed=seed)
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def elo(self) -> EloTracker:
        return self._elo

    @property
    def last_eval_step(self) -> int:
        return self._last_eval_step

    # ------------------------------------------------------------------
    # Drive
    # ------------------------------------------------------------------

    def maybe_run(self, current_step: int) -> dict[str, float] | None:
        """Run all benchmarks if ``every_steps`` has elapsed; else return None.

        Returns a flat metrics dict keyed by ``"{bench}/win_rate"`` and
        ``"{bench}/elo"`` — the same scalars logged to TB.
        """
        if current_step - self._last_eval_step < self._every_steps:
            return None
        self._last_eval_step = current_step

        learner_id = f"learner@step_{current_step}"
        learner = self._trainer.learner
        metrics: dict[str, float] = {}

        for name, bench in self._benchmarks.items():
            result = bench(learner)
            win_rate = _learner_win_rate(result)
            self._update_elo_from_result(result, learner_id, opponent_id=name)
            metrics[f"{name}/win_rate"] = win_rate
            metrics[f"{name}/elo"] = self._elo.rating(learner_id)

        for k, v in metrics.items():
            self._trainer.logger.log_scalar(f"eval/{k}", v, current_step)

        if self._ratings_path is not None:
            self._elo.save(self._ratings_path)

        if self._archive is not None:
            n_written = self._archive_some_games(current_step)
            metrics["archive/written"] = float(n_written)
            self._trainer.logger.log_scalar(
                "archive/written", n_written, current_step
            )

        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Archival
    # ------------------------------------------------------------------

    def _archive_some_games(self, current_step: int) -> int:
        """Play a small batch of vs-random games and archive interesting ones.

        Random opponents give the most diverse trajectories — checkpoint
        opponents tend to homogenise into a handful of opening patterns at
        the same training step. Returns the number of games actually
        written (i.e. that passed :func:`is_interesting`).
        """
        assert self._archive is not None
        written = 0
        base_seed = 900_000 + current_step  # distinct from benchmark seeds
        rng = random.Random(current_step)
        for i in range(self._archive_n_games):
            seed = base_seed + i
            env = self._archive_env_factory(seed)
            player_ids = list(env.state.config.player_ids)
            learner_seat = player_ids[0]
            opponents: dict[PlayerID, Agent] = {
                pid: RandomAgent(
                    random.Random(rng.randrange(2**32)), skip_proposals=True
                )
                for pid in player_ids[1:]
            }
            ep = play_episode(
                env=env,
                learner=self._trainer.learner,
                learner_seat=learner_seat,
                opponents=opponents,
                metadata={
                    "checkpoint_step": int(current_step),
                    "seed": int(seed),
                    "source": "eval_scheduler",
                },
            )
            if is_interesting(ep.final_vps, ep.winner):
                self._archive.write(ep)
                written += 1
        return written

    def _update_elo_from_result(
        self,
        result: TournamentResult,
        learner_id: str,
        opponent_id: str,
    ) -> None:
        """Per-game pairwise Elo updates with a 2-agent abstraction.

        We flatten the 4-player game into "learner vs opponent": the
        learner's score is 1.0 if it won this game, 0.0 otherwise. The
        opponent label aggregates the three baseline seats — a stand-in
        good enough for tracking learner improvement over training.
        """
        for game in result.games:
            learner_seat = _learner_seat(game)
            sa = 1.0 if game.winner == learner_seat else 0.0
            sb = 1.0 - sa if game.winner is not None else 0.0
            self._elo.update([learner_id, opponent_id], [sa, sb])


# ----------------------------------------------------------------------
# Default benchmark factories
# ----------------------------------------------------------------------


def make_bench_vs_random(
    env_factory: Callable[[int], CatanEnv],
    n_games: int = 20,
    base_seed: int = 100_000,
    *,
    skip_proposals: bool = True,
) -> BenchmarkCallable:
    """Closure: learner at seat 1 vs three RandomAgent seats."""

    def bench(learner: PolicyAgent) -> TournamentResult:
        agents = _learner_plus_three(
            learner,
            opponent_factory=lambda rng: RandomAgent(rng, skip_proposals=skip_proposals),
            seed=base_seed,
        )
        return Tournament(env_factory).play(
            agents, n_games=n_games, base_seed=base_seed
        )

    return bench


def make_bench_vs_heuristic(
    env_factory: Callable[[int], CatanEnv],
    n_games: int = 20,
    base_seed: int = 200_000,
) -> BenchmarkCallable:
    """Closure: learner at seat 1 vs three HeuristicAgent seats."""

    def bench(learner: PolicyAgent) -> TournamentResult:
        agents = _learner_plus_three(
            learner,
            opponent_factory=lambda _rng: HeuristicAgent(),
            seed=base_seed,
        )
        return Tournament(env_factory).play(
            agents, n_games=n_games, base_seed=base_seed
        )

    return bench


def make_bench_vs_pool(
    env_factory: Callable[[int], CatanEnv],
    pool: OpponentPool,
    n_games: int = 20,
    base_seed: int = 300_000,
) -> BenchmarkCallable:
    """Closure: learner at seat 1 vs three opponents sampled from ``pool``.

    Re-samples the three opponents *once* at benchmark-call time (not per
    game). This keeps the run reproducible at a given ``base_seed`` while
    still exercising the pool's distribution.
    """

    def bench(learner: PolicyAgent) -> TournamentResult:
        opponents = pool.sample_opponents(learner, n=3)
        seats = [PlayerID(i) for i in range(1, 5)]
        learner_seat = seats[0]
        agents: dict[PlayerID, Agent] = {learner_seat: learner}
        for pid, opp in zip(seats[1:], opponents):
            agents[pid] = opp
        return Tournament(env_factory).play(
            agents, n_games=n_games, base_seed=base_seed
        )

    return bench


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _learner_plus_three(
    learner: PolicyAgent,
    opponent_factory: Callable[[random.Random], Agent],
    seed: int,
) -> dict[PlayerID, Agent]:
    """Build a {seat → agent} dict with the learner at seat 1 and three
    freshly-seeded opponents at the rest."""
    seats = [PlayerID(i) for i in range(1, 5)]
    rng = random.Random(seed)
    agents: dict[PlayerID, Agent] = {seats[0]: learner}
    for pid in seats[1:]:
        agents[pid] = opponent_factory(random.Random(rng.randrange(2**32)))
    return agents


def _learner_win_rate(result: TournamentResult) -> float:
    """Win rate at seat 1 — the canonical learner seat in benchmark setups."""
    learner_seat = PlayerID(1)
    return result.win_rates.get(learner_seat, 0.0)


def _learner_seat(_game: GameStats) -> PlayerID:
    """The benchmark setups put the learner at seat 1; no per-game variation."""
    return PlayerID(1)
