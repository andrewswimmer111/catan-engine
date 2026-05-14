"""End-to-end PPO trainer: collect rollouts, run updates, log, snapshot.

The trainer is built around three callables / objects provided at construction:

- ``env_factory(seed) -> CatanEnv`` — produces a fresh env per episode so
  reseeding is deterministic and the four seats are always in the same
  config order. The learner's *seat assignment* is rotated each episode by
  the trainer (see below), not by the factory.
- ``opponent_pool: OpponentPool`` — source of opponents. The trainer samples
  ``n_opponents`` agents per episode and assigns them to the non-learner
  seats. With an empty pool every slot falls back to the live learner — the
  natural cold start for self-play.
- A :class:`PolicyAgent` for the learner — the only object whose weights
  are updated.

Seat rotation
-------------

PPO is sensitive to seat overfitting if the learner always plays seat 1
(setup-phase first-pick changes positional value). On each episode start
the trainer samples a learner seat uniformly at random; the worker
records every learner transition with the seat's PlayerID so per-agent
GAE naturally segments by episode boundary.

Self-play snapshots
-------------------

Every ``cfg.snapshot_every`` env steps the trainer writes a versioned
checkpoint (see :mod:`rl.training.checkpoint`) to ``snapshot_dir`` and
hands the path to the opponent pool's recent bucket. Every
``cfg.promote_every_n_snapshots`` snapshots one entry is promoted into
the historical reservoir. Snapshots are disabled when ``snapshot_dir`` is
``None``.

PPO update over the rollout
---------------------------

After the buffer fills, the trainer calls ``compute_advantages`` once and
hands the entire rollout as a single :class:`TrajectoryBatch` to
:func:`ppo_update`, which does its own shuffled minibatching internally.
Because opponents aren't stored, every transition in the buffer belongs
to the learner — no per-seat filtering is needed.

Vectorised rollouts
-------------------

Setting ``cfg.num_envs > 1`` switches the rollout from one in-process
:class:`RolloutWorker` to a :class:`SubprocVecEnv` of that many parallel
workers driven by a :class:`VecRolloutWorker`. The vec env is built lazily
on the first iteration and reused — subprocess spawn is expensive so we
pay it once. Each subprocess gets its opponent setup and learner seat
from the ``vec_factory`` callable (see :mod:`rl.training.vec_factories`),
which derives the seat from its incoming seed; with ``num_envs >= 4``
the four seats are exercised in parallel.

The vec path **does not** consult :class:`OpponentPool` for opponents —
the pool lives in the main process and pool-loaded agents aren't
picklable across the subprocess boundary. Snapshots are still written
into the pool so a downstream single-env eval / replay step can see
them, but opponent diversity in vec mode comes from the factory.

Eval
----

If ``eval_scheduler`` is set (via the constructor arg or the property),
the trainer calls ``scheduler.maybe_run(global_step)`` after every
iteration and skips its built-in vs-random ``_evaluate``. The scheduler
owns its cadence (``every_steps``), benchmark mix, Elo state, and replay
archive — the trainer is just the clock.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE
from rl.encoding.observation import OBS_LAYOUT_VERSION, OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.replay.buffer import TrajectoryBuffer
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    ModelArch,
    compute_config_hash,
    save_checkpoint,
)
from rl.training.config import TrainConfig
from rl.training.opponent_pool import OpponentPool
from rl.training.ppo import ppo_update
from rl.training.rollout import RolloutWorker
from rl.training.vec_env import SubprocVecEnv, VecEnvFactory
from rl.training.vec_rollout import VecRolloutWorker
from rl.utils.logging import NoOpLogger, TBLogger, make_logger

if TYPE_CHECKING:
    # rl.evaluation.scheduler imports Trainer at module level, so we avoid the
    # cycle by referring to it only under TYPE_CHECKING here.
    from rl.evaluation.scheduler import EvalScheduler

__all__ = ["Trainer"]


class Trainer:
    """Drives ``collect → compute_advantages → ppo_update`` to convergence."""

    def __init__(
        self,
        env_factory: Callable[[int], CatanEnv],
        learner: PolicyAgent,
        opponent_pool: OpponentPool,
        cfg: TrainConfig,
        log_dir: str | Path | None = None,
        snapshot_dir: str | Path | None = None,
        vec_factory: VecEnvFactory | None = None,
        eval_scheduler: "EvalScheduler | None" = None,
    ) -> None:
        if cfg.num_envs > 1 and vec_factory is None:
            raise ValueError(
                f"cfg.num_envs={cfg.num_envs} requires a vec_factory; pass "
                "rl.training.vec_factories.random_opponents_factory or a custom one"
            )
        self._env_factory = env_factory
        self._learner = learner
        self._opponent_pool = opponent_pool
        self._cfg = cfg
        self._vec_factory = vec_factory
        self._eval_scheduler: "EvalScheduler | None" = eval_scheduler

        # Discover player seats by spinning up a throwaway env. Re-using the
        # factory here keeps "what seats exist" out of TrainConfig.
        probe_env = env_factory(cfg.seed)
        self._player_ids: list[PlayerID] = list(probe_env.state.config.player_ids)

        self._rng = random.Random(cfg.seed)
        self._buffer = TrajectoryBuffer(
            capacity=cfg.rollout_steps,
            obs_dim=OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
        )
        self._optimizer = torch.optim.Adam(
            self._learner.model.parameters(), lr=cfg.ppo.lr
        )
        self._logger: TBLogger | NoOpLogger = make_logger(log_dir)
        self._global_step = 0
        self._iteration = 0
        self._seed_counter = cfg.seed

        self._snapshot_dir: Path | None = Path(snapshot_dir) if snapshot_dir else None
        if self._snapshot_dir is not None:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot_step: int = 0
        self._n_snapshots: int = 0
        self._config_hash: str = compute_config_hash(cfg)
        self._model_arch = ModelArch(
            obs_dim=OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            hidden=tuple(learner.model.hidden),
        )

        # Vec-env path is constructed lazily on the first rollout so a Trainer
        # with cfg.num_envs > 1 doesn't spawn subprocesses just by being created
        # (matters for tests that build a trainer to inspect config).
        self._vec_env: SubprocVecEnv | None = None
        self._vec_worker: VecRolloutWorker | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def opponent_pool(self) -> OpponentPool:
        return self._opponent_pool

    @property
    def learner(self) -> PolicyAgent:
        return self._learner

    @property
    def logger(self) -> TBLogger | NoOpLogger:
        return self._logger

    @property
    def eval_scheduler(self) -> "EvalScheduler | None":
        return self._eval_scheduler

    @eval_scheduler.setter
    def eval_scheduler(self, scheduler: "EvalScheduler | None") -> None:
        """Hand the trainer an :class:`EvalScheduler` to drive in-training eval.

        When set, the trainer calls ``scheduler.maybe_run(global_step)`` after
        every iteration and skips its built-in ``_evaluate`` (vs-random only).
        The scheduler owns its own cadence via ``every_steps``.
        """
        self._eval_scheduler = scheduler

    def train(self, total_steps: int) -> None:
        """Train until ``self.global_step`` reaches ``total_steps``."""
        last_eval_step = -self._cfg.eval_every  # force eval on first iter
        zero_win_streak = 0
        try:
            while self._global_step < total_steps:
                rollout_summary, bootstraps = self._collect_rollout_step()
                self._buffer.compute_advantages(
                    gamma=self._cfg.gamma,
                    lam=self._cfg.gae_lambda,
                    last_values=bootstraps,
                )
                update_metrics = self._ppo_step()
                self._log_iteration(rollout_summary, update_metrics)
                self._maybe_print_heartbeat(rollout_summary, update_metrics)
                zero_win_streak = self._check_watchdog(rollout_summary, zero_win_streak)
                self._maybe_snapshot()
                if self._eval_scheduler is not None:
                    self._eval_scheduler.maybe_run(self._global_step)
                elif (
                    self._cfg.eval_every > 0
                    and self._global_step - last_eval_step >= self._cfg.eval_every
                ):
                    self._evaluate()
                    last_eval_step = self._global_step
                self._iteration += 1
            self._logger.flush()
        finally:
            self.close()

    def _maybe_print_heartbeat(
        self,
        rollout_summary: dict[str, float],
        update_metrics: dict[str, float],
    ) -> None:
        """One-line stdout summary every ``cfg.print_every`` iterations.

        Catches "model not learning" cases without waiting for the next
        eval cadence — much cheaper than scraping TB to find a flat curve.
        """
        if self._cfg.print_every <= 0:
            return
        if self._iteration % self._cfg.print_every != 0:
            return
        step_k = self._global_step / 1000.0
        win_rate = rollout_summary.get("rollout/win_rate", 0.0)
        ret_mean = rollout_summary.get("rollout/return_mean", 0.0)
        episodes = int(rollout_summary.get("rollout/episodes", 0))
        entropy = update_metrics.get("entropy", float("nan"))
        kl = update_metrics.get("approx_kl", float("nan"))
        print(
            f"[train iter={self._iteration} step={step_k:.0f}k] "
            f"win_rate={win_rate:.3f} ret_mean={ret_mean:+.3f} "
            f"ep={episodes} entropy={entropy:.3f} kl={kl:.4f}",
            flush=True,
        )

    def _check_watchdog(
        self,
        rollout_summary: dict[str, float],
        zero_win_streak: int,
    ) -> int:
        """Track consecutive zero-win iterations; raise when threshold hit.

        Returns the new streak length. Counter resets on any iteration that
        recorded ≥1 learner win.
        """
        threshold = self._cfg.watchdog_zero_wins_iters
        if threshold <= 0:
            return 0
        wins = int(rollout_summary.get("rollout/wins", 0))
        if wins > 0:
            return 0
        zero_win_streak += 1
        if zero_win_streak >= threshold:
            raise RuntimeError(
                f"watchdog: rollout/wins == 0 for {zero_win_streak} "
                f"consecutive iterations (~{zero_win_streak * self._cfg.rollout_steps} "
                "env steps). Likely a degenerate policy or sparse-reward "
                "stalemate collapse — kill the run, change hyperparameters."
            )
        return zero_win_streak

    def close(self) -> None:
        """Tear down any subprocess vec env owned by the trainer.

        Safe to call repeatedly. ``train()`` calls this in a ``finally`` so
        subprocesses are reaped even when training raises.
        """
        if self._vec_env is not None:
            self._vec_env.close()
            self._vec_env = None
            self._vec_worker = None

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a versioned :class:`CheckpointMeta`-tagged checkpoint to ``path``."""
        save_checkpoint(self._learner, Path(path), self._build_meta())

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _collect_rollout_step(
        self,
    ) -> tuple[dict[str, float], dict[PlayerID, float]]:
        """Dispatch to the single- or vec-env rollout collector.

        Returns ``(summary, bootstraps)`` where ``summary`` is a flat scalar
        dict for TB and ``bootstraps`` is the ``{seat: V(s_T)}`` mapping fed
        to :meth:`TrajectoryBuffer.compute_advantages`.
        """
        if self._cfg.num_envs > 1:
            return self._collect_rollout_vec()
        return self._collect_rollout_single()

    def _collect_rollout_single(
        self,
    ) -> tuple[dict[str, float], dict[PlayerID, float]]:
        """Collect ``cfg.rollout_steps`` learner transitions into the buffer.

        New env, new opponent assignment, and new learner seat per episode —
        ``RolloutWorker.collect(..., stop_at_episode=True)`` returns after one
        completed game so the next iteration can re-sample.
        """
        self._buffer.clear()
        agg: dict[str, float] = defaultdict(float)
        all_returns: list[float] = []
        last_worker: RolloutWorker | None = None
        steps_before = self._global_step
        n_opponents = len(self._player_ids) - 1

        while len(self._buffer) < self._cfg.rollout_steps:
            learner_seat = self._rng.choice(self._player_ids)
            env = self._env_factory(self._seed_counter)
            opponent_agents = self._opponent_pool.sample_opponents(
                self._learner, n=n_opponents
            )
            opponents = {
                pid: a
                for pid, a in zip(
                    (p for p in self._player_ids if p != learner_seat),
                    opponent_agents,
                )
            }
            worker = RolloutWorker(env, self._learner, opponents, self._buffer)
            n_remaining = self._cfg.rollout_steps - len(self._buffer)
            stats = worker.collect(n_remaining, stop_at_episode=True)
            agg["episodes"] += stats.episodes_completed
            agg["wins"] += stats.learner_wins
            agg["losses"] += stats.learner_losses
            agg["stalemates"] += stats.stalemates
            all_returns.extend(stats.learner_returns)
            self._global_step += stats.learner_steps
            self._seed_counter += 1
            last_worker = worker

            # Safety: if the worker can't make progress (no transitions added
            # and no episode completed), break to avoid an infinite loop.
            if stats.learner_steps == 0 and stats.episodes_completed == 0:
                break

        if last_worker is None:
            raise RuntimeError("_collect_rollout_single produced no worker")

        summary = {
            "rollout/episodes": agg["episodes"],
            "rollout/wins": agg["wins"],
            "rollout/losses": agg["losses"],
            "rollout/stalemates": agg["stalemates"],
            "rollout/steps": float(self._global_step - steps_before),
            "pool/recent_size": float(len(self._opponent_pool.recent_paths)),
            "pool/historical_size": float(
                len(self._opponent_pool.historical_paths)
            ),
        }
        if agg["episodes"] > 0:
            summary["rollout/win_rate"] = agg["wins"] / agg["episodes"]
        if all_returns:
            summary["rollout/return_mean"] = sum(all_returns) / len(all_returns)
            summary["rollout/return_max"] = max(all_returns)
            summary["rollout/return_min"] = min(all_returns)
        return summary, {last_worker.learner_seat: last_worker.last_bootstrap_value}

    def _collect_rollout_vec(
        self,
    ) -> tuple[dict[str, float], dict[PlayerID, float]]:
        """Vec-env rollout: ``cfg.num_envs`` parallel subprocesses, one buffer.

        The :class:`SubprocVecEnv` is spawned lazily on first call and reused
        across iterations — subprocess startup is the dominant cost in the
        vec path, so we pay it once. Opponents are fixed at spawn time by the
        ``vec_factory``; in-training opponent diversity comes from the
        per-seed seat rotation baked into the shipped factories, not from
        :class:`OpponentPool` (which the vec workers can't see across the
        process boundary).
        """
        self._buffer.clear()
        steps_before = self._global_step

        if self._vec_worker is None:
            assert self._vec_factory is not None  # validated in __init__
            self._vec_env = SubprocVecEnv(
                factory=self._vec_factory,
                num_envs=self._cfg.num_envs,
                base_seed=self._seed_counter,
            )
            # Mirror the seat assignment the factory uses (seed % 4 + 1) so
            # the buffer's per-agent GAE walks each env's subsequence under
            # the correct seat key.
            learner_seats = [
                self._player_ids[(self._seed_counter + i) % len(self._player_ids)]
                for i in range(self._cfg.num_envs)
            ]
            self._vec_worker = VecRolloutWorker(
                vec_env=self._vec_env,
                learner=self._learner,
                buffer=self._buffer,
                learner_seats=learner_seats,
            )

        stats = self._vec_worker.collect(self._cfg.rollout_steps)
        self._global_step += stats.learner_steps
        # Bump the seed counter by the number of envs so a subsequent
        # close-and-respawn (e.g. tests) lands on a fresh seed window.
        self._seed_counter += self._cfg.num_envs

        summary: dict[str, float] = {
            "rollout/episodes": float(stats.episodes_completed),
            "rollout/wins": float(stats.learner_wins),
            "rollout/losses": float(stats.learner_losses),
            "rollout/stalemates": float(stats.stalemates),
            "rollout/steps": float(self._global_step - steps_before),
            "pool/recent_size": float(len(self._opponent_pool.recent_paths)),
            "pool/historical_size": float(
                len(self._opponent_pool.historical_paths)
            ),
        }
        if stats.episodes_completed > 0:
            summary["rollout/win_rate"] = (
                stats.learner_wins / stats.episodes_completed
            )
        return summary, self._vec_worker.last_bootstrap_values

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _ppo_step(self) -> dict[str, float]:
        size = len(self._buffer)
        if size == 0:
            return {}
        # Whole-rollout TrajectoryBatch. ppo_update handles shuffled
        # minibatching internally.
        batches = list(
            self._buffer.to_batches(batch_size=size, shuffle=False)
        )
        if not batches:
            return {}
        full_batch = batches[0]
        return ppo_update(
            model=self._learner.model,
            optimizer=self._optimizer,
            batch=full_batch,
            cfg=self._cfg.ppo,
            device=self._learner.device,
        )

    # ------------------------------------------------------------------
    # Snapshotting (self-play)
    # ------------------------------------------------------------------

    def _maybe_snapshot(self) -> None:
        if self._snapshot_dir is None:
            return
        if self._cfg.snapshot_every <= 0:
            return
        if self._global_step - self._last_snapshot_step < self._cfg.snapshot_every:
            return
        self._last_snapshot_step = self._global_step
        path = self._snapshot_dir / f"snapshot_step_{self._global_step}.pt"
        save_checkpoint(self._learner, path, self._build_meta())
        self._opponent_pool.add_checkpoint(path)
        self._n_snapshots += 1
        if (
            self._cfg.promote_every_n_snapshots > 0
            and self._n_snapshots % self._cfg.promote_every_n_snapshots == 0
        ):
            self._opponent_pool.promote_to_historical()

    def _build_meta(self) -> CheckpointMeta:
        return CheckpointMeta(
            obs_layout_version=OBS_LAYOUT_VERSION,
            action_layout_version=ACTION_LAYOUT_VERSION,
            model_arch=self._model_arch,
            train_step=self._global_step,
            timestamp=time.time(),
            config_hash=self._config_hash,
        )

    # ------------------------------------------------------------------
    # Logging / eval
    # ------------------------------------------------------------------

    def _log_iteration(
        self,
        rollout_summary: dict[str, float],
        update_metrics: dict[str, float],
    ) -> None:
        if (
            self._cfg.log_every > 0
            and self._iteration % self._cfg.log_every != 0
        ):
            return
        for k, v in rollout_summary.items():
            self._logger.log_scalar(k, float(v), self._global_step)
        for k, v in update_metrics.items():
            self._logger.log_scalar(f"ppo/{k}", float(v), self._global_step)

    def _evaluate(self) -> None:
        """Run a small benchmark and log the win rate.

        Imports the evaluator lazily so the smoke test doesn't depend on
        eval working — the eval path uses ``Tournament``, which calls
        the same ``PolicyAgent.choose`` we've already smoke-tested.
        """
        from rl.agents.random_agent import RandomAgent
        from rl.evaluation.tournament import Tournament

        opp_rng = random.Random(self._seed_counter)
        eval_seat = self._player_ids[0]
        agents: dict[PlayerID, Agent] = {}
        for pid in self._player_ids:
            if pid == eval_seat:
                agents[pid] = self._learner
            else:
                agents[pid] = RandomAgent(
                    random.Random(opp_rng.randrange(2**32)),
                    skip_proposals=True,
                )

        result = Tournament(self._env_factory).play(
            agents,
            n_games=self._cfg.eval_n_games,
            base_seed=self._seed_counter,
        )
        self._seed_counter += self._cfg.eval_n_games
        win_rate = result.win_rates.get(eval_seat, 0.0)
        self._logger.log_scalar("eval/win_rate_vs_random", win_rate, self._global_step)
        self._logger.log_scalar(
            "eval/mean_turns", result.mean_turns, self._global_step
        )
