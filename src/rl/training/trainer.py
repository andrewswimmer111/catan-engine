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
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

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
from rl.utils.logging import NoOpLogger, TBLogger, make_logger

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
    ) -> None:
        self._env_factory = env_factory
        self._learner = learner
        self._opponent_pool = opponent_pool
        self._cfg = cfg

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

    def train(self, total_steps: int) -> None:
        """Train until ``self.global_step`` reaches ``total_steps``."""
        last_eval_step = -self._cfg.eval_every  # force eval on first iter
        while self._global_step < total_steps:
            rollout_summary, last_worker = self._collect_rollout()
            self._compute_advantages(last_worker)
            update_metrics = self._ppo_step()
            self._log_iteration(rollout_summary, update_metrics)
            self._maybe_snapshot()
            if (
                self._cfg.eval_every > 0
                and self._global_step - last_eval_step >= self._cfg.eval_every
            ):
                self._evaluate()
                last_eval_step = self._global_step
            self._iteration += 1
        self._logger.flush()

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a versioned :class:`CheckpointMeta`-tagged checkpoint to ``path``."""
        save_checkpoint(self._learner, Path(path), self._build_meta())

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> tuple[dict[str, float], RolloutWorker]:
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
            raise RuntimeError("_collect_rollout produced no worker")

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
        return summary, last_worker

    def _compute_advantages(self, last_worker: RolloutWorker) -> None:
        bootstrap: dict[PlayerID, float] = {
            last_worker.learner_seat: last_worker.last_bootstrap_value
        }
        self._buffer.compute_advantages(
            gamma=self._cfg.gamma,
            lam=self._cfg.gae_lambda,
            last_values=bootstrap,
        )

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
