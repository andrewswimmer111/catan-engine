"""End-to-end PPO trainer: collect rollouts, run updates, log, checkpoint.

The trainer is built around three callables provided at construction:

- ``env_factory(seed) -> CatanEnv`` — produces a fresh env per episode so
  reseeding is deterministic and the four seats are always in the same
  config order. The learner's *seat assignment* is rotated each episode by
  the trainer (see below), not by the factory.
- ``opponent_factory(seed) -> dict[PlayerID, Agent]`` — returns an opponent
  for every seat keyed by ``PlayerID``. The trainer drops the entry that
  matches the learner's seat for that episode and uses the rest.
- A :class:`PolicyAgent` for the learner — the only object whose weights
  are updated.

Seat rotation
-------------

PPO is sensitive to seat overfitting if the learner always plays seat 1
(setup-phase first-pick changes positional value). On each episode start
the trainer samples a learner seat uniformly at random; the worker
records every learner transition with the seat's PlayerID so per-agent
GAE naturally segments by episode boundary.

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
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.replay.buffer import TrajectoryBuffer
from rl.training.config import TrainConfig
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
        opponent_factory: Callable[[int], dict[PlayerID, Agent]],
        cfg: TrainConfig,
        log_dir: str | Path | None = None,
    ) -> None:
        self._env_factory = env_factory
        self._learner = learner
        self._opponent_factory = opponent_factory
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def iteration(self) -> int:
        return self._iteration

    def train(self, total_steps: int) -> None:
        """Train until ``self.global_step`` reaches ``total_steps``."""
        last_eval_step = -self._cfg.eval_every  # force eval on first iter
        while self._global_step < total_steps:
            rollout_summary, last_worker = self._collect_rollout()
            self._compute_advantages(last_worker)
            update_metrics = self._ppo_step()
            self._log_iteration(rollout_summary, update_metrics)
            if (
                self._cfg.eval_every > 0
                and self._global_step - last_eval_step >= self._cfg.eval_every
            ):
                self._evaluate()
                last_eval_step = self._global_step
            self._iteration += 1
        self._logger.flush()

    def save_checkpoint(self, path: str | Path) -> None:
        """Write model weights, optimizer state, and trainer counters."""
        torch.save(
            {
                "model": self._learner.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "iteration": self._iteration,
                "global_step": self._global_step,
            },
            str(path),
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> tuple[dict[str, float], RolloutWorker]:
        """Collect ``cfg.rollout_steps`` learner transitions into the buffer."""
        self._buffer.clear()
        agg: dict[str, float] = defaultdict(float)
        all_returns: list[float] = []
        last_worker: RolloutWorker | None = None
        steps_before = self._global_step

        while len(self._buffer) < self._cfg.rollout_steps:
            learner_seat = self._rng.choice(self._player_ids)
            env = self._env_factory(self._seed_counter)
            factory_agents = self._opponent_factory(self._seed_counter)
            opponents = {
                pid: a for pid, a in factory_agents.items() if pid != learner_seat
            }
            if len(opponents) != len(self._player_ids) - 1:
                raise ValueError(
                    f"opponent_factory must return an agent for every seat; "
                    f"got {sorted(factory_agents.keys())}"
                )
            worker = RolloutWorker(env, self._learner, opponents, self._buffer)
            n_remaining = self._cfg.rollout_steps - len(self._buffer)
            stats = worker.collect(n_remaining)
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
            if (
                stats.learner_steps == 0
                and stats.episodes_completed == 0
            ):
                break

        if last_worker is None:
            raise RuntimeError("_collect_rollout produced no worker")

        summary = {
            "rollout/episodes": agg["episodes"],
            "rollout/wins": agg["wins"],
            "rollout/losses": agg["losses"],
            "rollout/stalemates": agg["stalemates"],
            "rollout/steps": float(self._global_step - steps_before),
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
        agents: dict[PlayerID, object] = {}
        for pid in self._player_ids:
            if pid == eval_seat:
                agents[pid] = self._learner
            else:
                agents[pid] = RandomAgent(
                    random.Random(opp_rng.randrange(2**32)),
                    skip_proposals=True,
                )

        result = Tournament(self._env_factory).play(
            agents,  # type: ignore[arg-type]
            n_games=self._cfg.eval_n_games,
            base_seed=self._seed_counter,
        )
        self._seed_counter += self._cfg.eval_n_games
        win_rate = result.win_rates.get(eval_seat, 0.0)
        self._logger.log_scalar("eval/win_rate_vs_random", win_rate, self._global_step)
        self._logger.log_scalar(
            "eval/mean_turns", result.mean_turns, self._global_step
        )
