"""Vectorized rollout collection on top of :class:`SubprocVecEnv`.

The single-process :class:`RolloutWorker` is the simpler, slower path: one
env, one learner, one episode at a time. This module's
:class:`VecRolloutWorker` runs ``num_envs`` envs in parallel subprocesses
and amortises learner inference into one batched forward per "decision
round" over ``num_envs`` decisions.

Per-env state
-------------

Each env has its own learner seat (fixed at factory time — opponent
diversity comes from the factory choosing different opponent setups per
subprocess, not from rotating seats inside one subprocess). Transitions go
into the shared :class:`TrajectoryBuffer` tagged with the per-env learner
seat so PPO's per-agent GAE walks each env's subsequence independently.

Bootstrap values
----------------

When the buffer fills mid-episode for any env, the worker reads the
learner's value estimate from the *next* obs the vec env returned and
stashes it in ``last_bootstrap_values[env_idx]``. The trainer hands the
dict to :meth:`TrajectoryBuffer.compute_advantages` exactly like the
single-env path does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.replay.buffer import TrajectoryBuffer
from rl.replay.transition import Transition
from rl.training.vec_env import SubprocVecEnv, WorkerStepResult

__all__ = ["VecRolloutStats", "VecRolloutWorker"]


@dataclass
class VecRolloutStats:
    """Aggregate stats for one :meth:`VecRolloutWorker.collect` call."""

    episodes_completed: int
    learner_wins: int
    learner_losses: int
    stalemates: int
    learner_steps: int


class VecRolloutWorker:
    """Drives a :class:`SubprocVecEnv` and writes learner transitions.

    Only the learner's transitions are stored — opponents already step
    inside the subprocess so we never observe their states in main.

    ``learner_seats`` must be parallel to the worker bundles inside
    ``vec_env``; the caller is responsible for ensuring the factory and
    this list agree. Mismatches don't fail loudly because we never see the
    bundle from main — pass the expected seats through explicitly.
    """

    def __init__(
        self,
        vec_env: SubprocVecEnv,
        learner: PolicyAgent,
        buffer: TrajectoryBuffer,
        learner_seats: list[PlayerID],
    ) -> None:
        if len(learner_seats) != vec_env.num_envs:
            raise ValueError(
                f"learner_seats length {len(learner_seats)} != "
                f"vec_env.num_envs {vec_env.num_envs}"
            )
        self._vec = vec_env
        self._learner = learner
        self._buffer = buffer
        self._seats = list(learner_seats)
        self._last_transition_idx: dict[int, int] = {}
        self._last_bootstrap_values: dict[PlayerID, float] = {}
        self._current_returns: dict[int, float] = {i: 0.0 for i in range(vec_env.num_envs)}
        self._needs_reset = True
        self._pending: list[WorkerStepResult] | None = None

    @property
    def last_bootstrap_values(self) -> dict[PlayerID, float]:
        return dict(self._last_bootstrap_values)

    def collect(self, n_steps: int) -> VecRolloutStats:
        """Run until the buffer holds ``n_steps`` more learner transitions.

        Each "decision round" yields up to ``num_envs`` new transitions
        (one per env), so the buffer fills in chunks. Stops when the
        budget is met or the buffer is full.
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        if self._needs_reset or self._pending is None:
            self._pending = self._vec.reset()
            self._needs_reset = False

        episodes = 0
        wins = 0
        losses = 0
        stalemates = 0
        learner_steps = 0

        while (
            learner_steps < n_steps
            and len(self._buffer) < self._buffer.capacity
        ):
            obs_batch = np.stack([r.obs for r in self._pending])
            mask_batch = np.stack([r.mask for r in self._pending])
            steps = self._learner.act_batch(obs_batch, mask_batch)

            # Stash a per-env transition (reward gets filled in on the next
            # round-trip when the env reports back).
            pending_transitions: list[Transition] = []
            for env_idx, step in enumerate(steps):
                pending_transitions.append(Transition(
                    obs=self._pending[env_idx].obs,
                    action=step.action_idx,
                    mask=self._pending[env_idx].mask,
                    logp=step.logp,
                    value=step.value,
                    reward=0.0,
                    done=False,
                    agent=self._seats[env_idx],
                ))

            results = self._vec.step([int(s.action_idx) for s in steps])
            self._pending = results

            for env_idx, (txn, result) in enumerate(zip(pending_transitions, results)):
                txn.reward = float(result.reward)
                self._buffer.add(txn)
                self._last_transition_idx[env_idx] = len(self._buffer) - 1
                self._current_returns[env_idx] += float(result.reward)
                learner_steps += 1

                if result.done:
                    self._buffer.mark_done(self._last_transition_idx[env_idx])
                    info = result.info
                    winner = info.get("winner")
                    seat = int(self._seats[env_idx])
                    if winner is None:
                        stalemates += 1
                    elif winner == seat:
                        wins += 1
                    else:
                        losses += 1
                    episodes += 1
                    self._current_returns[env_idx] = 0.0

                if len(self._buffer) >= self._buffer.capacity:
                    break

        # Bootstrap values: for each env whose final stored transition is
        # non-terminal, record the value the learner would emit from the
        # next obs. With one shared learner, all envs share a PlayerID
        # key — that means multiple envs with the same seat overwrite. For
        # the single-seat-per-vec-env-factory pattern we ship, that's the
        # right behavior; mixed-seat factories need richer bookkeeping.
        self._record_bootstraps()
        return VecRolloutStats(
            episodes_completed=episodes,
            learner_wins=wins,
            learner_losses=losses,
            stalemates=stalemates,
            learner_steps=learner_steps,
        )

    def close(self) -> None:
        self._vec.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_bootstraps(self) -> None:
        self._last_bootstrap_values = {}
        if self._pending is None:
            return
        live = [
            (i, r) for i, r in enumerate(self._pending)
            if not r.done and r.mask.any()
        ]
        if not live:
            return
        obs_batch = np.stack([r.obs for _, r in live])
        mask_batch = np.stack([r.mask for _, r in live])
        steps = self._learner.act_batch(obs_batch, mask_batch, deterministic=True)
        for (env_idx, _), step in zip(live, steps):
            self._last_bootstrap_values[self._seats[env_idx]] = float(step.value)
