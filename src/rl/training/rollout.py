"""Rollout collection — one env, one learner, three opponents.

A :class:`RolloutWorker` plays games on a single :class:`CatanEnv`, calling
the learner's :meth:`PolicyAgent.act` on every step the learner owns and
delegating opponent moves to plain ``Agent.choose`` callbacks. Only the
learner's transitions are written to the buffer — opponents are frozen in
self-play so storing their transitions would just waste memory.

Reward attribution
------------------

``CatanEnv.step`` returns the reward attributed to the seat that just
acted. The worker captures that reward on the learner's transition (when
the learner acted) and discards it otherwise (when an opponent acted —
their transitions aren't stored). At game termination the worker calls
``reward_fn.terminal_rewards(final_state)`` and uses
:meth:`TrajectoryBuffer.add_terminal_reward` to retroactively credit the
learner's last transition with the win / loss bonus — but only if the
learner did *not* act on the terminal step, since in that case the env's
own ``step_reward`` already credited them.

Bootstrap values
----------------

When the buffer fills before an episode ends, the GAE recursion needs a
``V(s_T)`` bootstrap for the learner — the worker reads it off the next
observation via :meth:`PolicyAgent.act` and stashes it in
``last_bootstrap_value`` for the trainer to pass to
``compute_advantages``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controller.agents import Agent
from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.engine.player_view import make_player_view
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.env.catan_env import CatanEnv
from rl.env.rewards import RewardFn
from rl.replay.buffer import TrajectoryBuffer
from rl.replay.transition import Transition

__all__ = ["RolloutWorker", "RolloutStats"]


@dataclass
class RolloutStats:
    """Diagnostics from one ``collect`` call.

    ``episodes_completed`` counts games that ended cleanly inside this
    call. ``learner_returns`` is one entry per completed episode in
    collection order, summing all rewards on the learner's trajectory
    for that episode (including the retroactive terminal credit).
    """

    episodes_completed: int
    learner_wins: int
    learner_losses: int
    stalemates: int
    learner_returns: list[float]
    learner_steps: int


class RolloutWorker:
    """Owns one env, one learner, and a fixed opponent assignment.

    The learner's seat is whichever PlayerID is not a key of ``opponents``.
    The worker is stateful across episodes — calling ``collect`` repeatedly
    accumulates transitions into the shared buffer until the requested
    step budget is met. Each ``collect`` call resets per-call statistics
    but preserves the buffer.

    For seat rotation, construct a new :class:`RolloutWorker` per episode
    (the :class:`Trainer` handles this).
    """

    def __init__(
        self,
        env: CatanEnv,
        learner: PolicyAgent,
        opponents: dict[PlayerID, Agent],
        buffer: TrajectoryBuffer,
        reward_fn: RewardFn | None = None,
    ) -> None:
        cfg_seats = set(env.state.config.player_ids)
        opp_seats = set(opponents.keys())
        if not opp_seats.issubset(cfg_seats):
            raise ValueError(
                f"opponents contain seats {opp_seats - cfg_seats} not in env config"
            )
        learner_seats = cfg_seats - opp_seats
        if len(learner_seats) != 1:
            raise ValueError(
                f"exactly one seat must be unassigned for the learner, got {learner_seats}"
            )
        self._env = env
        self._learner = learner
        self._opponents = dict(opponents)
        self._buffer = buffer
        self._reward_fn = reward_fn if reward_fn is not None else env.reward_fn
        self._learner_seat: PlayerID = next(iter(learner_seats))

        self._last_transition_idx: int | None = None
        self._current_episode_return: float = 0.0
        self._needs_reset: bool = False
        self._last_bootstrap_value: float = 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def learner_seat(self) -> PlayerID:
        return self._learner_seat

    @property
    def last_bootstrap_value(self) -> float:
        """V(s) for the learner's next obs when collect stopped mid-episode."""
        return self._last_bootstrap_value

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self, n_steps: int) -> RolloutStats:
        """Collect at most ``n_steps`` learner transitions into the buffer.

        Stops when either the requested learner-step count is met or the
        buffer fills, whichever comes first. May span multiple episodes;
        the env is reset internally on terminal steps.
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        learner_steps = 0
        episodes = 0
        wins = 0
        losses = 0
        stalemates = 0
        returns_per_episode: list[float] = []

        if self._needs_reset:
            self._reset_episode()
            self._needs_reset = False

        self._last_bootstrap_value = 0.0

        while learner_steps < n_steps and len(self._buffer) < self._buffer.capacity:
            acting = self._env.current_agent
            if acting == self._learner_seat:
                done, outcome = self._take_learner_step()
                learner_steps += 1
            else:
                done, outcome = self._take_opponent_step(acting)

            if done:
                episodes += 1
                returns_per_episode.append(self._current_episode_return)
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                else:
                    stalemates += 1
                if (
                    learner_steps < n_steps
                    and len(self._buffer) < self._buffer.capacity
                ):
                    self._reset_episode()
                else:
                    self._needs_reset = True

        if not self._needs_reset and self._last_transition_idx is not None:
            self._last_bootstrap_value = self._estimate_value()

        return RolloutStats(
            episodes_completed=episodes,
            learner_wins=wins,
            learner_losses=losses,
            stalemates=stalemates,
            learner_returns=returns_per_episode,
            learner_steps=learner_steps,
        )

    # ------------------------------------------------------------------
    # Per-step actions
    # ------------------------------------------------------------------

    def _take_learner_step(self) -> tuple[bool, str | None]:
        obs = self._encode_learner_obs()
        mask = self._env.action_mask()

        if not mask.any():
            # Nothing representable; fall back to first legal action (no
            # transition stored — the learner isn't actually deciding).
            legal = self._env.legal_actions()
            if not legal:
                return True, "stalemate"
            return self._apply_step(legal[0], stored=None)

        step_out = self._learner.act(obs, mask, deterministic=False)
        stored = Transition(
            obs=obs,
            action=step_out.action_idx,
            mask=mask,
            logp=step_out.logp,
            value=step_out.value,
            reward=0.0,  # patched by _apply_step from env.step's return
            done=False,
            agent=self._learner_seat,
        )
        return self._apply_step(step_out.action_idx, stored=stored)

    def _take_opponent_step(self, acting: PlayerID) -> tuple[bool, str | None]:
        agent = self._opponents[acting]
        snap = self._build_snapshot()
        legal = self._env.legal_actions()
        action = agent.choose(snap, legal)
        if action is None:
            # No action chosen — bail out as if the episode stalemated.
            return True, "stalemate"
        return self._apply_step(action, stored=None)

    def _apply_step(
        self,
        action: Action | int,
        stored: Transition | None,
    ) -> tuple[bool, str | None]:
        """Apply an action to the env. If ``stored`` is given, push it to
        the buffer with the env's returned reward filled in."""
        acting_pid = self._env.current_agent
        _, reward, done, _ = self._env.step(action)

        if stored is not None:
            stored.reward = float(reward)
            self._buffer.add(stored)
            self._last_transition_idx = len(self._buffer) - 1
            self._current_episode_return += float(reward)

        if done:
            outcome = self._episode_outcome()
            self._apply_terminal_credit(acting_pid_at_terminal=acting_pid)
            return True, outcome
        return False, None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._env.reset()
        self._last_transition_idx = None
        self._current_episode_return = 0.0

    def _apply_terminal_credit(self, acting_pid_at_terminal: PlayerID) -> None:
        """Mark the learner's last transition done and, if the learner
        wasn't the one who acted on the terminal step, retroactively add
        the per-seat terminal reward to it."""
        if self._last_transition_idx is None:
            return

        if acting_pid_at_terminal != self._learner_seat:
            payout = self._reward_fn.terminal_rewards(self._env.state)
            learner_reward = float(payout.get(self._learner_seat, 0.0))
            if learner_reward != 0.0:
                self._buffer.add_terminal_reward(
                    self._last_transition_idx, learner_reward
                )
                self._current_episode_return += learner_reward

        self._buffer.mark_done(self._last_transition_idx)

    def _episode_outcome(self) -> str:
        winner = self._env.state.winner
        if winner is None:
            return "stalemate"
        return "win" if winner == self._learner_seat else "loss"

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _encode_learner_obs(self) -> np.ndarray:
        view = make_player_view(self._env.state, self._learner_seat)
        return self._env.obs_encoder.encode(view)

    def _estimate_value(self) -> float:
        """V(s) for the learner's current obs — bootstrap for a non-terminal
        cut. Returns 0.0 if the learner doesn't own the next move (no
        bootstrap available; downstream GAE multiplies it out anyway when
        the episode has actually ended)."""
        if self._env.current_agent != self._learner_seat:
            return 0.0
        mask = self._env.action_mask()
        if not mask.any():
            return 0.0
        obs = self._encode_learner_obs()
        step = self._learner.act(obs, mask, deterministic=True)
        return step.value

    def _build_snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            state=self._env.state,
            step_index=0,
            last_action=None,
            last_events=(),
        )
