"""Pluggable reward functions for :class:`CatanEnv`.

A reward function maps ``(prev_state, action, result, agent) → float``. The
environment calls it once per ``step``, passing the player who just acted as
``agent``. Implementations should be pure; any internal state (e.g. running
moving averages) must live on the instance and survive ``env.reset`` only if
the same instance is passed back to a fresh env.

Two concrete reward fns ship here:

- :class:`SparseWinReward` — the canonical multi-agent terminal reward:
  ``+1`` for the winner, ``-1`` for losers (only at the terminal step), ``0``
  for stalemates and all non-terminal steps. The trainer is responsible for
  handling the multi-agent credit assignment.
- :class:`ShapedReward` — a denser signal for early bring-up:
  ``vp_coef × Δvp + turn_tick`` per step (using ``victory_points_public`` so
  hidden VP cards never leak), plus ``±win_bonus`` at the terminal step
  and ``-stalemate_penalty`` for everyone if the game stalemates. The
  stalemate penalty exists because, without it, a learner that accumulates
  even a small amount of shaping (e.g. from setup-phase settlements) finds
  "stall the game out" strictly better than "play and risk a loss" — a
  classic shaped-reward attractor that breaks training.

Per-seat credit decomposes into three orthogonal channels:

* ``step_reward(prev, action, result, agent)`` — reward to the *acting*
  seat, computed by the env per step.
* ``cross_step_rewards(prev, action, result, acting)`` — per-step rewards
  to *non-acting* seats (e.g. an opponent's road build dropping your
  Longest Road bonus). The rollout worker writes these onto each
  non-acting seat's last stored transition; without them, a seat's
  ``victory_points_public`` could decrease without any negative
  gradient signal.
* ``terminal_rewards(final_state)`` — the per-seat win/loss bonus at game
  end, again written by the worker onto non-acting seats' last
  transitions.

The three channels are *disjoint*: ``step_reward`` covers the acting seat
per step; ``cross_step_rewards`` covers non-acting seats per step (no
win/loss bonus); ``terminal_rewards`` covers non-acting seats at terminal
(no shaping). Acting + non-acting at terminal therefore receive the same
total reward, just split across two methods.
"""

from __future__ import annotations

from typing import Protocol

from domain.actions.base import Action
from domain.engine.step_result import StepResult
from domain.game.state import GameState
from domain.ids import PlayerID

__all__ = ["RewardFn", "SparseWinReward", "ShapedReward"]


class RewardFn(Protocol):
    """Three-channel per-seat reward contract.

    See the module docstring for how ``step_reward`` /
    ``cross_step_rewards`` / ``terminal_rewards`` decompose acting vs.
    non-acting seats and per-step vs. terminal credit.
    """

    def step_reward(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        agent: PlayerID,
    ) -> float: ...

    def cross_step_rewards(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        acting_agent: PlayerID,
    ) -> dict[PlayerID, float]:
        """Per-step rewards for *non-acting* seats only.

        Returns a dict ``{seat: reward}`` for seats whose side-effect
        rewards the rollout worker should retroactively credit. The
        acting seat is never included — its reward already came through
        ``step_reward``. Implementations that don't care about per-step
        non-acting effects may simply return ``{}``.
        """
        ...

    def terminal_rewards(self, final_state: GameState) -> dict[PlayerID, float]: ...


def _is_stalemate(result: StepResult) -> bool:
    """Terminal but no winner — game ended in stalemate."""
    return result.is_terminal and result.winner is None


def _vp_public(state: GameState, agent: PlayerID) -> int:
    """Read the agent's public VP (excludes hidden VP dev cards).

    Returns 0 if the agent isn't in the state's player table — guards against
    an env wired to a different config than the reward fn was instantiated
    with. Reaching that branch is a programmer error, but silently returning 0
    is safer than raising mid-rollout.
    """
    p = state.players.get(agent)
    return p.victory_points_public if p is not None else 0


class SparseWinReward:
    """+1 for the winner, -1 for every other seat at the terminal step.

    All non-terminal steps and stalemates return 0. The terminal-step penalty
    on losers is what makes this multi-agent — the trainer must call
    ``step_reward`` with each seat's id at the terminal step (or, equivalently,
    bookkeep losses outside the env) to credit the loss signal.
    """

    def step_reward(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        agent: PlayerID,
    ) -> float:
        if not result.is_terminal:
            return 0.0
        if result.winner is None:  # stalemate
            return 0.0
        return 1.0 if result.winner == agent else -1.0

    def cross_step_rewards(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        acting_agent: PlayerID,
    ) -> dict[PlayerID, float]:
        # Sparse rewards never credit non-acting seats mid-game.
        return {}

    def terminal_rewards(self, final_state: GameState) -> dict[PlayerID, float]:
        winner = final_state.winner
        if winner is None:
            return {pid: 0.0 for pid in final_state.config.player_ids}
        return {
            pid: (1.0 if pid == winner else -1.0)
            for pid in final_state.config.player_ids
        }


class ShapedReward:
    """VP-delta shaping with a small turn tick, terminal win/loss bonus, and stalemate penalty.

    Per-step reward:
        ``vp_coef × Δ victory_points_public(agent) + turn_tick``

    Terminal additions (added on the step that flips the game terminal):
        ``+ win_bonus`` if ``agent`` won, ``- win_bonus`` if a different seat
        won, ``- stalemate_penalty`` if the game stalemated. The stalemate
        penalty matters more than it might look: with ``stalemate_penalty=0``
        a learner with positive cumulative shaping from setup builds finds
        stalling strictly better than playing — see module docstring.

    Defaults (``vp_coef=0.05``, ``turn_tick=-0.001``, ``win_bonus=1.0``,
    ``stalemate_penalty=0.5``) are rough early-bring-up values. Set
    ``stalemate_penalty=0.0`` to reproduce the pre-penalty behaviour.
    """

    def __init__(
        self,
        vp_coef: float = 0.05,
        turn_tick: float = -0.001,
        win_bonus: float = 1.0,
        stalemate_penalty: float = 0.5,
    ) -> None:
        if stalemate_penalty < 0:
            raise ValueError(
                f"stalemate_penalty must be non-negative, got {stalemate_penalty}"
            )
        self.vp_coef = vp_coef
        self.turn_tick = turn_tick
        self.win_bonus = win_bonus
        self.stalemate_penalty = stalemate_penalty

    def step_reward(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        agent: PlayerID,
    ) -> float:
        if _is_stalemate(result):
            # Stalemate short-circuits shaping (matching the pre-penalty
            # contract) but applies the flat per-seat penalty so the
            # "stall to keep accumulated shaping" attractor is broken.
            return -self.stalemate_penalty

        prev_vp = _vp_public(prev_state, agent)
        new_vp = _vp_public(result.state, agent)
        shaped = self.vp_coef * (new_vp - prev_vp) + self.turn_tick

        if not result.is_terminal:
            return shaped

        if result.winner == agent:
            return shaped + self.win_bonus
        return shaped - self.win_bonus

    def cross_step_rewards(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        acting_agent: PlayerID,
    ) -> dict[PlayerID, float]:
        """``vp_coef × Δvp_public`` for every non-acting seat that gained or lost VP.

        The canonical use case is symmetric Longest Road / Largest Army
        title flips: when the acting seat takes LR from a non-acting
        seat, the non-acting seat's ``victory_points_public`` drops by 2,
        and they receive ``-2 × vp_coef`` here. Without this channel the
        loss is invisible to PPO because ``step_reward`` only credits the
        acting seat.

        Stalemate terminals zero out everywhere (matching ``step_reward``);
        non-stalemate terminals still shape so a non-acting seat whose VP
        moves on the final step gets that delta credited.
        """
        if _is_stalemate(result):
            return {}
        rewards: dict[PlayerID, float] = {}
        for pid in result.state.config.player_ids:
            if pid == acting_agent:
                continue
            delta = _vp_public(result.state, pid) - _vp_public(prev_state, pid)
            if delta != 0:
                rewards[pid] = self.vp_coef * delta
        return rewards

    def terminal_rewards(self, final_state: GameState) -> dict[PlayerID, float]:
        winner = final_state.winner
        if winner is None:
            # Stalemate — penalty applied to every seat. The rollout
            # worker writes this to each non-acting seat's last
            # transition; the acting seat at terminal already received
            # the same -stalemate_penalty via step_reward.
            return {
                pid: -self.stalemate_penalty
                for pid in final_state.config.player_ids
            }
        return {
            pid: (self.win_bonus if pid == winner else -self.win_bonus)
            for pid in final_state.config.player_ids
        }
