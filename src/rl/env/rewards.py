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
  hidden VP cards never leak), plus ``±win_bonus`` at the terminal step.
  Stalemate terminals reward 0 for everyone — log ``state.end_reason`` to
  decide whether to keep the trajectory.
"""

from __future__ import annotations

from typing import Protocol

from domain.actions.base import Action
from domain.engine.step_result import StepResult
from domain.game.state import GameState
from domain.ids import PlayerID

__all__ = ["RewardFn", "SparseWinReward", "ShapedReward"]


class RewardFn(Protocol):
    """Compute the per-step reward for ``agent`` after ``action`` was applied."""

    def step_reward(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        agent: PlayerID,
    ) -> float: ...


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


class ShapedReward:
    """VP-delta shaping with a small turn tick and a terminal win/loss bonus.

    Per-step reward:
        ``vp_coef × Δ victory_points_public(agent) + turn_tick``

    Terminal additions:
        ``+ win_bonus`` if ``agent`` won, ``- win_bonus`` if a different seat
        won. Stalemate terminals reward 0 across the board (the shaping signal
        is also dropped on the stalemate step so the trajectory ends cleanly
        at zero).

    Defaults (``vp_coef=0.05``, ``turn_tick=-0.001``, ``win_bonus=1.0``) are
    rough early-bring-up values; tune per training run.
    """

    def __init__(
        self,
        vp_coef: float = 0.05,
        turn_tick: float = -0.001,
        win_bonus: float = 1.0,
    ) -> None:
        self.vp_coef = vp_coef
        self.turn_tick = turn_tick
        self.win_bonus = win_bonus

    def step_reward(
        self,
        prev_state: GameState,
        action: Action,
        result: StepResult,
        agent: PlayerID,
    ) -> float:
        if _is_stalemate(result):
            return 0.0

        prev_vp = _vp_public(prev_state, agent)
        new_vp = _vp_public(result.state, agent)
        shaped = self.vp_coef * (new_vp - prev_vp) + self.turn_tick

        if not result.is_terminal:
            return shaped

        if result.winner == agent:
            return shaped + self.win_bonus
        return shaped - self.win_bonus
