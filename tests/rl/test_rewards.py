"""Reward-function unit tests + an end-to-end smoke run through CatanEnv."""

from __future__ import annotations

import copy
import random
from typing import cast

import numpy as np
import pytest

from domain.actions.all_actions import EndTurnAction
from domain.engine.step_result import StepResult
from domain.enums import EndReason, TurnPhase
from domain.events.all_events import GameStalled, GameWon
from domain.game.state import GameState
from domain.ids import PlayerID
from rl.env.catan_env import CatanEnv
from rl.env.rewards import RewardFn, ShapedReward, SparseWinReward


# ---------------------------------------------------------------------------
# Helpers — fabricate StepResult objects without driving a full game
# ---------------------------------------------------------------------------


def _fresh_state(seed: int = 0) -> GameState:
    """Return a fresh GameState (post-new_game, pre-setup actions)."""
    env = CatanEnv(seed=seed)
    env.reset(seed=seed)
    return env.state


def _player_ids(state: GameState) -> list[PlayerID]:
    return list(state.config.player_ids)


def _terminal_state(state: GameState, winner: PlayerID | None) -> GameState:
    """Deep-copy ``state`` and tag it as terminal under the given outcome.

    ``winner=None`` produces a stalemate. The mutated copy is the kind of
    state ``transitions.apply`` would return — sufficient for reward fns,
    which only inspect winner / phase / players.
    """
    s = copy.deepcopy(state)
    if winner is None:
        s.phase = TurnPhase.STALEMATE
        s.end_reason = EndReason.STALEMATE_NO_PROGRESS
    else:
        s.phase = TurnPhase.GAME_OVER
        s.end_reason = EndReason.WINNER
        s.winner = winner
    return s


def _make_step_result(
    new_state: GameState, events: list, action=None
) -> StepResult:
    pid = new_state.current_player
    return StepResult(
        state=new_state,
        events=list(events),
        is_terminal=new_state.is_terminal(),
        winner=new_state.winner,
        action=action or EndTurnAction(player_id=pid),
    )


# ---------------------------------------------------------------------------
# SparseWinReward
# ---------------------------------------------------------------------------


def test_sparse_winner_gets_plus_one_at_terminal() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)
    winner = pids[1]

    terminal = _terminal_state(state, winner=winner)
    result = _make_step_result(
        terminal, events=[GameWon(turn_number=42, player_id=winner, victory_points=10)]
    )

    fn: RewardFn = SparseWinReward()
    assert fn.step_reward(state, result.action, result, winner) == 1.0


def test_sparse_losers_get_minus_one_at_terminal() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)
    winner = pids[1]

    terminal = _terminal_state(state, winner=winner)
    result = _make_step_result(
        terminal, events=[GameWon(turn_number=42, player_id=winner, victory_points=10)]
    )

    fn = SparseWinReward()
    for loser in (pids[0], pids[2], pids[3]):
        assert fn.step_reward(state, result.action, result, loser) == -1.0


def test_sparse_zero_at_non_terminal() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)

    # Non-terminal "next" state — just a copy of the fresh state.
    next_state = copy.deepcopy(state)
    result = _make_step_result(next_state, events=[])
    assert not result.is_terminal

    fn = SparseWinReward()
    for pid in pids:
        assert fn.step_reward(state, result.action, result, pid) == 0.0


def test_sparse_zero_on_stalemate() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)

    terminal = _terminal_state(state, winner=None)
    result = _make_step_result(
        terminal, events=[GameStalled(turn_number=99, reason=EndReason.STALEMATE_VP_STALL)]
    )

    fn = SparseWinReward()
    for pid in pids:
        assert fn.step_reward(state, result.action, result, pid) == 0.0


# ---------------------------------------------------------------------------
# ShapedReward
# ---------------------------------------------------------------------------


def test_shaped_vp_delta_rewards_settlement_build() -> None:
    """A +1 VP gain on the active player yields ``vp_coef`` (plus the turn tick)."""
    state = _fresh_state(seed=0)
    agent = _player_ids(state)[0]

    next_state = copy.deepcopy(state)
    next_state.players[agent].victory_points_public = (
        state.players[agent].victory_points_public + 1
    )

    result = _make_step_result(next_state, events=[])

    fn = ShapedReward(vp_coef=0.05, turn_tick=-0.001, win_bonus=1.0)
    reward = fn.step_reward(state, result.action, result, agent)
    assert reward == pytest.approx(0.05 - 0.001)


def test_shaped_no_vp_delta_yields_only_turn_tick() -> None:
    state = _fresh_state(seed=0)
    agent = _player_ids(state)[0]
    next_state = copy.deepcopy(state)  # VPs unchanged
    result = _make_step_result(next_state, events=[])

    fn = ShapedReward(vp_coef=0.05, turn_tick=-0.001, win_bonus=1.0)
    reward = fn.step_reward(state, result.action, result, agent)
    assert reward == pytest.approx(-0.001)


def test_shaped_terminal_winner_includes_win_bonus_and_delta() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)
    winner = pids[2]

    terminal = _terminal_state(state, winner=winner)
    # Winner pushed to 10 VP at the same step.
    terminal.players[winner].victory_points_public = (
        state.players[winner].victory_points_public + 4
    )
    result = _make_step_result(
        terminal, events=[GameWon(turn_number=50, player_id=winner, victory_points=10)]
    )

    fn = ShapedReward(vp_coef=0.05, turn_tick=-0.001, win_bonus=1.0)
    reward = fn.step_reward(state, result.action, result, winner)
    assert reward == pytest.approx(0.05 * 4 - 0.001 + 1.0)


def test_shaped_terminal_loser_gets_minus_win_bonus() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)
    winner = pids[1]
    loser = pids[3]

    terminal = _terminal_state(state, winner=winner)
    result = _make_step_result(
        terminal, events=[GameWon(turn_number=50, player_id=winner, victory_points=10)]
    )

    fn = ShapedReward(vp_coef=0.05, turn_tick=-0.001, win_bonus=1.0)
    reward = fn.step_reward(state, result.action, result, loser)
    # Loser had no VP delta on this step, so reward = tick - win_bonus.
    assert reward == pytest.approx(-0.001 - 1.0)


def test_shaped_zero_on_stalemate_for_everyone() -> None:
    state = _fresh_state(seed=0)
    pids = _player_ids(state)

    terminal = _terminal_state(state, winner=None)
    # Even if a VP changed on the same step, stalemate flattens to zero.
    terminal.players[pids[0]].victory_points_public = (
        state.players[pids[0]].victory_points_public + 1
    )
    result = _make_step_result(
        terminal, events=[GameStalled(turn_number=99, reason=EndReason.STALEMATE_VP_STALL)]
    )

    fn = ShapedReward(vp_coef=0.05, turn_tick=-0.001, win_bonus=1.0)
    for pid in pids:
        assert fn.step_reward(state, result.action, result, pid) == 0.0


# ---------------------------------------------------------------------------
# Env integration — reward_fn is plumbed through end-to-end
# ---------------------------------------------------------------------------


def test_env_default_reward_is_sparse_zero_on_non_terminal() -> None:
    env = CatanEnv(seed=0)
    env.reset(seed=0)
    legal = env.legal_actions()
    _, reward, done, _ = env.step(legal[0])
    assert reward == 0.0
    assert not done


def test_env_uses_supplied_reward_fn() -> None:
    """ShapedReward returns the per-step turn tick on a non-VP-changing setup move."""
    env = CatanEnv(seed=0, reward_fn=ShapedReward(vp_coef=0.05, turn_tick=-0.001))
    env.reset(seed=0)
    # PlaceSettlementAction in INITIAL_SETTLEMENT increments settlements_built
    # *and* victory_points_public by 1, so we expect a positive shaped reward.
    legal = env.legal_actions()
    _, reward, _, _ = env.step(legal[0])
    assert reward == pytest.approx(0.05 - 0.001)


def test_env_smoke_run_rewards_finite_under_each_reward_fn() -> None:
    """50-step random rollout under each reward fn — never NaN/Inf."""
    for fn in (SparseWinReward(), ShapedReward()):
        env = CatanEnv(seed=11, reward_fn=fn)
        _, info = env.reset(seed=11)
        rng = random.Random(11)
        for _ in range(50):
            mask = info["action_mask"]
            idxs = np.argwhere(mask).flatten().tolist()
            if not idxs:
                break
            _, reward, done, info = env.step(int(rng.choice(idxs)))
            assert np.isfinite(reward)
            if done:
                break
