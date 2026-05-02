"""Basic correctness tests for CatanEnv.reset() / .step()."""

from __future__ import annotations

import pytest

from domain.engine.game_engine import IllegalActionError
from domain.engine.player_view import PlayerView
from domain.enums import TurnPhase
from rl.agents.random_agent import make_random_agents
from rl.env.catan_env import CatanEnv


def _play_setup(env: CatanEnv) -> None:
    """Drive the env through the initial-placement phase by always picking the first legal action."""
    while env.state.phase in (TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD):
        legal = env.legal_actions()
        obs, _, done, _ = env.step(legal[0])
        if done:
            return


def test_reset_returns_player_view_in_initial_settlement() -> None:
    env = CatanEnv(seed=0)
    obs, info = env.reset()
    assert isinstance(obs, PlayerView)
    assert env.state.phase == TurnPhase.INITIAL_SETTLEMENT
    assert info["current_phase"] == TurnPhase.INITIAL_SETTLEMENT


def test_step_legal_action_advances_turn_number() -> None:
    env = CatanEnv(seed=1)
    env.reset()
    _play_setup(env)

    # Now in ROLL phase, turn_number == 1
    assert env.state.turn_number == 1

    legal = env.legal_actions()
    env.step(legal[0])  # RollDiceAction → MAIN

    # Find and apply EndTurnAction
    from domain.actions.all_actions import EndTurnAction
    end_actions = [a for a in env.legal_actions() if isinstance(a, EndTurnAction)]
    assert end_actions, "EndTurnAction must be legal in MAIN phase"
    env.step(end_actions[0])

    assert env.state.turn_number == 2


def test_step_illegal_action_raises() -> None:
    env = CatanEnv(seed=0)
    env.reset()
    # INITIAL_SETTLEMENT expects PlaceSettlementAction; EndTurnAction is illegal here.
    from domain.actions.all_actions import EndTurnAction
    from domain.ids import PlayerID
    bad = EndTurnAction(player_id=PlayerID(1))
    with pytest.raises(IllegalActionError):
        env.step(bad)


def test_smoke_no_crash_across_seeds() -> None:
    """50 random seeds, 200 steps each — verifies the env raises no exceptions."""
    for seed in range(50):
        env = CatanEnv(seed=seed)
        obs, info = env.reset()
        agents = make_random_agents(list(env.state.config.player_ids), seed=seed)
        for _ in range(200):
            legal = env.legal_actions()
            pid = env.current_agent
            action = agents[pid].choose(obs, legal)
            assert action is not None
            obs, _reward, done, info = env.step(action)
            if done:
                break


def test_smoke_game_reaches_terminal() -> None:
    """One seeded game must reach a terminal state (VP stall triggers by turn 1500)."""
    env = CatanEnv(seed=2)
    obs, info = env.reset()
    agents = make_random_agents(list(env.state.config.player_ids), seed=2)
    done = False
    for _ in range(200_000):
        legal = env.legal_actions()
        pid = env.current_agent
        action = agents[pid].choose(obs, legal)
        assert action is not None
        obs, _reward, done, info = env.step(action)
        if done:
            break
    assert done, "game did not reach terminal state"
