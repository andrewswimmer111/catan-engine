"""Tests for Agent protocol, HumanAgent, and ScriptedAgent."""

from __future__ import annotations

import random
from typing import runtime_checkable

import pytest

from controller.agents import Agent, HumanAgent, ScriptedAgent
from controller.session import GameSession, GameSnapshot
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID


def _make_snapshot(seed: int = 0, n_players: int = 3) -> tuple[GameSnapshot, list]:
    pids = [PlayerID(i) for i in range(n_players)]
    cfg = GameConfig(player_ids=pids, seed=seed)
    engine = GameEngine(SeededRandomizer(seed))
    state = engine.new_game(cfg)
    legals = engine.legal_actions(state)
    snap = GameSnapshot(state, 0, None, ())
    return snap, legals


class TestAgentProtocol:
    def test_human_agent_satisfies_protocol(self):
        agent: Agent = HumanAgent()
        snap, legals = _make_snapshot()
        assert agent.choose(snap, legals) is None

    def test_scripted_agent_satisfies_protocol(self):
        agent: Agent = ScriptedAgent(random.Random(0))
        snap, legals = _make_snapshot()
        result = agent.choose(snap, legals)
        assert result in legals

    def test_concrete_classes_do_not_inherit_agent(self):
        assert Agent not in HumanAgent.__mro__
        assert Agent not in ScriptedAgent.__mro__


class TestHumanAgent:
    def test_always_returns_none(self):
        agent = HumanAgent()
        snap, legals = _make_snapshot()
        assert agent.choose(snap, legals) is None
        assert agent.choose(snap, []) is None


class TestScriptedAgent:
    def test_returns_legal_action(self):
        snap, legals = _make_snapshot(seed=1)
        agent = ScriptedAgent(random.Random(1))
        action = agent.choose(snap, legals)
        assert action in legals

    def test_independent_rng_streams(self):
        snap, legals = _make_snapshot(seed=42)
        agent_a = ScriptedAgent(random.Random(7))
        agent_b = ScriptedAgent(random.Random(7))
        # Advance agent_a's rng independently
        agent_a.choose(snap, legals)
        agent_a.choose(snap, legals)
        # agent_b should still produce the same first choice as original agent_a
        agent_c = ScriptedAgent(random.Random(7))
        result_b = agent_b.choose(snap, legals)
        result_c = agent_c.choose(snap, legals)
        assert result_b == result_c

    def test_deterministic_given_seed(self):
        snap, legals = _make_snapshot(seed=5)
        result1 = ScriptedAgent(random.Random(5)).choose(snap, legals)
        result2 = ScriptedAgent(random.Random(5)).choose(snap, legals)
        assert result1 == result2

    def test_end_to_end_output_unchanged(self):
        """ScriptedAgent reproduces the same --seed 2 game result as the old _pick_action."""
        from src.main import play
        state = play(seed=2)
        from domain.rules import victory
        vp = {int(pid): victory.compute_victory_points(state, pid) for pid in state.config.player_ids}
        assert state.turn_number == 379
        assert vp == {0: 4, 1: 4, 2: 5, 3: 2}
        assert int(state.longest_road_holder) == 2
        assert int(state.largest_army_holder) == 1
