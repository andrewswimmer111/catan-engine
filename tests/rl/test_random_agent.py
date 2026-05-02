import random

import pytest

from domain.actions.base import Action
from rl.agents.random_agent import RandomAgent, make_random_agents


class _FakeAction(Action):
    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeAction) and self.n == other.n


LEGAL = [_FakeAction(i) for i in range(5)]


def test_returns_action_from_legal():
    agent = RandomAgent(random.Random(0))
    result = agent.choose(snap=None, legal=LEGAL)  # type: ignore[arg-type]
    assert result in LEGAL


def test_returns_none_when_legal_empty():
    agent = RandomAgent(random.Random(0))
    assert agent.choose(snap=None, legal=[]) is None  # type: ignore[arg-type]


def test_same_seed_produces_identical_sequence():
    snaps = [None] * 20  # snap is unused by RandomAgent
    agent_a = RandomAgent(random.Random(42))
    agent_b = RandomAgent(random.Random(42))
    for snap in snaps:
        assert agent_a.choose(snap, LEGAL) == agent_b.choose(snap, LEGAL)  # type: ignore[arg-type]


def test_make_random_agents_keys_match_player_ids():
    pids = [1, 2, 3, 4]
    agents = make_random_agents(pids, seed=7)
    assert set(agents.keys()) == set(pids)
    assert all(isinstance(a, RandomAgent) for a in agents.values())
