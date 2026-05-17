"""Integration smoke for :class:`rl.agents.search_agent.SearchAgent`.

Validates that the MCTS wrapper composes with the tournament harness end
to end: SearchAgent takes a trained PolicyAgent, runs PUCT MCTS at every
non-forced decision, and produces typed actions the engine accepts for a
full game. Marked ``slow`` — even with a small rollout budget this runs
~10–30 seconds, well above the fast-suite budget.
"""

from __future__ import annotations

import random

import pytest

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.hybrid_agent import PlacementOverrideAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.search_agent import SearchAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.tournament import Tournament
from rl.search.mcts import MCTSConfig

PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


@pytest.fixture(scope="module")
def loaded_policy():
    """Load the run #5 checkpoint once for the module's tests.

    Skips the suite if the checkpoint isn't on disk — the test exists for
    the smoke property of SearchAgent against a real model; without the
    artifact there's nothing to integrate against.
    """
    from pathlib import Path

    from rl.training.checkpoint import load_checkpoint

    path = Path("runs/overnight_20260515_2006/final.pt")
    if not path.is_file():
        pytest.skip(f"checkpoint not present at {path}")
    agent, _meta = load_checkpoint(path)
    return agent


@pytest.mark.slow
def test_search_agent_finishes_full_game_vs_random(loaded_policy):
    """A SearchAgent wrapper plays out a full game vs three random
    opponents without producing illegal/None actions, and the engine
    reaches a winner-or-stalemate terminal state.
    """
    search = SearchAgent(loaded_policy, MCTSConfig(rollouts=12, seed=0))
    learner_seat = PLAYER_IDS[0]
    agents: dict[PlayerID, Agent] = {learner_seat: search}
    rng = random.Random(0)
    for pid in PLAYER_IDS[1:]:
        agents[pid] = RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)

    result = Tournament(_env_factory).play(agents, n_games=1, base_seed=0)
    game = result.games[0]
    # The game terminated cleanly — either with a winner or a stalemate
    # the engine assigned an end_reason to.
    assert game.end_reason is not None
    # The search seat actually built something during the game (initial
    # placements alone produce 2 settlements + 2 roads).
    per_seat = game.per_seat_action_histogram[learner_seat]
    assert per_seat.get("PlaceSettlementAction", 0) >= 2
    assert per_seat.get("PlaceRoadAction", 0) >= 2


@pytest.mark.slow
def test_search_agent_composes_with_placement_override(loaded_policy):
    """``PlacementOverrideAgent(SearchAgent(learner), heuristic)`` plays out
    cleanly: heuristic handles the openings, MCTS handles MAIN-phase decisions.
    Validates the two wrappers stack without interfering."""
    search = SearchAgent(loaded_policy, MCTSConfig(rollouts=8, seed=0))
    composite = PlacementOverrideAgent(
        main_agent=search, placement_agent=HeuristicAgent()
    )
    learner_seat = PLAYER_IDS[0]
    agents: dict[PlayerID, Agent] = {learner_seat: composite}
    rng = random.Random(0)
    for pid in PLAYER_IDS[1:]:
        agents[pid] = RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)

    result = Tournament(_env_factory).play(agents, n_games=1, base_seed=1)
    game = result.games[0]
    assert game.end_reason is not None
    per_seat = game.per_seat_action_histogram[learner_seat]
    assert per_seat.get("PlaceSettlementAction", 0) >= 2
