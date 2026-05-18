"""Integration smoke for :class:`rl.agents.search_agent.SearchAgent`.

Validates that the MCTS wrapper composes with the tournament harness end
to end: SearchAgent takes a trained PolicyAgent, runs PUCT MCTS at every
non-forced decision, and produces typed actions the engine accepts for a
full game. Marked ``slow`` — even with a small rollout budget this runs
~10–30 seconds, well above the fast-suite budget.

Also covers fast unit tests for :class:`NetworkEvaluator`'s two
value-head paths (``scalar`` vs ``vector``): they must produce identical
absolute-seat value vectors when wrapped around the same underlying
weights, modulo the unrotation arithmetic the vector path applies.
"""

from __future__ import annotations

import random
from dataclasses import replace as _dc_replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from controller.agents import Agent  # noqa: E402
from domain.ids import PlayerID  # noqa: E402
from rl.agents.heuristic_agent import HeuristicAgent  # noqa: E402
from rl.agents.hybrid_agent import PlacementOverrideAgent  # noqa: E402
from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.agents.random_agent import RandomAgent  # noqa: E402
from rl.agents.search_agent import SearchAgent, NetworkEvaluator  # noqa: E402
from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv  # noqa: E402
from rl.evaluation.tournament import Tournament  # noqa: E402
from rl.models.gnn import DEFAULT_GNN_ARCH, GNNArch, GNNPolicyValue  # noqa: E402
from rl.search.mcts import MCTSConfig  # noqa: E402

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


# ----------------------------------------------------------------------
# NetworkEvaluator unit tests — both value-head paths
# ----------------------------------------------------------------------


_TINY_GNN = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
)


def _make_tiny_policy(value_kind: str, seed: int = 0) -> PolicyAgent:
    """A small graph-encoder PolicyAgent for fast NetworkEvaluator tests."""
    arch = _dc_replace(_TINY_GNN, value_kind=value_kind)  # type: ignore[arg-type]
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=arch,
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )


def _advance_env(seed: int, steps: int) -> CatanEnv:
    """Drive an env forward ``steps`` random actions; useful for getting
    past the deterministic initial-settlement phase so the acting seat is
    not always seat 0."""
    env = CatanEnv(seed=seed, obs_encoder=GraphObservationEncoder())
    env.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))
    return env


def test_network_evaluator_vector_path_value_vec_shape_and_priors() -> None:
    """Vector path: one forward, value_vec is per-seat and sums sensibly."""
    policy = _make_tiny_policy("vector")
    evaluator = NetworkEvaluator(policy)
    env = _advance_env(seed=11, steps=10)
    legal = env.legal_actions()
    assert legal, "test setup: env should still have legal actions"

    priors, value_vec = evaluator.evaluate(env.state, legal)
    assert priors.shape == (len(legal),)
    assert pytest.approx(priors.sum(), rel=1e-6) == 1.0
    assert (priors >= 0).all()
    assert value_vec.shape == (len(PLAYER_IDS),)
    assert np.isfinite(value_vec).all()


def test_network_evaluator_scalar_path_value_vec_shape_and_priors() -> None:
    """Scalar path: one batched 4-perspective forward; same Evaluator contract."""
    policy = _make_tiny_policy("scalar")
    evaluator = NetworkEvaluator(policy)
    env = _advance_env(seed=13, steps=10)
    legal = env.legal_actions()
    assert legal

    priors, value_vec = evaluator.evaluate(env.state, legal)
    assert priors.shape == (len(legal),)
    assert pytest.approx(priors.sum(), rel=1e-6) == 1.0
    assert value_vec.shape == (len(PLAYER_IDS),)
    assert np.isfinite(value_vec).all()


def test_network_evaluator_vector_unrotation_matches_acting_seat() -> None:
    """The vector path must place the acting seat's prediction in
    ``value_vec[acting_idx]``, not slot 0. We feed the model's output
    through a controlled stub-replacement on the underlying value head
    and verify ``np.roll`` rotates correctly for a non-zero acting seat.
    """
    policy = _make_tiny_policy("vector")

    # Find a state where the acting seat is non-zero — that's where the
    # unrotation arithmetic actually does work. The initial-settlement
    # phase steps through each seat in turn, so a few random steps
    # reliably produce a non-zero acting_idx.
    from rl.acting_player import acting_player

    acting_idx = 0
    env = None
    for steps in range(1, 10):
        env = _advance_env(seed=17, steps=steps)
        acting_pid = acting_player(env.state)
        acting_idx = list(env.state.config.player_ids).index(acting_pid)
        if acting_idx != 0:
            break
    assert env is not None and acting_idx != 0, (
        "could not reach a non-zero-acting-seat state — unrotation test is "
        "degenerate without rotation"
    )
    legal = env.legal_actions()
    assert legal

    # Replace the model's value_head with a constant module returning
    # a known rotated vector [1, 2, 3, 4]. The evaluator's vector path
    # should unrotate this to the absolute-seat layout.
    rotated = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    class _ConstantValueHead(torch.nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("_out", out)
            # The PyTorch Linear API the evaluator inspects has an
            # ``out_features``; we mimic it so other code paths stay
            # happy if they introspect.
            self.out_features = out.size(-1)

        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            return self._out

    policy.model.value_head = _ConstantValueHead(rotated)  # type: ignore[assignment]

    evaluator = NetworkEvaluator(policy)
    _priors, value_vec = evaluator.evaluate(env.state, legal)

    # The model emits [1,2,3,4] rotated to viewer-as-slot-0 (so slot 0 =
    # acting seat). After unrotation, value_vec[acting_idx] == 1.0 and
    # value_vec[(acting_idx + k) % 4] == k+1.
    n_players = len(PLAYER_IDS)
    expected = np.zeros(n_players)
    for k in range(n_players):
        expected[(acting_idx + k) % n_players] = float(k + 1)
    np.testing.assert_allclose(value_vec, expected)
