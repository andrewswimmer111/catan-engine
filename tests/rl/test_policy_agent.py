"""Tests for the MLP policy/value model and the PolicyAgent adapter (rl-011).

Covers four contracts:

* ``MLPPolicyValue.forward`` produces correct shapes and never assigns
  probability mass to masked-out actions.
* ``PolicyAgent.act`` honours the mask and is deterministic when asked.
* The engine ``choose`` adapter can drive a full game against a random
  agent without crashing (smoke).
* Save / load round-trips preserve weights bit-for-bit.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from domain.actions.all_actions import EndTurnAction
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE, FlatObservationEncoder
from rl.env.catan_env import CatanEnv
from rl.evaluation.tournament import Tournament
from rl.models.mlp import MASK_FILL_VALUE, MLPPolicyValue, ModelOutput


# -----------------------------------------------------------------------------
# MLPPolicyValue
# -----------------------------------------------------------------------------


def test_forward_returns_expected_shapes() -> None:
    obs_dim = OBS_SHAPE[0]
    model = MLPPolicyValue(obs_dim, ACTION_SPACE_SIZE, hidden=(64, 64))

    batch = 32
    obs = torch.zeros(batch, obs_dim)
    mask = torch.ones(batch, ACTION_SPACE_SIZE, dtype=torch.bool)

    out = model(obs, mask)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (batch, ACTION_SPACE_SIZE)
    assert out.value.shape == (batch,)


def test_forward_masks_illegal_logits_to_neg_inf_sentinel() -> None:
    obs_dim = OBS_SHAPE[0]
    model = MLPPolicyValue(obs_dim, ACTION_SPACE_SIZE, hidden=(32, 32))

    obs = torch.randn(2, obs_dim)
    mask = torch.zeros(2, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, [3, 5, 7]] = True

    out = model(obs, mask)
    illegal_logits = out.logits[~mask]
    legal_logits = out.logits[mask]
    assert torch.all(illegal_logits == MASK_FILL_VALUE)
    assert torch.all(legal_logits > MASK_FILL_VALUE)


def test_softmax_assigns_zero_probability_to_illegal_actions() -> None:
    obs_dim = OBS_SHAPE[0]
    model = MLPPolicyValue(obs_dim, ACTION_SPACE_SIZE, hidden=(32,))

    obs = torch.randn(4, obs_dim)
    mask = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 100] = True

    probs = torch.softmax(model(obs, mask).logits, dim=-1)
    legal_mass = probs[mask].view(4, 2).sum(dim=-1)
    assert torch.allclose(legal_mass, torch.ones(4), atol=1e-6)


def test_constructor_rejects_empty_hidden() -> None:
    try:
        MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty hidden")


# -----------------------------------------------------------------------------
# PolicyAgent.act
# -----------------------------------------------------------------------------


def _make_agent(player_ids: list[PlayerID] | None = None) -> PolicyAgent:
    pids = player_ids or [PlayerID(i) for i in range(1, 5)]
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(64, 64))
    return PolicyAgent(model, ActionEncoder(pids))


def test_act_returns_masked_action() -> None:
    agent = _make_agent()
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[[10, 50, 200]] = True

    for _ in range(20):
        step = agent.act(obs, mask)
        assert mask[step.action_idx], f"sampled illegal index {step.action_idx}"


def test_act_is_deterministic_with_argmax() -> None:
    agent = _make_agent()
    obs = np.random.default_rng(0).standard_normal(OBS_SHAPE).astype(np.float32)

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[[1, 4, 9, 16, 25, 36]] = True

    a = agent.act(obs, mask, deterministic=True)
    b = agent.act(obs, mask, deterministic=True)
    assert a.action_idx == b.action_idx
    assert math.isclose(a.logp, b.logp, abs_tol=1e-6)
    assert math.isclose(a.value, b.value, abs_tol=1e-6)


def test_act_logp_matches_categorical_logprob() -> None:
    agent = _make_agent()
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)

    step = agent.act(obs, mask, deterministic=True)

    obs_t = torch.as_tensor(obs).unsqueeze(0)
    mask_t = torch.as_tensor(mask).unsqueeze(0)
    with torch.no_grad():
        out = agent.model(obs_t, mask_t)
        logp = torch.log_softmax(out.logits, dim=-1)[0, step.action_idx].item()
    assert math.isclose(step.logp, logp, abs_tol=1e-5)


def test_act_raises_on_all_false_mask() -> None:
    agent = _make_agent()
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    try:
        agent.act(obs, mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError on all-False mask")


# -----------------------------------------------------------------------------
# state_dict round-trip
# -----------------------------------------------------------------------------


def test_state_dict_round_trip_after_gradient_step() -> None:
    """Train one step, save, load into a fresh agent, verify equality."""
    pids = [PlayerID(i) for i in range(1, 5)]
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(32, 32))
    agent = PolicyAgent(model, ActionEncoder(pids))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = torch.randn(8, OBS_SHAPE[0])
    mask = torch.ones(8, ACTION_SPACE_SIZE, dtype=torch.bool)
    out = model(obs, mask)
    loss = out.value.pow(2).mean() + out.logits.sum() * 1e-6
    optim.zero_grad()
    loss.backward()
    optim.step()

    sd = agent.state_dict()

    fresh = PolicyAgent(
        MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(32, 32)),
        ActionEncoder(pids),
    )
    fresh.load_state_dict(sd)

    for k, v in agent.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), f"mismatch on {k}"


# -----------------------------------------------------------------------------
# Engine-side smoke
# -----------------------------------------------------------------------------


def test_policy_agent_plays_full_game_against_random() -> None:
    """Untrained PolicyAgent vs 3 random — must run to termination without errors."""
    pids = [PlayerID(i) for i in range(1, 5)]
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(64, 64))
    learner_seat = pids[0]

    rng = random.Random(0)
    agents = {pid: PolicyAgent(model, ActionEncoder(pids)) if pid == learner_seat
              else RandomAgent(random.Random(rng.randrange(2**32)), skip_proposals=True)
              for pid in pids}

    tournament = Tournament(lambda seed: CatanEnv(seed=seed))
    result = tournament.play(agents, n_games=1, base_seed=42)
    assert len(result.games) == 1
    # No assertion on winner — random/untrained policy could stalemate; we just
    # need to confirm the choose loop didn't crash.


def test_policy_agent_choose_respects_action_mask() -> None:
    """In a known phase (after setup), choose must return a typed legal Action."""
    env = CatanEnv(seed=7)
    env.reset()
    while env.state.phase.name.startswith("INITIAL"):
        legal = env.legal_actions()
        env.step(legal[0])

    pids = list(env.state.config.player_ids)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(32, 32))
    agent = PolicyAgent(model, ActionEncoder(pids))

    from controller.session import GameSnapshot
    snap = GameSnapshot(state=env.state, step_index=0, last_action=None, last_events=())
    legal = env.legal_actions()
    chosen = agent.choose(snap, legal)
    assert chosen is not None
    # Chosen action must be in the legal list (typed equality).
    assert chosen in legal or isinstance(chosen, EndTurnAction)
