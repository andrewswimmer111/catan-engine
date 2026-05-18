"""Unit tests for :class:`GNNPolicyValue`.

Covers forward-pass shapes, batch invariance, mask-fill correctness, and
that the model interoperates with :class:`PolicyAgent` through the same
``ModelOutput`` contract the flat MLP exposes.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from dataclasses import replace as _dc_replace  # noqa: E402

from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    N_PLAYERS,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv  # noqa: E402
from rl.models.gnn import DEFAULT_GNN_ARCH, GNNArch, GNNPolicyValue  # noqa: E402
from rl.models.mlp import MASK_FILL_VALUE  # noqa: E402


def _make_model(arch: GNNArch | None = None) -> GNNPolicyValue:
    """Construct a GNNPolicyValue with the project default arch.

    DEFAULT_GNN_ARCH defaults to ``value_kind="vector"`` (AlphaZero-shape
    head). Tests that need the legacy ``"scalar"`` head pass a custom
    arch.
    """
    return GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=arch or DEFAULT_GNN_ARCH,
    )


def _scalar_arch(base: GNNArch | None = None) -> GNNArch:
    """Returns ``base`` (or the default) with ``value_kind="scalar"``."""
    return _dc_replace(base or DEFAULT_GNN_ARCH, value_kind="scalar")


def _encode_states(seeds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (obs_batch, mask_batch) from a few independent random rollouts."""
    obs_enc = GraphObservationEncoder()
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    for seed in seeds:
        env = CatanEnv(seed=seed)
        env.reset(seed=seed)
        rng = random.Random(seed)
        for _ in range(15):
            legal = env.legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))
        viewer = env.current_agent
        view = env._engine.player_view(env.state, viewer)
        obs_list.append(obs_enc.encode(view))
        mask_list.append(
            ActionEncoder.for_state(env.state).mask(env.legal_actions())
        )
    obs = torch.from_numpy(np.stack(obs_list))
    mask = torch.from_numpy(np.stack(mask_list))
    return obs, mask


def test_forward_shape_and_dtype_vector_default() -> None:
    """Default arch is ``value_kind="vector"``; value is (B, N_PLAYERS)."""
    model = _make_model()
    assert model.value_kind == "vector"
    obs, mask = _encode_states([1, 2, 3])
    out = model(obs, mask)
    assert out.logits.shape == (3, ACTION_SPACE_SIZE)
    assert out.value.shape == (3, N_PLAYERS)
    assert out.logits.dtype == torch.float32
    assert out.value.dtype == torch.float32


def test_forward_shape_and_dtype_scalar_mode() -> None:
    """``value_kind="scalar"`` collapses the value head back to (B,)."""
    model = _make_model(_scalar_arch())
    assert model.value_kind == "scalar"
    obs, mask = _encode_states([1, 2, 3])
    out = model(obs, mask)
    assert out.logits.shape == (3, ACTION_SPACE_SIZE)
    assert out.value.shape == (3,)
    assert out.value.dtype == torch.float32


def test_illegal_logits_are_masked_out() -> None:
    """Where ``mask=False``, the returned logits must be the fill value."""
    model = _make_model()
    obs, mask = _encode_states([5])
    out = model(obs, mask)
    illegal = ~mask.bool()
    masked_vals = out.logits[illegal]
    assert torch.all(masked_vals == MASK_FILL_VALUE)


def test_legal_logits_are_finite() -> None:
    model = _make_model()
    obs, mask = _encode_states([7])
    out = model(obs, mask)
    legal_vals = out.logits[mask.bool()]
    assert torch.isfinite(legal_vals).all()
    assert torch.isfinite(out.value).all()


def test_batched_and_single_forward_agree() -> None:
    """Stacking two single-row forward passes must equal one batched pass.

    Catches mistakes in the batched-edge tiling: if the per-sample offsets
    are wrong, samples bleed into each other and this comparison breaks.
    """
    model = _make_model()
    model.eval()  # deactivate dropout (we use 0 dropout but be explicit)
    obs, mask = _encode_states([11, 13])

    with torch.no_grad():
        out_batch = model(obs, mask)
        out_a = model(obs[:1], mask[:1])
        out_b = model(obs[1:], mask[1:])

    torch.testing.assert_close(out_batch.logits[0], out_a.logits[0])
    torch.testing.assert_close(out_batch.logits[1], out_b.logits[0])
    torch.testing.assert_close(out_batch.value[0:1], out_a.value)
    torch.testing.assert_close(out_batch.value[1:2], out_b.value)


def test_forward_rejects_wrong_obs_dim() -> None:
    model = _make_model()
    bad_obs = torch.zeros(2, GRAPH_OBS_SHAPE[0] + 1, dtype=torch.float32)
    mask = torch.ones(2, ACTION_SPACE_SIZE, dtype=torch.bool)
    with pytest.raises(ValueError):
        model(bad_obs, mask)


def test_forward_rejects_wrong_mask_shape() -> None:
    model = _make_model()
    obs = torch.zeros(2, GRAPH_OBS_SHAPE[0], dtype=torch.float32)
    bad_mask = torch.ones(2, ACTION_SPACE_SIZE + 1, dtype=torch.bool)
    with pytest.raises(ValueError):
        model(obs, bad_mask)


def test_constructor_rejects_wrong_obs_dim() -> None:
    with pytest.raises(ValueError):
        GNNPolicyValue(
            obs_dim=GRAPH_OBS_SHAPE[0] + 1, action_dim=ACTION_SPACE_SIZE
        )


def test_constructor_rejects_wrong_action_dim() -> None:
    with pytest.raises(ValueError):
        GNNPolicyValue(obs_dim=GRAPH_OBS_SHAPE[0], action_dim=42)


def test_constructor_rejects_indivisible_node_hidden() -> None:
    bad = GNNArch(node_hidden=130, n_heads=4)  # 130 % 4 != 0
    with pytest.raises(ValueError):
        GNNPolicyValue(
            obs_dim=GRAPH_OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, arch=bad
        )


def test_constructor_rejects_unknown_value_kind() -> None:
    """Typos in ``value_kind`` must fail loudly at constructor time,
    not silently miscompute downstream tensor shapes."""
    bad = _dc_replace(DEFAULT_GNN_ARCH, value_kind="per-seat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_kind"):
        GNNPolicyValue(
            obs_dim=GRAPH_OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, arch=bad
        )


def test_policy_agent_can_use_gnn_model() -> None:
    """The PolicyAgent path treats the model as a black box; swapping in
    the GNN model and the graph encoder must produce a usable agent."""
    from rl.agents.policy_agent import PolicyAgent
    from rl.encoding.action import ActionEncoder

    env = CatanEnv(seed=21)
    env.reset(seed=21)
    rng = random.Random(21)
    for _ in range(10):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))

    model = _make_model()
    agent = PolicyAgent(
        model=model,
        action_encoder=ActionEncoder.for_state(env.state),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )

    obs = GraphObservationEncoder().encode(
        env._engine.player_view(env.state, env.current_agent)
    )
    mask = ActionEncoder.for_state(env.state).mask(env.legal_actions())
    step = agent.act(obs, mask, deterministic=True)
    assert 0 <= step.action_idx < ACTION_SPACE_SIZE
    assert mask[step.action_idx]


def test_default_arch_param_count_is_in_target_band() -> None:
    """Sanity check that ``DEFAULT_GNN_ARCH`` lands near the MLP baseline (~3M).

    Catches accidental size explosions (e.g. dropping ``concat=True`` off
    GATv2Conv would 4x the params). 1M..6M is a healthy range; if this
    fails after a deliberate arch change, update the bounds.
    """
    model = _make_model()
    n = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < n < 6_000_000, (
        f"DEFAULT_GNN_ARCH param count {n:,} outside expected band; "
        "adjust DEFAULT_GNN_ARCH or update this bound."
    )
