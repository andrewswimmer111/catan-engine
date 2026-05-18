"""Cross-device correctness smoke tests for the GNN policy/value network.

Verifies that the same weights produce identical (within fp32 tolerance)
outputs on CPU vs the accelerator backends (MPS / CUDA). Catches silent
divergence from things like fused-kernel mismatches in GATv2 message
passing, LayerNorm numerics, or unsupported ops that silently no-op on
one backend.

Both accelerator tests skip when the backend isn't available — MPS only
exists on macOS with Metal-supporting GPUs and CUDA only when a CUDA
device is present.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv  # noqa: E402
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402


_TINY_GNN = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
)


def _seeded_model(seed: int) -> GNNPolicyValue:
    torch.manual_seed(seed)
    return GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=_TINY_GNN,
    )


def _sample_obs_and_mask(batch: int = 3, seed: int = 0):
    """A few seeded random env states encoded as (obs, mask) torch tensors."""
    obs_enc = GraphObservationEncoder()
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    rng = random.Random(seed)
    for i in range(batch):
        env = CatanEnv(seed=seed + i)
        env.reset(seed=seed + i)
        for _ in range(8):
            legal = env.legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))
        view = env._engine.player_view(env.state, env.current_agent)
        obs_list.append(obs_enc.encode(view))
        mask_list.append(
            ActionEncoder.for_state(env.state).mask(env.legal_actions())
        )
    obs = torch.from_numpy(np.stack(obs_list))
    mask = torch.from_numpy(np.stack(mask_list))
    return obs, mask


def _forward_on_device(model: GNNPolicyValue, obs: torch.Tensor, mask: torch.Tensor, device: torch.device):
    """Move ``model`` to ``device``, run a single forward, return CPU tensors."""
    model = model.to(device)
    obs_d = obs.to(device)
    mask_d = mask.to(device)
    model.eval()
    with torch.no_grad():
        out = model(obs_d, mask_d)
    return out.logits.detach().cpu(), out.value.detach().cpu()


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> None:
    """Compare with backend-tolerant tolerances; fp32 accelerator kernels can
    diverge from CPU by tens of ulps on accumulated ops like GATv2 attention,
    so we don't expect bit-exact equality.
    """
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


# ----------------------------------------------------------------------
# MPS
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    reason="MPS backend not available on this host",
)
def test_cpu_and_mps_forward_agree_within_fp32_tolerance() -> None:
    """The GNN forward on MPS must produce the same logits/value as CPU.

    Tolerance is loose (1e-4 absolute) because accelerator fp32 kernels
    on Metal can diverge from CPU by tens of ulps on accumulated
    attention/scatter ops. Anything tighter than this would be flaky
    without bit-stable kernels we don't control.
    """
    obs, mask = _sample_obs_and_mask(batch=2, seed=11)
    cpu_logits, cpu_value = _forward_on_device(_seeded_model(0), obs, mask, torch.device("cpu"))
    mps_logits, mps_value = _forward_on_device(_seeded_model(0), obs, mask, torch.device("mps"))

    _assert_close(cpu_logits, mps_logits, atol=1e-4, rtol=1e-4)
    _assert_close(cpu_value, mps_value, atol=1e-4, rtol=1e-4)
    # And shapes — guards against silent backend-only squeezes.
    assert cpu_logits.shape == mps_logits.shape
    assert cpu_value.shape == mps_value.shape


@pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    reason="MPS backend not available on this host",
)
def test_mps_policy_agent_act_returns_valid_action() -> None:
    """:meth:`PolicyAgent.act` works end-to-end on MPS — exercises the
    masked sampling path, the .item() boundary back to Python scalars,
    and the value head reduction for a vector-mode model."""
    from rl.agents.policy_agent import PolicyAgent
    from domain.ids import PlayerID

    agent = PolicyAgent(
        _seeded_model(7),
        ActionEncoder([PlayerID(i) for i in range(1, 5)]),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
        device="mps",
    )
    obs, mask = _sample_obs_and_mask(batch=1, seed=3)
    step = agent.act(obs[0].numpy(), mask[0].numpy(), deterministic=True)
    assert 0 <= step.action_idx < ACTION_SPACE_SIZE
    assert bool(mask[0, step.action_idx])
    assert np.isfinite(step.value)


# ----------------------------------------------------------------------
# CUDA (skipped on macOS CI; included for completeness)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available on this host"
)
def test_cpu_and_cuda_forward_agree_within_fp32_tolerance() -> None:
    obs, mask = _sample_obs_and_mask(batch=2, seed=23)
    cpu_logits, cpu_value = _forward_on_device(_seeded_model(0), obs, mask, torch.device("cpu"))
    cuda_logits, cuda_value = _forward_on_device(_seeded_model(0), obs, mask, torch.device("cuda"))
    _assert_close(cpu_logits, cuda_logits, atol=1e-4, rtol=1e-4)
    _assert_close(cpu_value, cuda_value, atol=1e-4, rtol=1e-4)
