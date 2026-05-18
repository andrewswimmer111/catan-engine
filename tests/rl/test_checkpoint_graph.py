"""Tests for graph-encoder checkpoint round-trip and version dispatch.

Mirrors ``test_checkpoint.py`` but for the GNN encoder/model pair. Also
guards back-compat: old checkpoints without ``encoder_kind`` must continue
to load as flat-encoder artifacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_LAYOUT_VERSION,
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.encoding.observation import OBS_LAYOUT_VERSION, OBS_SHAPE  # noqa: E402
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.models.mlp import MLPPolicyValue  # noqa: E402
from rl.training import checkpoint as ckpt_mod  # noqa: E402
from rl.training.checkpoint import (  # noqa: E402
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    IncompatibleCheckpointError,
    ModelArch,
    compute_config_hash,
    load_checkpoint,
    save_checkpoint,
)
from rl.training.config import TrainConfig  # noqa: E402
from domain.ids import PlayerID  # noqa: E402

PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
# Default SMALL_GNN exercises the AZ-shape vector value head — the new
# default and what new graph checkpoints will be written with.
SMALL_GNN = GNNArch(
    node_hidden=32,
    player_hidden=16,
    n_mp_layers=2,
    n_heads=4,
    global_mlp_hidden=32,
)
# Mirror with an explicit scalar value head for the PPO back-compat path.
SMALL_GNN_SCALAR = GNNArch(
    node_hidden=32,
    player_hidden=16,
    n_mp_layers=2,
    n_heads=4,
    global_mlp_hidden=32,
    value_kind="scalar",
)


def _make_graph_agent(seed: int = 0, arch: GNNArch = SMALL_GNN) -> PolicyAgent:
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


def _graph_meta(train_step: int = 10, arch: GNNArch = SMALL_GNN) -> CheckpointMeta:
    return CheckpointMeta(
        obs_layout_version=GRAPH_OBS_LAYOUT_VERSION,
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=ModelArch(
            obs_dim=GRAPH_OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="graph",
            gnn_arch=arch,
        ),
        train_step=train_step,
        timestamp=time.time(),
        config_hash=compute_config_hash(TrainConfig()),
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_graph_checkpoint_round_trip_preserves_weights(tmp_path: Path) -> None:
    """Vector-value-head round trip (the new default for graph checkpoints)."""
    agent = _make_graph_agent(seed=7)
    meta = _graph_meta(train_step=1234)
    path = tmp_path / "gnn.pt"
    save_checkpoint(agent, path, meta)
    assert path.exists()

    loaded, loaded_meta = load_checkpoint(path)
    for k, v in agent.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"mismatch on {k}"

    assert loaded_meta.model_arch.encoder_kind == "graph"
    assert loaded_meta.model_arch.gnn_arch == SMALL_GNN
    assert loaded_meta.model_arch.gnn_arch.value_kind == "vector"
    assert loaded_meta.obs_layout_version == GRAPH_OBS_LAYOUT_VERSION
    assert loaded_meta.train_step == 1234


def test_graph_checkpoint_round_trip_scalar_value_head(tmp_path: Path) -> None:
    """PPO-back-compat path: ``value_kind="scalar"`` round-trips with a
    ``(out_features=1,)`` value head, distinct from the vector path."""
    agent = _make_graph_agent(seed=8, arch=SMALL_GNN_SCALAR)
    meta = _graph_meta(train_step=99, arch=SMALL_GNN_SCALAR)
    path = tmp_path / "gnn_scalar.pt"
    save_checkpoint(agent, path, meta)

    loaded, loaded_meta = load_checkpoint(path)
    for k, v in agent.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"mismatch on {k}"
    assert loaded_meta.model_arch.gnn_arch == SMALL_GNN_SCALAR
    assert loaded_meta.model_arch.gnn_arch.value_kind == "scalar"
    # Sanity: scalar value head has one output feature, vector has N_PLAYERS.
    assert isinstance(loaded.model, GNNPolicyValue)
    assert loaded.model.value_head.out_features == 1


def test_old_graph_checkpoint_without_value_kind_defaults_to_scalar(
    tmp_path: Path,
) -> None:
    """A graph checkpoint written before ``value_kind`` existed (i.e. the
    pre-az-001 PPO runs like ``runs/overnight_20260515_2006/final.pt``)
    must load as ``value_kind="scalar"``. Without the back-compat default
    the weight shapes would mismatch the new vector default and load_state_dict
    would fail."""
    agent = _make_graph_agent(seed=9, arch=SMALL_GNN_SCALAR)
    # Write a payload with a gnn_arch dict that lacks ``value_kind`` — the
    # exact shape pre-az-001 PPO checkpoints have on disk.
    legacy_gnn_arch = {
        "node_hidden": SMALL_GNN_SCALAR.node_hidden,
        "player_hidden": SMALL_GNN_SCALAR.player_hidden,
        "n_mp_layers": SMALL_GNN_SCALAR.n_mp_layers,
        "n_heads": SMALL_GNN_SCALAR.n_heads,
        "global_mlp_hidden": SMALL_GNN_SCALAR.global_mlp_hidden,
    }
    legacy_meta = {
        "obs_layout_version": GRAPH_OBS_LAYOUT_VERSION,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "model_arch": {
            "obs_dim": GRAPH_OBS_SHAPE[0],
            "action_dim": ACTION_SPACE_SIZE,
            "encoder_kind": "graph",
            "hidden": [],
            "gnn_arch": legacy_gnn_arch,
        },
        "train_step": 42,
        "timestamp": float(time.time()),
        "config_hash": compute_config_hash(TrainConfig()),
    }
    path = tmp_path / "legacy_graph.pt"
    torch.save({"model": agent.state_dict(), "meta": legacy_meta}, str(path))

    loaded, meta = load_checkpoint(path)
    assert meta.model_arch.encoder_kind == "graph"
    assert meta.model_arch.gnn_arch is not None
    assert meta.model_arch.gnn_arch.value_kind == "scalar"
    assert isinstance(loaded.model, GNNPolicyValue)
    assert loaded.model.value_head.out_features == 1


def test_load_rejects_checkpoint_with_unknown_value_kind(tmp_path: Path) -> None:
    """A hand-edited checkpoint with an unrecognised ``value_kind`` must
    fail loudly rather than silently defaulting."""
    agent = _make_graph_agent(seed=10)
    bad_gnn_arch = {
        "node_hidden": SMALL_GNN.node_hidden,
        "player_hidden": SMALL_GNN.player_hidden,
        "n_mp_layers": SMALL_GNN.n_mp_layers,
        "n_heads": SMALL_GNN.n_heads,
        "global_mlp_hidden": SMALL_GNN.global_mlp_hidden,
        "value_kind": "per-seat",  # typo
    }
    bad_meta = {
        "obs_layout_version": GRAPH_OBS_LAYOUT_VERSION,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "model_arch": {
            "obs_dim": GRAPH_OBS_SHAPE[0],
            "action_dim": ACTION_SPACE_SIZE,
            "encoder_kind": "graph",
            "hidden": [],
            "gnn_arch": bad_gnn_arch,
        },
        "train_step": 0,
        "timestamp": float(time.time()),
        "config_hash": compute_config_hash(TrainConfig()),
    }
    path = tmp_path / "bad.pt"
    torch.save({"model": agent.state_dict(), "meta": bad_meta}, str(path))

    with pytest.raises(IncompatibleCheckpointError, match="value_kind"):
        load_checkpoint(path)


def test_loaded_graph_agent_uses_graph_obs_encoder(tmp_path: Path) -> None:
    """Confirm the loader wires a GraphObservationEncoder into the agent."""
    agent = _make_graph_agent()
    meta = _graph_meta()
    path = tmp_path / "gnn.pt"
    save_checkpoint(agent, path, meta)

    loaded, _ = load_checkpoint(path)
    assert isinstance(loaded.obs_encoder, GraphObservationEncoder)
    assert isinstance(loaded.model, GNNPolicyValue)


# ---------------------------------------------------------------------------
# Version dispatch
# ---------------------------------------------------------------------------


def test_load_checks_graph_obs_layout_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph checkpoint must be validated against GRAPH_OBS_LAYOUT_VERSION,
    not OBS_LAYOUT_VERSION."""
    agent = _make_graph_agent()
    meta = _graph_meta()
    path = tmp_path / "gnn.pt"
    save_checkpoint(agent, path, meta)

    # Bump only the *graph* obs layout version. The flat constant is
    # unchanged — a buggy loader that checks the flat constant would let
    # this load.
    monkeypatch.setattr(
        ckpt_mod, "GRAPH_OBS_LAYOUT_VERSION", meta.obs_layout_version + 1
    )

    with pytest.raises(IncompatibleCheckpointError) as exc:
        load_checkpoint(path)
    assert "obs_layout_version" in str(exc.value)


def test_old_flat_checkpoint_without_encoder_kind_still_loads(
    tmp_path: Path,
) -> None:
    """A checkpoint written before ``encoder_kind`` existed must default to ``"flat"``.

    Simulates an old payload by writing the legacy meta dict shape (no
    ``encoder_kind`` key) directly to disk, then loading through the
    current code path.
    """
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(32, 32))
    agent = PolicyAgent(model, ActionEncoder(PLAYER_IDS))

    legacy_meta = {
        "obs_layout_version": OBS_LAYOUT_VERSION,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "model_arch": {
            "obs_dim": OBS_SHAPE[0],
            "action_dim": ACTION_SPACE_SIZE,
            "hidden": [32, 32],
        },
        "train_step": 999,
        "timestamp": float(time.time()),
        "config_hash": compute_config_hash(TrainConfig()),
    }
    path = tmp_path / "legacy.pt"
    torch.save({"model": agent.state_dict(), "meta": legacy_meta}, str(path))

    loaded, meta = load_checkpoint(path)
    assert meta.model_arch.encoder_kind == "flat"
    assert meta.model_arch.hidden == (32, 32)
    assert isinstance(loaded.model, MLPPolicyValue)


# ---------------------------------------------------------------------------
# ModelArch invariants
# ---------------------------------------------------------------------------


def test_model_arch_flat_requires_hidden() -> None:
    with pytest.raises(ValueError):
        ModelArch(
            obs_dim=OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="flat",
            hidden=(),
        )


def test_model_arch_flat_rejects_gnn_arch() -> None:
    with pytest.raises(ValueError):
        ModelArch(
            obs_dim=OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="flat",
            hidden=(32,),
            gnn_arch=SMALL_GNN,
        )


def test_model_arch_graph_requires_gnn_arch() -> None:
    with pytest.raises(ValueError):
        ModelArch(
            obs_dim=GRAPH_OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="graph",
        )


def test_model_arch_graph_rejects_hidden() -> None:
    with pytest.raises(ValueError):
        ModelArch(
            obs_dim=GRAPH_OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="graph",
            hidden=(32,),
            gnn_arch=SMALL_GNN,
        )


def test_model_arch_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ModelArch(
            obs_dim=GRAPH_OBS_SHAPE[0],
            action_dim=ACTION_SPACE_SIZE,
            encoder_kind="hetero",  # type: ignore[arg-type]
            gnn_arch=SMALL_GNN,
        )
