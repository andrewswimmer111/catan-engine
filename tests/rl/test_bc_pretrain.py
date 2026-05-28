"""Tests for :mod:`rl.training.bc_pretrain`.

Two layers:

* **Fast unit tests** drive the supervised loss on hand-crafted synthetic
  transitions (no real games), pin the constructor guards, and verify the
  loss actually descends on a memorisable batch. Milliseconds each.
* **Slow integration tests** roll out real heuristic games through
  :func:`generate_bc_transitions`, pin the per-sample invariants
  (one-hot legal targets, value/vp-aux shapes), and confirm a trained
  checkpoint round-trips as an AZ-compatible (graph encoder, vector value
  head) artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from pathlib import Path  # noqa: E402

from domain.ids import PlayerID  # noqa: E402
from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.stalemate_value import StalemateValueConfig  # noqa: E402
from rl.training.bc_pretrain import (  # noqa: E402
    BCConfig,
    generate_bc_transitions,
    train_bc,
)
from rl.training.checkpoint import (  # noqa: E402
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    load_checkpoint,
    model_arch_from,
    obs_layout_version_for,
    save_checkpoint,
)
from rl.training.self_play import SelfPlayTransition  # noqa: E402


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
N_PLAYERS = len(PLAYER_IDS)


_TINY_GNN_VECTOR = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
)
_TINY_GNN_SCALAR = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
    value_kind="scalar",
)


def _make_agent(arch: GNNArch, seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0], action_dim=ACTION_SPACE_SIZE, arch=arch
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
        device="cpu",
    )


def _synthetic_transition(
    rng: np.random.Generator, n_legal: int = 5
) -> SelfPlayTransition:
    """A self-contained BC sample with a one-hot target on a legal slot."""
    obs = rng.standard_normal(GRAPH_OBS_SHAPE[0]).astype(np.float32)
    legal = rng.choice(ACTION_SPACE_SIZE, size=n_legal, replace=False)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    mask[legal] = True
    target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    target[legal[0]] = 1.0
    value = np.zeros(N_PLAYERS, dtype=np.float32)
    value[0] = 1.0
    vp_aux = rng.random(N_PLAYERS).astype(np.float32)
    return SelfPlayTransition(
        obs=obs,
        action_mask=mask,
        mcts_policy=target,
        acting_seat_idx=0,
        value_target=value,
        vp_aux_target=vp_aux,
    )


# ----------------------------------------------------------------------
# Fast: constructor guards
# ----------------------------------------------------------------------


def test_train_bc_rejects_scalar_value_head() -> None:
    """A scalar-head model can't carry the per-seat BC value target."""
    agent = _make_agent(_TINY_GNN_SCALAR)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="vector value head"):
        train_bc(agent, [_synthetic_transition(rng)], BCConfig(epochs=1))


def test_train_bc_rejects_empty_transitions() -> None:
    agent = _make_agent(_TINY_GNN_VECTOR)
    with pytest.raises(ValueError, match="empty transition list"):
        train_bc(agent, [], BCConfig(epochs=1))


# ----------------------------------------------------------------------
# Fast: the supervised loss descends on a memorisable batch
# ----------------------------------------------------------------------


def test_train_bc_descends_on_synthetic_batch() -> None:
    """Cloning a fixed, memorisable set of samples should reduce policy
    loss and raise top-1 accuracy across epochs."""
    rng = np.random.default_rng(7)
    transitions = [_synthetic_transition(rng) for _ in range(8)]
    agent = _make_agent(_TINY_GNN_VECTOR)
    cfg = BCConfig(epochs=30, batch_size=8, lr=1e-2, value_coef=1.0, seed=0)

    history: list[dict[str, float]] = []
    train_bc(agent, transitions, cfg, on_epoch_end=history.append)

    assert len(history) == cfg.epochs
    first, last = history[0], history[-1]
    assert all(np.isfinite(v) for v in last.values())
    assert last["policy_loss"] < first["policy_loss"]
    assert last["accuracy"] >= first["accuracy"]
    # Per-epoch summaries expose what the driver prints / tables.
    assert last["epoch"] == float(cfg.epochs)
    assert last["n_samples"] == float(len(transitions))


# ----------------------------------------------------------------------
# Slow: real heuristic-game generation invariants
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_generate_bc_transitions_invariants() -> None:
    """Every recorded sample is a genuine, encodable, one-hot expert choice."""
    cfg = BCConfig(n_games=3, victory_point_target=6, max_moves=300, seed=0)
    transitions = generate_bc_transitions(cfg)

    assert transitions, "expected a non-empty expert corpus"
    for t in transitions:
        assert t.obs.shape == (GRAPH_OBS_SHAPE[0],)
        assert t.action_mask.shape == (ACTION_SPACE_SIZE,)
        assert t.action_mask.dtype == np.bool_
        assert t.mcts_policy.shape == (ACTION_SPACE_SIZE,)
        # One-hot policy target on a legal slot.
        assert t.mcts_policy.sum() == pytest.approx(1.0)
        hot = int(t.mcts_policy.argmax())
        assert t.action_mask[hot]
        # Only genuine choices were recorded (forced / collapsed-discard
        # states have a single encodable legal action and are skipped).
        assert int(t.action_mask.sum()) > 1
        assert t.value_target.shape == (N_PLAYERS,)
        assert t.vp_aux_target.shape == (N_PLAYERS,)


@pytest.mark.slow
def test_generate_bc_transitions_seed_is_deterministic() -> None:
    cfg = BCConfig(n_games=2, victory_point_target=6, max_moves=200, seed=3)
    a = generate_bc_transitions(cfg)
    b = generate_bc_transitions(cfg)
    assert len(a) == len(b)
    assert np.array_equal(a[0].mcts_policy, b[0].mcts_policy)
    assert np.array_equal(a[-1].obs, b[-1].obs)


# ----------------------------------------------------------------------
# Slow: trained checkpoint round-trips as an AZ-compatible artefact
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_bc_checkpoint_round_trips_as_az_compatible(tmp_path: Path) -> None:
    """A BC-trained checkpoint must load as a graph-encoder, vector-value
    model so ``train_alphazero.py --init-from`` accepts it unchanged."""
    cfg = BCConfig(
        n_games=2, victory_point_target=6, max_moves=200, epochs=1, batch_size=64
    )
    transitions = generate_bc_transitions(cfg)
    agent = _make_agent(_TINY_GNN_VECTOR)
    train_bc(agent, transitions, cfg)

    arch = model_arch_from(agent.model)
    meta = CheckpointMeta(
        obs_layout_version=obs_layout_version_for(arch.encoder_kind),
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=arch,
        train_step=0,
        timestamp=0.0,
        config_hash="bc_pretrain",
    )
    ckpt = tmp_path / "final.pt"
    save_checkpoint(agent, ckpt, meta)

    loaded, loaded_meta = load_checkpoint(ckpt, device="cpu")
    assert loaded_meta.model_arch.encoder_kind == "graph"
    assert loaded_meta.model_arch.gnn_arch is not None
    assert loaded_meta.model_arch.gnn_arch.value_kind == "vector"
    assert loaded.model.value_kind == "vector"
