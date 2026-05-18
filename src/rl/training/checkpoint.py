"""Versioned checkpoint serialization for :class:`PolicyAgent`.

A checkpoint is a single torch ``.pt`` file containing ``{"model": state_dict,
"meta": meta_dict}`` — both plain dicts of tensors / primitives. No arbitrary
Python objects are pickled so the file can be loaded under
``weights_only=False`` without surprises.

Self-describing meta
--------------------

:class:`CheckpointMeta` carries every field the loader needs to rebuild the
policy from disk with no external config: layout versions for the observation
and action layouts (refused at load if they mismatch the current code), the
:class:`ModelArch` knobs needed to size the policy (the architecture is
either ``MLPPolicyValue`` for the flat encoder or :class:`GNNPolicyValue`
for the graph encoder — see ``encoder_kind`` below), the ``train_step`` the
snapshot was taken at, a wall-clock ``timestamp``, and a ``config_hash``
over the training config so accidental drift between runs shows up as a
visible field difference.

Player IDs are intentionally **not** stored. The model is seat-agnostic —
the observation encoder rotates the viewer into "self" on every forward
pass — and the action layout is fixed to four opponent slots by
:data:`ACTION_LAYOUT_VERSION`. Runtime player IDs come from the game
config, not the checkpoint.

Encoder kind
------------

``ModelArch.encoder_kind`` discriminates which (encoder, model) pair the
checkpoint was trained against. ``"flat"`` is the original
:class:`FlatObservationEncoder` + :class:`MLPPolicyValue` path; ``"graph"``
is the :class:`GraphObservationEncoder` + :class:`GNNPolicyValue` path
introduced in Phase 1. ``obs_layout_version`` is checked against the
matching constant for the chosen encoder. Old checkpoints written before
``encoder_kind`` existed are treated as ``"flat"`` for back-compat.

Value-head kind
---------------

``GNNArch.value_kind`` (carried inside ``ModelArch.gnn_arch`` for the
graph encoder) records whether the value head is ``"scalar"`` (PPO
back-compat: ``(B,)``) or ``"vector"`` (AlphaZero per-seat: ``(B,
N_PLAYERS)``). The loader reconstructs the right head shape from this
field, so a vector-trained model and a scalar-trained model of the same
GNN dimensions are not interchangeable: loading a scalar state-dict into
a vector model (or vice versa) raises at ``load_state_dict`` time
because the value-head linear has a different out-feature count. Old
graph checkpoints written before this field existed are treated as
``"scalar"`` for back-compat.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding._action_layout import _ACTION_LAYOUT_VERSION
from rl.encoding.action import ActionEncoder
from rl.encoding.graph_observation import (
    GRAPH_OBS_LAYOUT_VERSION,
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.encoding.observation import OBS_LAYOUT_VERSION
from rl.models.gnn import GNNArch, GNNPolicyValue
from rl.models.mlp import MLPPolicyValue
from rl.training.config import TrainConfig

__all__ = [
    "ACTION_LAYOUT_VERSION",
    "OBS_LAYOUT_VERSION",
    "GRAPH_OBS_LAYOUT_VERSION",
    "EncoderKind",
    "CheckpointMeta",
    "IncompatibleCheckpointError",
    "ModelArch",
    "build_agent_for_arch",
    "compute_config_hash",
    "load_checkpoint",
    "save_checkpoint",
    "model_arch_from",
    "obs_layout_version_for",
]


ACTION_LAYOUT_VERSION: int = _ACTION_LAYOUT_VERSION

EncoderKind = Literal["flat", "graph"]


# Default 4-seat player_ids used by ``CatanEnv``. The action layout's steal
# slots are seat-indexed, so the loaded encoder needs *some* PlayerID list to
# decode index→Action; the env's default config uses these IDs, and a policy
# trained against a non-default config is the caller's problem to handle by
# rebuilding the encoder externally.
_DEFAULT_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]


@dataclass(frozen=True)
class ModelArch:
    """Architecture knobs needed to reconstruct the policy/value network.

    The discriminator is :attr:`encoder_kind`. For ``"flat"`` the
    :attr:`hidden` tuple sizes a :class:`MLPPolicyValue`; for ``"graph"``
    the :attr:`gnn_arch` dataclass sizes a :class:`GNNPolicyValue`.
    Exactly one of the two carries the per-encoder knobs (the other is
    empty/None) — this is enforced in ``__post_init__``.
    """

    obs_dim: int
    action_dim: int
    encoder_kind: EncoderKind = "flat"
    hidden: tuple[int, ...] = ()
    gnn_arch: GNNArch | None = None

    def __post_init__(self) -> None:
        if self.encoder_kind == "flat":
            if not self.hidden:
                raise ValueError(
                    "encoder_kind='flat' requires non-empty hidden sizes"
                )
            if self.gnn_arch is not None:
                raise ValueError(
                    "encoder_kind='flat' must not carry gnn_arch"
                )
        elif self.encoder_kind == "graph":
            if self.gnn_arch is None:
                raise ValueError(
                    "encoder_kind='graph' requires gnn_arch"
                )
            if self.hidden:
                raise ValueError(
                    "encoder_kind='graph' must not carry hidden sizes"
                )
        else:
            raise ValueError(f"unknown encoder_kind: {self.encoder_kind!r}")


@dataclass(frozen=True)
class CheckpointMeta:
    """Self-describing checkpoint metadata.

    Layout version fields gate compatibility: a checkpoint trained against a
    different obs or action layout would have shape-mismatched weights or
    mis-mapped action indices, so :func:`load_checkpoint` refuses it. The
    obs layout version is checked against the constant that matches the
    encoder kind recorded in :attr:`model_arch`.
    """

    obs_layout_version: int
    action_layout_version: int
    model_arch: ModelArch
    train_step: int
    timestamp: float
    config_hash: str


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint's layout versions don't match the loader."""


def model_arch_from(model: torch.nn.Module) -> ModelArch:
    """Derive a :class:`ModelArch` spec from a constructed policy/value model.

    Single source of truth for "what kind of model is this" — the trainer
    and the warm-start path both consult this rather than inspecting
    ``model`` directly, so adding a new encoder kind only touches this
    function plus the loader dispatch in :func:`load_checkpoint`.
    """
    if isinstance(model, MLPPolicyValue):
        return ModelArch(
            obs_dim=model.obs_dim,
            action_dim=model.action_dim,
            encoder_kind="flat",
            hidden=tuple(model.hidden),
        )
    if isinstance(model, GNNPolicyValue):
        return ModelArch(
            obs_dim=model.obs_dim,
            action_dim=model.action_dim,
            encoder_kind="graph",
            gnn_arch=model.arch,
        )
    raise TypeError(
        f"don't know how to build ModelArch for {type(model).__name__}; "
        "extend rl.training.checkpoint.model_arch_from to support it."
    )


def obs_layout_version_for(kind: EncoderKind) -> int:
    """Return the current obs-layout version constant matching ``kind``.

    Exposed so callers (trainer, scripts) don't reach into the loader's
    private dispatch table when building :class:`CheckpointMeta`.
    """
    return _expected_obs_layout_version(kind)


def compute_config_hash(train_cfg: TrainConfig) -> str:
    """SHA256 over a stable JSON serialization of the training config.

    Includes the nested :class:`PPOConfig` automatically (it's a dataclass
    field on ``TrainConfig``). The hash is purely a *drift detector*: two
    runs with the same hash had the same config, two runs with different
    hashes did not. The hash is not interpretable on its own.
    """
    payload = _stable_jsonable(asdict(train_cfg))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_checkpoint(agent: PolicyAgent, path: Path, meta: CheckpointMeta) -> None:
    """Write ``agent``'s model weights and ``meta`` to ``path`` as a torch ``.pt``.

    Parent directories are created if needed. The on-disk payload has exactly
    two top-level keys: ``"model"`` (state_dict) and ``"meta"`` (plain dict).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": agent.state_dict(),
        "meta": _meta_to_dict(meta),
    }
    torch.save(payload, str(out_path))


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyAgent, CheckpointMeta]:
    """Load a checkpoint, validate compatibility, and return ``(agent, meta)``.

    The encoder kind recorded in ``meta.model_arch`` selects both the
    obs-layout version to check and the (model, obs_encoder) pair to
    instantiate. Old checkpoints written before ``encoder_kind`` existed
    are treated as ``"flat"`` so existing artifacts keep loading.

    ``device`` selects where the constructed model lives. We always
    deserialize the raw tensors with ``map_location="cpu"`` and then
    move the assembled :class:`PolicyAgent` onto the requested device —
    this avoids deserialization-time issues on non-CPU backends and lets
    a single checkpoint round-trip cleanly to any device.

    Raises :class:`IncompatibleCheckpointError` if either layout version on
    disk differs from the current codebase. Raises ``FileNotFoundError`` if
    ``path`` does not exist.
    """
    in_path = Path(path)
    payload = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "meta" not in payload or "model" not in payload:
        raise ValueError(
            f"checkpoint at {in_path} is not a {{model, meta}} payload"
        )
    meta = _meta_from_dict(payload["meta"])
    _validate_layout_versions(meta)

    agent = _build_agent_for_arch(meta.model_arch, device=device)
    agent.load_state_dict(payload["model"])
    return agent, meta


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _expected_obs_layout_version(kind: EncoderKind) -> int:
    if kind == "flat":
        return OBS_LAYOUT_VERSION
    if kind == "graph":
        return GRAPH_OBS_LAYOUT_VERSION
    raise ValueError(f"unknown encoder_kind: {kind!r}")


def _validate_layout_versions(meta: CheckpointMeta) -> None:
    expected_obs = _expected_obs_layout_version(meta.model_arch.encoder_kind)
    if meta.obs_layout_version != expected_obs:
        raise IncompatibleCheckpointError(
            f"checkpoint obs_layout_version={meta.obs_layout_version} "
            f"!= current obs_layout_version={expected_obs} "
            f"(encoder_kind={meta.model_arch.encoder_kind!r})"
        )
    if meta.action_layout_version != ACTION_LAYOUT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint action_layout_version={meta.action_layout_version} "
            f"!= current ACTION_LAYOUT_VERSION={ACTION_LAYOUT_VERSION}"
        )


def build_agent_for_arch(
    arch: ModelArch,
    *,
    device: str | torch.device = "cpu",
) -> PolicyAgent:
    """Construct a fresh :class:`PolicyAgent` matching ``arch`` on ``device``.

    Public counterpart of the loader's internal dispatch — useful for
    subprocess workers (az-009) and any other code path that needs to
    rebuild a PolicyAgent from a serialised arch + state-dict pair
    without writing the checkpoint to disk first.
    """
    return _build_agent_for_arch(arch, device=device)


def _build_agent_for_arch(
    arch: ModelArch,
    *,
    device: str | torch.device = "cpu",
) -> PolicyAgent:
    if arch.encoder_kind == "flat":
        model: torch.nn.Module = MLPPolicyValue(
            obs_dim=arch.obs_dim,
            action_dim=arch.action_dim,
            hidden=arch.hidden,
        )
        encoder = ActionEncoder(list(_DEFAULT_PLAYER_IDS))
        return PolicyAgent(model, encoder, device=device)  # type: ignore[arg-type]
    if arch.encoder_kind == "graph":
        assert arch.gnn_arch is not None  # validated by ModelArch.__post_init__
        if arch.obs_dim != GRAPH_OBS_SHAPE[0]:
            raise IncompatibleCheckpointError(
                f"graph checkpoint obs_dim={arch.obs_dim} != "
                f"GRAPH_OBS_SHAPE[0]={GRAPH_OBS_SHAPE[0]}"
            )
        model = GNNPolicyValue(
            obs_dim=arch.obs_dim,
            action_dim=arch.action_dim,
            arch=arch.gnn_arch,
        )
        encoder = ActionEncoder(list(_DEFAULT_PLAYER_IDS))
        return PolicyAgent(
            model,  # type: ignore[arg-type]
            encoder,
            obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
            device=device,
        )
    raise ValueError(f"unknown encoder_kind: {arch.encoder_kind!r}")


def _meta_to_dict(meta: CheckpointMeta) -> dict[str, Any]:
    return {
        "obs_layout_version": int(meta.obs_layout_version),
        "action_layout_version": int(meta.action_layout_version),
        "model_arch": _model_arch_to_dict(meta.model_arch),
        "train_step": int(meta.train_step),
        "timestamp": float(meta.timestamp),
        "config_hash": str(meta.config_hash),
    }


def _model_arch_to_dict(arch: ModelArch) -> dict[str, Any]:
    out: dict[str, Any] = {
        "obs_dim": int(arch.obs_dim),
        "action_dim": int(arch.action_dim),
        "encoder_kind": arch.encoder_kind,
        "hidden": [int(h) for h in arch.hidden],
    }
    if arch.gnn_arch is not None:
        out["gnn_arch"] = _gnn_arch_to_dict(arch.gnn_arch)
    return out


def _gnn_arch_to_dict(arch: GNNArch) -> dict[str, Any]:
    return {
        "node_hidden": int(arch.node_hidden),
        "player_hidden": int(arch.player_hidden),
        "n_mp_layers": int(arch.n_mp_layers),
        "n_heads": int(arch.n_heads),
        "global_mlp_hidden": int(arch.global_mlp_hidden),
        "value_kind": str(arch.value_kind),
    }


def _meta_from_dict(d: dict[str, Any]) -> CheckpointMeta:
    arch = _model_arch_from_dict(d["model_arch"])
    return CheckpointMeta(
        obs_layout_version=int(d["obs_layout_version"]),
        action_layout_version=int(d["action_layout_version"]),
        model_arch=arch,
        train_step=int(d["train_step"]),
        timestamp=float(d["timestamp"]),
        config_hash=str(d["config_hash"]),
    )


def _model_arch_from_dict(d: dict[str, Any]) -> ModelArch:
    encoder_kind: EncoderKind = d.get("encoder_kind", "flat")  # back-compat
    if encoder_kind not in ("flat", "graph"):
        raise IncompatibleCheckpointError(
            f"unknown encoder_kind in checkpoint: {encoder_kind!r}"
        )
    hidden = tuple(int(h) for h in d.get("hidden", ()))
    gnn_arch = (
        _gnn_arch_from_dict(d["gnn_arch"]) if "gnn_arch" in d else None
    )
    return ModelArch(
        obs_dim=int(d["obs_dim"]),
        action_dim=int(d["action_dim"]),
        encoder_kind=encoder_kind,
        hidden=hidden,
        gnn_arch=gnn_arch,
    )


def _gnn_arch_from_dict(d: dict[str, Any]) -> GNNArch:
    # Back-compat: checkpoints written before az-001 (Phase 3) didn't
    # carry ``value_kind``. They were always scalar-value PPO artifacts,
    # so defaulting missing values to ``"scalar"`` is the lossless choice;
    # without this default they'd silently load as vector and fail at
    # state-dict size-mismatch time.
    value_kind = d.get("value_kind", "scalar")
    if value_kind not in ("scalar", "vector"):
        raise IncompatibleCheckpointError(
            f"unknown value_kind in checkpoint gnn_arch: {value_kind!r}"
        )
    return GNNArch(
        node_hidden=int(d["node_hidden"]),
        player_hidden=int(d["player_hidden"]),
        n_mp_layers=int(d["n_mp_layers"]),
        n_heads=int(d["n_heads"]),
        global_mlp_hidden=int(d["global_mlp_hidden"]),
        value_kind=value_kind,
    )


def _stable_jsonable(x: Any) -> Any:
    """Recursively normalize tuples → lists so JSON dumps are order-stable."""
    if isinstance(x, tuple):
        return [_stable_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_stable_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _stable_jsonable(v) for k, v in x.items()}
    return x


