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
:class:`ModelArch` knobs needed to size :class:`MLPPolicyValue`, the
``train_step`` the snapshot was taken at, a wall-clock ``timestamp``, and a
``config_hash`` over the training config so accidental drift between runs
shows up as a visible field difference.

Player IDs are intentionally **not** stored. The model is seat-agnostic —
:mod:`rl.encoding.observation` rotates the viewer into "self" at every
forward pass — and the action layout is fixed to four opponent slots by
:data:`ACTION_LAYOUT_VERSION`. Runtime player IDs come from the game config,
not the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding._action_layout import _ACTION_LAYOUT_VERSION
from rl.encoding.action import ActionEncoder
from rl.encoding.observation import OBS_LAYOUT_VERSION
from rl.models.mlp import MLPPolicyValue
from rl.training.config import TrainConfig

__all__ = [
    "ACTION_LAYOUT_VERSION",
    "CheckpointMeta",
    "IncompatibleCheckpointError",
    "ModelArch",
    "compute_config_hash",
    "load_checkpoint",
    "save_checkpoint",
]


ACTION_LAYOUT_VERSION: int = _ACTION_LAYOUT_VERSION


# Default 4-seat player_ids used by ``CatanEnv``. The action layout's steal
# slots are seat-indexed, so the loaded encoder needs *some* PlayerID list to
# decode index→Action; the env's default config uses these IDs, and a policy
# trained against a non-default config is the caller's problem to handle by
# rebuilding the encoder externally.
_DEFAULT_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]


@dataclass(frozen=True)
class ModelArch:
    """Architecture knobs needed to reconstruct an :class:`MLPPolicyValue`."""

    obs_dim: int
    action_dim: int
    hidden: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointMeta:
    """Self-describing checkpoint metadata.

    Layout version fields gate compatibility: a checkpoint trained against a
    different obs or action layout would have shape-mismatched weights or
    mis-mapped action indices, so :func:`load_checkpoint` refuses it.
    """

    obs_layout_version: int
    action_layout_version: int
    model_arch: ModelArch
    train_step: int
    timestamp: float
    config_hash: str


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint's layout versions don't match the loader."""


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


def load_checkpoint(path: Path) -> tuple[PolicyAgent, CheckpointMeta]:
    """Load a checkpoint, validate compatibility, and return ``(agent, meta)``.

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

    if meta.obs_layout_version != OBS_LAYOUT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint obs_layout_version={meta.obs_layout_version} "
            f"!= current OBS_LAYOUT_VERSION={OBS_LAYOUT_VERSION}"
        )
    if meta.action_layout_version != ACTION_LAYOUT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint action_layout_version={meta.action_layout_version} "
            f"!= current ACTION_LAYOUT_VERSION={ACTION_LAYOUT_VERSION}"
        )

    model = MLPPolicyValue(
        obs_dim=meta.model_arch.obs_dim,
        action_dim=meta.model_arch.action_dim,
        hidden=meta.model_arch.hidden,
    )
    encoder = ActionEncoder(list(_DEFAULT_PLAYER_IDS))
    agent = PolicyAgent(model, encoder)
    agent.load_state_dict(payload["model"])
    return agent, meta


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _meta_to_dict(meta: CheckpointMeta) -> dict[str, Any]:
    return {
        "obs_layout_version": int(meta.obs_layout_version),
        "action_layout_version": int(meta.action_layout_version),
        "model_arch": {
            "obs_dim": int(meta.model_arch.obs_dim),
            "action_dim": int(meta.model_arch.action_dim),
            "hidden": list(int(h) for h in meta.model_arch.hidden),
        },
        "train_step": int(meta.train_step),
        "timestamp": float(meta.timestamp),
        "config_hash": str(meta.config_hash),
    }


def _meta_from_dict(d: dict[str, Any]) -> CheckpointMeta:
    arch_d = d["model_arch"]
    return CheckpointMeta(
        obs_layout_version=int(d["obs_layout_version"]),
        action_layout_version=int(d["action_layout_version"]),
        model_arch=ModelArch(
            obs_dim=int(arch_d["obs_dim"]),
            action_dim=int(arch_d["action_dim"]),
            hidden=tuple(int(h) for h in arch_d["hidden"]),
        ),
        train_step=int(d["train_step"]),
        timestamp=float(d["timestamp"]),
        config_hash=str(d["config_hash"]),
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
