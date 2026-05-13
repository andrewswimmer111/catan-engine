"""On-disk archive of self-play / eval games with per-step policy outputs.

A :class:`ReplayDataset` is a directory where each subdirectory is one
episode. Per-episode contents:

* ``replay.json`` — the existing :class:`ReplayLog` action stream (loaded by
  the GUI's replay machinery as-is).
* ``steps.npz`` — packed numpy arrays for every step's observation, mask,
  action distribution, value, reward, agent id, and the integer action.
* ``meta.json`` — final VPs, winner, and a free-form ``metadata`` dict (the
  trainer fills this with the checkpoint id, seed, etc.).

Episode IDs are derived from a monotonically-increasing zero-padded counter
plus a timestamp slug so listings are stable in chronological order without
the dataset having to lock a separate index file. Stale or partially-written
episodes are detected by missing files; :meth:`list_episodes` skips them.

Auto-archival
-------------

The trainer should **not** archive every game — disk space blows up fast.
:func:`is_interesting` codifies the spec's selection rule (close win = ≤2 VP
margin) so callers can wire it into their rollout-worker callback without
re-encoding the heuristic inline. The matching rollout-worker hook is
:class:`~rl.training.rollout.ArchiveHook`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.ids import PlayerID
from serialization.replay import ReplayLog, load_replay, save_replay

__all__ = [
    "EpisodeRecord",
    "ReplayDataset",
    "StepRecord",
    "is_interesting",
]


@dataclass(frozen=True)
class StepRecord:
    """One step the learner took inside a single episode."""

    obs: np.ndarray
    action: int
    mask: np.ndarray
    action_dist: np.ndarray
    value: float
    reward: float
    agent: PlayerID


@dataclass(frozen=True)
class EpisodeRecord:
    """A complete archived game.

    ``replay_log`` is the canonical action stream — it can be replayed
    through the engine without any of the RL-side per-step data. ``steps`` is
    sparse: only steps where the learner acted appear, in chronological
    order. The pair is meant to be consumed together (the GUI overlay
    aligns ``steps`` against ``replay_log`` by integer step index keyed in
    ``metadata['learner_step_indices']``).
    """

    replay_log: ReplayLog
    steps: list[StepRecord]
    final_vps: dict[PlayerID, int]
    winner: PlayerID | None
    metadata: dict[str, Any] = field(default_factory=dict)


# ID format: 6-digit counter underscore unix-seconds, e.g. "000042_1715600000".
_EPISODE_ID_PATTERN = re.compile(r"^\d{6}_\d+$")


class ReplayDataset:
    """Directory-backed archive of :class:`EpisodeRecord`s.

    The dataset is single-writer; concurrent writers can race on the
    counter. Self-play training writes from one process at a time so this
    is fine for now — if rollout parallelism (rl-023) ever shares an
    archive, we'd add a lockfile.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write(self, ep: EpisodeRecord) -> Path:
        """Persist ``ep`` and return the new episode directory path.

        The counter component of the id is the smallest unused 6-digit
        number; the timestamp component breaks ties so reruns at the same
        counter (e.g. a fresh trainer pointed at an existing dataset) sort
        chronologically.
        """
        counter = self._next_counter()
        episode_id = f"{counter:06d}_{int(time.time())}"
        ep_dir = self._root / episode_id
        ep_dir.mkdir(parents=True, exist_ok=False)

        save_replay(ep.replay_log, str(ep_dir / "replay.json"))
        _save_steps(ep.steps, ep_dir / "steps.npz")
        _save_meta(ep, ep_dir / "meta.json")
        return ep_dir

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read(self, episode_id: str) -> EpisodeRecord:
        ep_dir = self._root / episode_id
        if not ep_dir.is_dir():
            raise FileNotFoundError(f"no episode {episode_id!r} in {self._root}")
        replay = load_replay(str(ep_dir / "replay.json"))
        steps = _load_steps(ep_dir / "steps.npz")
        meta = _load_meta(ep_dir / "meta.json")
        return EpisodeRecord(
            replay_log=replay,
            steps=steps,
            final_vps=meta["final_vps"],
            winner=meta["winner"],
            metadata=meta["metadata"],
        )

    def list_episodes(self) -> list[str]:
        """Return episode IDs in ascending counter order.

        Subdirectories whose names don't match the id pattern or are missing
        a required file (e.g. half-written) are silently skipped.
        """
        ids: list[str] = []
        for entry in self._root.iterdir():
            if not entry.is_dir():
                continue
            if not _EPISODE_ID_PATTERN.match(entry.name):
                continue
            required = ["replay.json", "steps.npz", "meta.json"]
            if not all((entry / fname).exists() for fname in required):
                continue
            ids.append(entry.name)
        ids.sort()
        return ids

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_counter(self) -> int:
        existing = self.list_episodes()
        if not existing:
            return 0
        last_counter = int(existing[-1].split("_", 1)[0])
        return last_counter + 1


# ----------------------------------------------------------------------
# Storage helpers
# ----------------------------------------------------------------------


def _save_steps(steps: list[StepRecord], path: Path) -> None:
    """Pack a list of :class:`StepRecord` into a single .npz blob.

    Empty step lists serialize to a zero-length npz; ``_load_steps`` returns
    ``[]`` for them. That lets the dataset archive games where the learner
    never acted (rare but possible during cold start).
    """
    if not steps:
        # ``np.savez`` requires at least one array; write an empty marker.
        np.savez(str(path), empty=np.array(True))
        return
    obs = np.stack([s.obs for s in steps]).astype(np.float32, copy=False)
    masks = np.stack([s.mask for s in steps]).astype(np.bool_, copy=False)
    dists = np.stack([s.action_dist for s in steps]).astype(np.float32, copy=False)
    actions = np.array([s.action for s in steps], dtype=np.int64)
    values = np.array([s.value for s in steps], dtype=np.float32)
    rewards = np.array([s.reward for s in steps], dtype=np.float32)
    agents = np.array([int(s.agent) for s in steps], dtype=np.int64)
    np.savez(
        str(path),
        obs=obs,
        masks=masks,
        action_dist=dists,
        actions=actions,
        values=values,
        rewards=rewards,
        agents=agents,
    )


def _load_steps(path: Path) -> list[StepRecord]:
    with np.load(str(path)) as data:
        if "empty" in data.files and len(data.files) == 1:
            return []
        obs = data["obs"]
        masks = data["masks"]
        dists = data["action_dist"]
        actions = data["actions"]
        values = data["values"]
        rewards = data["rewards"]
        agents = data["agents"]
    return [
        StepRecord(
            obs=np.asarray(obs[i]),
            action=int(actions[i]),
            mask=np.asarray(masks[i]),
            action_dist=np.asarray(dists[i]),
            value=float(values[i]),
            reward=float(rewards[i]),
            agent=PlayerID(int(agents[i])),
        )
        for i in range(obs.shape[0])
    ]


def _save_meta(ep: EpisodeRecord, path: Path) -> None:
    payload = {
        "final_vps": {str(int(pid)): int(v) for pid, v in ep.final_vps.items()},
        "winner": None if ep.winner is None else int(ep.winner),
        "metadata": _json_safe(ep.metadata),
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def _load_meta(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {
        "final_vps": {
            PlayerID(int(k)): int(v) for k, v in data["final_vps"].items()
        },
        "winner": None if data["winner"] is None else PlayerID(int(data["winner"])),
        "metadata": data["metadata"],
    }


def _json_safe(x: Any) -> Any:
    """Coerce ``metadata`` into something JSON can serialize.

    The metadata dict is free-form, so callers might stash Paths, PlayerIDs,
    or numpy scalars. We round-trip everything we recognise; unknown types
    raise TypeError at write time (caught upstream).
    """
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


# ----------------------------------------------------------------------
# Auto-archive predicate
# ----------------------------------------------------------------------


def is_interesting(
    final_vps: dict[PlayerID, int],
    winner: PlayerID | None,
    *,
    close_win_margin: int = 2,
) -> bool:
    """Spec's "interesting game" rule: close finish.

    Returns True when the top score - second-best score ≤ ``close_win_margin``.
    Stalemates (no winner) are also flagged as interesting since they're rare
    and worth post-hoc inspection.
    """
    if winner is None:
        return True
    if not final_vps:
        return False
    ranked = sorted(final_vps.values(), reverse=True)
    if len(ranked) < 2:
        return False
    return (ranked[0] - ranked[1]) <= close_win_margin
