"""Adapter from :class:`EpisodeRecord` to the existing GUI replay machinery.

The GUI already knows how to load a :class:`ReplayLog` into a
:class:`GameSession` via :meth:`GameSession.from_replay`. This module wraps
that path and additionally exposes the per-step overlay data (action
distribution, value estimate) the policy widget renders alongside.

This file deliberately does **not** import PySide6 — keeping the data path
GUI-free lets the unit tests below exercise it without a Qt runtime, and the
widget itself stays in :mod:`gui.widgets.policy_overlay`.
"""

from __future__ import annotations

from pathlib import Path

from controller.session import GameSession
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from rl.replay.dataset import ReplayDataset, StepRecord

__all__ = ["EpisodeOverlay", "load_episode_into_session"]


from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeOverlay:
    """Bundle returned to GUI callers.

    ``session`` is the replay-loaded :class:`GameSession` ready to plug into
    ``MainWindow._replace_session``. ``overlay`` maps integer
    ``snapshot.step_index`` → the :class:`StepRecord` the learner emitted at
    that state. Snapshots that weren't the learner's turn have no key in
    ``overlay``.
    """

    session: GameSession
    overlay: dict[int, StepRecord]


def load_episode_into_session(ep_dir: Path) -> EpisodeOverlay:
    """Load an archived episode and produce a session + overlay map.

    ``ep_dir`` is the per-episode directory inside a :class:`ReplayDataset`
    (the one returned by :meth:`ReplayDataset.write`). The function looks at
    the parent directory to construct a dataset and reads via the dataset's
    standard interface — this keeps the gui_hook agnostic to the on-disk
    layout details.
    """
    ep_dir = Path(ep_dir)
    if not ep_dir.is_dir():
        raise FileNotFoundError(f"episode directory not found: {ep_dir}")

    dataset = ReplayDataset(ep_dir.parent)
    episode = dataset.read(ep_dir.name)

    engine = GameEngine(SeededRandomizer(seed=episode.replay_log.config.seed))
    session = GameSession.from_replay(engine, episode.replay_log)

    learner_step_indices = episode.metadata.get("learner_step_indices", [])
    if len(learner_step_indices) != len(episode.steps):
        # Metadata drift — fall back to assigning steps in order, which is the
        # best we can do without the explicit mapping. Real archives written
        # by the current recorder always carry the matching indices, so this
        # only fires for hand-edited or legacy episodes.
        learner_step_indices = list(range(len(episode.steps)))

    overlay = {
        int(idx): step for idx, step in zip(learner_step_indices, episode.steps)
    }
    return EpisodeOverlay(session=session, overlay=overlay)
