"""Side panel showing the policy's top-K action probabilities + value estimate.

The widget is read-only and stateful only in that it caches the overlay map
:class:`rl.utils.gui_hook.load_episode_into_session` produced. It listens for
``GameSession.on_change`` (delegated by :class:`MainWindow`) and refreshes
against the current snapshot's ``step_index``.

When the active snapshot has no overlay record (e.g. an opponent's turn) the
widget shows a placeholder string rather than blanking — the change is
visible to the user without them having to track which steps were the
learner's.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from controller.session import GameSnapshot
from rl.encoding.action import ActionEncoder, DiscardSentinel
from rl.replay.dataset import StepRecord

__all__ = ["PolicyOverlayWidget"]


_NO_OVERLAY = "(no policy overlay for this step)"


class PolicyOverlayWidget(QWidget):
    """List of top-5 action probabilities + value for the current step."""

    def __init__(self) -> None:
        super().__init__()
        self._overlay: dict[int, StepRecord] = {}
        self._encoder: ActionEncoder | None = None

        self._title = QLabel("Policy overlay")
        self._title.setAlignment(Qt.AlignLeft)
        self._value_label = QLabel(_NO_OVERLAY)
        self._actions_list = QListWidget()
        self._actions_list.setUniformItemSizes(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._value_label)
        layout.addWidget(self._actions_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_overlay(
        self,
        overlay: dict[int, StepRecord],
        action_encoder: ActionEncoder,
    ) -> None:
        """Install a new overlay map and the encoder used to render names.

        Call this once on replay load. The encoder must be constructed
        against the same player_ids as the loaded session — the GUI uses
        :meth:`ActionEncoder.for_state` against the initial snapshot.
        """
        self._overlay = dict(overlay)
        self._encoder = action_encoder
        self._actions_list.clear()
        self._value_label.setText(_NO_OVERLAY)

    def clear(self) -> None:
        self._overlay = {}
        self._encoder = None
        self._actions_list.clear()
        self._value_label.setText(_NO_OVERLAY)

    def on_step_changed(self, snap: GameSnapshot) -> None:
        """Refresh against the snapshot the user just navigated to."""
        if self._encoder is None:
            return
        record = self._overlay.get(snap.step_index)
        if record is None:
            self._value_label.setText(_NO_OVERLAY)
            self._actions_list.clear()
            return
        self._value_label.setText(f"V(s) = {record.value:+.3f}")
        self._actions_list.clear()
        for label, prob in _top_k_actions(record, self._encoder, snap, k=5):
            self._actions_list.addItem(QListWidgetItem(f"{label}: {prob:.1%}"))


def _top_k_actions(
    record: StepRecord,
    encoder: ActionEncoder,
    snap: GameSnapshot,
    *,
    k: int,
) -> list[tuple[str, float]]:
    """Decode the top-``k`` probabilities into human-readable labels.

    Probabilities below 1e-4 are filtered (the masked softmax pushes
    illegal actions to ~0 but they're still nonzero floats). Decoding can
    raise for indices that don't fit the current state (e.g. a stale
    discard-phase index); we catch and skip those.
    """
    if record.action_dist.size == 0:
        return []
    dist = np.asarray(record.action_dist)
    indices = np.argsort(-dist)[: k * 2]  # over-sample so filtering still yields k
    out: list[tuple[str, float]] = []
    for idx in indices:
        p = float(dist[int(idx)])
        if p < 1e-4:
            continue
        label = _decode_label(int(idx), encoder, snap)
        if label is None:
            continue
        out.append((label, p))
        if len(out) >= k:
            break
    return out


def _decode_label(idx: int, encoder: ActionEncoder, snap: GameSnapshot) -> str | None:
    """Render a discrete action index as a short human-readable string."""
    try:
        decoded = encoder.decode(idx, snap.state)
    except (ValueError, KeyError):
        return None
    if isinstance(decoded, DiscardSentinel):
        return f"Discard (P{int(decoded.player_id)})"
    return _format_action(decoded)


def _format_action(action: object) -> str:
    """Compact ``ClassName field=value`` rendering for the widget."""
    name = type(action).__name__
    # Pull integer-valued fields off the dataclass shell — this is enough
    # for "BuildRoadAction edge=12" / "BuildSettlementAction vertex=42"
    # style summaries without us hard-coding every action class.
    bits: list[str] = []
    for attr in ("edge_id", "vertex_id", "tile_id"):
        value = getattr(action, attr, None)
        if value is not None:
            bits.append(f"{attr.split('_')[0]}={int(value)}")
    for attr in ("resource", "give", "receive", "resource1", "resource2", "card_type"):
        value = getattr(action, attr, None)
        if value is not None:
            bits.append(getattr(value, "name", str(value)).lower())
    target_pid = getattr(action, "target_player_id", None)
    if target_pid is not None:
        bits.append(f"target=P{int(target_pid)}")
    return f"{name}({', '.join(bits)})" if bits else name
