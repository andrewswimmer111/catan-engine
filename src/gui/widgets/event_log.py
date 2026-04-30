from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

import domain.events.all_events as E
from controller.session import GameSession, GameSnapshot

# ── filter options ────────────────────────────────────────────────────────────

_ALL_EVENT_TYPES: list[str] = [
    "DiceRolled", "ResourcesDistributed", "BankShortfall",
    "RoadBuilt", "SettlementBuilt", "CityBuilt",
    "DevCardBought", "DevCardPlayed", "PlayerDiscarded",
    "RobberMoved", "ResourceStolen", "TradeCompleted",
    "MaritimeTradeCompleted", "LongestRoadAwarded", "LargestArmyAwarded",
    "TurnEnded", "GameWon", "GameStalled",
]

_ROLE_STEP = Qt.UserRole
_ROLE_TYPE = Qt.UserRole + 1


# ── human-readable summary per event type ────────────────────────────────────

def _short_repr(event: object) -> str:
    if isinstance(event, E.DiceRolled):
        return f"total={event.total} ({event.die1}+{event.die2})"
    if isinstance(event, E.ResourcesDistributed):
        parts = []
        for pid, res in sorted(event.distributions.items(), key=lambda x: int(x[0])):
            chunk = " ".join(f"{r.name}×{n}" for r, n in res.items() if n > 0)
            if chunk:
                parts.append(f"p{int(pid)}: {chunk}")
        return ", ".join(parts) if parts else "none"
    if isinstance(event, E.BankShortfall):
        return f"{event.resource.name} req={event.requested} got={event.given}"
    if isinstance(event, E.RoadBuilt):
        return f"p{int(event.player_id)} edge={int(event.edge_id)}"
    if isinstance(event, E.SettlementBuilt):
        return f"p{int(event.player_id)} vertex={int(event.vertex_id)}"
    if isinstance(event, E.CityBuilt):
        return f"p{int(event.player_id)} vertex={int(event.vertex_id)}"
    if isinstance(event, E.DevCardBought):
        return f"p{int(event.player_id)} {event.card_type.name}"
    if isinstance(event, E.DevCardPlayed):
        return f"p{int(event.player_id)} {event.card_type.name}"
    if isinstance(event, E.PlayerDiscarded):
        chunk = " ".join(f"{r.name}×{n}" for r, n in event.resources.items() if n > 0)
        return f"p{int(event.player_id)} [{chunk}]"
    if isinstance(event, E.RobberMoved):
        return f"p{int(event.player_id)} tile={int(event.tile_id)}"
    if isinstance(event, E.ResourceStolen):
        return (
            f"p{int(event.by_player_id)}→p{int(event.from_player_id)}"
            f" {event.resource.name}"
        )
    if isinstance(event, E.TradeCompleted):
        g1 = " ".join(f"{r.name}×{n}" for r, n in event.player1_gives.items() if n > 0)
        g2 = " ".join(f"{r.name}×{n}" for r, n in event.player2_gives.items() if n > 0)
        return f"p{int(event.player1_id)} [{g1}] ↔ p{int(event.player2_id)} [{g2}]"
    if isinstance(event, E.MaritimeTradeCompleted):
        return f"p{int(event.player_id)} {event.gave.name}→{event.received.name}"
    if isinstance(event, E.LongestRoadAwarded):
        return f"p{int(event.player_id)} len={event.length}"
    if isinstance(event, E.LargestArmyAwarded):
        return f"p{int(event.player_id)} count={event.count}"
    if isinstance(event, E.TurnEnded):
        return f"p{int(event.player_id)}"
    if isinstance(event, E.GameWon):
        return f"p{int(event.player_id)} vp={event.victory_points}"
    if isinstance(event, E.GameStalled):
        return f"{event.reason.name}"
    return ""


# ── widget ────────────────────────────────────────────────────────────────────

class EventLogWidget(QWidget):
    jumped = Signal(object)  # emits GameSnapshot after jump_to

    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self._next_expected_step = 1  # next step we'd append in normal flow

        self._filter = QComboBox()
        self._filter.addItem("All")
        self._filter.addItems(_ALL_EVENT_TYPES)

        self._list = QListWidget()
        self._list.setUniformItemSizes(True)

        header = QHBoxLayout()
        header.addWidget(QLabel("Filter:"))
        header.addWidget(self._filter)
        header.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self._list)

        self._filter.currentTextChanged.connect(self._apply_filter)
        self._list.itemDoubleClicked.connect(self._on_double_clicked)

    # ── public API ────────────────────────────────────────────────────────────

    def set_session(self, session: GameSession) -> None:
        self._session = session
        self._rebuild()

    def on_applied(self, snap: GameSnapshot) -> None:
        """Called after session.apply(). Appends normally, or rebuilds on branch."""
        if snap.step_index == self._next_expected_step:
            self._append_snap(snap)
            self._next_expected_step += 1
        else:
            self._rebuild()

    def rebuild(self) -> None:
        """Full rebuild from session.history(). Call on replay-load or jump-to-zero."""
        self._rebuild()

    # ── internals ─────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._list.clear()
        history = self._session.history()
        for snap in history[1:]:  # step 0 carries no events
            self._append_snap(snap)
        self._next_expected_step = len(history)

    def _append_snap(self, snap: GameSnapshot) -> None:
        current_filter = self._filter.currentText()
        for event in snap.last_events:
            event_type = type(event).__name__
            label = f"[t{event.turn_number}] {event_type}: {_short_repr(event)}"
            item = QListWidgetItem(label)
            item.setData(_ROLE_STEP, snap.step_index)
            item.setData(_ROLE_TYPE, event_type)
            self._list.addItem(item)
            if current_filter != "All" and event_type != current_filter:
                item.setHidden(True)
        self._list.scrollToBottom()

    def _apply_filter(self) -> None:
        f = self._filter.currentText()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(f != "All" and item.data(_ROLE_TYPE) != f)

    def _on_double_clicked(self, item: QListWidgetItem) -> None:
        step = item.data(_ROLE_STEP)
        snap = self._session.jump_to(step)
        self.jumped.emit(snap)
