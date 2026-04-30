from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from controller.session import GameSession, GameSnapshot


class TimelineWidget(QWidget):
    jumped = Signal(object)  # emits GameSnapshot after each jump

    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session

        self._btn_start = QPushButton("<<")
        self._btn_back = QPushButton("<")
        self._btn_forward = QPushButton(">")
        self._btn_end = QPushButton(">>")

        for btn in (self._btn_start, self._btn_back, self._btn_forward, self._btn_end):
            btn.setFixedWidth(36)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)

        self._label = QLabel()
        self._label.setMinimumWidth(280)

        row = QHBoxLayout()
        row.addWidget(self._btn_start)
        row.addWidget(self._btn_back)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._btn_forward)
        row.addWidget(self._btn_end)
        row.addWidget(self._label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addLayout(row)

        self._btn_start.clicked.connect(lambda: self._do_jump(0))
        self._btn_back.clicked.connect(
            lambda: self._do_jump(self._session.current().step_index - 1)
        )
        self._btn_forward.clicked.connect(
            lambda: self._do_jump(self._session.current().step_index + 1)
        )
        self._btn_end.clicked.connect(
            lambda: self._do_jump(len(self._session.history()) - 1)
        )
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._sync_controls(session.current())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, session: GameSession) -> None:
        self._session = session
        self._sync_controls(session.current())

    def refresh(self, snap: GameSnapshot) -> None:
        """Call after any external state change (apply or jump) to keep controls in sync."""
        self._sync_controls(snap)

    def step_back(self) -> None:
        step = self._session.current().step_index
        if step > 0:
            self._do_jump(step - 1)

    def step_forward(self) -> None:
        step = self._session.current().step_index
        if step < len(self._session.history()) - 1:
            self._do_jump(step + 1)

    def jump_start(self) -> None:
        if self._session.current().step_index > 0:
            self._do_jump(0)

    def jump_end(self) -> None:
        last = len(self._session.history()) - 1
        if self._session.current().step_index < last:
            self._do_jump(last)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_jump(self, step: int) -> None:
        snap = self._session.jump_to(step)
        self._sync_controls(snap)
        self.jumped.emit(snap)

    def _on_slider_changed(self, value: int) -> None:
        if value != self._session.current().step_index:
            self._do_jump(value)

    def _sync_controls(self, snap: GameSnapshot) -> None:
        total = len(self._session.history()) - 1
        current = snap.step_index

        self._slider.blockSignals(True)
        self._slider.setMaximum(total)
        self._slider.setValue(current)
        self._slider.blockSignals(False)

        state = snap.state
        self._label.setText(
            f"step {current} / {total}"
            f"  —  phase={state.phase.name}  player={int(state.current_player)}"
        )

        self._btn_start.setEnabled(current > 0)
        self._btn_back.setEnabled(current > 0)
        self._btn_forward.setEnabled(current < total)
        self._btn_end.setEnabled(current < total)
