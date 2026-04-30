from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import domain.actions.all_actions as A
from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.enums import DomesticTradeState
from domain.turn.pending import DomesticTradePending


def _res_summary(resources: dict) -> str:
    return ", ".join(f"{v} {r.value.capitalize()}" for r, v in resources.items() if v > 0) or "—"


class TradePanel(QWidget):
    """Purely pending-driven view of an in-progress domestic trade negotiation."""

    action_chosen = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._body: QWidget | None = None
        self.hide()

    def refresh(self, snap: GameSnapshot, legal: list[Action]) -> None:
        pending = snap.state.pending
        if not isinstance(pending, DomesticTradePending):
            self._swap_body(None)
            self.hide()
            return

        self.show()
        self._swap_body(self._build_body(snap, pending, legal))

    # ------------------------------------------------------------------ build

    def _build_body(
        self,
        snap: GameSnapshot,
        pending: DomesticTradePending,
        legal: list[Action],
    ) -> QWidget:
        all_pids = set(snap.state.players.keys())
        proposer = (all_pids - set(pending.responses.keys())).pop()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header
        layout.addWidget(QLabel(f"<b>Trade proposal from Player {int(proposer)}</b>"))
        layout.addWidget(QLabel(f"Offering:   {_res_summary(pending.offer)}"))
        layout.addWidget(QLabel(f"Requesting: {_res_summary(pending.request)}"))

        layout.addWidget(_hline())

        # Response rows
        resp_group = QGroupBox("Responses")
        resp_layout = QVBoxLayout(resp_group)
        for pid, state in pending.responses.items():
            resp_layout.addWidget(self._response_row(pid, state, legal))
        layout.addWidget(resp_group)

        layout.addWidget(_hline())

        # Proposer confirmation bar (current_player is always the proposer during trade)
        layout.addLayout(self._confirm_bar(legal))

        return body

    def _response_row(
        self,
        pid,
        state: DomesticTradeState,
        legal: list[Action],
    ) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(f"Player {int(pid)}:"))
        h.addWidget(QLabel(state.value.capitalize()))
        h.addStretch()

        for label, response in (("Accept", DomesticTradeState.ACCEPTED), ("Reject", DomesticTradeState.REJECTED)):
            candidate = A.RespondDomesticTradeAction(player_id=pid, response=response)
            legal_action = next((a for a in legal if a == candidate), None)
            btn = QPushButton(label)
            btn.setEnabled(legal_action is not None)
            if legal_action is not None:
                btn.clicked.connect(lambda checked=False, a=legal_action: self.action_chosen.emit(a))
            h.addWidget(btn)

        return row

    def _confirm_bar(self, legal: list[Action]) -> QHBoxLayout:
        bar = QHBoxLayout()
        for action in legal:
            if isinstance(action, A.ConfirmDomesticTradeAction):
                btn = QPushButton(f"Confirm with Player {int(action.trade_with)}")
                btn.clicked.connect(lambda checked=False, a=action: self.action_chosen.emit(a))
                bar.addWidget(btn)

        cancel = next((a for a in legal if isinstance(a, A.CancelDomesticTradeAction)), None)
        cancel_btn = QPushButton("Cancel Trade")
        cancel_btn.setEnabled(cancel is not None)
        if cancel is not None:
            cancel_btn.clicked.connect(lambda checked=False, a=cancel: self.action_chosen.emit(a))
        bar.addWidget(cancel_btn)
        bar.addStretch()
        return bar

    # ------------------------------------------------------------------ util

    def _swap_body(self, widget: QWidget | None) -> None:
        if self._body is not None:
            self._layout.removeWidget(self._body)
            self._body.deleteLater()
            self._body = None
        if widget is not None:
            self._layout.addWidget(widget)
        self._body = widget


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line
