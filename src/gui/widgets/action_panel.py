from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import controller.selectors as selectors
import domain.actions.all_actions as A
from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.turn.pending import DiscardPending
from gui.widgets.modal_dialogs import (
    DiscardDialog,
    MaritimeTradeDialog,
    MonopolyDialog,
    ProposeDomesticTradeDialog,
    YearOfPlentyDialog,
)

_BUTTON_GROUPS: dict[str, tuple[type[Action], ...]] = {
    "Turn": selectors.ACTION_GROUPS["Turn"],
    "Dev Card": selectors.ACTION_GROUPS["DevCard"],
    "Trade": (A.MaritimeTradeAction, A.ProposeDomesticTradeAction),
}

_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _label(cls: type) -> str:
    name = cls.__name__.removesuffix("Action")
    return _CAMEL_SPLIT.sub(" ", name)


class ActionPanel(QWidget):
    action_chosen = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._legal: list[Action] = []
        self._snap: GameSnapshot | None = None
        self._last_discard_step = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_box = QGroupBox("Actions")
        btn_layout = QVBoxLayout(btn_box)
        btn_layout.setSpacing(2)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        self._buttons: dict[type[Action], QPushButton] = {}
        for group_label, group_types in _BUTTON_GROUPS.items():
            label = QLabel(f"<b>{group_label}</b>")
            btn_layout.addWidget(label)
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            for cls in group_types:
                btn = QPushButton(_label(cls))
                btn.setEnabled(False)
                btn.clicked.connect(lambda checked=False, c=cls: self._on_button(c))
                row.addWidget(btn)
                self._buttons[cls] = btn
            row.addStretch()
            btn_layout.addLayout(row)
        layout.addWidget(btn_box)

        self._show_raw = QCheckBox("Show raw list")
        self._show_raw.setChecked(True)
        self._show_raw.toggled.connect(self._list_toggle)
        layout.addWidget(self._show_raw)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

    def refresh(self, snap: GameSnapshot, legal: list[Action]) -> None:
        self._snap = snap
        self._legal = legal

        by_type: dict[type[Action], list[Action]] = {}
        for action in legal:
            by_type.setdefault(type(action), []).append(action)

        for cls, btn in self._buttons.items():
            btn.setEnabled(cls in by_type)

        self._list.clear()
        for action in legal:
            item = QListWidgetItem(repr(action))
            item.setData(Qt.UserRole, action)
            self._list.addItem(item)

        if (
            isinstance(snap.state.pending, DiscardPending)
            and A.DiscardResourcesAction in by_type
            and snap.step_index != self._last_discard_step
        ):
            self._last_discard_step = snap.step_index
            self._show_discard_dialog()

    def _on_button(self, cls: type[Action]) -> None:
        if cls is A.PlayYearOfPlentyAction:
            self._show_year_of_plenty_dialog()
        elif cls is A.PlayMonopolyAction:
            self._show_monopoly_dialog()
        elif cls is A.MaritimeTradeAction:
            self._show_maritime_trade_dialog()
        elif cls is A.ProposeDomesticTradeAction:
            self._show_propose_trade_dialog()
        else:
            for action in self._legal:
                if type(action) is cls:
                    self.action_chosen.emit(action)
                    return

    def _on_double_click(self, item: QListWidgetItem) -> None:
        action = item.data(Qt.UserRole)
        if action is not None:
            self.action_chosen.emit(action)

    def _list_toggle(self, visible: bool) -> None:
        self._list.setVisible(visible)

    def _show_discard_dialog(self) -> None:
        snap = self._snap
        action = next((a for a in self._legal if isinstance(a, A.DiscardResourcesAction)), None)
        if action is None:
            return
        player_id = action.player_id
        must_discard = snap.state.pending.cards_to_discard[player_id]
        hand = snap.state.players[player_id].resources
        dlg = DiscardDialog(must_discard, hand, parent=self)
        if dlg.exec():
            self.action_chosen.emit(
                A.DiscardResourcesAction(player_id=player_id, resources=dlg.chosen())
            )

    def _show_year_of_plenty_dialog(self) -> None:
        dlg = YearOfPlentyDialog(parent=self)
        if dlg.exec():
            r1, r2 = dlg.chosen()
            player_id = self._snap.state.current_player
            self.action_chosen.emit(
                A.PlayYearOfPlentyAction(player_id=player_id, resource1=r1, resource2=r2)
            )

    def _show_monopoly_dialog(self) -> None:
        dlg = MonopolyDialog(parent=self)
        if dlg.exec():
            player_id = self._snap.state.current_player
            self.action_chosen.emit(
                A.PlayMonopolyAction(player_id=player_id, resource=dlg.chosen())
            )

    def _show_maritime_trade_dialog(self) -> None:
        trades = [a for a in self._legal if isinstance(a, A.MaritimeTradeAction)]
        if not trades:
            return
        dlg = MaritimeTradeDialog(trades, parent=self)
        if dlg.exec():
            self.action_chosen.emit(dlg.chosen())

    def _show_propose_trade_dialog(self) -> None:
        player_id = self._snap.state.current_player
        hand = self._snap.state.players[player_id].resources
        dlg = ProposeDomesticTradeDialog(hand=hand, parent=self)
        if dlg.exec():
            offer, request = dlg.chosen()
            self.action_chosen.emit(
                A.ProposeDomesticTradeAction(player_id=player_id, offer=offer, request=request)
            )
