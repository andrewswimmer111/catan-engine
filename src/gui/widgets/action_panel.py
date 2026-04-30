from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import controller.selectors as selectors
from controller.session import GameSnapshot
from domain.actions.base import Action

_BUTTON_GROUPS: dict[str, tuple[type[Action], ...]] = {
    "Turn": selectors.ACTION_GROUPS["Turn"],
    "Dev Card": selectors.ACTION_GROUPS["DevCard"],
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_box = QGroupBox("Actions")
        btn_layout = QHBoxLayout(btn_box)
        btn_layout.setAlignment(Qt.AlignLeft)
        self._buttons: dict[type[Action], QPushButton] = {}
        for group_label, group_types in _BUTTON_GROUPS.items():
            group_box = QGroupBox(group_label)
            group_inner = QHBoxLayout(group_box)
            for cls in group_types:
                btn = QPushButton(_label(cls))
                btn.setEnabled(False)
                btn.clicked.connect(lambda checked=False, c=cls: self._on_button(c))
                group_inner.addWidget(btn)
                self._buttons[cls] = btn
            btn_layout.addWidget(group_box)
        layout.addWidget(btn_box)

        self._show_raw = QCheckBox("Show raw list")
        self._show_raw.setChecked(True)
        self._show_raw.toggled.connect(self._list_toggle)
        layout.addWidget(self._show_raw)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

    def refresh(self, snap: GameSnapshot, legal: list[Action]) -> None:
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

    def _on_button(self, cls: type[Action]) -> None:
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
