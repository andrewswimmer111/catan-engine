from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
)

from domain.actions.all_actions import MaritimeTradeAction, ProposeDomesticTradeAction
from domain.enums import Resource, tradeable_resources

_RESOURCES: list[Resource] = list(tradeable_resources())
_LABEL: dict[Resource, str] = {r: r.value.capitalize() for r in _RESOURCES}


class DiscardDialog(QDialog):
    """Forces the current player to choose exactly `must_discard` cards to discard."""

    def __init__(self, must_discard: int, hand: dict[Resource, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Discard {must_discard} cards")
        self.setModal(True)

        self._must = must_discard
        self._spins: dict[Resource, QSpinBox] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        for r in _RESOURCES:
            spin = QSpinBox()
            spin.setRange(0, hand.get(r, 0))
            spin.valueChanged.connect(self._update_ok)
            form.addRow(_LABEL[r], spin)
            self._spins[r] = spin
        layout.addLayout(form)

        self._total_label = QLabel()
        layout.addWidget(self._total_label)

        self._ok_btn = QDialogButtonBox(QDialogButtonBox.Ok)
        self._ok_btn.accepted.connect(self.accept)
        layout.addWidget(self._ok_btn)

        self._update_ok()

    def _update_ok(self) -> None:
        total = sum(s.value() for s in self._spins.values())
        self._total_label.setText(f"Selected: {total} / {self._must}")
        self._ok_btn.button(QDialogButtonBox.Ok).setEnabled(total == self._must)

    def chosen(self) -> dict[Resource, int]:
        return {r: s.value() for r, s in self._spins.items() if s.value() > 0}


class YearOfPlentyDialog(QDialog):
    """Lets the player pick two resources to receive from the bank."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Year of Plenty — pick two resources")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._combos: list[QComboBox] = []
        for label in ("First resource:", "Second resource:"):
            combo = QComboBox()
            for r in _RESOURCES:
                combo.addItem(_LABEL[r], r)
            form.addRow(label, combo)
            self._combos.append(combo)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def chosen(self) -> tuple[Resource, Resource]:
        return (self._combos[0].currentData(), self._combos[1].currentData())


class MonopolyDialog(QDialog):
    """Lets the player pick one resource to monopolize."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Monopoly — pick a resource")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._combo = QComboBox()
        for r in _RESOURCES:
            self._combo.addItem(_LABEL[r], r)
        form.addRow("Resource:", self._combo)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def chosen(self) -> Resource:
        return self._combo.currentData()


class ProposeDomesticTradeDialog(QDialog):
    """Two resource grids (offer / request) for proposing a domestic trade."""

    def __init__(self, hand: dict[Resource, int] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Propose Domestic Trade")
        self.setModal(True)

        self._offer_spins: dict[Resource, QSpinBox] = {}
        self._request_spins: dict[Resource, QSpinBox] = {}

        outer = QVBoxLayout(self)

        grids = QHBoxLayout()
        for title, spins, cap in (
            ("You offer", self._offer_spins, hand),
            ("You request", self._request_spins, None),
        ):
            box = QGroupBox(title)
            form = QFormLayout(box)
            for r in _RESOURCES:
                spin = QSpinBox()
                spin.setRange(0, cap.get(r, 0) if cap else 19)
                spin.valueChanged.connect(self._update_ok)
                form.addRow(_LABEL[r], spin)
                spins[r] = spin
            grids.addWidget(box)
        outer.addLayout(grids)

        self._btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        outer.addWidget(self._btns)

        self._update_ok()

    def _update_ok(self) -> None:
        has_offer = any(s.value() > 0 for s in self._offer_spins.values())
        has_request = any(s.value() > 0 for s in self._request_spins.values())
        self._btns.button(QDialogButtonBox.Ok).setEnabled(has_offer and has_request)

    def chosen(self) -> tuple[dict[Resource, int], dict[Resource, int]]:
        offer = {r: s.value() for r, s in self._offer_spins.items() if s.value() > 0}
        request = {r: s.value() for r, s in self._request_spins.items() if s.value() > 0}
        return offer, request


class MaritimeTradeDialog(QDialog):
    """Shows the enumerated set of legal maritime trades and lets the player pick one."""

    def __init__(self, legal: list[MaritimeTradeAction], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Maritime Trade")
        self.setModal(True)

        self._actions = legal

        layout = QVBoxLayout(self)
        self._list = QListWidget()
        for action in legal:
            text = f"{action.give_count}× {_LABEL[action.give]} → {_LABEL[action.receive]}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, action)
            self._list.addItem(item)
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def chosen(self) -> MaritimeTradeAction:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item else self._actions[0]
