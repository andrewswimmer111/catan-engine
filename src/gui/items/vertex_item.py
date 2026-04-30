from __future__ import annotations

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem

from domain.enums import BuildingType
from domain.ids import PlayerID, VertexID
from gui import theme

_RADIUS = 6.0
_SETTLEMENT_WIDTH = 1
_CITY_WIDTH = 3
_EMPTY_COLOR = "#cccccc"
_HIGHLIGHT_COLOR = "#FFD700"
_HIGHLIGHT_WIDTH = 3


class VertexItem(QGraphicsEllipseItem):
    def __init__(self, vertex_id: VertexID, cx: float, cy: float) -> None:
        super().__init__(cx - _RADIUS, cy - _RADIUS, _RADIUS * 2, _RADIUS * 2)
        self.setData(0, int(vertex_id))
        self.setZValue(2)
        self._owner: PlayerID | None = None
        self._kind: BuildingType | None = None
        self._highlighted = False
        self._apply()

    def set_building(self, owner: PlayerID | None, kind: BuildingType | None) -> None:
        self._owner = owner
        self._kind = kind
        self._apply()

    def set_highlight(self, on: bool) -> None:
        self._highlighted = on
        self._apply()

    def _apply(self) -> None:
        owner, kind = self._owner, self._kind
        if owner is None:
            self.setBrush(QBrush(QColor(_EMPTY_COLOR)))
        else:
            self.setBrush(QBrush(QColor(theme.PLAYER_COLORS[int(owner) % len(theme.PLAYER_COLORS)])))

        if self._highlighted:
            self.setPen(QPen(QColor(_HIGHLIGHT_COLOR), _HIGHLIGHT_WIDTH))
        elif owner is None:
            self.setPen(QPen(QColor(_EMPTY_COLOR), _SETTLEMENT_WIDTH))
        else:
            color = QColor(theme.PLAYER_COLORS[int(owner) % len(theme.PLAYER_COLORS)])
            width = _CITY_WIDTH if kind == BuildingType.CITY else _SETTLEMENT_WIDTH
            self.setPen(QPen(color.darker(150), width))
