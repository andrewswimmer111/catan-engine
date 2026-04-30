from __future__ import annotations

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem

from domain.enums import BuildingType
from domain.ids import VertexID

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
        self._color: QColor | None = None
        self._kind: BuildingType | None = None
        self._highlighted = False
        self._apply()

    def set_building(self, color: QColor | None, kind: BuildingType | None) -> None:
        self._color = color
        self._kind = kind
        self._apply()

    def set_highlight(self, on: bool) -> None:
        self._highlighted = on
        self._apply()

    def _apply(self) -> None:
        if self._color is None:
            self.setBrush(QBrush(QColor(_EMPTY_COLOR)))
        else:
            self.setBrush(QBrush(self._color))

        if self._highlighted:
            self.setPen(QPen(QColor(_HIGHLIGHT_COLOR), _HIGHLIGHT_WIDTH))
        elif self._color is None:
            self.setPen(QPen(QColor(_EMPTY_COLOR), _SETTLEMENT_WIDTH))
        else:
            width = _CITY_WIDTH if self._kind == BuildingType.CITY else _SETTLEMENT_WIDTH
            self.setPen(QPen(self._color.darker(150), width))
