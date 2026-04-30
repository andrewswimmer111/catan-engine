from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsLineItem

from domain.ids import EdgeID

_EMPTY_COLOR = "#dddddd"
_EMPTY_WIDTH = 1
_ROAD_WIDTH = 4
_HIGHLIGHT_COLOR = "#FFD700"
_HIGHLIGHT_WIDTH = 4


class EdgeItem(QGraphicsLineItem):
    def __init__(self, edge_id: EdgeID, p1: QPointF, p2: QPointF) -> None:
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())
        self.setData(0, int(edge_id))
        self.setZValue(1)
        self._color: QColor | None = None
        self._highlighted = False
        self._apply()

    def set_road(self, color: QColor | None) -> None:
        self._color = color
        self._apply()

    def set_highlight(self, on: bool) -> None:
        self._highlighted = on
        self._apply()

    def _apply(self) -> None:
        if self._color is not None:
            self.setPen(QPen(self._color, _ROAD_WIDTH))
        elif self._highlighted:
            self.setPen(QPen(QColor(_HIGHLIGHT_COLOR), _HIGHLIGHT_WIDTH))
        else:
            self.setPen(QPen(QColor(_EMPTY_COLOR), _EMPTY_WIDTH))
