from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsLineItem

from domain.ids import EdgeID, PlayerID
from gui import theme

_EMPTY_COLOR = "#dddddd"
_EMPTY_WIDTH = 1
_ROAD_WIDTH = 4


class EdgeItem(QGraphicsLineItem):
    def __init__(self, edge_id: EdgeID, p1: QPointF, p2: QPointF) -> None:
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())
        self.setData(0, int(edge_id))
        self.setZValue(1)
        self._apply(None)

    def set_road(self, owner: PlayerID | None) -> None:
        self._apply(owner)

    def _apply(self, owner: PlayerID | None) -> None:
        if owner is None:
            self.setPen(QPen(QColor(_EMPTY_COLOR), _EMPTY_WIDTH))
        else:
            color = QColor(theme.PLAYER_COLORS[int(owner) % len(theme.PLAYER_COLORS)])
            self.setPen(QPen(color, _ROAD_WIDTH))
