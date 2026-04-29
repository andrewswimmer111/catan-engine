from __future__ import annotations

import math

from PySide6.QtGui import QBrush, QColor, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem

from domain.ids import TileID
from gui.geometry import tile_centers_px

_RADIUS = 14.0
_COLOR = "#616161"


def _hex_polygon(cx: float, cy: float, r: float) -> QPolygonF:
    from PySide6.QtCore import QPointF
    points = []
    for i in range(6):
        angle = math.pi / 6 + math.pi / 3 * i
        points.append(QPointF(cx + r * math.cos(angle), cy - r * math.sin(angle)))
    return QPolygonF(points)


class RobberItem(QGraphicsPolygonItem):
    def __init__(self) -> None:
        super().__init__()
        self.setZValue(3)
        self.setBrush(QBrush(QColor(_COLOR)))

    def move_to(self, tile_id: TileID) -> None:
        center = tile_centers_px()[tile_id]
        self.setPolygon(_hex_polygon(center.x(), center.y(), _RADIUS))
