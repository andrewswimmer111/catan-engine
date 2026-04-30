from __future__ import annotations

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QGraphicsPolygonItem

from domain.ids import TileID
from gui.geometry import hex_polygon_px, tile_centers_px

_RADIUS = 14.0
_COLOR = "#616161"


class RobberItem(QGraphicsPolygonItem):
    def __init__(self) -> None:
        super().__init__()
        self.setZValue(3)
        self.setBrush(QBrush(QColor(_COLOR)))

    def move_to(self, tile_id: TileID) -> None:
        self.setPolygon(hex_polygon_px(tile_centers_px()[tile_id], _RADIUS))
