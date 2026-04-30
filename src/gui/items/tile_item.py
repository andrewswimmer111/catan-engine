from __future__ import annotations

from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsTextItem

from domain.board.tile import Tile
from gui import theme

_FALLBACK_COLOR = "#9e9e9e"
_HIGHLIGHT_COLOR = "#FFD700"
_HIGHLIGHT_WIDTH = 3


class TileItem(QGraphicsPolygonItem):
    def __init__(self, tile: Tile, polygon: QPolygonF) -> None:
        super().__init__(polygon)
        self.setData(0, int(tile.tile_id))
        self._highlighted = False

        color = (
            theme.RESOURCE_FILL.get(tile.resource, _FALLBACK_COLOR)
            if tile.resource is not None
            else _FALLBACK_COLOR
        )
        self.setBrush(QBrush(QColor(color)))

        if tile.dice_number is not None:
            label = QGraphicsTextItem(str(tile.dice_number), self)
            br = label.boundingRect()
            center = polygon.boundingRect().center()
            label.setPos(center.x() - br.width() / 2, center.y() - br.height() / 2)

    def set_highlight(self, on: bool) -> None:
        self._highlighted = on
        if on:
            self.setPen(QPen(QColor(_HIGHLIGHT_COLOR), _HIGHLIGHT_WIDTH))
        else:
            self.setPen(QPen())
