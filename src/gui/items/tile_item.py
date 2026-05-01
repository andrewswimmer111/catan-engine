from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
    QGraphicsSimpleTextItem,
)

from domain.board.tile import Tile
from gui import theme

_FALLBACK_COLOR = "#9e9e9e"
_HIGHLIGHT_COLOR = "#FFD700"
_HIGHLIGHT_WIDTH = 3
_NUMBER_DISC_RADIUS = 18
_NUMBER_COLOR = QColor("#000000")


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
            center = polygon.boundingRect().center()
            r = _NUMBER_DISC_RADIUS

            disc = QGraphicsEllipseItem(
                QRectF(center.x() - r, center.y() - r, r * 2, r * 2), self
            )
            disc.setBrush(QBrush(QColor("#fffde7")))
            disc.setPen(QPen(QColor("#bdbdbd"), 1))
            disc.setZValue(-1)

            # Sibling of the disc (not a child): QGraphicsSimpleTextItem paints more
            # reliably here than QGraphicsTextItem on some platforms.
            label = QGraphicsSimpleTextItem(str(tile.dice_number), self)
            font = QFont()
            font.setBold(True)
            font.setPointSize(11)
            label.setFont(font)
            label.setBrush(QBrush(_NUMBER_COLOR))
            label.setZValue(1)
            br = label.boundingRect()
            label.setPos(center.x() - br.width() / 2, center.y() - br.height() / 2)

    def set_highlight(self, on: bool) -> None:
        self._highlighted = on
        if on:
            self.setPen(QPen(QColor(_HIGHLIGHT_COLOR), _HIGHLIGHT_WIDTH))
        else:
            self.setPen(QPen())
