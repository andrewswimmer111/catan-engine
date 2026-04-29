from __future__ import annotations

import math

from PySide6.QtCore import QPointF
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItemGroup, QGraphicsTextItem

from domain.board.port import Port
from domain.ids import VertexID
from gui import theme

_CIRCLE_RADIUS = 8.0
_SEA_OFFSET = 18.0  # pixels away from board center so marker sits in the sea area


class PortItem(QGraphicsItemGroup):
    def __init__(self, port: Port, vpos: dict[VertexID, QPointF]) -> None:
        super().__init__()

        p1 = vpos[port.vertices[0]]
        p2 = vpos[port.vertices[1]]
        mx = (p1.x() + p2.x()) / 2
        my = (p1.y() + p2.y()) / 2

        # Offset away from board center (0, 0) so the marker sits in the sea.
        dist = math.hypot(mx, my)
        if dist > 0:
            ox = mx / dist * _SEA_OFFSET
            oy = my / dist * _SEA_OFFSET
        else:
            ox, oy = 0.0, 0.0

        cx, cy = mx + ox, my + oy
        r = _CIRCLE_RADIUS

        circle = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2)
        circle.setBrush(QBrush(QColor(theme.PORT_COLOR)))
        self.addToGroup(circle)

        if port.port_type is not None:
            label = QGraphicsTextItem(port.port_type.value)
            font = QFont()
            font.setPointSize(6)
            label.setFont(font)
            br = label.boundingRect()
            # Label floats further outward from the circle (same outward direction).
            label.setPos(cx + ox / _SEA_OFFSET * (r + 2) - br.width() / 2,
                         cy + oy / _SEA_OFFSET * (r + 2) - br.height() / 2)
            self.addToGroup(label)
