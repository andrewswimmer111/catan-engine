from __future__ import annotations

import math

from PySide6.QtCore import QPointF
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItemGroup, QGraphicsTextItem

from domain.board.port import Port
from domain.ids import VertexID
from gui import theme

_CIRCLE_RADIUS = 8.0
_SEA_OFFSET = 18.0  # pixels away from board center so marker sits in the sea area
_LABEL_OFFSET = 28.0  # additional outward push so label clears the circle


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
        circle.setPen(QPen(QColor("#3e2723"), 1))
        self.addToGroup(circle)

        if port.port_type is not None:
            label = QGraphicsTextItem(port.port_type.value)
            font = QFont()
            font.setPointSize(7)
            font.setBold(True)
            label.setFont(font)
            br = label.boundingRect()
            # Push the label further outward from the board center.
            unit_x = ox / _SEA_OFFSET if dist > 0 else 0.0
            unit_y = oy / _SEA_OFFSET if dist > 0 else 0.0
            lx = cx + unit_x * _LABEL_OFFSET - br.width() / 2
            ly = cy + unit_y * _LABEL_OFFSET - br.height() / 2
            label.setPos(lx, ly)
            self.addToGroup(label)
