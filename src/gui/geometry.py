from __future__ import annotations

import math
from functools import lru_cache

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from domain.board.layout import standard_board_coordinates
from domain.board.topology import BoardTopology
from domain.ids import EdgeID, TileID, VertexID

PIXELS_PER_UNIT = 80.0
_HEX_SIZE = 1.0  # must match domain.board.hex_geometry.SIZE


def _to_screen(x: float, y: float) -> QPointF:
    # Domain uses y-up; Qt uses y-down, so negate y.
    return QPointF(x * PIXELS_PER_UNIT, -y * PIXELS_PER_UNIT)


@lru_cache(maxsize=1)
def _board_coords() -> tuple[
    dict[TileID, tuple[float, float]],
    dict[VertexID, tuple[float, float]],
]:
    return standard_board_coordinates()


@lru_cache(maxsize=1)
def tile_centers_px() -> dict[TileID, QPointF]:
    tile_centers, _ = _board_coords()
    return {tid: _to_screen(x, y) for tid, (x, y) in tile_centers.items()}


@lru_cache(maxsize=1)
def vertex_positions_px() -> dict[VertexID, QPointF]:
    _, vertex_coords = _board_coords()
    return {vid: _to_screen(x, y) for vid, (x, y) in vertex_coords.items()}


def edge_midpoint_px(topology: BoardTopology, edge_id: EdgeID) -> QPointF:
    vpos = vertex_positions_px()
    edge = topology.edges[edge_id]
    p1, p2 = vpos[edge.v1], vpos[edge.v2]
    return QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)


def hex_polygon_px(center: QPointF, radius: float) -> QPolygonF:
    """Pointy-top hex polygon centered at ``center`` with corner radius ``radius``."""
    points: list[QPointF] = []
    for i in range(6):
        angle = math.pi / 6 + math.pi / 3 * i
        # cos is x (same direction), sin negated for y-down screen space
        points.append(QPointF(
            center.x() + radius * math.cos(angle),
            center.y() - radius * math.sin(angle),
        ))
    return QPolygonF(points)


def tile_polygon_px(tile_id: TileID) -> QPolygonF:
    """6-corner pointy-top hex polygon for a board tile in screen pixel space."""
    return hex_polygon_px(tile_centers_px()[tile_id], _HEX_SIZE * PIXELS_PER_UNIT)
