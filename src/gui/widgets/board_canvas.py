from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from controller.session import GameSession, GameSnapshot
from gui.geometry import tile_polygon_px, vertex_positions_px
from gui.items.port_item import PortItem
from gui.items.tile_item import TileItem


class BoardCanvas(QGraphicsView):
    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._build_static_layer()
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _build_static_layer(self) -> None:
        topology = self._session.current().state.topology
        vpos = vertex_positions_px()

        for tile_id, tile in topology.tiles.items():
            self._scene.addItem(TileItem(tile, tile_polygon_px(tile_id)))

        for port in topology.ports:
            self._scene.addItem(PortItem(port, vpos))

    def refresh(self, snap: GameSnapshot) -> None:
        pass  # overlay layer added in ticket 2.3
