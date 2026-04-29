from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from controller.session import GameSession, GameSnapshot
from domain.ids import EdgeID, VertexID
from gui.geometry import tile_polygon_px, vertex_positions_px
from gui.items.edge_item import EdgeItem
from gui.items.port_item import PortItem
from gui.items.robber_item import RobberItem
from gui.items.tile_item import TileItem
from gui.items.vertex_item import VertexItem


class BoardCanvas(QGraphicsView):
    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._build_static_layer()
        self._build_overlay_layer()
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _build_static_layer(self) -> None:
        topology = self._session.current().state.topology
        vpos = vertex_positions_px()

        for tile_id, tile in topology.tiles.items():
            item = TileItem(tile, tile_polygon_px(tile_id))
            item.setZValue(0)
            self._scene.addItem(item)

        for port in topology.ports:
            self._scene.addItem(PortItem(port, vpos))

    def _build_overlay_layer(self) -> None:
        topology = self._session.current().state.topology
        vpos = vertex_positions_px()

        self._edge_items: dict[EdgeID, EdgeItem] = {}
        for edge_id, edge in topology.edges.items():
            item = EdgeItem(edge_id, vpos[edge.v1], vpos[edge.v2])
            self._scene.addItem(item)
            self._edge_items[edge_id] = item

        self._vertex_items: dict[VertexID, VertexItem] = {}
        for vertex_id in topology.vertices:
            p = vpos[vertex_id]
            item = VertexItem(vertex_id, p.x(), p.y())
            self._scene.addItem(item)
            self._vertex_items[vertex_id] = item

        self._robber = RobberItem()
        self._scene.addItem(self._robber)
        self._robber.move_to(self._session.current().state.occupancy.robber_tile)

    def refresh(self, snap: GameSnapshot) -> None:
        occ = snap.state.occupancy

        for vid, item in self._vertex_items.items():
            b = occ.building_at(vid)
            item.set_building(*(b if b else (None, None)))

        for eid, item in self._edge_items.items():
            item.set_road(occ.road_owner(eid))

        self._robber.move_to(occ.robber_tile)
