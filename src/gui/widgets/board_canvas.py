from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from controller import selectors
from controller.session import GameSession, GameSnapshot
from domain.ids import EdgeID, TileID, VertexID
from gui import theme
from gui.geometry import tile_polygon_px, vertex_positions_px
from gui.items.edge_item import EdgeItem
from gui.items.port_item import PortItem
from gui.items.robber_item import RobberItem
from gui.items.tile_item import TileItem
from gui.items.vertex_item import VertexItem


class BoardCanvas(QGraphicsView):
    vertex_clicked = Signal(int)
    edge_clicked = Signal(int)
    tile_clicked = Signal(int)

    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._highlighted_vertices: set[VertexID] = set()
        self._highlighted_edges: set[EdgeID] = set()
        self._highlighted_tiles: set[TileID] = set()
        self._build_static_layer()
        self._build_overlay_layer()
        self.refresh(session.current())
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _build_static_layer(self) -> None:
        topology = self._session.current().state.topology
        vpos = vertex_positions_px()

        self._tile_items: dict[TileID, TileItem] = {}
        for tile_id, tile in topology.tiles.items():
            item = TileItem(tile, tile_polygon_px(tile_id))
            item.setZValue(0)
            self._scene.addItem(item)
            self._tile_items[tile_id] = item

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

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            item = next(iter(self.items(event.pos())), None)
            while item is not None and not isinstance(item, (VertexItem, EdgeItem, TileItem)):
                item = item.parentItem()

            if isinstance(item, VertexItem):
                vid = VertexID(item.data(0))
                if vid in self._highlighted_vertices:
                    self.vertex_clicked.emit(int(vid))
            elif isinstance(item, EdgeItem):
                eid = EdgeID(item.data(0))
                if eid in self._highlighted_edges:
                    self.edge_clicked.emit(int(eid))
            elif isinstance(item, TileItem):
                tid = TileID(item.data(0))
                if tid in self._highlighted_tiles:
                    self.tile_clicked.emit(int(tid))
        super().mousePressEvent(event)

    def refresh(self, snap: GameSnapshot) -> None:
        occ = snap.state.occupancy

        for vid, item in self._vertex_items.items():
            b = occ.building_at(vid)
            if b:
                owner, kind = b
                color = QColor(theme.PLAYER_COLORS[int(owner) % len(theme.PLAYER_COLORS)])
                item.set_building(color, kind)
            else:
                item.set_building(None, None)

        for eid, item in self._edge_items.items():
            owner = occ.road_owner(eid)
            color = QColor(theme.PLAYER_COLORS[int(owner) % len(theme.PLAYER_COLORS)]) if owner is not None else None
            item.set_road(color)

        self._robber.move_to(occ.robber_tile)

        legal = self._session.legal_actions()
        v_targets = selectors.vertex_targets(legal)
        e_targets = selectors.edge_targets(legal)
        t_targets = selectors.tile_targets(legal)

        self._highlighted_vertices = set().union(*v_targets.values()) if v_targets else set()
        self._highlighted_edges = set().union(*e_targets.values()) if e_targets else set()
        self._highlighted_tiles = t_targets

        for vid, item in self._vertex_items.items():
            item.set_highlight(vid in self._highlighted_vertices)
        for eid, item in self._edge_items.items():
            item.set_highlight(eid in self._highlighted_edges)
        for tid, item in self._tile_items.items():
            item.set_highlight(tid in self._highlighted_tiles)
