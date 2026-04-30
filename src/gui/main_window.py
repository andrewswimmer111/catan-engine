from __future__ import annotations

from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QMainWindow, QMenu, QSplitter

import controller.selectors as selectors
import domain.actions.all_actions as A
from controller.session import GameSession, GameSnapshot
from domain.ids import EdgeID, TileID, VertexID
from gui.widgets.action_panel import ActionPanel
from gui.widgets.board_canvas import BoardCanvas


class MainWindow(QMainWindow):
    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self.setWindowTitle("Catan Engine")

        self._canvas = BoardCanvas(session)
        self._panel = ActionPanel()

        splitter = QSplitter()
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._canvas.vertex_clicked.connect(self._on_vertex_clicked)
        self._canvas.edge_clicked.connect(self._on_edge_clicked)
        self._canvas.tile_clicked.connect(self._on_tile_clicked)
        self._panel.action_chosen.connect(self._on_action_chosen)

        menu = self.menuBar().addMenu("File")
        menu.addAction("Quit", self.close)

        self.statusBar()
        snap = session.current()
        self._update_status(snap)
        self._panel.refresh(snap, session.legal_actions())

    def _update_status(self, snap: GameSnapshot) -> None:
        state = snap.state
        self.statusBar().showMessage(
            f"phase={state.phase.name}  player={int(state.current_player)}"
        )

    def refresh(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
        self._canvas.refresh(snap)
        self._panel.refresh(snap, self._session.legal_actions())

    def _on_action_chosen(self, action: object) -> None:
        self._session.apply(action)

    def _on_vertex_clicked(self, vertex_id_int: int) -> None:
        vid = VertexID(vertex_id_int)
        legal = self._session.legal_actions()
        v_targets = selectors.vertex_targets(legal)
        candidates = [cls for cls, ids in v_targets.items() if vid in ids]
        if not candidates:
            return
        player_id = self._session.current().state.current_player
        if len(candidates) == 1:
            self._session.apply(candidates[0](player_id=player_id, vertex_id=vid))
        else:
            menu = QMenu(self)
            for cls in candidates:
                menu.addAction(
                    cls.__name__,
                    lambda c=cls: self._session.apply(c(player_id=player_id, vertex_id=vid)),
                )
            menu.exec(QCursor.pos())

    def _on_edge_clicked(self, edge_id_int: int) -> None:
        eid = EdgeID(edge_id_int)
        legal = self._session.legal_actions()
        e_targets = selectors.edge_targets(legal)
        candidates = [cls for cls, ids in e_targets.items() if eid in ids]
        if not candidates:
            return
        player_id = self._session.current().state.current_player
        if len(candidates) == 1:
            self._session.apply(candidates[0](player_id=player_id, edge_id=eid))
        else:
            menu = QMenu(self)
            for cls in candidates:
                menu.addAction(
                    cls.__name__,
                    lambda c=cls: self._session.apply(c(player_id=player_id, edge_id=eid)),
                )
            menu.exec(QCursor.pos())

    def _on_tile_clicked(self, tile_id_int: int) -> None:
        tid = TileID(tile_id_int)
        player_id = self._session.current().state.current_player
        self._session.apply(A.MoveRobberAction(player_id=player_id, tile_id=tid))
