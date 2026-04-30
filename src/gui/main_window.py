from __future__ import annotations

from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import controller.selectors as selectors
import domain.actions.all_actions as A
from controller.session import GameSession, GameSnapshot
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.ids import EdgeID, TileID, VertexID
from gui.widgets.action_panel import ActionPanel
from gui.widgets.board_canvas import BoardCanvas
from gui.widgets.timeline import TimelineWidget
from gui.widgets.trade_panel import TradePanel
from serialization.replay import load_replay, save_replay


class MainWindow(QMainWindow):
    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self.setWindowTitle("Catan Engine")

        self._canvas = BoardCanvas(session)
        self._panel = ActionPanel()
        self._trade = TradePanel()
        self._timeline = TimelineWidget(session)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._panel, stretch=2)
        right_layout.addWidget(self._trade, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(self._canvas)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([750, 250])

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(splitter, stretch=1)
        central_layout.addWidget(self._timeline)
        self.setCentralWidget(central)

        self._canvas.vertex_clicked.connect(self._on_vertex_clicked)
        self._canvas.edge_clicked.connect(self._on_edge_clicked)
        self._canvas.tile_clicked.connect(self._on_tile_clicked)
        self._panel.action_chosen.connect(self._on_action_chosen)
        self._trade.action_chosen.connect(self._on_action_chosen)
        self._timeline.jumped.connect(self._on_jumped)

        self._setup_menu()
        self._setup_shortcuts()

        self.statusBar()
        snap = session.current()
        legal = session.legal_actions()
        self._update_status(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)

    # ------------------------------------------------------------------
    # Menu & shortcuts
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        menu.addAction("Save Replay…", self._save_replay)
        menu.addAction("Load Replay…", self._load_replay)
        menu.addSeparator()
        menu.addAction("Quit", self.close)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Left"), self).activated.connect(self._timeline.step_back)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._timeline.step_forward)
        QShortcut(QKeySequence("Home"), self).activated.connect(self._timeline.jump_start)
        QShortcut(QKeySequence("End"), self).activated.connect(self._timeline.jump_end)

    # ------------------------------------------------------------------
    # Replay I/O
    # ------------------------------------------------------------------

    def _save_replay(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Replay", "", "JSON Files (*.json)"
        )
        if path:
            log = self._session.export_replay()
            save_replay(log, path)

    def _load_replay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Replay", "", "JSON Files (*.json)"
        )
        if path:
            log = load_replay(path)
            engine = GameEngine(SeededRandomizer(seed=log.config.seed))
            new_session = GameSession.from_replay(engine, log)
            self._replace_session(new_session)

    def _replace_session(self, session: GameSession) -> None:
        self._session = session
        self._canvas._session = session
        session.on_change = self.refresh
        self._timeline.set_session(session)
        snap = session.current()
        self.refresh(snap)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_status(self, snap: GameSnapshot) -> None:
        state = snap.state
        self.statusBar().showMessage(
            f"phase={state.phase.name}  player={int(state.current_player)}"
        )

    def refresh(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
        legal = self._session.legal_actions()
        self._canvas.refresh(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)
        self._timeline.refresh(snap)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_jumped(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
        legal = self._session.legal_actions()
        self._canvas.refresh(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)

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
