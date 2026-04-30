from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
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
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from gui.widgets.action_panel import ActionPanel
from gui.widgets.board_canvas import BoardCanvas
from gui.widgets.event_log import EventLogWidget
from gui.widgets.player_panel import PlayerPanel
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
        self._event_log = EventLogWidget(session)

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

        # Player panels dock (left edge)
        player_ids = sorted(session.current().state.players.keys())
        self._panels: dict[PlayerID, PlayerPanel] = {}
        players_container = QWidget()
        players_layout = QVBoxLayout(players_container)
        players_layout.setContentsMargins(4, 4, 4, 4)
        players_layout.setSpacing(4)
        for pid in player_ids:
            panel = PlayerPanel(int(pid))
            self._panels[pid] = panel
            players_layout.addWidget(panel)
        players_layout.addStretch()

        players_dock = QDockWidget("Players", self)
        players_dock.setWidget(players_container)
        players_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, players_dock)

        # Event log dock (bottom)
        log_dock = QDockWidget("Event Log", self)
        log_dock.setWidget(self._event_log)
        log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        self._canvas.vertex_clicked.connect(self._on_vertex_clicked)
        self._canvas.edge_clicked.connect(self._on_edge_clicked)
        self._canvas.tile_clicked.connect(self._on_tile_clicked)
        self._panel.action_chosen.connect(self._on_action_chosen)
        self._trade.action_chosen.connect(self._on_action_chosen)
        self._timeline.jumped.connect(self._on_jumped)
        self._event_log.jumped.connect(self._on_jumped)

        self._setup_menu()
        self._setup_toolbar(player_ids)
        self._setup_shortcuts()

        self.statusBar()
        snap = session.current()
        legal = session.legal_actions()
        self._update_status(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)
        self._refresh_player_panels(snap)

    # ------------------------------------------------------------------
    # Menu, toolbar & shortcuts
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        menu.addAction("Save Replay…", self._save_replay)
        menu.addAction("Load Replay…", self._load_replay)
        menu.addSeparator()
        menu.addAction("Quit", self.close)

    def _setup_toolbar(self, player_ids: list[PlayerID]) -> None:
        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("View as:  "))
        self._view_combo = QComboBox()
        self._view_combo.addItem("GOD")
        for pid in player_ids:
            self._view_combo.addItem(f"P{int(pid)}")
        toolbar.addWidget(self._view_combo)
        self._view_combo.currentTextChanged.connect(
            lambda _: self._refresh_player_panels(self._session.current())
        )

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
        self._event_log.set_session(session)
        snap = session.current()
        self.refresh(snap)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _prev_state(self, snap: GameSnapshot) -> GameState | None:
        if snap.step_index == 0:
            return None
        return self._session.history()[snap.step_index - 1].state

    def _update_status(self, snap: GameSnapshot) -> None:
        state = snap.state
        self.statusBar().showMessage(
            f"phase={state.phase.name}  player={int(state.current_player)}"
        )

    def _refresh_player_panels(self, snap: GameSnapshot) -> None:
        state = snap.state
        selection = self._view_combo.currentText()

        if selection == "GOD":
            for pid, panel in self._panels.items():
                panel.render_full(
                    state.players[pid],
                    longest_road=(state.longest_road_holder == pid),
                    largest_army=(state.largest_army_holder == pid),
                )
        else:
            viewer_id = PlayerID(int(selection[1:]))  # "P0" → PlayerID(0)
            view = make_player_view(state, viewer_id)
            for pid, panel in self._panels.items():
                panel.render_perspective(
                    view.players[pid],
                    longest_road=(state.longest_road_holder == pid),
                    largest_army=(state.largest_army_holder == pid),
                )

    # ------------------------------------------------------------------
    # Public refresh (called by session.on_change after apply())
    # ------------------------------------------------------------------

    def refresh(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
        legal = self._session.legal_actions()
        self._canvas.refresh(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)
        self._timeline.refresh(snap)
        self._event_log.on_applied(snap)
        self._refresh_player_panels(snap)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_jumped(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
        legal = self._session.legal_actions()
        self._canvas.refresh(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)
        self._timeline.refresh(snap)
        self._refresh_player_panels(snap)
        if snap.step_index == 0:
            self._event_log.rebuild()

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
