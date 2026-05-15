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
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import controller.selectors as selectors
import domain.actions.all_actions as A
from controller.agents import HumanAgent, make_default_agents, make_scripted_agent
from controller.orchestrator import Orchestrator
from controller.session import GameSession, GameSnapshot
from domain.engine.game_engine import GameEngine
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from gui.widgets.action_panel import ActionPanel
from gui.widgets.board_canvas import BoardCanvas
from gui.widgets.event_log import EventLogWidget
from gui.widgets.player_panel import PlayerPanel
from gui.widgets.policy_overlay import PolicyOverlayWidget
from gui.widgets.timeline import TimelineWidget
from gui.widgets.trade_panel import TradePanel
from rl.encoding.action import ActionEncoder
from rl.utils.gui_hook import load_episode_into_session
from serialization.replay import load_replay, save_replay


class MainWindow(QMainWindow):

    _TIMELINE_TO_BOTTOM_DOCK_GAP_PX = 8
    _BOTTOM_DOCK_MIN_HEIGHT_PX = 220
    _MAIN_SPLITTER_STRETCH = (4, 1)
    _MAIN_SPLITTER_INITIAL_SIZES = (820, 180)

    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        session.on_change = self.refresh
        self.setWindowTitle("Catan Engine")

        self._canvas = BoardCanvas(session)
        self._panel = ActionPanel()
        self._trade = TradePanel()
        self._timeline = TimelineWidget(session)
        self._event_log = EventLogWidget(session)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(
            0, 0, 0, self._TIMELINE_TO_BOTTOM_DOCK_GAP_PX
        )
        central_layout.setSpacing(0)
        central_layout.addWidget(self._build_main_splitter(), stretch=1)
        self.setCentralWidget(central)

        # Player panels dock (left edge). Panels themselves are (re)built
        # on every session swap by ``_rebuild_player_panels`` — see the
        # _replace_session path for why.
        self._panels: dict[PlayerID, PlayerPanel] = {}
        players_container = QWidget()
        self._players_layout = QVBoxLayout(players_container)
        self._players_layout.setContentsMargins(4, 4, 4, 4)
        self._players_layout.setSpacing(4)

        players_dock = QDockWidget("Players", self)
        players_dock.setWidget(players_container)
        players_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, players_dock)

        player_ids = sorted(session.current().state.players.keys())
        self._rebuild_player_panels(player_ids)

        # Event log dock (bottom-left half)
        log_dock = QDockWidget("Event Log", self)
        log_dock.setWidget(self._event_log)
        log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # Policy overlay dock — populated only when an RL replay is loaded.
        # Shares the bottom area with the event log, split horizontally 50/50.
        self._policy_overlay = PolicyOverlayWidget()
        overlay_dock = QDockWidget("Policy Overlay", self)
        overlay_dock.setWidget(self._policy_overlay)
        overlay_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, overlay_dock)
        self._configure_bottom_docks(log_dock, overlay_dock)

        self._canvas.vertex_clicked.connect(self._on_vertex_clicked)
        self._canvas.edge_clicked.connect(self._on_edge_clicked)
        self._canvas.tile_clicked.connect(self._on_tile_clicked)
        self._panel.action_chosen.connect(self._on_action_chosen)
        self._trade.action_chosen.connect(self._on_action_chosen)
        self._timeline.jumped.connect(self._on_jumped)
        self._event_log.jumped.connect(self._on_jumped)

        self._agents = make_default_agents(player_ids)
        self._orchestrator = Orchestrator(session, self._agents)

        self._setup_menu()
        self._setup_toolbar(player_ids)
        self._setup_shortcuts()

        self.statusBar()
        self._render(session.current())

    def _configure_bottom_docks(
        self, log_dock: QDockWidget, overlay_dock: QDockWidget
    ) -> None:
        self.splitDockWidget(log_dock, overlay_dock, Qt.Horizontal)
        self.resizeDocks([log_dock, overlay_dock], [1, 1], Qt.Horizontal)

        # Minimum heights reliably enforce a taller shared bottom dock row.
        for dock in (log_dock, overlay_dock):
            dock.setMinimumHeight(self._BOTTOM_DOCK_MIN_HEIGHT_PX)

    def _build_main_splitter(self) -> QSplitter:
        board_pane = QWidget()
        board_layout = QVBoxLayout(board_pane)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(0)
        board_layout.addWidget(self._canvas, stretch=1)
        board_layout.addWidget(self._timeline)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._panel, stretch=2)
        right_layout.addWidget(self._trade, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(board_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, self._MAIN_SPLITTER_STRETCH[0])
        splitter.setStretchFactor(1, self._MAIN_SPLITTER_STRETCH[1])
        splitter.setSizes(list(self._MAIN_SPLITTER_INITIAL_SIZES))
        return splitter

    # ------------------------------------------------------------------
    # Menu, toolbar & shortcuts
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        menu.addAction("Save Replay…", self._save_replay)
        menu.addAction("Load Replay…", self._load_replay)
        menu.addAction("Open RL Replay…", self._load_rl_replay)
        menu.addSeparator()
        menu.addAction("Quit", self.close)

    def _setup_toolbar(self, player_ids: list[PlayerID]) -> None:
        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)

        toolbar.addWidget(QLabel("View as:  "))
        self._view_combo = QComboBox()
        toolbar.addWidget(self._view_combo)
        self._view_combo.currentTextChanged.connect(
            lambda _: self._refresh_player_panels(self._session.current())
        )
        self._populate_view_combo(player_ids)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Seats:  "))
        self._seat_combos: dict[PlayerID, QComboBox] = {}
        for pid in player_ids:
            toolbar.addWidget(QLabel(f"P{int(pid)}:"))
            combo = QComboBox()
            combo.addItems(["HUMAN", "SCRIPTED"])
            combo.currentTextChanged.connect(
                lambda text, p=pid: self._on_seat_changed(p, text)
            )
            self._seat_combos[pid] = combo
            toolbar.addWidget(combo)

        toolbar.addSeparator()
        step1_btn = QPushButton("Step 1")
        step1_btn.clicked.connect(self._on_step_once)
        toolbar.addWidget(step1_btn)

        step_all_btn = QPushButton("Step until Human")
        step_all_btn.clicked.connect(self._on_step_until_human)
        toolbar.addWidget(step_all_btn)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Left"), self).activated.connect(self._timeline.step_back)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._timeline.step_forward)
        QShortcut(QKeySequence("Home"), self).activated.connect(self._timeline.jump_start)
        QShortcut(QKeySequence("End"), self).activated.connect(self._timeline.jump_end)

    # ------------------------------------------------------------------
    # Per-session widgets (panels + view dropdown)
    # ------------------------------------------------------------------

    def _rebuild_player_panels(self, player_ids: list[PlayerID]) -> None:
        """Tear down any existing player panels and rebuild for ``player_ids``.

        Replays may use a different PlayerID set than the initial session
        (e.g. RL episodes use 1..4 while the GUI's fresh game historically
        used 0..3). Without rebuilding, ``self._panels`` would stay keyed
        by stale IDs and panel lookups against the new state would miss.
        """
        while self._players_layout.count():
            item = self._players_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._panels.clear()
        for pid in player_ids:
            panel = PlayerPanel(int(pid))
            self._panels[pid] = panel
            self._players_layout.addWidget(panel)
        self._players_layout.addStretch()

    def _populate_view_combo(self, player_ids: list[PlayerID]) -> None:
        """Replace the View-as dropdown's items to match ``player_ids``."""
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItem("GOD")
        for pid in player_ids:
            self._view_combo.addItem(f"P{int(pid)}")
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(False)

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
            self._policy_overlay.clear()
            self._replace_session(new_session)

    def _load_rl_replay(self) -> None:
        """Open an :class:`EpisodeRecord` directory and overlay its policy data."""
        from pathlib import Path

        ep_dir = QFileDialog.getExistingDirectory(self, "Open RL Replay")
        if not ep_dir:
            return
        loaded = load_episode_into_session(Path(ep_dir))
        new_session = loaded.session
        encoder = ActionEncoder(list(new_session.current().state.config.player_ids))
        self._policy_overlay.set_overlay(loaded.overlay, encoder)
        self._replace_session(new_session)

    def _replace_session(self, session: GameSession) -> None:
        self._session = session
        session.on_change = self.refresh
        self._canvas.set_session(session)
        self._timeline.set_session(session)
        self._event_log.set_session(session)
        self._orchestrator.set_session(session)
        player_ids = sorted(session.current().state.players.keys())
        self._rebuild_player_panels(player_ids)
        self._populate_view_combo(player_ids)
        self._render(session.current())

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

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

    def _render(self, snap: GameSnapshot) -> None:
        """Sync every widget except the event log against ``snap``."""
        self._update_status(snap)
        legal = self._session.legal_actions()
        self._canvas.refresh(snap)
        self._panel.refresh(snap, legal)
        self._trade.refresh(snap, legal)
        self._timeline.refresh(snap)
        self._refresh_player_panels(snap)
        self._policy_overlay.on_step_changed(snap)

    def refresh(self, snap: GameSnapshot) -> None:
        self._render(snap)
        self._event_log.on_applied(snap)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_seat_changed(self, player_id: PlayerID, text: str) -> None:
        if text == "SCRIPTED":
            self._orchestrator.set_agent(player_id, make_scripted_agent(player_id))
        else:
            self._orchestrator.set_agent(player_id, HumanAgent())

    def _on_step_once(self) -> None:
        self._orchestrator.step_once()

    def _on_step_until_human(self) -> None:
        self._orchestrator.run_until_human()

    def _on_jumped(self, snap: GameSnapshot) -> None:
        self._render(snap)

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
