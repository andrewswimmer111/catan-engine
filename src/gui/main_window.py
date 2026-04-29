from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow

from controller.session import GameSession, GameSnapshot


class MainWindow(QMainWindow):
    def __init__(self, session: GameSession) -> None:
        super().__init__()
        self._session = session
        self.setWindowTitle("Catan Engine")

        placeholder = QLabel("board goes here")
        placeholder.setAlignment(placeholder.alignment())
        self.setCentralWidget(placeholder)

        menu = self.menuBar().addMenu("File")
        menu.addAction("Quit", self.close)

        self.statusBar()
        self._update_status(session.current())

    def _update_status(self, snap: GameSnapshot) -> None:
        state = snap.state
        self.statusBar().showMessage(
            f"phase={state.phase.name}  player={int(state.current_player)}"
        )

    def refresh(self, snap: GameSnapshot) -> None:
        self._update_status(snap)
