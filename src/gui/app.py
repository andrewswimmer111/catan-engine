from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from controller.session import GameSession
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    engine = GameEngine(SeededRandomizer(seed=2))
    config = GameConfig(player_ids=[PlayerID(i) for i in range(4)], seed=2)
    session = GameSession(engine, config)
    win = MainWindow(session)
    session.on_change = win.refresh
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
