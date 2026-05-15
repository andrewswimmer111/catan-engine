"""Regression tests for replay session swaps in :mod:`gui.main_window`."""

from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

from controller.session import GameSession
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from gui.main_window import MainWindow


def _make_session(player_ids: list[PlayerID], seed: int = 2) -> GameSession:
    engine = GameEngine(SeededRandomizer(seed))
    config = GameConfig(player_ids=player_ids, seed=seed)
    return GameSession(engine, config)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_replace_session_rebuilds_panels_for_new_player_ids(qapp: QApplication) -> None:
    window = MainWindow(_make_session([PlayerID(i) for i in range(4)]))
    try:
        replay_ids = [PlayerID(i) for i in range(1, 5)]
        window._replace_session(_make_session(replay_ids))

        assert set(window._panels.keys()) == set(replay_ids)
        assert [window._view_combo.itemText(i) for i in range(window._view_combo.count())] == [
            "GOD",
            "P1",
            "P2",
            "P3",
            "P4",
        ]
    finally:
        window.close()
