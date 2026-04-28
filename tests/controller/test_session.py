"""Tests for GameSnapshot and GameSession."""

from __future__ import annotations

import pytest

from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from controller.session import GameSession, GameSnapshot
from tests.fixtures.states import _first_setup_action


def _make_session(seed: int = 0, n_players: int = 3) -> GameSession:
    pids = [PlayerID(i) for i in range(n_players)]
    config = GameConfig(player_ids=pids, seed=seed)
    engine = GameEngine(SeededRandomizer(seed))
    return GameSession(engine, config)


def _advance(session: GameSession, steps: int) -> None:
    for _ in range(steps):
        actions = session.legal_actions()
        session.apply(_first_setup_action(actions))


# ---------------------------------------------------------------------------
# forward step
# ---------------------------------------------------------------------------

def test_initial_snapshot_is_step_zero() -> None:
    session = _make_session()
    snap = session.current()
    assert snap.step_index == 0
    assert snap.last_action is None
    assert snap.last_events == ()


def test_forward_step_advances_cursor() -> None:
    session = _make_session()
    _advance(session, 1)
    assert session.current().step_index == 1
    assert len(session.history()) == 2


def test_forward_step_records_action_and_events() -> None:
    session = _make_session()
    actions = session.legal_actions()
    action = _first_setup_action(actions)
    snap = session.apply(action)
    assert snap.last_action == action
    assert isinstance(snap.last_events, tuple)


def test_legal_actions_matches_engine() -> None:
    session = _make_session()
    engine = GameEngine(SeededRandomizer(0))
    pids = [PlayerID(i) for i in range(3)]
    config = GameConfig(player_ids=pids, seed=0)
    state = engine.new_game(config)
    assert session.legal_actions() == engine.legal_actions(state)


def test_actions_log_grows_with_steps() -> None:
    session = _make_session()
    _advance(session, 3)
    assert len(session.actions_log()) == 3


# ---------------------------------------------------------------------------
# jump back + forward
# ---------------------------------------------------------------------------

def test_jump_to_step_zero_from_later() -> None:
    session = _make_session()
    _advance(session, 4)
    snap = session.jump_to(0)
    assert snap.step_index == 0
    assert snap.last_action is None


def test_jump_forward_within_history() -> None:
    session = _make_session()
    _advance(session, 4)
    session.jump_to(0)
    snap = session.jump_to(3)
    assert snap.step_index == 3


def test_jump_preserves_full_history() -> None:
    session = _make_session()
    _advance(session, 4)
    session.jump_to(1)
    assert len(session.history()) == 5  # snapshots 0..4 intact


def test_jump_out_of_range_raises() -> None:
    session = _make_session()
    with pytest.raises(IndexError):
        session.jump_to(5)


# ---------------------------------------------------------------------------
# jump back + apply (truncates future)
# ---------------------------------------------------------------------------

def test_apply_after_jump_back_truncates_future() -> None:
    session = _make_session()
    _advance(session, 4)
    session.jump_to(2)
    actions = session.legal_actions()
    session.apply(_first_setup_action(actions))
    # history should be 0,1,2 + new step = 4 snapshots total
    assert len(session.history()) == 4
    assert session.current().step_index == 3


def test_apply_after_jump_back_truncates_actions_log() -> None:
    session = _make_session()
    _advance(session, 4)
    session.jump_to(2)
    actions = session.legal_actions()
    session.apply(_first_setup_action(actions))
    assert len(session.actions_log()) == 3


def test_apply_after_jump_back_forward_jump_works() -> None:
    session = _make_session()
    _advance(session, 4)
    session.jump_to(1)
    actions = session.legal_actions()
    session.apply(_first_setup_action(actions))
    # After truncation + 1 apply: snapshots 0,1,new => 3
    assert len(session.history()) == 3
    session.jump_to(0)
    assert session.current().step_index == 0
    session.jump_to(2)
    assert session.current().step_index == 2


# ---------------------------------------------------------------------------
# on_change callback
# ---------------------------------------------------------------------------

def test_on_change_fired_on_apply() -> None:
    session = _make_session()
    received: list[GameSnapshot] = []
    session.on_change = received.append
    _advance(session, 2)
    assert len(received) == 2
    assert received[-1].step_index == 2


def test_on_change_not_fired_on_jump() -> None:
    session = _make_session()
    _advance(session, 3)
    received: list[GameSnapshot] = []
    session.on_change = received.append
    session.jump_to(1)
    assert received == []
