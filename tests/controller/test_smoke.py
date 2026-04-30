"""Full-stack smoke test: headless game loop with four ScriptedAgents.

Must not import anything from gui/ — verifies the controller/domain layering
boundary.
"""

from __future__ import annotations

import random

import pytest

from controller.agents import ScriptedAgent
from controller.orchestrator import Orchestrator
from controller.session import GameSession
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from serialization.codec import encode_state

_SEED = 2
_MAX_STEPS = 20_000


def _make_session_and_orchestrator(seed: int) -> tuple[GameSession, Orchestrator]:
    pids = [PlayerID(i) for i in range(4)]
    config = GameConfig(player_ids=pids, seed=seed)
    engine = GameEngine(SeededRandomizer(seed))
    session = GameSession(engine, config)
    agents = {pid: ScriptedAgent(random.Random(seed + int(pid))) for pid in pids}
    orchestrator = Orchestrator(session, agents)
    return session, orchestrator


def test_full_game_reaches_terminal() -> None:
    session, orchestrator = _make_session_and_orchestrator(_SEED)

    steps = 0
    while steps < _MAX_STEPS:
        if not orchestrator.step_once():
            break
        steps += 1

    assert steps < _MAX_STEPS, "Game did not terminate within step budget"
    assert session.current().state.is_terminal()


def test_history_length_matches_actions() -> None:
    session, orchestrator = _make_session_and_orchestrator(_SEED)

    while orchestrator.step_once():
        pass

    # history() includes the initial snapshot (step 0) plus one per action applied
    assert len(session.history()) == len(session.actions_log()) + 1


def test_replay_round_trip() -> None:
    session, orchestrator = _make_session_and_orchestrator(_SEED)

    while orchestrator.step_once():
        pass

    original_encoded = encode_state(session.current().state)
    log = session.export_replay()

    restored = GameSession.from_replay(GameEngine(SeededRandomizer(_SEED)), log)

    assert encode_state(restored.current().state) == original_encoded


def test_deterministic_across_runs() -> None:
    """Running the same seed twice must produce identical final states."""
    def run(seed: int) -> object:
        session, orchestrator = _make_session_and_orchestrator(seed)
        while orchestrator.step_once():
            pass
        return encode_state(session.current().state)

    assert run(_SEED) == run(_SEED)
