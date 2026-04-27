"""
Record and verify games by stepping with the same seed and matching emitted events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.engine.game_engine import GameEngine
from domain.engine.step_result import StepResult
from domain.game.config import GameConfig
from domain.game.state import GameState

from serialization.codec import (
    decode_action,
    decode_config,
    decode_event,
    encode_config,
)


@dataclass(frozen=True)
class ReplayLog:
    """
    * ``actions`` and ``events`` are parallel: ``events[i]`` are the encoded
      events from applying ``actions[i]`` to the state reached after actions
      ``0 .. i-1`` (or from :meth:`GameEngine.new_game` when ``i == 0``).
    * Each inner event list is JSON-ready dicts from :func:`encode_event
      <serialization.codec.encode_event>`.
    """

    config: GameConfig
    actions: list[dict[str, Any]]
    events: list[list[dict[str, Any]]]


def save_replay(log: ReplayLog, path: str) -> None:
    payload: dict[str, Any] = {
        "config": encode_config(log.config),
        "actions": log.actions,
        "events": log.events,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_replay(path: str) -> ReplayLog:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return ReplayLog(
        config=decode_config(data["config"]),
        actions=data["actions"],
        events=data["events"],
    )


def replay_game(log: ReplayLog, engine: GameEngine) -> list[StepResult]:
    """
    Rebuild from ``log.config`` with ``engine`` (use the same RNG/seed the log
    was recorded under), re-apply each action, and assert the emitted event
    list matches ``log.events`` at each step. Returns every :class:`StepResult`.
    """
    if len(log.actions) != len(log.events):
        raise ValueError("ReplayLog.actions and ReplayLog.events must have the same length")
    state: GameState = engine.new_game(log.config)
    out: list[StepResult] = []
    for i, act_data in enumerate(log.actions):
        action = decode_action(act_data)
        result = engine.apply_action(state, action)
        expected = [decode_event(d) for d in log.events[i]]
        if list(result.events) != expected:
            raise AssertionError(
                f"replay step {i}: event stream mismatch. expected {expected!r} got {list(result.events)!r}"
            )
        out.append(result)
        state = result.state
    return out
