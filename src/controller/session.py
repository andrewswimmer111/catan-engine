"""
Lifecycle management for a single game session.

Owns the :class:`~domain.engine.game_engine.GameEngine` instance and acts as the
single entry point through which external callers advance the game, keeping
engine construction, seeding, and teardown in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.step_result import StepResult
from domain.events.base import GameEvent
from domain.game.config import GameConfig
from domain.game.state import GameState


__all__ = ["GameSnapshot", "GameSession"]


@dataclass(frozen=True)
class GameSnapshot:
    state: GameState
    step_index: int          # 0 = initial state
    last_action: Action | None
    last_events: tuple[GameEvent, ...]


class GameSession:
    def __init__(self, engine: GameEngine, config: GameConfig) -> None:
        self._engine = engine
        self._config = config
        self._actions: list[Action] = []
        self._snapshots: list[GameSnapshot] = [
            GameSnapshot(engine.new_game(config), 0, None, ())
        ]
        self._cursor: int = 0
        self.on_change: Callable[[GameSnapshot], None] | None = None

    def current(self) -> GameSnapshot:
        return self._snapshots[self._cursor]

    def legal_actions(self) -> list[Action]:
        return self._engine.legal_actions(self.current().state)

    def apply(self, action: Action) -> GameSnapshot:
        """Append a new snapshot; truncates any forward history past the cursor."""
        if self._cursor < len(self._snapshots) - 1:
            self._snapshots = self._snapshots[: self._cursor + 1]
            self._actions = self._actions[: self._cursor]

        result: StepResult = self._engine.apply_action(self.current().state, action)
        snapshot = GameSnapshot(
            state=result.state,
            step_index=self._cursor + 1,
            last_action=action,
            last_events=tuple(result.events),
        )
        self._snapshots.append(snapshot)
        self._actions.append(action)
        self._cursor += 1

        if self.on_change is not None:
            self.on_change(snapshot)
        return snapshot

    def jump_to(self, step_index: int) -> GameSnapshot:
        """Move cursor to step_index within [0, len(snapshots)-1]; never re-runs the engine."""
        if not (0 <= step_index < len(self._snapshots)):
            raise IndexError(
                f"step_index {step_index} out of range [0, {len(self._snapshots) - 1}]"
            )
        self._cursor = step_index
        return self._snapshots[self._cursor]

    def history(self) -> tuple[GameSnapshot, ...]:
        return tuple(self._snapshots)

    def actions_log(self) -> tuple[Action, ...]:
        return tuple(self._actions)
