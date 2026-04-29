"""
Lifecycle management for a single game session.

Owns the :class:`~domain.engine.game_engine.GameEngine` instance and acts as the
single entry point through which external callers advance the game, keeping
engine construction, seeding, and teardown in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.step_result import StepResult
from domain.events.base import GameEvent
from domain.game.config import GameConfig
from domain.game.state import GameState

if TYPE_CHECKING:
    from serialization.replay import ReplayLog


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

    @classmethod
    def from_replay(
        cls,
        engine: GameEngine,
        log: "ReplayLog",
    ) -> "GameSession":
        """Replay every action; the resulting session has the full history
        loaded and the cursor at the final step.

        The engine must use the same RNG seed the log was recorded under —
        passing an engine with a different seed will produce incorrect state
        because board layout, dev-deck order, and dice rolls are all seeded.
        """
        from serialization.codec import decode_action

        session: GameSession = cls.__new__(cls)
        session._engine = engine
        session._config = log.config
        session.on_change = None

        initial_state = engine.new_game(log.config)
        session._snapshots = [GameSnapshot(initial_state, 0, None, ())]
        session._actions = []
        session._cursor = 0

        state = initial_state
        for i, act_data in enumerate(log.actions):
            action = decode_action(act_data)
            result: StepResult = engine.apply_action(state, action)
            snapshot = GameSnapshot(
                state=result.state,
                step_index=i + 1,
                last_action=action,
                last_events=tuple(result.events),
            )
            session._snapshots.append(snapshot)
            session._actions.append(action)
            state = result.state

        session._cursor = len(session._snapshots) - 1
        return session

    def export_replay(self) -> "ReplayLog":
        """Build a ReplayLog from the actions applied so far (cursor-aware:
        only includes actions up to the cursor)."""
        from serialization.codec import encode_action, encode_event
        from serialization.replay import ReplayLog

        actions = []
        events = []
        for i, action in enumerate(self._actions[: self._cursor]):
            actions.append(encode_action(action))
            events.append([encode_event(e) for e in self._snapshots[i + 1].last_events])

        return ReplayLog(config=self._config, actions=actions, events=events)
