"""Gym-shaped environment wrapper around :class:`GameEngine`.

``reset`` and ``step`` return an encoded ``np.ndarray`` observation plus an
:class:`Info` dict carrying the current agent, the boolean action mask, the
typed legal-actions list, the last events, and the phase.

``step`` accepts either an ``int`` (decoded via the action encoder) or a
typed :class:`Action`. The dual API exists so neural-net agents can drive the
env from masked categorical samples while heuristic / scripted agents (and
the existing tournament harness) keep passing typed actions through.

Reward stays hard-coded at 0.0 until rl-008.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from domain.actions.all_actions import DiscardResourcesAction
from domain.actions.base import Action
from domain.engine.game_engine import (
    GameEngine,
    IllegalActionError as IllegalActionError,
)
from domain.engine.player_view import PlayerView
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.events.base import GameEvent
from domain.game.config import GameConfig
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending
from rl.encoding.action import ActionEncoder, DiscardSentinel
from rl.encoding.observation import FlatObservationEncoder
from rl.env.spec import Info

__all__ = ["CatanEnv", "IllegalActionError"]

_DEFAULT_PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]

ActionInput = Union[int, np.integer, Action]


def _default_config(seed: int) -> GameConfig:
    return GameConfig(player_ids=list(_DEFAULT_PLAYER_IDS), seed=seed)


class CatanEnv:
    """Single-process Catan environment with encoded observations and masks."""

    def __init__(
        self,
        config: GameConfig | None = None,
        seed: int | None = None,
        obs_encoder: FlatObservationEncoder | None = None,
        action_encoder: ActionEncoder | None = None,
    ) -> None:
        self._base_config = config
        self._seed: int = seed if seed is not None else 0
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(self._seed)
        self._state: GameState = self._engine.new_game(cfg)
        self._last_events: list[GameEvent] = []

        self._obs_encoder = obs_encoder or FlatObservationEncoder()
        # If the caller supplied an action encoder we trust their seat mapping;
        # otherwise build one matching the current config so steal indices line
        # up with config.player_ids.
        self._user_action_encoder = action_encoder
        self._action_encoder: ActionEncoder = (
            action_encoder if action_encoder is not None
            else ActionEncoder.for_state(self._state)
        )

        # Legal actions are computed once per step boundary and reused for the
        # mask, info["legal_actions"], and any int → Action decoding. Cleared
        # at each step entry so a stale list never feeds the next decision.
        self._legal_cache: list[Action] | None = None

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Info]:
        if seed is not None:
            self._seed = seed
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(self._seed)
        self._state = self._engine.new_game(cfg)
        self._last_events = []
        self._legal_cache = None
        if self._user_action_encoder is None:
            self._action_encoder = ActionEncoder.for_state(self._state)
        return self._encode_obs(), self._info()

    def step(self, action: ActionInput) -> tuple[np.ndarray, float, bool, Info]:
        typed_action = self._resolve_action(action)
        result = self._engine.apply_action(self._state, typed_action)
        self._state = result.state
        self._last_events = list(result.events)
        self._legal_cache = None
        return self._encode_obs(), 0.0, result.is_terminal, self._info()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_agent(self) -> PlayerID:
        """Player expected to act next.

        In DISCARD phase several players may owe discards; returns the first
        player still listed in DiscardPending rather than the dice-roller
        (state.current_player), so callers can dispatch to the correct agent.
        """
        if (
            self._state.phase == TurnPhase.DISCARD
            and isinstance(self._state.pending, DiscardPending)
        ):
            return next(iter(self._state.pending.cards_to_discard))
        return self._state.current_player

    def legal_actions(self) -> list[Action]:
        """Typed legal-action list for the current state, cached per step."""
        if self._legal_cache is None:
            self._legal_cache = self._engine.legal_actions(self._state)
        return self._legal_cache

    def action_mask(self) -> np.ndarray:
        """Boolean mask of length ``ACTION_SPACE_SIZE`` for the current state."""
        return self._action_encoder.mask(self.legal_actions())

    @property
    def state(self) -> GameState:
        """Full game state — for tests and debugging only; do not mutate."""
        return self._state

    @property
    def obs_encoder(self) -> FlatObservationEncoder:
        return self._obs_encoder

    @property
    def action_encoder(self) -> ActionEncoder:
        return self._action_encoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _player_view(self, pid: PlayerID) -> PlayerView:
        return self._engine.player_view(self._state, pid)

    def _encode_obs(self) -> np.ndarray:
        return self._obs_encoder.encode(self._player_view(self.current_agent))

    def _info(self) -> Info:
        return Info(
            current_agent=self.current_agent,
            action_mask=self.action_mask(),
            legal_actions=list(self.legal_actions()),
            last_events=list(self._last_events),
            phase=self._state.phase,
        )

    def _resolve_action(self, action: ActionInput) -> Action:
        """Coerce an int index or typed Action into the typed Action the engine wants.

        Discards: when an integer index decodes to a :class:`DiscardSentinel`,
        we resolve it against the currently-cached legal actions by picking
        the first :class:`DiscardResourcesAction` for the owing player. This
        is a placeholder until the rl-009 heuristic ships; documented in the
        encoder design as the env's job, not the encoder's.
        """
        if isinstance(action, Action):
            return action
        if isinstance(action, (int, np.integer)):
            decoded = self._action_encoder.decode(int(action), self._state)
            if isinstance(decoded, DiscardSentinel):
                return self._resolve_discard_sentinel(decoded)
            return decoded
        raise TypeError(
            f"step() expected int or Action, got {type(action).__name__}"
        )

    def _resolve_discard_sentinel(
        self, sentinel: DiscardSentinel
    ) -> DiscardResourcesAction:
        """Placeholder discard policy — pick the first matching legal discard.

        Replaced by a proper heuristic in rl-009. Raises ``IllegalActionError``
        if no matching discard is legal (e.g. discard idx fired outside the
        DISCARD phase).
        """
        for a in self.legal_actions():
            if (
                isinstance(a, DiscardResourcesAction)
                and a.player_id == sentinel.player_id
            ):
                return a
        raise IllegalActionError(
            f"discard sentinel for player {sentinel.player_id} has no matching "
            f"legal DiscardResourcesAction in the current state"
        )
