"""Gym-shaped environment wrapper around :class:`GameEngine`.

``reset`` and ``step`` return an encoded ``np.ndarray`` observation plus an
:class:`Info` dict carrying the current agent, the boolean action mask, the
typed legal-actions list, the last events, and the phase.

``step`` accepts either an ``int`` (decoded via the action encoder) or a
typed :class:`Action`. The dual API exists so neural-net agents can drive the
env from masked categorical samples while heuristic / scripted agents (and
the existing tournament harness) keep passing typed actions through.
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
from domain.events.base import GameEvent
from domain.game.config import GameConfig
from domain.game.state import GameState
from domain.ids import PlayerID
from rl.acting_player import acting_player
from rl.agents.heuristic_agent import heuristic_discard
from rl.encoding.action import ActionEncoder, DiscardSentinel
from rl.encoding.observation import FlatObservationEncoder
from rl.encoding.protocol import ObservationEncoder
from rl.env.rewards import RewardFn, SparseWinReward
from rl.env.spec import Info

__all__ = ["CatanEnv", "IllegalActionError"]

_DEFAULT_PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]

ActionInput = Union[int, np.integer, Action]


def _default_config(seed: int, victory_point_target: int = 10) -> GameConfig:
    return GameConfig(
        player_ids=list(_DEFAULT_PLAYER_IDS),
        seed=seed,
        victory_point_target=victory_point_target,
    )


class CatanEnv:
    """Single-process Catan environment with encoded observations and masks."""

    def __init__(
        self,
        config: GameConfig | None = None,
        seed: int | None = None,
        obs_encoder: ObservationEncoder | None = None,
        action_encoder: ActionEncoder | None = None,
        reward_fn: RewardFn | None = None,
        victory_point_target: int = 10,
    ) -> None:
        self._base_config = config
        self._seed: int = seed if seed is not None else 0
        self._victory_point_target = victory_point_target
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(
            self._seed, victory_point_target=victory_point_target
        )
        self._state: GameState = self._engine.new_game(cfg)
        self._last_events: list[GameEvent] = []

        self._obs_encoder: ObservationEncoder = (
            obs_encoder if obs_encoder is not None else FlatObservationEncoder()
        )
        # If the caller supplied an action encoder we trust their seat mapping;
        # otherwise build one matching the current config so steal indices line
        # up with config.player_ids.
        self._user_action_encoder = action_encoder
        self._action_encoder: ActionEncoder = (
            action_encoder if action_encoder is not None
            else ActionEncoder.for_state(self._state)
        )
        self._reward_fn: RewardFn = reward_fn if reward_fn is not None else SparseWinReward()

        # Legal actions are computed once per step boundary and reused for the
        # mask, info["legal_actions"], and any int → Action decoding. Cleared
        # at each step entry so a stale list never feeds the next decision.
        self._legal_cache: list[Action] | None = None

        # Per-step rewards owed to non-acting seats (the canonical case is
        # losing Longest Road / Largest Army to whoever just acted). Set
        # by ``step()`` after applying the action; consumed by the rollout
        # workers between learner turns to retroactively credit the
        # learner's last stored transition. Empty after ``reset()``.
        self._last_cross_rewards: dict[PlayerID, float] = {}

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Info]:
        if seed is not None:
            self._seed = seed
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(
            self._seed, victory_point_target=self._victory_point_target
        )
        self._state = self._engine.new_game(cfg)
        self._last_events = []
        self._legal_cache = None
        self._last_cross_rewards = {}
        if self._user_action_encoder is None:
            self._action_encoder = ActionEncoder.for_state(self._state)
        return self._encode_obs(), self._info()

    def step(self, action: ActionInput) -> tuple[np.ndarray, float, bool, Info]:
        typed_action = self._resolve_action(action)
        # Capture decision context BEFORE the engine swaps in a new state. The
        # acting agent is the env's current_agent at step entry — for DISCARD,
        # that's the player who actually owes the discard, not the dice-roller.
        acting_agent = self.current_agent
        prev_state = self._state
        result = self._engine.apply_action(prev_state, typed_action)
        reward = float(self._reward_fn.step_reward(prev_state, typed_action, result, acting_agent))
        self._last_cross_rewards = dict(
            self._reward_fn.cross_step_rewards(
                prev_state, typed_action, result, acting_agent
            )
        )
        self._state = result.state
        self._last_events = list(result.events)
        self._legal_cache = None
        return self._encode_obs(), reward, result.is_terminal, self._info()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_agent(self) -> PlayerID:
        """Player expected to act next (see :func:`rl.acting_player.acting_player`)."""
        return acting_player(self._state)

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
    def obs_encoder(self) -> ObservationEncoder:
        return self._obs_encoder

    @property
    def action_encoder(self) -> ActionEncoder:
        return self._action_encoder

    @property
    def reward_fn(self) -> RewardFn:
        return self._reward_fn

    @property
    def last_cross_rewards(self) -> dict[PlayerID, float]:
        """Per-step rewards owed to non-acting seats from the most recent step.

        Returns a fresh dict on each call so callers can't accidentally
        clobber internal state. Empty after ``reset()`` and on steps where
        no non-acting seat saw a side-effect VP change.
        """
        return dict(self._last_cross_rewards)

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
        we resolve it through :func:`heuristic_discard`. The engine never
        sees the sentinel — the encoder's design declares this resolution as
        the env's responsibility, not the encoder's.
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
        """Delegate to the heuristic discard policy (see rl-009)."""
        return heuristic_discard(self._state, sentinel.player_id)
