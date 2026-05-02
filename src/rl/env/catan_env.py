"""Gym-shaped environment wrapper around GameEngine.

Observations are raw PlayerView objects; action encoding (tensor indices) arrives
in rl-007. Reward is hard-coded to 0.0 until rl-008.
"""

from __future__ import annotations

from domain.actions.base import Action
from domain.engine.game_engine import GameEngine, IllegalActionError as IllegalActionError
from domain.engine.player_view import PlayerView
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.game.config import GameConfig
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending
from rl.env.spec import Info

__all__ = ["CatanEnv"]

_DEFAULT_PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _default_config(seed: int) -> GameConfig:
    return GameConfig(player_ids=list(_DEFAULT_PLAYER_IDS), seed=seed)


class CatanEnv:
    """Single-process Catan environment.

    step() accepts a typed Action for now. Index-based API (for neural nets)
    arrives in rl-007 once the encoder exists.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self._base_config = config
        self._seed: int = seed if seed is not None else 0
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(self._seed)
        self._state: GameState = self._engine.new_game(cfg)
        self._last_events: list = []

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[PlayerView, Info]:
        if seed is not None:
            self._seed = seed
        self._engine = GameEngine(SeededRandomizer(self._seed))
        cfg = self._base_config or _default_config(self._seed)
        self._state = self._engine.new_game(cfg)
        self._last_events = []
        return self._engine.player_view(self._state, self.current_agent), self._info()

    def step(self, action: Action) -> tuple[PlayerView, float, bool, Info]:
        result = self._engine.apply_action(self._state, action)
        self._state = result.state
        self._last_events = result.events
        obs = self._engine.player_view(self._state, self.current_agent)
        return obs, 0.0, result.is_terminal, self._info()

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
        return self._engine.legal_actions(self._state)

    @property
    def state(self) -> GameState:
        """Full game state — for tests and debugging only; do not mutate."""
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _info(self) -> Info:
        return Info(
            current_agent=self.current_agent,
            legal_actions=self.legal_actions(),
            last_events=list(self._last_events),
            current_phase=self._state.phase,
        )
