"""
Public entry point for the game engine.

Application code, agents, and the RL layer should import only from here. Rule
mechanics live in :mod:`domain.rules` (``legal_actions`` and ``transitions``);
this class wires RNG, state construction, and validation.
"""

from __future__ import annotations

from domain.actions.base import Action
from domain.board.layout import build_standard_board
from domain.board.scenario import assign_random_scenario, desert_tile_id
from domain.board.occupancy import BoardOccupancy
from domain.enums import TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck, standard_dev_deck_composition
from domain.game.state import GameState
from domain.game.player_state import PlayerState
from domain.engine.player_view import PlayerView, make_player_view
from domain.engine.randomizer import Randomizer
from domain.engine.step_result import StepResult
from domain.ids import PlayerID, TileID
from domain.rules import transitions
from domain.rules.legal_actions import legal_actions as rules_legal_actions


__all__ = [
    "GameEngine",
    "IllegalActionError",
    "PlayerView",
]


class IllegalActionError(ValueError):
    """Raised when :meth:`GameEngine.apply_action` is passed an action not in ``legal_actions(state)``."""


def _snake_setup_order(player_ids: list[PlayerID]) -> list[PlayerID]:
    """Order of initial settlement/road turns: *forward* then *reverse* the seat list."""
    return list(player_ids) + list(reversed(player_ids))


class GameEngine:
    def __init__(self, rng: Randomizer) -> None:
        self._rng = rng

    def new_game(self, config: GameConfig) -> GameState:
        if config.board_variant != "standard":
            raise NotImplementedError(
                f"board variant {config.board_variant!r} is not supported; only 'standard' is implemented"
            )
        raw = build_standard_board()
        topology = assign_random_scenario(raw, self._rng)
        pids = list(config.player_ids)
        players: dict[PlayerID, PlayerState] = {pid: PlayerState(player_id=pid) for pid in pids}
        setup_order = _snake_setup_order(pids)
        shuffled = self._rng.shuffle_dev_deck(standard_dev_deck_composition())
        current = setup_order[0] if setup_order else pids[0]
        robber = desert_tile_id(topology)
        return GameState(
            config=config,
            topology=topology,
            occupancy=BoardOccupancy(robber_tile=robber),
            players=players,
            bank=Bank(),
            dev_deck=DevelopmentDeck(cards=shuffled),
            current_player=current,
            phase=TurnPhase.INITIAL_SETTLEMENT,
            turn_number=0,
            setup_order=setup_order,
            setup_index=0,
        )

    def legal_actions(self, state: GameState) -> list[Action]:
        return rules_legal_actions(state)

    def apply_action(self, state: GameState, action: Action) -> StepResult:
        allowed = rules_legal_actions(state)
        if action not in allowed:
            raise IllegalActionError("action is not legal in the current state")
        return transitions.apply(self._rng, state, action)

    def resolve_if_no_legal_actions(self, state: GameState) -> GameState:
        """Terminal stalemate when the position has no legal moves (engine source of truth)."""
        return transitions.resolve_no_legal_actions(state)

    def player_view(self, state: GameState, player_id: PlayerID) -> PlayerView:
        return make_player_view(state, player_id)
