"""
The authoritative, full-game :class:`GameState` model: board, bank, players,
phase, and pending effect.  Defines only data and simple read-only helpers, not
rules or transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.board.occupancy import BoardOccupancy
from domain.board.topology import BoardTopology
from domain.enums import TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck
from domain.game.player_state import PlayerState
from domain.ids import PlayerID, TileID
from domain.turn.pending import PendingEffect


@dataclass
class GameState:
    """
    Authoritative snapshot of a game. No rule logic — only data and
    read-only views (:meth:`is_terminal`, :meth:`active_player`).

    The robber position is stored on :attr:`occupancy` only. :attr:`robber_tile`
    is a convenience alias.
    """

    config: GameConfig
    topology: BoardTopology
    occupancy: BoardOccupancy
    players: dict[PlayerID, PlayerState]
    bank: Bank
    dev_deck: DevelopmentDeck
    current_player: PlayerID
    phase: TurnPhase
    turn_number: int
    pending: Optional[PendingEffect] = None
    setup_order: list[PlayerID] = field(default_factory=list)
    setup_index: int = 0
    longest_road_holder: Optional[PlayerID] = None
    largest_army_holder: Optional[PlayerID] = None
    winner: Optional[PlayerID] = None

    @property
    def robber_tile(self) -> TileID:
        return self.occupancy.robber_tile

    def is_terminal(self) -> bool:
        return self.phase == TurnPhase.GAME_OVER or self.winner is not None

    def active_player(self) -> PlayerState:
        return self.players[self.current_player]
