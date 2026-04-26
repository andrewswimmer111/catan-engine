"""
Per-seat, information-hiding projection of :class:`GameState`.

Opponents' unplayed development cards are reduced to a count; the dev deck
contents are reduced to a remaining-card count. Everything else matches the
public table state.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.board.occupancy import BoardOccupancy
from domain.board.topology import BoardTopology
from domain.enums import DevCardType, Resource, TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import PlayerID, VertexID
from domain.turn.pending import PendingEffect


@dataclass(frozen=True)
class PlayerPerspectiveState:
    """
    A single player's public stats, with ``dev_cards_in_hand`` either the full
    list (this row is the viewer) or the count of unplayed dev cards (opponent).
    """

    player_id: PlayerID
    resources: dict[Resource, int]
    dev_cards_in_hand: list[tuple[DevCardType, int]] | int
    dev_cards_played: list[DevCardType]
    roads_built: int
    settlements_built: int
    cities_built: int
    knights_played: int
    has_played_dev_card_this_turn: bool
    victory_points_public: int


@dataclass(frozen=True)
class PlayerView:
    """
    Game snapshot visible to one player: same structure as :class:`GameState`
    except hidden dev information.
    """

    config: GameConfig
    topology: BoardTopology
    occupancy: BoardOccupancy
    bank: Bank
    dev_deck_remaining: int
    players: dict[PlayerID, PlayerPerspectiveState]
    current_player: PlayerID
    phase: TurnPhase
    turn_number: int
    pending: PendingEffect | None
    setup_order: list[PlayerID]
    setup_index: int
    last_settlement_vertex: VertexID | None
    longest_road_holder: PlayerID | None
    largest_army_holder: PlayerID | None
    winner: PlayerID | None


def _perspective_row(ps: PlayerState, viewer: PlayerID) -> PlayerPerspectiveState:
    if ps.player_id == viewer:
        dev: list[tuple[DevCardType, int]] | int = list(ps.dev_cards_in_hand)
    else:
        dev = len(ps.dev_cards_in_hand)
    return PlayerPerspectiveState(
        player_id=ps.player_id,
        resources=dict(ps.resources),
        dev_cards_in_hand=dev,
        dev_cards_played=list(ps.dev_cards_played),
        roads_built=ps.roads_built,
        settlements_built=ps.settlements_built,
        cities_built=ps.cities_built,
        knights_played=ps.knights_played,
        has_played_dev_card_this_turn=ps.has_played_dev_card_this_turn,
        victory_points_public=ps.victory_points_public,
    )


def make_player_view(state: GameState, player_id: PlayerID) -> PlayerView:
    """Build a :class:`PlayerView` for the given seat."""
    return PlayerView(
        config=state.config,
        topology=state.topology,
        occupancy=state.occupancy,
        bank=state.bank,
        dev_deck_remaining=state.dev_deck.remaining(),
        players={pid: _perspective_row(p, player_id) for pid, p in state.players.items()},
        current_player=state.current_player,
        phase=state.phase,
        turn_number=state.turn_number,
        pending=state.pending,
        setup_order=list(state.setup_order),
        setup_index=state.setup_index,
        last_settlement_vertex=state.last_settlement_vertex,
        longest_road_holder=state.longest_road_holder,
        largest_army_holder=state.largest_army_holder,
        winner=state.winner,
    )
