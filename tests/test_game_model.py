"""Smoke tests for Task 3 game configuration and state containers."""

from __future__ import annotations

import pytest

from domain.board.layout import build_standard_board
from domain.board.occupancy import BoardOccupancy
from domain.enums import Resource, TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck, standard_dev_deck_composition
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import PlayerID, TileID
from domain.turn.phase import is_setup_phase, requires_active_player_only


def test_game_config_accepts_3_and_4_players() -> None:
    GameConfig(player_ids=[PlayerID(i) for i in range(3)], seed=0)
    GameConfig(player_ids=[PlayerID(i) for i in range(4)], seed=0)


def test_game_config_rejects_invalid_player_count() -> None:
    with pytest.raises(ValueError, match="3 or 4"):
        GameConfig(player_ids=[PlayerID(0), PlayerID(1)], seed=0)


def test_game_config_defaults_to_standard_ten_vp_target() -> None:
    """Existing call sites that didn't pass ``victory_point_target``
    must keep getting the standard Catan 10 VP threshold."""
    cfg = GameConfig(player_ids=[PlayerID(i) for i in range(4)], seed=0)
    assert cfg.victory_point_target == 10


def test_game_config_accepts_lower_vp_target_for_curriculum() -> None:
    """Curriculum training uses lower targets (e.g. 6) so games actually
    terminate in real wins; the constructor must accept them."""
    cfg = GameConfig(
        player_ids=[PlayerID(i) for i in range(4)],
        seed=0,
        victory_point_target=6,
    )
    assert cfg.victory_point_target == 6


def test_game_config_rejects_vp_target_below_two() -> None:
    """A 2-seat initial-placement already yields 2 VP per player —
    any target < 2 would award the win at the first check. Refuse it."""
    with pytest.raises(ValueError, match="victory_point_target"):
        GameConfig(
            player_ids=[PlayerID(i) for i in range(4)],
            seed=0,
            victory_point_target=1,
        )


def test_bank_starts_at_19_per_tradeable_resource() -> None:
    from domain.enums import tradeable_resources
    b = Bank()
    for r in tradeable_resources():
        assert b.resources[r] == 19


def test_bank_withdraw_rejects_overspend() -> None:
    b = Bank()
    with pytest.raises(ValueError, match="insufficient"):
        b.withdraw({Resource.WOOD: 20})


def test_development_deck_composition() -> None:
    deck = standard_dev_deck_composition()
    assert len(deck) == 25


def test_fresh_game_state_is_not_terminal() -> None:
    topo = build_standard_board()
    pids = [PlayerID(0), PlayerID(1), PlayerID(2), PlayerID(3)]
    players = {pid: PlayerState(player_id=pid) for pid in pids}
    state = GameState(
        config=GameConfig(player_ids=pids, seed=1),
        topology=topo,
        occupancy=BoardOccupancy(robber_tile=TileID(0)),
        players=players,
        bank=Bank(),
        dev_deck=DevelopmentDeck(cards=standard_dev_deck_composition()),
        current_player=pids[0],
        phase=TurnPhase.INITIAL_SETTLEMENT,
        turn_number=0,
    )
    assert not state.is_terminal()
    assert state.active_player() is players[pids[0]]
    assert state.robber_tile == TileID(0)


def test_phase_helpers() -> None:
    assert is_setup_phase(TurnPhase.INITIAL_SETTLEMENT) is True
    assert is_setup_phase(TurnPhase.MAIN) is False
    assert requires_active_player_only(TurnPhase.DISCARD) is False
    assert requires_active_player_only(TurnPhase.MAIN) is True
