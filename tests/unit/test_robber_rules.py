"""Unit tests for robber, discard, and seven / steal rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from domain.actions import all_actions as A
from domain.board.layout import build_standard_board
from domain.board.occupancy import BoardOccupancy
from domain.enums import BuildingType, DevCardType, Resource, TurnPhase
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck, standard_dev_deck_composition
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import PlayerID, TileID
from domain.rules import robber_rules, transitions
from domain.turn.pending import DiscardPending, RobberMovePending, StealPending


def _base_state(*, phase: TurnPhase = TurnPhase.ROLL, turn_number: int = 1) -> GameState:
    topo = build_standard_board()
    pids = [PlayerID(0), PlayerID(1), PlayerID(2), PlayerID(3)]
    players = {pid: PlayerState(player_id=pid) for pid in pids}
    return GameState(
        config=GameConfig(player_ids=pids, seed=1),
        topology=topo,
        occupancy=BoardOccupancy(robber_tile=TileID(0)),
        players=players,
        bank=Bank(),
        dev_deck=DevelopmentDeck(cards=standard_dev_deck_composition()),
        current_player=pids[0],
        phase=phase,
        turn_number=turn_number,
    )


@dataclass
class _AlwaysRollSeven:
    def roll_dice(self) -> tuple[int, int]:
        return (3, 4)

    def shuffle_dev_deck(self, cards: list) -> list:
        return list(cards)

    def choose_stolen_resource(self, resources: list) -> Any:
        return resources[0]

    def shuffled(self, items: list) -> list:
        return list(items)

    def choice(self, items: list) -> Any:
        return items[0]


def test_rolling_seven_with_hands_seven_or_under_skips_discard_phase() -> None:
    s = _base_state()
    assert robber_rules.cards_to_discard_on_seven(s) == {}


def test_rolling_seven_with_eight_cards_requires_discard_of_four() -> None:
    s = _base_state()
    s.players[PlayerID(0)].resources[Resource.WHEAT] = 8
    assert robber_rules.cards_to_discard_on_seven(s) == {PlayerID(0): 4}


def test_legal_steal_actions_exclude_active_player_id_as_victim() -> None:
    s = _base_state()
    s.phase = TurnPhase.STEAL
    s.pending = StealPending(
        valid_targets=frozenset({PlayerID(1), PlayerID(2)}),
        return_phase=TurnPhase.MAIN,
    )
    for a in robber_rules.legal_steal_actions(s):
        assert a.target_player_id is not a.player_id


def test_legal_move_robber_omits_tile_where_robber_already_sits() -> None:
    s = _base_state()
    s.phase = TurnPhase.MOVE_ROBBER
    s.pending = RobberMovePending(return_phase=TurnPhase.MAIN)
    here = s.occupancy.robber_tile
    move_tiles = {a.tile_id for a in robber_rules.legal_robber_moves(s)}
    assert here not in move_tiles


def test_legal_discard_list_matches_fixed_hand_size() -> None:
    s = _base_state()
    s.phase = TurnPhase.DISCARD
    s.pending = DiscardPending(cards_to_discard={PlayerID(0): 2})
    s.players[PlayerID(0)].resources = {Resource.WOOD: 1, Resource.BRICK: 1}
    legals = robber_rules.legal_discard_actions(s)
    assert len(legals) == 1
    assert sum(legals[0].resources.values()) == 2


def test_roll_seven_routes_to_discard_when_over_limit() -> None:
    s = _base_state()
    s.players[PlayerID(0)].resources[Resource.WHEAT] = 8
    rng = _AlwaysRollSeven()
    out = transitions.apply(rng, s, A.RollDiceAction(player_id=PlayerID(0)))
    assert out.state.phase is TurnPhase.DISCARD
    assert isinstance(out.state.pending, DiscardPending)
    from domain.events.all_events import ResourcesDistributed

    assert not any(isinstance(e, ResourcesDistributed) for e in out.events)


def test_roll_seven_with_no_discard_enters_move_robber() -> None:
    s = _base_state()
    rng = _AlwaysRollSeven()
    out = transitions.apply(rng, s, A.RollDiceAction(player_id=PlayerID(0)))
    assert out.state.phase is TurnPhase.MOVE_ROBBER
    assert isinstance(out.state.pending, RobberMovePending)


def test_knight_resolves_to_roll_phase_after_full_robber_flow() -> None:
    s = _base_state(phase=TurnPhase.ROLL)
    s.players[PlayerID(0)].dev_cards_in_hand.append((DevCardType.KNIGHT, 0))
    v_on_0 = next(
        vid
        for vid, vx in s.topology.vertices.items()
        if TileID(0) in vx.adjacent_tiles
    )
    s.occupancy.buildings = {v_on_0: (PlayerID(1), BuildingType.SETTLEMENT)}
    s.occupancy.robber_tile = TileID(1)
    s.players[PlayerID(1)].resources[Resource.ORE] = 1
    rng = _AlwaysRollSeven()
    out1 = transitions.apply(
        rng, copy.deepcopy(s), A.PlayKnightAction(player_id=PlayerID(0))
    )
    assert out1.state.phase is TurnPhase.MOVE_ROBBER
    out2 = transitions.apply(
        rng, out1.state, A.MoveRobberAction(player_id=PlayerID(0), tile_id=TileID(0))
    )
    assert out2.state.phase is TurnPhase.STEAL
    out3 = transitions.apply(
        rng,
        out2.state,
        A.StealResourceAction(
            player_id=PlayerID(0), target_player_id=PlayerID(1)
        ),
    )
    assert out3.state.phase is TurnPhase.ROLL
    assert out3.state.pending is None
