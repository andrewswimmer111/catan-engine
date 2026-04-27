"""Tests for robber, discards, and seven / knight transitions."""

from __future__ import annotations

import copy

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


def test_cards_to_discard_on_seven_more_than_seven_cards() -> None:
    s = _base_state()
    s.players[PlayerID(0)].resources[Resource.WHEAT] = 8
    assert robber_rules.cards_to_discard_on_seven(s) == {PlayerID(0): 4}


def test_cards_to_discard_on_seven_empty_when_at_most_seven() -> None:
    s = _base_state()
    s.players[PlayerID(0)].resources[Resource.WHEAT] = 7
    assert robber_rules.cards_to_discard_on_seven(s) == {}


def test_enumerate_discard_options_multiset() -> None:
    s = _base_state()
    s.phase = TurnPhase.DISCARD
    s.pending = DiscardPending(cards_to_discard={PlayerID(0): 2})
    s.players[PlayerID(0)].resources = {Resource.WOOD: 1, Resource.BRICK: 1}
    legals = robber_rules.legal_discard_actions(s)
    assert len(legals) == 1
    r0 = legals[0].resources
    assert sum(r0.values()) == 2


class _AlwaysRollSeven:
    def roll_dice(self) -> tuple[int, int]:
        return (3, 4)

    def shuffle_dev_deck(self, cards):
        return list(cards)

    def choose_stolen_resource(self, resources):
        return resources[0]

    def shuffled(self, items):
        return list(items)

    def choice(self, items):
        return items[0]


def test_roll_seven_skips_production_and_sets_discard() -> None:
    s = _base_state()
    s.players[PlayerID(0)].resources[Resource.WHEAT] = 8
    rng = _AlwaysRollSeven()
    out = transitions.apply(rng, s, A.RollDiceAction(player_id=PlayerID(0)))
    assert out.state.phase is TurnPhase.DISCARD
    assert isinstance(out.state.pending, DiscardPending)
    assert PlayerID(0) in out.state.pending.cards_to_discard
    from domain.events.all_events import ResourcesDistributed

    assert not any(isinstance(e, ResourcesDistributed) for e in out.events)


def test_roll_seven_no_discard_goes_to_move_robber() -> None:
    s = _base_state()
    rng = _AlwaysRollSeven()
    out = transitions.apply(rng, s, A.RollDiceAction(player_id=PlayerID(0)))
    assert out.state.phase is TurnPhase.MOVE_ROBBER
    assert isinstance(out.state.pending, RobberMovePending)
    assert out.state.pending.return_phase is TurnPhase.MAIN


def test_knight_before_roll_returns_to_roll_after_robber_flow() -> None:
    s = _base_state(phase=TurnPhase.ROLL)
    p0 = s.players[PlayerID(0)]
    p0.dev_cards_in_hand.append((DevCardType.KNIGHT, 0))
    # Robber not on 0; P1 has a settlement on a vertex touching tile 0
    v_on_0 = next(
        vid
        for vid, vx in s.topology.vertices.items()
        if TileID(0) in vx.adjacent_tiles
    )
    s.occupancy.buildings = {v_on_0: (PlayerID(1), BuildingType.SETTLEMENT)}
    s.occupancy.robber_tile = TileID(1)
    p1 = s.players[PlayerID(1)]
    p1.resources[Resource.ORE] = 1
    rng = _AlwaysRollSeven()
    out = transitions.apply(rng, copy.deepcopy(s), A.PlayKnightAction(player_id=PlayerID(0)))
    assert out.state.phase is TurnPhase.MOVE_ROBBER
    assert isinstance(out.state.pending, RobberMovePending)
    assert out.state.pending.return_phase is TurnPhase.ROLL
    out2 = transitions.apply(
        rng, out.state, A.MoveRobberAction(player_id=PlayerID(0), tile_id=TileID(0))
    )
    assert out2.state.phase is TurnPhase.STEAL
    assert isinstance(out2.state.pending, StealPending)
    assert out2.state.pending.return_phase is TurnPhase.ROLL
    out3 = transitions.apply(
        rng,
        out2.state,
        A.StealResourceAction(player_id=PlayerID(0), target_player_id=PlayerID(1)),
    )
    assert out3.state.phase is TurnPhase.ROLL
    assert out3.state.pending is None
