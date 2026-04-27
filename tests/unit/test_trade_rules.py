"""Unit tests for :mod:`domain.rules.trade_rules` enumeration coverage."""

from __future__ import annotations

import copy

from domain.actions import all_actions as A
from domain.enums import Resource, TurnPhase
from domain.rules import trade_rules
from tests.fixtures.states import post_setup_state


def _main_phase_state(seed: int = 0):
    s = copy.deepcopy(post_setup_state(seed=seed))
    s.phase = TurnPhase.MAIN
    s.pending = None
    s.turn_number = 2
    s.players[s.current_player].has_played_dev_card_this_turn = False
    return s


def test_maritime_default_4_to_1_when_no_ports_owned() -> None:
    s = _main_phase_state()
    pid = s.current_player
    s.occupancy.buildings = {}
    s.players[pid].resources = {Resource.WOOD: 4}

    trades = trade_rules.legal_maritime_trades(s)
    woods = [t for t in trades if t.give is Resource.WOOD]
    assert woods, "expected at least one wood-giving maritime trade"
    assert all(t.give_count == 4 for t in woods)
    receives = {t.receive for t in woods}
    assert receives == {r for r in Resource if r is not Resource.DESERT and r is not Resource.WOOD}


def test_maritime_4_to_1_unavailable_below_threshold() -> None:
    s = _main_phase_state()
    pid = s.current_player
    s.occupancy.buildings = {}
    s.players[pid].resources = {Resource.WOOD: 3}

    trades = trade_rules.legal_maritime_trades(s)
    assert not any(t.give is Resource.WOOD for t in trades)


def test_domestic_enumerates_five_ratios_per_pair_when_hand_supports() -> None:
    s = _main_phase_state()
    pid = s.current_player
    s.players[pid].resources = {r: 3 for r in (
        Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE,
    )}

    proposals = trade_rules.legal_propose_domestic_single_resource(s)
    ratios_by_pair: dict[tuple[Resource, Resource], set[tuple[int, int]]] = {}
    for p in proposals:
        assert len(p.offer) == 1 and len(p.request) == 1
        ((give, gc),) = p.offer.items()
        ((recv, rc),) = p.request.items()
        ratios_by_pair.setdefault((give, recv), set()).add((gc, rc))

    expected_ratios = {(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)}
    pairs = [(a, b) for a in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE)
             for b in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE) if a != b]
    assert set(ratios_by_pair.keys()) == set(pairs)
    for pair in pairs:
        assert ratios_by_pair[pair] == expected_ratios


def test_domestic_give_count_bounded_by_proposer_hand() -> None:
    s = _main_phase_state()
    pid = s.current_player
    s.players[pid].resources = {Resource.WOOD: 2, Resource.BRICK: 0, Resource.SHEEP: 1,
                                Resource.WHEAT: 0, Resource.ORE: 0}

    proposals = trade_rules.legal_propose_domestic_single_resource(s)
    by_give: dict[Resource, set[int]] = {}
    for p in proposals:
        ((give, gc),) = p.offer.items()
        by_give.setdefault(give, set()).add(gc)

    assert by_give.get(Resource.WOOD) == {1, 2}
    assert by_give.get(Resource.SHEEP) == {1}
    assert Resource.BRICK not in by_give
    assert Resource.WHEAT not in by_give
    assert Resource.ORE not in by_give


def test_domestic_request_count_not_filtered_by_responder_hand() -> None:
    s = _main_phase_state()
    pid = s.current_player
    s.players[pid].resources = {Resource.WOOD: 1}
    for other_id, other in s.players.items():
        if other_id != pid:
            other.resources = {}

    proposals = trade_rules.legal_propose_domestic_single_resource(s)
    request_counts = {tuple(p.request.items())[0][1] for p in proposals}
    assert request_counts == {1, 2, 3}


def test_domestic_no_proposals_outside_main_or_with_pending() -> None:
    s = _main_phase_state()
    s.players[s.current_player].resources = {Resource.WOOD: 3}

    s_roll = copy.deepcopy(s)
    s_roll.phase = TurnPhase.ROLL
    assert trade_rules.legal_propose_domestic_single_resource(s_roll) == []

    s_pending = copy.deepcopy(s)
    s_pending.pending = object()  # any non-None pending blocks proposal
    assert trade_rules.legal_propose_domestic_single_resource(s_pending) == []


def test_main_phase_legal_actions_includes_n_to_1_and_1_to_n_proposals() -> None:
    from domain.rules.legal_actions import legal_actions

    s = _main_phase_state()
    pid = s.current_player
    s.players[pid].resources = {Resource.WOOD: 3}

    leg = legal_actions(s)
    proposals = [a for a in leg if isinstance(a, A.ProposeDomesticTradeAction)]
    seen_ratios = {
        (tuple(p.offer.items())[0][1], tuple(p.request.items())[0][1])
        for p in proposals
        if tuple(p.offer.keys()) == (Resource.WOOD,)
    }
    assert {(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)}.issubset(seen_ratios)
