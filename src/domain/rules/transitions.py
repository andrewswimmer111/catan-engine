"""
State transitions: apply actions to a copied :class:`GameState` and collect events.

Each public action is routed by type to a small handler function; handlers are
registered in :data:`_HANDLERS` so adding a new action does not require editing
a long if/elif chain.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.randomizer import Randomizer
from domain.engine.step_result import StepResult
from domain.enums import (
    BuildingType,
    DevCardType,
    DomesticTradeState,
    EndReason,
    Resource,
    TurnPhase,
    tradeable_resources,
)
import domain.events.all_events as E
from domain.events.base import GameEvent
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.rules import build_rules, resource_rules, robber_rules, setup_rules, victory
from domain.rules.legal_actions import legal_actions as rules_legal_actions
from domain.rules import longest_road as lr_mod
from domain.turn.pending import (
    DiscardPending,
    DomesticTradePending,
    RobberMovePending,
    RoadBuildingPending,
    StealPending,
)


# ----------------------------------------------------------------------
# Post-action bookkeeping (awards + winner check)
# ----------------------------------------------------------------------

VP_STALL_TURN_THRESHOLD = 1000


def _all_players_at_full_build_expansion(state: GameState) -> bool:
    """True when every player has placed all settlement tokens and city upgrades."""
    for pl in state.players.values():
        if pl.settlements_built + pl.cities_built < build_rules.MAX_SETTLEMENTS:
            return False
        if pl.cities_built < build_rules.MAX_CITIES:
            return False
    return True


def _no_progress_possible(state: GameState) -> bool:
    """
    Conservative deadlock heuristic: full build expansion for all seats, no dev
    draws possible, and no one is at win threshold on the full VP tally.

    Unapplicable in standard 10 VP Catan game, here for extension.
    """
    if state.winner is not None:
        return False
    if any(
        victory.compute_victory_points(state, p) >= 10
        for p in state.config.player_ids
    ):
        return False
    if not _all_players_at_full_build_expansion(state):
        return False
    if state.dev_deck.remaining() == 0:
        return True
    return not any(
        pl.can_afford(build_rules.DEV_CARD_COST) for pl in state.players.values()
    )


def _maybe_stalemate(s: GameState, ev: list[GameEvent]) -> None:
    if s.winner is not None or s.phase is TurnPhase.GAME_OVER:
        return
    if s.phase is TurnPhase.STALEMATE:
        return
    if _no_progress_possible(s):
        s.phase = TurnPhase.STALEMATE
        s.end_reason = EndReason.STALEMATE_NO_PROGRESS
        ev.append(
            E.GameStalled(turn_number=s.turn_number, reason=EndReason.STALEMATE_NO_PROGRESS)
        )
    elif s.turns_since_vp_change > VP_STALL_TURN_THRESHOLD:
        s.phase = TurnPhase.STALEMATE
        s.end_reason = EndReason.STALEMATE_VP_STALL
        ev.append(
            E.GameStalled(turn_number=s.turn_number, reason=EndReason.STALEMATE_VP_STALL)
        )


def _sync_public_vp(s: GameState) -> None:
    """Recompute victory_points_public for every player from authoritative state."""
    for pid, p in s.players.items():
        p.victory_points_public = (
            p.settlements_built
            + 2 * p.cities_built
            + (2 if s.longest_road_holder == pid else 0)
            + (2 if s.largest_army_holder == pid else 0)
        )


def _post_action(s: GameState, ev: list[GameEvent]) -> None:
    """Recompute special awards and end the game if the current player has won."""
    _, lrc = lr_mod.update_longest_road_award(s)
    if lrc and s.longest_road_holder is not None:
        ev.append(
            E.LongestRoadAwarded(
                turn_number=s.turn_number,
                player_id=s.longest_road_holder,
                length=lr_mod.compute_longest_road(s, s.longest_road_holder),
            )
        )
    _, lac = victory.update_largest_army_award(s)
    if lac and s.largest_army_holder is not None:
        ev.append(
            E.LargestArmyAwarded(
                turn_number=s.turn_number,
                player_id=s.largest_army_holder,
                count=s.players[s.largest_army_holder].knights_played,
            )
        )
    _sync_public_vp(s)
    w = victory.check_winner(s)
    if w is not None:
        s.winner = w
        s.phase = TurnPhase.GAME_OVER
        s.end_reason = EndReason.WINNER
        ev.append(
            E.GameWon(
                turn_number=s.turn_number,
                player_id=w,
                victory_points=victory.compute_victory_points(s, w),
            )
        )


def _emit_distribution(
    distributions: dict[PlayerID, dict[Resource, int]], turn: int
) -> E.ResourcesDistributed | None:
    """Build a :class:`ResourcesDistributed` event, or ``None`` if no one received anything."""
    clean = {p: m for p, m in distributions.items() if m}
    if not clean:
        return None
    return E.ResourcesDistributed(turn_number=turn, distributions=clean)


# ----------------------------------------------------------------------
# Action handlers
# ----------------------------------------------------------------------


def _place_settlement(
    s: GameState, action: A.PlaceSettlementAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    events: list[GameEvent] = []
    s.occupancy.buildings[action.vertex_id] = (action.player_id, BuildingType.SETTLEMENT)
    p = s.players[action.player_id]
    p.settlements_built += 1
    p.victory_points_public += 1
    events.append(
        E.SettlementBuilt(
            turn_number=s.turn_number,
            player_id=action.player_id,
            vertex_id=action.vertex_id,
        )
    )
    s.last_settlement_vertex = action.vertex_id
    s.phase = TurnPhase.INITIAL_ROAD
    if setup_rules.is_second_settlement_turn(s):
        wanted = setup_rules.second_settlement_resources(s, action.vertex_id)
        granted, sfs = resource_rules.grant_from_bank(
            s, action.player_id, wanted, s.turn_number
        )
        events.extend(sfs)
        if granted:
            er = _emit_distribution({action.player_id: granted}, s.turn_number)
            if er is not None:
                events.append(er)
    _post_action(s, events)
    return s, events


def _place_road(
    s: GameState, action: A.PlaceRoadAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    events: list[GameEvent] = [
        E.RoadBuilt(
            turn_number=s.turn_number,
            player_id=action.player_id,
            edge_id=action.edge_id,
        )
    ]
    s.occupancy.roads[action.edge_id] = action.player_id
    s.players[action.player_id].roads_built += 1
    s.last_settlement_vertex = None
    s.setup_index += 1
    if s.setup_index >= len(s.setup_order):
        s.phase = TurnPhase.ROLL
        s.current_player = s.setup_order[0]
        s.turn_number = 1
    else:
        s.phase = TurnPhase.INITIAL_SETTLEMENT
        s.current_player = setup_rules.next_setup_player(s)
    _post_action(s, events)
    return s, events


def _roll_dice(
    s: GameState, action: A.RollDiceAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = action
    d1, d2 = rng.roll_dice()
    total = d1 + d2
    events: list[GameEvent] = [
        E.DiceRolled(
            turn_number=s.turn_number,
            player_id=s.current_player,
            die1=d1,
            die2=d2,
            total=total,
        )
    ]
    if total == 7:
        cards = robber_rules.cards_to_discard_on_seven(s)
        if cards:
            s.phase = TurnPhase.DISCARD
            s.pending = DiscardPending(cards_to_discard=dict(cards))
        else:
            s.phase = TurnPhase.MOVE_ROBBER
            s.pending = RobberMovePending(return_phase=TurnPhase.MAIN)
        return s, events
    dists, sfs = resource_rules.distribute_resources(s, total)
    events.extend(sfs)
    if any(m for m in dists.values() if m):
        block = {p: m for p, m in dists.items() if m}
        resource_rules.apply_distribution(s, block)
        er = _emit_distribution(block, s.turn_number)
        if er is not None:
            events.append(er)
    s.phase = TurnPhase.MAIN
    _post_action(s, events)
    return s, events


def _build_road(
    s: GameState, action: A.BuildRoadAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    if s.phase is TurnPhase.MAIN:
        p.pay(build_rules.ROAD_COST)
    s.occupancy.roads[action.edge_id] = pid
    p.roads_built += 1
    events: list[GameEvent] = [
        E.RoadBuilt(turn_number=s.turn_number, player_id=pid, edge_id=action.edge_id)
    ]
    if s.phase is TurnPhase.BUILD_ROADS and isinstance(s.pending, RoadBuildingPending):
        nxt = s.pending.roads_remaining - 1
        if nxt <= 0:
            s.pending = None
            s.phase = TurnPhase.MAIN
        else:
            s.pending = RoadBuildingPending(roads_remaining=nxt)
            # Auto-end if no further legal road placements remain (e.g., player is
            # boxed in or has placed their 15th piece). Without this the engine
            # would deadlock — legal_actions returns [] in BUILD_ROADS with no roads.
            if not build_rules.legal_build_roads(s):
                s.pending = None
                s.phase = TurnPhase.MAIN
    _post_action(s, events)
    return s, events


def _build_settlement(
    s: GameState, action: A.BuildSettlementAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    p = s.players[action.player_id]
    p.pay(build_rules.SETTLEMENT_COST)
    s.occupancy.buildings[action.vertex_id] = (action.player_id, BuildingType.SETTLEMENT)
    p.settlements_built += 1
    p.victory_points_public += 1
    events: list[GameEvent] = [
        E.SettlementBuilt(
            turn_number=s.turn_number,
            player_id=action.player_id,
            vertex_id=action.vertex_id,
        )
    ]
    _post_action(s, events)
    return s, events


def _build_city(
    s: GameState, action: A.BuildCityAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    p = s.players[action.player_id]
    p.pay(build_rules.CITY_COST)
    s.occupancy.buildings[action.vertex_id] = (action.player_id, BuildingType.CITY)
    p.settlements_built -= 1
    p.cities_built += 1
    # Settlement => City: +1 VP (settlement is 1, city is 2).
    p.victory_points_public += 1
    events: list[GameEvent] = [
        E.CityBuilt(
            turn_number=s.turn_number,
            player_id=action.player_id,
            vertex_id=action.vertex_id,
        )
    ]
    _post_action(s, events)
    return s, events


def _buy_dev(
    s: GameState, action: A.BuyDevCardAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    p.pay(build_rules.DEV_CARD_COST)
    c = s.dev_deck.draw()
    p.dev_cards_in_hand.append((c, s.turn_number))
    events: list[GameEvent] = [
        E.DevCardBought(turn_number=s.turn_number, player_id=pid, card_type=c)
    ]
    _post_action(s, events)
    return s, events


def _maritime(
    s: GameState, action: A.MaritimeTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    give = {action.give: action.give_count}
    receive = {action.receive: 1}
    p.pay(give)
    s.bank.deposit(give)
    s.bank.withdraw(receive)
    p.gain(receive)
    events: list[GameEvent] = [
        E.MaritimeTradeCompleted(
            turn_number=s.turn_number,
            player_id=pid,
            gave=action.give,
            received=action.receive,
        )
    ]
    _post_action(s, events)
    return s, events


def _propose(
    s: GameState, action: A.ProposeDomesticTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    s.pending = DomesticTradePending(
        offer=dict(action.offer),
        request=dict(action.request),
        responses={},
    )
    return s, []


def _respond(
    s: GameState, action: A.RespondDomesticTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pend = s.pending
    assert isinstance(pend, DomesticTradePending)
    s.pending = replace(
        pend, responses={**pend.responses, action.player_id: action.response}
    )
    return s, []


def _confirm_domestic(
    s: GameState, action: A.ConfirmDomesticTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    assert isinstance(s.pending, DomesticTradePending)
    other = action.trade_with
    proposer = s.current_player
    if other == proposer:
        raise ValueError("cannot trade with self")
    if s.pending.responses.get(other) is not DomesticTradeState.ACCEPTED:
        raise ValueError("target did not accept")
    pend = s.pending
    a, b = s.players[proposer], s.players[other]
    if not a.can_afford(pend.offer) or not b.can_afford(pend.request):
        raise ValueError("insufficient resources to confirm")
    a.pay(pend.offer)
    b.gain(pend.offer)
    b.pay(pend.request)
    a.gain(pend.request)
    s.pending = None
    events: list[GameEvent] = [
        E.TradeCompleted(
            turn_number=s.turn_number,
            player1_id=proposer,
            player2_id=other,
            player1_gives=dict(pend.offer),
            player2_gives=dict(pend.request),
        )
    ]
    _post_action(s, events)
    return s, events


def _cancel_domestic(
    s: GameState, action: A.CancelDomesticTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    _ = action
    s.pending = None
    return s, []


def _end_turn(
    s: GameState, action: A.EndTurnAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    _ = action
    ended_turn = s.turn_number
    ended_pid = s.current_player
    s.players[ended_pid].has_played_dev_card_this_turn = False
    pids = s.config.player_ids
    s.current_player = pids[(pids.index(s.current_player) + 1) % len(pids)]
    s.turn_number += 1
    s.phase = TurnPhase.ROLL
    s.turns_since_vp_change += 1
    events: list[GameEvent] = [E.TurnEnded(turn_number=ended_turn, player_id=ended_pid)]
    _post_action(s, events)
    return s, events


def _discard_resources(
    s: GameState, action: A.DiscardResourcesAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    if s.phase is not TurnPhase.DISCARD or not isinstance(s.pending, DiscardPending):
        raise ValueError("not in discard")
    pend = s.pending
    pid = action.player_id
    if pid not in pend.cards_to_discard:
        raise ValueError("no discard expected for this player")
    need = pend.cards_to_discard[pid]
    given = {r: c for r, c in action.resources.items() if c}
    if sum(given.values()) != need:
        raise ValueError("wrong discard size")
    p = s.players[pid]
    if not p.can_afford(given):
        raise ValueError("cannot pay discard")
    p.pay(given)
    s.bank.deposit(given)
    events: list[GameEvent] = [
        E.PlayerDiscarded(
            turn_number=s.turn_number, player_id=pid, resources=dict(given)
        )
    ]
    rest = {k: v for k, v in pend.cards_to_discard.items() if k != pid}
    if not rest:
        s.pending = RobberMovePending(return_phase=TurnPhase.MAIN)
        s.phase = TurnPhase.MOVE_ROBBER
    else:
        s.pending = DiscardPending(cards_to_discard=rest)
    return s, events


def _move_robber(
    s: GameState, action: A.MoveRobberAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    if s.phase is not TurnPhase.MOVE_ROBBER or not isinstance(
        s.pending, RobberMovePending
    ):
        raise ValueError("not moving robber")
    pid = action.player_id
    if pid != s.current_player:
        raise ValueError("only current player may move the robber")
    if action.tile_id == s.occupancy.robber_tile:
        raise ValueError("must move the robber to a different tile")
    return_phase = s.pending.return_phase
    s.occupancy.robber_tile = action.tile_id
    events: list[GameEvent] = [
        E.RobberMoved(
            turn_number=s.turn_number, player_id=pid, tile_id=action.tile_id
        )
    ]
    victims = robber_rules.players_adjacent_to_tile(s, action.tile_id, pid)
    if not victims:
        s.pending = None
        s.phase = return_phase
        _post_action(s, events)
        return s, events
    s.pending = StealPending(valid_targets=victims, return_phase=return_phase)
    s.phase = TurnPhase.STEAL
    return s, events


def _steal_resource(
    s: GameState, action: A.StealResourceAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    if not isinstance(s.pending, StealPending):
        raise ValueError("not stealing")
    by = s.current_player
    if action.player_id != by:
        raise ValueError("steal is by current player only")
    target = action.target_player_id
    pend = s.pending
    if target not in pend.valid_targets:
        raise ValueError("invalid theft target")
    victim = s.players[target]
    bag: list[Resource] = []
    for r, c in victim.resources.items():
        bag.extend([r] * c)
    if not bag:
        s.phase = pend.return_phase
        s.pending = None
        events_empty: list[GameEvent] = []
        _post_action(s, events_empty)
        return s, events_empty
    stolen = rng.choose_stolen_resource(bag)
    victim.pay({stolen: 1})
    s.players[by].gain({stolen: 1})
    events: list[GameEvent] = [
        E.ResourceStolen(
            turn_number=s.turn_number,
            by_player_id=by,
            from_player_id=target,
            resource=stolen,
        )
    ]
    s.pending = None
    s.phase = pend.return_phase
    _post_action(s, events)
    return s, events


def _play_knight(
    s: GameState, action: A.PlayKnightAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    return_phase = s.phase
    if return_phase not in (TurnPhase.ROLL, TurnPhase.MAIN):
        raise ValueError("knight in wrong phase")
    p.remove_dev_card(DevCardType.KNIGHT)
    p.dev_cards_played.append(DevCardType.KNIGHT)
    p.knights_played += 1
    p.has_played_dev_card_this_turn = True
    events: list[GameEvent] = [
        E.DevCardPlayed(
            turn_number=s.turn_number, player_id=pid, card_type=DevCardType.KNIGHT
        )
    ]
    # Set the next phase BEFORE recomputing awards so a winning largest-army
    # award (which would set phase=GAME_OVER inside _post_action) is preserved.
    s.pending = RobberMovePending(return_phase=return_phase)
    s.phase = TurnPhase.MOVE_ROBBER
    _post_action(s, events)
    return s, events


def _play_road_building(
    s: GameState, action: A.PlayRoadBuildingAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    p.remove_dev_card(DevCardType.ROAD_BUILDING)
    p.dev_cards_played.append(DevCardType.ROAD_BUILDING)
    p.has_played_dev_card_this_turn = True
    events: list[GameEvent] = [
        E.DevCardPlayed(
            turn_number=s.turn_number,
            player_id=pid,
            card_type=DevCardType.ROAD_BUILDING,
        )
    ]
    free = min(2, max(0, build_rules.MAX_ROADS - p.roads_built))
    if free == 0:
        s.phase = TurnPhase.MAIN
        s.pending = None
        _post_action(s, events)
        return s, events
    s.pending = RoadBuildingPending(roads_remaining=free)
    s.phase = TurnPhase.BUILD_ROADS
    # If no legal placements at all, immediately revert.
    if not build_rules.legal_build_roads(s):
        s.pending = None
        s.phase = TurnPhase.MAIN
    _post_action(s, events)
    return s, events


def _play_yop(
    s: GameState, action: A.PlayYearOfPlentyAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    p.remove_dev_card(DevCardType.YEAR_OF_PLENTY)
    p.dev_cards_played.append(DevCardType.YEAR_OF_PLENTY)
    p.has_played_dev_card_this_turn = True

    bank_total = sum(s.bank.resources.get(r, 0) for r in tradeable_resources())
    if bank_total == 0:
        raise ValueError("bank has no tradeable resources")

    got: dict[Resource, int] = {}
    if bank_total == 1:
        for r in tradeable_resources():
            if s.bank.resources.get(r, 0) >= 1:
                take = {r: 1}
                s.bank.withdraw(take)
                p.gain(take)
                got = dict(take)
                break
    else:
        r1, r2 = action.resource1, action.resource2
        if r1 is r2:
            if s.bank.resources.get(r1, 0) < 2:
                raise ValueError("not enough of that resource in the bank")
        else:
            if s.bank.resources.get(r1, 0) < 1 or s.bank.resources.get(r2, 0) < 1:
                raise ValueError("bank cannot pay that pair")
        for r in (r1, r2):
            take = {r: 1}
            s.bank.withdraw(take)
            p.gain(take)
            got[r] = got.get(r, 0) + 1

    events: list[GameEvent] = [
        E.DevCardPlayed(
            turn_number=s.turn_number,
            player_id=pid,
            card_type=DevCardType.YEAR_OF_PLENTY,
        )
    ]
    er = _emit_distribution({pid: got}, s.turn_number)
    if er is not None:
        events.append(er)
    _post_action(s, events)
    return s, events


def _play_monopoly(
    s: GameState, action: A.PlayMonopolyAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    p.remove_dev_card(DevCardType.MONOPOLY)
    p.dev_cards_played.append(DevCardType.MONOPOLY)
    p.has_played_dev_card_this_turn = True
    r = action.resource
    for opid in s.config.player_ids:
        if opid == pid:
            continue
        other = s.players[opid]
        n = other.resources.get(r, 0)
        if n:
            other.resources.pop(r, None)
            p.resources[r] = p.resources.get(r, 0) + n
    events: list[GameEvent] = [
        E.DevCardPlayed(
            turn_number=s.turn_number, player_id=pid, card_type=DevCardType.MONOPOLY
        )
    ]
    _post_action(s, events)
    return s, events


# ----------------------------------------------------------------------
# Action -> handler dispatch
# ----------------------------------------------------------------------

_Handler = Callable[[GameState, Action, Randomizer], tuple[GameState, list[GameEvent]]]

_HANDLERS: dict[type, _Handler] = {
    A.PlaceSettlementAction: _place_settlement,  # type: ignore[dict-item]
    A.PlaceRoadAction: _place_road,  # type: ignore[dict-item]
    A.RollDiceAction: _roll_dice,  # type: ignore[dict-item]
    A.DiscardResourcesAction: _discard_resources,  # type: ignore[dict-item]
    A.MoveRobberAction: _move_robber,  # type: ignore[dict-item]
    A.StealResourceAction: _steal_resource,  # type: ignore[dict-item]
    A.BuildRoadAction: _build_road,  # type: ignore[dict-item]
    A.BuildSettlementAction: _build_settlement,  # type: ignore[dict-item]
    A.BuildCityAction: _build_city,  # type: ignore[dict-item]
    A.BuyDevCardAction: _buy_dev,  # type: ignore[dict-item]
    A.MaritimeTradeAction: _maritime,  # type: ignore[dict-item]
    A.ProposeDomesticTradeAction: _propose,  # type: ignore[dict-item]
    A.RespondDomesticTradeAction: _respond,  # type: ignore[dict-item]
    A.ConfirmDomesticTradeAction: _confirm_domestic,  # type: ignore[dict-item]
    A.CancelDomesticTradeAction: _cancel_domestic,  # type: ignore[dict-item]
    A.EndTurnAction: _end_turn,  # type: ignore[dict-item]
    A.PlayKnightAction: _play_knight,  # type: ignore[dict-item]
    A.PlayRoadBuildingAction: _play_road_building,  # type: ignore[dict-item]
    A.PlayYearOfPlentyAction: _play_yop,  # type: ignore[dict-item]
    A.PlayMonopolyAction: _play_monopoly,  # type: ignore[dict-item]
}


def _route(
    s: GameState, action: Action, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    handler = _HANDLERS.get(type(action))
    if handler is None:
        raise NotImplementedError(f"no transition for {type(action).__name__}")
    return handler(s, action, rng)


def apply(rng: Randomizer, state: GameState, action: Action) -> StepResult:
    """Apply ``action`` to a deep-copied ``state`` and return the resulting :class:`StepResult`."""
    old_vp = {
        pid: victory.compute_victory_points(state, pid) for pid in state.config.player_ids
    }
    new_state = copy.deepcopy(state)
    new_state, events = _route(new_state, action, rng)
    new_vp = {
        pid: victory.compute_victory_points(new_state, pid)
        for pid in new_state.config.player_ids
    }
    if any(new_vp[p] > old_vp[p] for p in old_vp):
        new_state.turns_since_vp_change = 0
    # Stalemate runs after VP stall accounting so a same-step VP gain cannot
    # incorrectly trip the VP-stall threshold (see _maybe_stalemate).
    if not (
        new_state.winner is not None
        or new_state.phase
        in (TurnPhase.GAME_OVER, TurnPhase.STALEMATE)
    ):
        _maybe_stalemate(new_state, events)
    return StepResult(
        state=new_state,
        events=events,
        is_terminal=new_state.is_terminal(),
        winner=new_state.winner,
        action=action,
    )


def resolve_no_legal_actions(state: GameState) -> GameState:
    """
    Terminal stalemate for a non-terminal state with no legal actions (e.g. setup edge case).
    Does not emit events; use after ``legal_actions`` is verified empty.
    """
    s = copy.deepcopy(state)
    if s.is_terminal():
        return s
    if rules_legal_actions(s):
        return s
    s.phase = TurnPhase.STALEMATE
    s.end_reason = EndReason.STALEMATE_NO_PROGRESS
    return s
