"""
State transitions: apply actions to a copied :class:`GameState` and collect events.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.randomizer import Randomizer
from domain.engine.step_result import StepResult
from domain.enums import (
    BuildingType,
    DevCardType,
    DomesticTradeState,
    Resource,
    TurnPhase,
    tradeable_resources,
)
import domain.events.all_events as E
from domain.events.base import GameEvent
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.rules import build_rules, resource_rules, robber_rules, setup_rules, victory
from domain.rules import longest_road as lr_mod
from domain.turn.pending import (
    DiscardPending,
    DomesticTradePending,
    RobberMovePending,
    RoadBuildingPending,
    StealPending,
)


def _pay(pl, cost: dict[Resource, int]) -> None:
    for r, c in cost.items():
        pl.resources[r] = pl.resources.get(r, 0) - c
        if pl.resources.get(r, 0) <= 0:
            pl.resources.pop(r, None)


def _take(pl, g: dict[Resource, int]) -> None:
    for r, c in g.items():
        pl.resources[r] = pl.resources.get(r, 0) + c


def _remove_dev(p, card: DevCardType) -> None:
    for i, (c, _t) in enumerate(p.dev_cards_in_hand):
        if c is card:
            del p.dev_cards_in_hand[i]
            return
    raise ValueError("dev card not in hand")


def _post_action(s: GameState, ev: list[GameEvent]) -> None:
    _, lrc = lr_mod.update_longest_road_award(s)
    if lrc and s.longest_road_holder is not None:
        hold = s.longest_road_holder
        ev.append(
            E.LongestRoadAwarded(
                turn_number=s.turn_number,
                player_id=hold,
                length=lr_mod.compute_longest_road(s, hold),
            )
        )
    _la, lac = victory.update_largest_army_award(s)
    if lac and s.largest_army_holder is not None:
        hold2 = s.largest_army_holder
        ev.append(
            E.LargestArmyAwarded(
                turn_number=s.turn_number,
                player_id=hold2,
                count=s.players[hold2].knights_played,
            )
        )
    w = victory.check_winner(s)
    if w is not None:
        s.winner = w
        s.phase = TurnPhase.GAME_OVER
        ev.append(
            E.GameWon(
                turn_number=s.turn_number,
                player_id=w,
                victory_points=victory.compute_victory_points(s, w),
            )
        )


def _emit_res(dists: dict[PlayerID, dict[Resource, int]], turn: int) -> E.ResourcesDistributed | None:
    if not any(dists.get(p, {}) for p in dists):
        return None
    clean = {p: m for p, m in dists.items() if m}
    if not clean:
        return None
    return E.ResourcesDistributed(turn_number=turn, distributions=clean)


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
            turn_number=s.turn_number, player_id=action.player_id, vertex_id=action.vertex_id
        )
    )
    s.last_settlement_vertex = action.vertex_id
    s.phase = TurnPhase.INITIAL_ROAD
    if setup_rules.is_second_settlement_turn(s):
        wanted = setup_rules.second_settlement_resources(s, action.vertex_id)
        granted, sfs = resource_rules.grant_from_bank(
            s, action.player_id, wanted, s.turn_number
        )
        for sf in sfs:
            events.append(sf)
        if granted:
            er = _emit_res({action.player_id: granted}, s.turn_number)
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
            turn_number=s.turn_number, player_id=action.player_id, edge_id=action.edge_id
        )
    ]
    s.occupancy.roads[action.edge_id] = action.player_id
    s.players[action.player_id].roads_built += 1
    s.last_settlement_vertex = None
    s.setup_index += 1
    n = len(s.setup_order)
    if s.setup_index >= n:
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
    for sf in sfs:
        events.append(sf)
    if any(m for m in dists.values() if m):
        apply_block = {p: m for p, m in dists.items() if m}
        resource_rules.apply_distribution(s, apply_block)
        er = _emit_res(apply_block, s.turn_number)
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
        _pay(p, build_rules.ROAD_COST)
    s.occupancy.roads[action.edge_id] = pid
    p.roads_built += 1
    events: list[GameEvent] = [
        E.RoadBuilt(turn_number=s.turn_number, player_id=pid, edge_id=action.edge_id)
    ]
    if s.phase is TurnPhase.BUILD_ROADS and isinstance(
        s.pending, RoadBuildingPending
    ):
        nxt = s.pending.roads_remaining - 1
        if nxt <= 0:
            s.pending = None
            s.phase = TurnPhase.MAIN
        else:
            s.pending = RoadBuildingPending(roads_remaining=nxt)
    _post_action(s, events)
    return s, events


def _build_settlement(
    s: GameState, action: A.BuildSettlementAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    _pay(s.players[action.player_id], build_rules.SETTLEMENT_COST)
    s.occupancy.buildings[action.vertex_id] = (action.player_id, BuildingType.SETTLEMENT)
    p = s.players[action.player_id]
    p.settlements_built += 1
    ev: list[GameEvent] = [
        E.SettlementBuilt(
            turn_number=s.turn_number, player_id=action.player_id, vertex_id=action.vertex_id
        )
    ]
    _post_action(s, ev)
    return s, ev


def _build_city(
    s: GameState, action: A.BuildCityAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    _pay(s.players[pid], build_rules.CITY_COST)
    s.occupancy.buildings[action.vertex_id] = (pid, BuildingType.CITY)
    p = s.players[pid]
    p.settlements_built -= 1
    p.cities_built += 1
    ev: list[GameEvent] = [
        E.CityBuilt(turn_number=s.turn_number, player_id=pid, vertex_id=action.vertex_id)
    ]
    _post_action(s, ev)
    return s, ev


def _buy_dev(
    s: GameState, action: A.BuyDevCardAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    _pay(s.players[pid], build_rules.DEV_CARD_COST)
    c = s.dev_deck.draw()
    s.players[pid].dev_cards_in_hand.append((c, s.turn_number))
    ev: list[GameEvent] = [
        E.DevCardBought(turn_number=s.turn_number, player_id=pid, card_type=c)
    ]
    _post_action(s, ev)
    return s, ev


def _maritime(
    s: GameState, action: A.MaritimeTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    pl = s.players[pid]
    b = s.bank
    g = {action.give: action.give_count}
    r = {action.receive: 1}
    _pay(pl, g)
    b.deposit(g)
    b.withdraw(r)
    _take(pl, r)
    ev: list[GameEvent] = [
        E.MaritimeTradeCompleted(
            turn_number=s.turn_number, player_id=pid, gave=action.give, received=action.receive
        )
    ]
    _post_action(s, ev)
    return s, ev


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
    oth = action.trade_with
    proposer = s.current_player
    if oth == proposer:
        raise ValueError("cannot trade with self")
    if s.pending.responses.get(oth) is not DomesticTradeState.ACCEPTED:
        raise ValueError("target did not accept")
    pend = s.pending
    a, b = s.players[proposer], s.players[oth]
    if not a.can_afford(pend.offer) or not b.can_afford(pend.request):
        raise ValueError("insufficient resources to confirm")
    _pay(a, pend.offer)
    _take(b, pend.offer)
    _pay(b, pend.request)
    _take(a, pend.request)
    s.pending = None
    ev: list[GameEvent] = [
        E.TradeCompleted(
            turn_number=s.turn_number,
            player1_id=proposer,
            player2_id=oth,
            player1_gives=dict(pend.offer),
            player2_gives=dict(pend.request),
        )
    ]
    _post_action(s, ev)
    return s, ev


def _cancel_domestic(
    s: GameState, action: A.CancelDomesticTradeAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    s.pending = None
    return s, []


def _end_turn(
    s: GameState, action: A.EndTurnAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    ended = s.turn_number
    ended_p = s.current_player
    s.players[ended_p].has_played_dev_card_this_turn = False
    pids = s.config.player_ids
    i = pids.index(s.current_player)
    s.current_player = pids[(i + 1) % len(pids)]
    s.turn_number += 1
    s.phase = TurnPhase.ROLL
    ev: list[GameEvent] = [E.TurnEnded(turn_number=ended, player_id=ended_p)]
    return s, ev


def _discard_resources(
    s: GameState, action: A.DiscardResourcesAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    if s.phase is not TurnPhase.DISCARD or not isinstance(s.pending, DiscardPending):
        raise ValueError("not in discard")
    p = action.player_id
    pend = s.pending
    if p not in pend.cards_to_discard:
        raise ValueError("no discard expected for this player")
    need = pend.cards_to_discard[p]
    g = {r: c for r, c in action.resources.items() if c}
    if sum(g.values()) != need:
        raise ValueError("wrong discard size")
    pl = s.players[p]
    if not pl.can_afford(g):
        raise ValueError("cannot pay discard")
    _pay(pl, g)
    s.bank.deposit(g)
    ev: list[GameEvent] = [
        E.PlayerDiscarded(turn_number=s.turn_number, player_id=p, resources=dict(g))
    ]
    rest = {k: v for k, v in pend.cards_to_discard.items() if k != p}
    if not rest:
        s.pending = RobberMovePending(return_phase=TurnPhase.MAIN)
        s.phase = TurnPhase.MOVE_ROBBER
    else:
        s.pending = DiscardPending(cards_to_discard=rest)
    return s, ev


def _move_robber(
    s: GameState, action: A.MoveRobberAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    if s.phase is not TurnPhase.MOVE_ROBBER or not isinstance(s.pending, RobberMovePending):
        raise ValueError("not moving robber")
    pid = action.player_id
    if pid != s.current_player:
        raise ValueError("only current player may move the robber")
    dest = action.tile_id
    if dest == s.occupancy.robber_tile:
        raise ValueError("must move the robber to a different tile")
    ret = s.pending.return_phase
    s.occupancy.robber_tile = dest
    ev: list[GameEvent] = [
        E.RobberMoved(turn_number=s.turn_number, player_id=pid, tile_id=dest)
    ]
    victims = robber_rules.players_adjacent_to_tile(s, dest, pid)
    if not victims:
        s.pending = None
        s.phase = ret
        _post_action(s, ev)
        return s, ev
    s.pending = StealPending(valid_targets=victims, return_phase=ret)
    s.phase = TurnPhase.STEAL
    return s, ev


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
        ev_empty: list[GameEvent] = []
        _post_action(s, ev_empty)
        return s, ev_empty
    st = rng.choose_stolen_resource(bag)
    _pay(victim, {st: 1})
    _take(s.players[by], {st: 1})
    ev: list[GameEvent] = [
        E.ResourceStolen(
            turn_number=s.turn_number, by_player_id=by, from_player_id=target, resource=st
        )
    ]
    s.pending = None
    s.phase = pend.return_phase
    _post_action(s, ev)
    return s, ev


def _play_knight(
    s: GameState, action: A.PlayKnightAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    return_phase = s.phase
    if return_phase is not TurnPhase.ROLL and return_phase is not TurnPhase.MAIN:
        raise ValueError("knight in wrong phase")
    _remove_dev(p, DevCardType.KNIGHT)
    p.knights_played += 1
    p.has_played_dev_card_this_turn = True
    ev: list[GameEvent] = [
        E.DevCardPlayed(turn_number=s.turn_number, player_id=pid, card_type=DevCardType.KNIGHT)
    ]
    _post_action(s, ev)
    s.pending = RobberMovePending(return_phase=return_phase)
    s.phase = TurnPhase.MOVE_ROBBER
    return s, ev


def _play_road_building(
    s: GameState, action: A.PlayRoadBuildingAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    _remove_dev(p, DevCardType.ROAD_BUILDING)
    p.has_played_dev_card_this_turn = True
    left = 15 - p.roads_built
    n = min(2, max(0, left))
    ev1: list[GameEvent] = [
        E.DevCardPlayed(turn_number=s.turn_number, player_id=pid, card_type=DevCardType.ROAD_BUILDING)
    ]
    if n == 0:
        s.phase = TurnPhase.MAIN
        s.pending = None
        _post_action(s, ev1)
        return s, ev1
    s.pending = RoadBuildingPending(roads_remaining=n)
    s.phase = TurnPhase.BUILD_ROADS
    _post_action(s, ev1)
    return s, ev1


def _play_yop(
    s: GameState, action: A.PlayYearOfPlentyAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    b = s.bank
    r1, r2 = action.resource1, action.resource2
    _remove_dev(p, DevCardType.YEAR_OF_PLENTY)
    p.has_played_dev_card_this_turn = True
    tbank = sum(b.resources.get(x, 0) for x in tradeable_resources())
    if tbank == 0:
        raise ValueError("bank has no tradeable resources")
    got: dict[Resource, int] = {}
    if tbank == 1:
        for r in tradeable_resources():
            if b.resources.get(r, 0) >= 1:
                take = {r: 1}
                b.withdraw(take)
                _take(p, take)
                got = dict(take)
                break
    else:
        if r1 is r2:
            if b.resources.get(r1, 0) < 2:
                raise ValueError("not enough of that resource in the bank")
        else:
            if b.resources.get(r1, 0) < 1 or b.resources.get(r2, 0) < 1:
                raise ValueError("bank cannot pay that pair")
        for r in (r1, r2):
            w = {r: 1}
            b.withdraw(w)
            _take(p, w)
            got[r] = got.get(r, 0) + 1
    ev2 = _emit_res({pid: got}, s.turn_number)
    out: list[GameEvent] = [
        E.DevCardPlayed(
            turn_number=s.turn_number, player_id=pid, card_type=DevCardType.YEAR_OF_PLENTY
        )
    ]
    if ev2 is not None:
        out.append(ev2)
    _post_action(s, out)
    return s, out


def _play_monopoly(
    s: GameState, action: A.PlayMonopolyAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    pid = action.player_id
    p = s.players[pid]
    r = action.resource
    _remove_dev(p, DevCardType.MONOPOLY)
    p.has_played_dev_card_this_turn = True
    for opid in s.config.player_ids:
        if opid == pid:
            continue
        o = s.players[opid]
        n = o.resources.get(r, 0)
        if n:
            o.resources.pop(r, None)
            p.resources[r] = p.resources.get(r, 0) + n
    ev: list[GameEvent] = [
        E.DevCardPlayed(turn_number=s.turn_number, player_id=pid, card_type=DevCardType.MONOPOLY)
    ]
    _post_action(s, ev)
    return s, ev


def _route(s: GameState, action: Action, rng: Randomizer) -> tuple[GameState, list[GameEvent]]:
    if isinstance(action, A.PlaceSettlementAction):
        return _place_settlement(s, action, rng)
    if isinstance(action, A.PlaceRoadAction):
        return _place_road(s, action, rng)
    if isinstance(action, A.RollDiceAction):
        return _roll_dice(s, action, rng)
    if isinstance(action, A.DiscardResourcesAction):
        return _discard_resources(s, action, rng)
    if isinstance(action, A.MoveRobberAction):
        return _move_robber(s, action, rng)
    if isinstance(action, A.StealResourceAction):
        return _steal_resource(s, action, rng)
    if isinstance(action, A.BuildRoadAction):
        return _build_road(s, action, rng)
    if isinstance(action, A.BuildSettlementAction):
        return _build_settlement(s, action, rng)
    if isinstance(action, A.BuildCityAction):
        return _build_city(s, action, rng)
    if isinstance(action, A.BuyDevCardAction):
        return _buy_dev(s, action, rng)
    if isinstance(action, A.MaritimeTradeAction):
        return _maritime(s, action, rng)
    if isinstance(action, A.ProposeDomesticTradeAction):
        return _propose(s, action, rng)
    if isinstance(action, A.RespondDomesticTradeAction):
        return _respond(s, action, rng)
    if isinstance(action, A.ConfirmDomesticTradeAction):
        return _confirm_domestic(s, action, rng)
    if isinstance(action, A.CancelDomesticTradeAction):
        return _cancel_domestic(s, action, rng)
    if isinstance(action, A.EndTurnAction):
        return _end_turn(s, action, rng)
    if isinstance(action, A.PlayKnightAction):
        return _play_knight(s, action, rng)
    if isinstance(action, A.PlayRoadBuildingAction):
        return _play_road_building(s, action, rng)
    if isinstance(action, A.PlayYearOfPlentyAction):
        return _play_yop(s, action, rng)
    if isinstance(action, A.PlayMonopolyAction):
        return _play_monopoly(s, action, rng)
    raise NotImplementedError(f"no transition for {type(action).__name__}")


def apply(rng: Randomizer, state: GameState, action: Action) -> StepResult:
    new_state = copy.deepcopy(state)
    new_state, events = _route(new_state, action, rng)
    return StepResult(
        state=new_state,
        events=events,
        is_terminal=new_state.is_terminal(),
        winner=new_state.winner,
        action=action,
    )
