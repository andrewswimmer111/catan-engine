"""
State transitions: apply actions to a copied :class:`GameState` and collect events.
"""

from __future__ import annotations

import copy

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.randomizer import Randomizer
from domain.engine.step_result import StepResult
from domain.enums import BuildingType, Resource, TurnPhase
from domain.events.all_events import (
    BankShortfall,
    DiceRolled,
    ResourcesDistributed,
    RoadBuilt,
    SettlementBuilt,
)
from domain.events.base import GameEvent
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.rules import resource_rules
from domain.rules import setup_rules


def _emit_resources(dists: dict[PlayerID, dict[Resource, int]], turn: int) -> ResourcesDistributed | None:
    if not any(dists.get(p, {}) for p in dists):
        return None
    clean = {p: m for p, m in dists.items() if m}
    if not clean:
        return None
    return ResourcesDistributed(turn_number=turn, distributions=clean)


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
        SettlementBuilt(
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
            ev = _emit_resources({action.player_id: granted}, s.turn_number)
            if ev is not None:
                events.append(ev)
    return s, events


def _place_road(
    s: GameState, action: A.PlaceRoadAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = rng
    events: list[GameEvent] = [
        RoadBuilt(
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
    return s, events


def _roll_dice(
    s: GameState, action: A.RollDiceAction, rng: Randomizer
) -> tuple[GameState, list[GameEvent]]:
    _ = action
    d1, d2 = rng.roll_dice()
    total = d1 + d2
    events: list[GameEvent] = [
        DiceRolled(
            turn_number=s.turn_number,
            player_id=s.current_player,
            die1=d1,
            die2=d2,
            total=total,
        )
    ]
    dists, sfs = resource_rules.distribute_resources(s, total)
    for sf in sfs:
        events.append(sf)
    if any(m for m in dists.values() if m):
        apply_block = {p: m for p, m in dists.items() if m}
        resource_rules.apply_distribution(s, apply_block)
        ev = _emit_resources(apply_block, s.turn_number)
        if ev is not None:
            events.append(ev)
    s.phase = TurnPhase.MAIN
    return s, events


def _route(s: GameState, action: Action, rng: Randomizer) -> tuple[GameState, list[GameEvent]]:
    if isinstance(action, A.PlaceSettlementAction):
        return _place_settlement(s, action, rng)
    if isinstance(action, A.PlaceRoadAction):
        return _place_road(s, action, rng)
    if isinstance(action, A.RollDiceAction):
        return _roll_dice(s, action, rng)
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
