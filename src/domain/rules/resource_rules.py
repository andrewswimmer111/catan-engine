"""
Bank-backed resource movement: dice production, shortfall, and applying batches
to player hands.

Tile terrain, numbers, and harbors are assigned in
:mod:`domain.board.scenario` during :meth:`GameEngine.new_game`. Production and
second-settlement logic here assume that step has run.
"""

from __future__ import annotations

from collections import defaultdict

from domain.enums import BuildingType, Resource
from domain.events.all_events import BankShortfall
from domain.game.state import GameState
from domain.ids import PlayerID


def distribute_resources(
    state: GameState, dice_total: int
) -> tuple[dict[PlayerID, dict[Resource, int]], list[BankShortfall]]:
    """
    For each *numbered* production tile that matches *dice_total* and is not under
    the robber, grant 1/2 of that tile's resource to adjacent settlements/cities.
    If the bank cannot satisfy **total** demand for a given resource, **no
    player** receives that resource this roll.
    """
    turn = state.turn_number
    per_player: dict[PlayerID, dict[Resource, int]] = defaultdict(
        lambda: defaultdict(int)  # type: ignore[assignment]
    )
    raw_demand: dict[Resource, int] = defaultdict(int)  # type: ignore[assignment]

    for tile in state.topology.tiles.values():
        if tile.tile_id == state.occupancy.robber_tile:
            continue
        if not tile.produces_on_roll(dice_total):
            continue
        if tile.resource is None or tile.is_desert():
            continue
        res = tile.resource
        for vid, (owner, btype) in state.occupancy.buildings.items():
            if tile.tile_id not in state.topology.vertices[vid].adjacent_tiles:
                continue
            n = 2 if btype is BuildingType.CITY else 1
            per_player[owner][res] += n
            raw_demand[res] += n

    shortfalls: list[BankShortfall] = []
    for res, need in list(raw_demand.items()):
        if need <= state.bank.resources.get(res, 0):
            continue
        shortfalls.append(
            BankShortfall(turn_number=turn, resource=res, requested=need, given=0)
        )
        for p in per_player:
            if res in per_player[p] and per_player[p][res] > 0:
                per_player[p][res] = 0

    merged: dict[PlayerID, dict[Resource, int]] = {}
    for p, d in per_player.items():
        m = {r: c for r, c in d.items() if c > 0}
        if m:
            merged[p] = m
    return merged, shortfalls


def grant_from_bank(
    state: GameState, player_id: PlayerID, wanted: dict[Resource, int], turn: int
) -> tuple[dict[Resource, int], list[BankShortfall]]:
    """
    Grant each resource type only if the bank can pay the full amount for that
    type; otherwise emit a :class:`BankShortfall` and skip that type.
    (Used for second-settlement starting resources, where partial grants are required.)
    """
    granted: dict[Resource, int] = {}
    shortfalls: list[BankShortfall] = []
    for r, c in wanted.items():
        if c <= 0:
            continue
        if state.bank.resources.get(r, 0) >= c:
            state.bank.withdraw({r: c})
            pl = state.players[player_id]
            pl.resources[r] = pl.resources.get(r, 0) + c
            granted[r] = c
        else:
            shortfalls.append(
                BankShortfall(turn_number=turn, resource=r, requested=c, given=0)
            )
    return granted, shortfalls


def apply_distribution(
    state: GameState, distributions: dict[PlayerID, dict[Resource, int]]
) -> GameState:
    """Move resources from the bank to player hands; mutates and returns ``state``."""
    bank = state.bank
    for player_id, res_map in distributions.items():
        if not res_map:
            continue
        bank.withdraw(res_map)
        pl = state.players[player_id]
        for r, c in res_map.items():
            pl.resources[r] = pl.resources.get(r, 0) + c
    return state
