"""Tests for controller.diff.changed_paths."""

from __future__ import annotations

import copy

from domain.enums import BuildingType, TurnPhase
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from controller.diff import changed_paths
from tests.fixtures.states import fresh_game_state, post_setup_state


def _state():
    return post_setup_state(seed=0, n_players=3)


def test_no_op_yields_empty_set() -> None:
    s = _state()
    assert changed_paths(s, s) == set()


def test_no_op_deep_copy_yields_empty_set() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    assert changed_paths(s, s2) == set()


def test_road_diff_detected() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    free_edge = next(e for e in s.topology.edges if e not in s.occupancy.roads)
    s2.occupancy.roads[free_edge] = PlayerID(0)

    paths = changed_paths(s, s2)
    # occupancy.roads.<edge_id> should appear
    assert any(p[0] == "occupancy" and p[1] == "roads" for p in paths)


def test_building_diff_detected() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    free_vertex = next(v for v in s.topology.vertices if v not in s.occupancy.buildings)
    s2.occupancy.buildings[free_vertex] = (PlayerID(0), BuildingType.SETTLEMENT)

    paths = changed_paths(s, s2)
    assert any(p[0] == "occupancy" and p[1] == "buildings" for p in paths)


def test_robber_diff_detected() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    current_robber = s.occupancy.robber_tile
    other_tile = next(t for t in s.topology.tiles if t != current_robber)
    s2.occupancy.robber_tile = other_tile

    paths = changed_paths(s, s2)
    assert ("occupancy", "robber_tile") in paths


def test_resource_diff_detected() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    pid = PlayerID(0)
    from domain.enums import Resource
    s2.players[pid].resources[Resource.WOOD] = (s.players[pid].resources.get(Resource.WOOD, 0) + 1)

    paths = changed_paths(s, s2)
    assert any(p[0] == "players" and "resources" in p and "WOOD" in p for p in paths)


def test_only_changed_fields_reported() -> None:
    s = _state()
    s2 = copy.deepcopy(s)
    current_robber = s.occupancy.robber_tile
    other_tile = next(t for t in s.topology.tiles if t != current_robber)
    s2.occupancy.robber_tile = other_tile

    paths = changed_paths(s, s2)
    # topology and players should be untouched
    assert not any(p[0] == "topology" for p in paths)
    assert not any(p[0] == "players" for p in paths)
