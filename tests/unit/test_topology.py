"""Unit tests for :class:`BoardTopology` and ``build_standard_board``."""

from __future__ import annotations

import math

import pytest

from domain.board.hex_geometry import SIZE, axial_tiles, hex_distance, RADIUS
from domain.board.layout import build_standard_board, standard_board_coordinates
from domain.ids import EdgeID, TileID, VertexID


@pytest.fixture(scope="module")
def topo():
    return build_standard_board()


def test_each_vertex_has_two_or_three_adjacent_vertices_only(topo):
    for v in topo.vertices.values():
        d = len(v.adjacent_vertices)
        assert d in (2, 3), f"vertex {v.vertex_id} has degree {d}, expected 2 or 3"


def test_interior_hex_vertices_have_exactly_three_adjacent_vertices(topo):
    """Any vertex touching three hexes is an interior node with degree 3."""
    for v in topo.vertices.values():
        if len(v.adjacent_tiles) == 3:
            assert len(v.adjacent_vertices) == 3


def test_there_are_eighteen_two_degree_vertices(topo):
    n = sum(1 for v in topo.vertices.values() if len(v.adjacent_vertices) == 2)
    assert n == 18


def test_there_are_thirty_six_three_degree_vertices(topo):
    n = sum(1 for v in topo.vertices.values() if len(v.adjacent_vertices) == 3)
    assert n == 36


def test_no_vertex_has_fewer_than_two_adjacent_vertices(topo):
    for v in topo.vertices.values():
        assert len(v.adjacent_vertices) >= 2


def test_no_vertex_has_more_than_three_adjacent_vertices(topo):
    for v in topo.vertices.values():
        assert len(v.adjacent_vertices) <= 3


def test_each_edge_connects_two_distinct_endpoints(topo):
    for e in topo.edges.values():
        assert len(e.vertices) == 2
        assert e.v1 != e.v2


def test_vertices_adjacent_to_edge_alternate_returns_two_endpoints(topo):
    e = topo.edges[EdgeID(0)]
    a, b = e.v1, e.v2
    v1, v2 = topo.vertices_adjacent_to_edge(EdgeID(0))
    assert {v1.vertex_id, v2.vertex_id} == {a, b}


def test_vertices_within_distance_two_matches_manual_expansion_for_vertex_zero(topo):
    v0 = VertexID(0)
    o = topo.vertices[v0]
    d1 = set(o.adjacent_vertices)
    d2: set[VertexID] = set()
    for w in d1:
        d2 |= topo.vertices[w].adjacent_vertices
    d2.discard(v0)
    assert topo.vertices_within_distance_two(v0) == d1 | d2


def test_standard_board_has_54_vertices_72_edges_19_tiles(topo):
    assert len(topo.vertices) == 54
    assert len(topo.edges) == 72
    assert len(topo.tiles) == 19


# ---------------------------------------------------------------------------
# standard_board_coordinates tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coords():
    return standard_board_coordinates()


def test_every_vertex_id_has_a_coordinate(topo, coords):
    _, vertex_coords = coords
    assert vertex_coords.keys() == topo.vertices.keys()


def test_tile_center_count_matches_topology(topo, coords):
    tile_centers, _ = coords
    assert len(tile_centers) == len(topo.tiles)


def test_hex_adjacent_tile_centers_are_size_sqrt3_apart(coords):
    tile_centers, _ = coords
    axial_coords = axial_tiles(RADIUS)
    tile_id_to_axial = {TileID(i): coord for i, coord in enumerate(axial_coords)}
    expected_dist = SIZE * math.sqrt(3)

    tile_ids = list(tile_id_to_axial.keys())
    adjacent_pairs_found = 0
    for i, t1 in enumerate(tile_ids):
        q1, r1 = tile_id_to_axial[t1]
        for t2 in tile_ids[i + 1:]:
            q2, r2 = tile_id_to_axial[t2]
            if hex_distance(q1 - q2, r1 - r2) == 1:
                x1, y1 = tile_centers[t1]
                x2, y2 = tile_centers[t2]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                assert abs(dist - expected_dist) < 1e-6, (
                    f"tiles {t1},{t2} distance {dist:.8f} != {expected_dist:.8f}"
                )
                adjacent_pairs_found += 1

    assert adjacent_pairs_found > 0, "no hex-adjacent tile pairs found"
