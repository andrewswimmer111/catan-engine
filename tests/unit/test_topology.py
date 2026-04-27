"""Unit tests for :class:`BoardTopology` and ``build_standard_board``."""

from __future__ import annotations

import pytest

from domain.board.layout import build_standard_board
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
