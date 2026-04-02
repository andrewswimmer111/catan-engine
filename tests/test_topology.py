"""Pytest tests for BoardTopology and build_standard_board().

Each test covers exactly one property.  Geometry helpers are duplicated
from layout.py (which keeps them private) rather than imported, so the
tests remain independent of internal implementation details.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from domain.board.layout import build_standard_board
from domain.board.topology import BoardTopology
from domain.board.edge import Edge
from domain.board.vertex import Vertex
from domain.board.tile import Tile
from domain.ids import EdgeID, TileID, VertexID


# Geometry helpers (mirrors of layout.py internals — kept private there)

_SIZE = 1.0
_PREC = 6


def _axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
    x = _SIZE * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = _SIZE * (3 / 2 * r)
    return x, y


def _hex_corners(cx: float, cy: float) -> list[tuple[float, float]]:
    corners = []
    for i in range(6):
        angle_rad = math.pi / 6 + math.pi / 3 * i
        corners.append((
            cx + _SIZE * math.cos(angle_rad),
            cy + _SIZE * math.sin(angle_rad),
        ))
    return corners


def _quantize(pt: tuple[float, float]) -> tuple[float, float]:
    return (round(pt[0], _PREC), round(pt[1], _PREC))


def _corner_coord(q: int, r: int, corner_index: int) -> tuple[float, float]:
    """Return the quantised Cartesian coordinate of a specific corner of a tile."""
    cx, cy = _axial_to_cartesian(q, r)
    return _quantize(_hex_corners(cx, cy)[corner_index])


def _find_vertex_at(topo: BoardTopology, coord: tuple[float, float]) -> Optional[VertexID]:
    """
    Return the VertexID whose stored coordinate matches ``coord``, or None.

    This requires inverting the coord_to_vertex_id map, which is internal to
    layout.py.  We reconstruct it here by recomputing all tile corners and
    matching against the topology's vertex count.  This is acceptable in
    tests — we are deliberately exercising the same geometry from the outside.
    """
    from collections import defaultdict
    import math as _math

    _R = 2

    # recompute the full coord->vid map the same way layout.py does.
    axial: list[tuple[int, int]] = []
    for q in range(-_R, _R + 1):
        for r in range(max(-_R, -q - _R), min(_R, -q + _R) + 1):
            axial.append((q, r))

    seen: dict[tuple[float, float], int] = {}
    for i, (q, r) in enumerate(axial):
        cx, cy = _axial_to_cartesian(q, r)
        for j in range(6):
            angle_rad = _math.pi / 6 + _math.pi / 3 * j
            raw = (cx + _SIZE * _math.cos(angle_rad), cy + _SIZE * _math.sin(angle_rad))
            qpt = (_round(raw[0], _PREC), _round(raw[1], _PREC))
            if qpt not in seen:
                seen[qpt] = len(seen)

    vid = seen.get(coord)
    if vid is None:
        return None
    return VertexID(vid)


def _round(x: float, prec: int) -> float:
    return round(x, prec)


# Fixtures
@pytest.fixture(scope="session")
def topo() -> BoardTopology:
    return build_standard_board()


# Structural count tests

def test_tile_count(topo):
    assert len(topo.tiles) == 19


def test_vertex_count(topo):
    assert len(topo.vertices) == 54


def test_edge_count(topo):
    assert len(topo.edges) == 72


def test_port_count(topo):
    assert len(topo.ports) == 9


# ID integrity

def test_tile_ids_are_contiguous(topo):
    assert set(topo.tiles.keys()) == {TileID(i) for i in range(19)}


def test_vertex_ids_are_contiguous(topo):
    assert set(topo.vertices.keys()) == {VertexID(i) for i in range(54)}


def test_edge_ids_are_contiguous(topo):
    assert set(topo.edges.keys()) == {EdgeID(i) for i in range(72)}


# Tile fields

def test_all_tiles_have_no_resource(topo):
    assert all(t.resource is None for t in topo.tiles.values())


def test_all_tiles_have_no_dice_number(topo):
    assert all(t.dice_number is None for t in topo.tiles.values())


# Edge canonical ordering

def test_all_edges_have_canonical_vertex_order(topo):
    for edge in topo.edges.values():
        assert edge.vertices[0] < edge.vertices[1], (
            f"Edge {edge.edge_id} has non-canonical order {edge.vertices}"
        )


# Vertex degree distribution

def test_interior_vertex_count(topo):
    interior = [v for v in topo.vertices.values() if len(v.adjacent_vertices) == 3]
    assert len(interior) == 36


def test_coastal_vertex_count(topo):
    coastal = [v for v in topo.vertices.values() if len(v.adjacent_vertices) == 2]
    assert len(coastal) == 18


# Vertex–edge consistency (bidirectional)

def test_vertex_lists_only_incident_edges(topo):
    for v in topo.vertices.values():
        for eid in v.adjacent_edges:
            edge = topo.edges[eid]
            assert v.vertex_id in edge.vertices, (
                f"Vertex {v.vertex_id} lists edge {eid} but is not an endpoint"
            )


def test_edge_endpoints_list_the_edge(topo):
    for edge in topo.edges.values():
        for vid in edge.vertices:
            assert edge.edge_id in topo.vertices[vid].adjacent_edges, (
                f"Edge {edge.edge_id} lists vertex {vid} but vertex does not reciprocate"
            )


# Edge–edge adjacency

def test_edge_adjacency_is_symmetric(topo):
    for edge in topo.edges.values():
        for adj_eid in edge.adjacent_edges:
            assert edge.edge_id in topo.edges[adj_eid].adjacent_edges, (
                f"Edge {edge.edge_id} lists {adj_eid} as adjacent but not vice-versa"
            )


def test_edge_not_adjacent_to_itself(topo):
    for edge in topo.edges.values():
        assert edge.edge_id not in edge.adjacent_edges


def test_edges_adjacent_iff_share_a_vertex(topo):
    from collections import defaultdict
    vid_to_eids: dict[VertexID, set[EdgeID]] = defaultdict(set)
    for edge in topo.edges.values():
        for vid in edge.vertices:
            vid_to_eids[vid].add(edge.edge_id)

    for edge in topo.edges.values():
        expected: set[EdgeID] = set()
        for vid in edge.vertices:
            expected |= vid_to_eids[vid]
        expected.discard(edge.edge_id)
        assert edge.adjacent_edges == frozenset(expected), (
            f"Edge {edge.edge_id} adjacency mismatch"
        )


# Vertex–tile adjacency distribution

def test_vertices_adjacent_to_three_tiles(topo):
    count = sum(1 for v in topo.vertices.values() if len(v.adjacent_tiles) == 3)
    assert count == 24


def test_vertices_adjacent_to_two_tiles(topo):
    count = sum(1 for v in topo.vertices.values() if len(v.adjacent_tiles) == 2)
    assert count == 12


def test_vertices_adjacent_to_one_tile(topo):
    count = sum(1 for v in topo.vertices.values() if len(v.adjacent_tiles) == 1)
    assert count == 18


def test_all_tile_ids_in_vertex_adjacency_are_valid(topo):
    for v in topo.vertices.values():
        for tid in v.adjacent_tiles:
            assert tid in topo.tiles


# ---------------------------------------------------------------------------
# Port validity
# ---------------------------------------------------------------------------

def test_all_port_types_are_none(topo):
    assert all(p.port_type is None for p in topo.ports)


def test_all_port_vertices_are_in_graph(topo):
    for port in topo.ports:
        assert port.vertices[0] in topo.vertices
        assert port.vertices[1] in topo.vertices


def test_all_port_vertices_are_coastal(topo):
    for port in topo.ports:
        for vid in port.vertices:
            assert len(topo.vertices[vid].adjacent_tiles) < 3, (
                f"Port vertex {vid} is interior"
            )


def test_all_port_vertex_pairs_are_adjacent(topo):
    for port in topo.ports:
        va, vb = port.vertices
        assert vb in topo.vertices[va].adjacent_vertices, (
            f"Port vertices {va} and {vb} are not adjacent"
        )


def test_no_two_ports_share_a_vertex(topo):
    all_port_vids = [vid for port in topo.ports for vid in port.vertices]
    assert len(all_port_vids) == len(set(all_port_vids))


# Port placement — tile (-2, 2), corners at 90° (index 1) and 150° (index 2)

def _port_vertex_ids(topo: BoardTopology) -> set[VertexID]:
    return {vid for port in topo.ports for vid in port.vertices}


def test_tile_minus2_2_corner1_is_a_port_vertex(topo):
    """Corner at 90° (index 1) of tile (-2, 2) must be a port vertex."""
    coord = _corner_coord(-2, 2, 1)
    vid = _find_vertex_at(topo, coord)
    assert vid is not None, "No vertex found at corner 1 of tile (-2, 2)"
    assert vid in _port_vertex_ids(topo), (
        f"Vertex {vid} at corner 1 (90°) of tile (-2, 2) is not a port vertex"
    )


def test_tile_minus2_2_corner2_is_a_port_vertex(topo):
    """Corner at 150° (index 2) of tile (-2, 2) must be a port vertex."""
    coord = _corner_coord(-2, 2, 2)
    vid = _find_vertex_at(topo, coord)
    assert vid is not None, "No vertex found at corner 2 of tile (-2, 2)"
    assert vid in _port_vertex_ids(topo), (
        f"Vertex {vid} at corner 2 (150°) of tile (-2, 2) is not a port vertex"
    )


def test_tile_minus2_2_corners_1_and_2_form_a_port(topo):
    """Corners 1 and 2 of tile (-2, 2) must form a single Port object together."""
    coord1 = _corner_coord(-2, 2, 1)
    coord2 = _corner_coord(-2, 2, 2)
    vid1 = _find_vertex_at(topo, coord1)
    vid2 = _find_vertex_at(topo, coord2)
    assert vid1 is not None, "No vertex found at corner 1 of tile (-2, 2)"
    assert vid2 is not None, "No vertex found at corner 2 of tile (-2, 2)"
    port_pairs = {frozenset(p.vertices) for p in topo.ports}
    assert frozenset({vid1, vid2}) in port_pairs, (
        f"Vertices {vid1} and {vid2} at corners 1 and 2 of tile (-2, 2) "
        "do not form a port together"
    )


# Topology query methods

def test_tiles_adjacent_to_vertex_returns_tile_objects(topo):
    result = topo.tiles_adjacent_to_vertex(VertexID(0))
    assert all(isinstance(t, Tile) for t in result)


def test_tiles_adjacent_to_vertex_count_matches_adjacency_list(topo):
    v = topo.vertices[VertexID(0)]
    result = topo.tiles_adjacent_to_vertex(VertexID(0))
    assert len(result) == len(v.adjacent_tiles)


def test_vertices_adjacent_to_edge_returns_two_vertices(topo):
    result = topo.vertices_adjacent_to_edge(EdgeID(0))
    assert len(result) == 2
    assert all(isinstance(v, Vertex) for v in result)


def test_vertices_adjacent_to_edge_match_edge_endpoints(topo):
    edge = topo.edges[EdgeID(0)]
    v1, v2 = topo.vertices_adjacent_to_edge(EdgeID(0))
    assert {v1.vertex_id, v2.vertex_id} == set(edge.vertices)


def test_edges_adjacent_to_vertex_returns_edge_objects(topo):
    result = topo.edges_adjacent_to_vertex(VertexID(0))
    assert all(isinstance(e, Edge) for e in result)


def test_edges_adjacent_to_vertex_count_matches_adjacency_list(topo):
    v = topo.vertices[VertexID(0)]
    result = topo.edges_adjacent_to_vertex(VertexID(0))
    assert len(result) == len(v.adjacent_edges)


# Distance rule

def test_distance_two_excludes_origin(topo):
    for v in topo.vertices.values():
        assert v.vertex_id not in topo.vertices_within_distance_two(v.vertex_id)


def test_distance_two_includes_all_direct_neighbours(topo):
    for v in topo.vertices.values():
        d2 = topo.vertices_within_distance_two(v.vertex_id)
        assert v.adjacent_vertices.issubset(d2)


def test_distance_two_is_non_empty_for_all_vertices(topo):
    for v in topo.vertices.values():
        assert len(topo.vertices_within_distance_two(v.vertex_id)) > 0