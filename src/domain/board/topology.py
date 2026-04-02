"""
``BoardTopology`` is the immutable board graph used throughout the engine:
``GameState``, ``legal_actions``, ``transitions``, and ``longest_road`` all
read from it.  It is never mutated after construction.

All query methods are O(1) dict-lookups or small set-operations over the
pre-computed adjacency lists stored in Vertex and Edge objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.board.edge import Edge
from domain.board.port import Port
from domain.board.tile import Tile
from domain.board.vertex import Vertex
from domain.ids import EdgeID, TileID, VertexID


@dataclass(frozen=True)
class BoardTopology:
    """
    The complete, immutable graph representation of the board.

    ``tiles``    — maps TileID   -> Tile
    ``vertices`` — maps VertexID -> Vertex
    ``edges``    — maps EdgeID   -> Edge
    ``ports``    — ordered list of all Port objects (order is stable but
                   carries no semantic weight beyond that)
    """

    tiles: dict[TileID, Tile]
    vertices: dict[VertexID, Vertex]
    edges: dict[EdgeID, Edge]
    ports: tuple[Port, ...]

    # ------------------------------------------------------------------
    # Query helpers — all callers should go through these rather than
    # indexing the dicts directly, so that the access pattern is explicit.
    # ------------------------------------------------------------------

    def tiles_adjacent_to_vertex(self, vertex_id: VertexID) -> list[Tile]:
        """Return all tiles that share the given vertex."""
        vertex = self.vertices[vertex_id]
        return [self.tiles[tid] for tid in vertex.adjacent_tiles]

    def vertices_adjacent_to_edge(self, edge_id: EdgeID) -> tuple[Vertex, Vertex]:
        """Return the two vertices that form the endpoints of the given edge."""
        edge = self.edges[edge_id]
        return (self.vertices[edge.v1], self.vertices[edge.v2])

    def edges_adjacent_to_vertex(self, vertex_id: VertexID) -> list[Edge]:
        """Return all edges that touch the given vertex."""
        vertex = self.vertices[vertex_id]
        return [self.edges[eid] for eid in vertex.adjacent_edges]

    def vertices_within_distance_two(self, vertex_id: VertexID) -> frozenset[VertexID]:
        """
        Return all vertex IDs reachable from ``vertex_id`` in at most two
        edge-hops, **excluding** ``vertex_id`` itself.

        Used by the distance rule: a settlement may not be placed on any
        vertex in this set if another settlement already occupies
        ``vertex_id``, and vice versa.
        """
        origin = self.vertices[vertex_id]
        distance_one: frozenset[VertexID] = origin.adjacent_vertices
        distance_two: set[VertexID] = set()
        for vid in distance_one:
            distance_two.update(self.vertices[vid].adjacent_vertices)
        # Remove the origin itself; distance-one vertices are already inside.
        distance_two -= {vertex_id}
        return distance_one | distance_two