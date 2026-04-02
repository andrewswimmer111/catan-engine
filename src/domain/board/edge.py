"""
Immutable edge (road slot) in the board graph.

Vertex ordering within ``vertices`` is canonical — always
``(min_id, max_id)`` — and is enforced in ``__post_init__``.  This means
any code constructing a lookup key for an edge gets the same result
regardless of which endpoint it starts from.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.ids import EdgeID, VertexID


@dataclass(frozen=True)
class Edge:
    """
    A single road slot connecting two adjacent vertices.

    ``vertices``        — the two endpoint vertex IDs, always stored as
                         ``(min_id, max_id)`` (canonical order, enforced in
                         ``__post_init__``).
    ``adjacent_edges``  — all edges that share at least one vertex with this
                         edge (i.e. roads that can be chained).
    """

    edge_id: EdgeID
    vertices: tuple[VertexID, VertexID]
    adjacent_edges: frozenset[EdgeID]

    def __post_init__(self) -> None:
        # Enforce canonical (min, max) ordering on the immutable tuple field.
        # We must use object.__setattr__ because the dataclass is frozen.
        a, b = self.vertices
        if a > b:
            object.__setattr__(self, "vertices", (b, a))

    @property
    def v1(self) -> VertexID:
        return self.vertices[0]

    @property
    def v2(self) -> VertexID:
        return self.vertices[1]