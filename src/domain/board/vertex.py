"""
Immutable vertex node in the board graph.

A Vertex stores *all* adjacency as explicit frozen sets of IDs so that
every topology query is a plain dict-lookup or set-operation — no geometry
is recomputed at query time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain.enums import PortType
from domain.ids import EdgeID, TileID, VertexID


@dataclass(frozen=True)
class Vertex:
    """
    A single intersection point on the board.

    ``adjacent_vertices``  — the 2 or 3 vertices directly connected to this
                             vertex by an edge (degree-2 on the coast,
                             degree-3 in the interior).
    ``adjacent_edges``     — the edges that touch this vertex (one per
                             adjacent vertex).
    ``adjacent_tiles``     — the tiles that share this vertex (1–3).
    ``port``               — the :class:`PortType` for this corner after setup
                             randomization, or ``None`` in the bare board (even on
                             port corners until types are assigned).
    """

    vertex_id: VertexID
    adjacent_vertices: frozenset[VertexID]
    adjacent_edges: frozenset[EdgeID]
    adjacent_tiles: frozenset[TileID]
    port: Optional[PortType]