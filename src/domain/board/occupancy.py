"""
Mutable placement of pieces on a fixed :class:`~domain.board.topology.BoardTopology`.

The robber position lives here (single source of truth; :class:`GameState` exposes
it via a read-only view).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.enums import BuildingType
from domain.ids import EdgeID, PlayerID, TileID, VertexID


@dataclass
class BoardOccupancy:
    """
    Per-edge roads, per-vertex buildings, and the robber tile.
    This object is mutably updated by game transitions; the topology never is.
    """

    roads: dict[EdgeID, PlayerID] = field(default_factory=dict)
    buildings: dict[VertexID, tuple[PlayerID, BuildingType]] = field(default_factory=dict)
    robber_tile: TileID = TileID(0)
