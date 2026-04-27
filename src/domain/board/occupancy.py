"""
Mutable placement of pieces on a fixed :class:`~domain.board.topology.BoardTopology`.

The robber position lives here (single source of truth; :class:`GameState` exposes
it via a read-only view).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

    def has_road_at(self, edge_id: EdgeID) -> bool:
        return edge_id in self.roads

    def road_owner(self, edge_id: EdgeID) -> Optional[PlayerID]:
        return self.roads.get(edge_id)

    def has_building_at(self, vertex_id: VertexID) -> bool:
        return vertex_id in self.buildings

    def building_at(self, vertex_id: VertexID) -> Optional[tuple[PlayerID, BuildingType]]:
        return self.buildings.get(vertex_id)

    def building_owner(self, vertex_id: VertexID) -> Optional[PlayerID]:
        b = self.buildings.get(vertex_id)
        return None if b is None else b[0]
