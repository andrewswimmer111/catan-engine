"""
Immutable tile node in the board graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain.enums import Resource
from domain.ids import TileID


@dataclass(frozen=True)
class Tile:
    """
    A single hex tile on the board.

    ``resource`` is ``None`` when terrain has not been assigned yet (engine
    randomization). ``Resource.DESERT`` is the final label for the desert tile.
    ``dice_number`` is ``None`` until the scenario assigns production numbers
    (desert has no number).
    """

    tile_id: TileID
    resource: Optional[Resource]
    dice_number: Optional[int]

    def is_desert(self) -> bool:
        return self.resource is Resource.DESERT

    def produces_on_roll(self, roll: int) -> bool:
        """True when this tile distributes resources for the given dice roll."""
        if self.resource is None or self.is_desert() or self.dice_number is None:
            return False
        return self.dice_number == roll