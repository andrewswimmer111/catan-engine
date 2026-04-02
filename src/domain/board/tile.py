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

    ``resource`` is ``None`` for the desert.
    ``dice_number`` is ``None`` for the desert and for any tile whose
    resources have not yet been assigned (e.g. the bare topology returned
    by ``build_standard_board()``).
    """

    tile_id: TileID
    resource: Optional[Resource]
    dice_number: Optional[int]

    def is_desert(self) -> bool:
        return self.resource is None

    def produces_on_roll(self, roll: int) -> bool:
        """True when this tile distributes resources for the given dice roll."""
        return self.dice_number == roll and not self.is_desert()