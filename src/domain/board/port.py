"""
Immutable port descriptor.  A Port is a named trading ratio available at two
adjacent coastal vertices.  It is part of the static board topology; port
positions never change during a game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain.enums import PortType
from domain.ids import VertexID


@dataclass(frozen=True)
class Port:
    """
    A port slot on the board coastline.

    ``vertices`` is the pair of adjacent coastal vertices that grant access to
    this port.  The ordering within the tuple is arbitrary but stable once
    assigned by ``layout.py``.

    ``port_type`` is ``None`` when the bare topology has been constructed but
    port types have not yet been assigned.  A separate assignment step
    (analogous to resource/number assignment for tiles) fills this in before
    a game begins.
    """

    port_type: Optional[PortType]
    vertices: tuple[VertexID, VertexID]