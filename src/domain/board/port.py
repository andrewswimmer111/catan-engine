"""
Immutable port descriptor.  A Port is a trading ratio at two adjacent coastal
vertices.  ``layout`` only fixes *where* ports sit; ``port_type`` is filled when
the engine randomizes the scenario (same phase as tile resources and numbers).
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

    ``port_type`` is ``None`` in the bare topology from :func:`build_standard_board`
    and is set when ports are shuffled during game setup.
    """

    port_type: Optional[PortType]
    vertices: tuple[VertexID, VertexID]