"""
Randomized assignment of production terrain, number tokens, and harbor types.

:func:`build_standard_board` only fixes the graph: tile shapes, which vertices
and edges exist, and where the nine :class:`Port` vertex-pairs are.  This
module shuffles a standard 4E-style scenario onto that skeleton and writes
:attr:`~domain.board.vertex.Vertex.port` for each port corner.  The engine calls
:func:`assign_random_scenario` during :meth:`GameEngine.new_game`.
"""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar

from domain.board.port import Port
from domain.board.tile import Tile
from domain.board.topology import BoardTopology
from domain.board.vertex import Vertex
from domain.enums import PortType, Resource
from domain.ids import TileID, VertexID

T_co = TypeVar("T_co")


class ScenarioRandom(Protocol):
    """Minimal RNG surface for :func:`assign_random_scenario` (implemented by :class:`SeededRandomizer`)."""

    def shuffled(self, items: list[T_co]) -> list[T_co]:
        ...

    def choice(self, items: Sequence[T_co]) -> T_co:
        ...


# 18 production hexes: standard distribution (4th ed–style)
_STANDARD_TERRAIN: tuple[Resource, ...] = (
    (Resource.WOOD,) * 4
    + (Resource.SHEEP,) * 4
    + (Resource.WHEAT,) * 4
    + (Resource.BRICK,) * 3
    + (Resource.ORE,) * 3
)
# 18 pips: one 2, one 12, two of each 3‥6 and 8‥11, two 6s
_STANDARD_NUMBERS: tuple[int, ...] = (2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12)

# Nine harbors: four 3:1, one 2:1 per resource
_STANDARD_PORT_TYPES: tuple[PortType, ...] = (
    (PortType.THREE_TO_ONE,) * 4
    + (
        PortType.WOOD_TWO,
        PortType.BRICK_TWO,
        PortType.SHEEP_TWO,
        PortType.WHEAT_TWO,
        PortType.ORE_TWO,
    )
)


def desert_tile_id(topology: BoardTopology) -> TileID:
    """Return the unique desert tile; requires :func:`assign_random_scenario` to have run."""
    for tid, t in topology.tiles.items():
        if t.is_desert():
            return tid
    raise RuntimeError("no desert on board — was the scenario applied?")


def assign_random_scenario(topology: BoardTopology, rng: ScenarioRandom) -> BoardTopology:
    """
    Fill ``resource`` / ``dice_number`` for every :class:`Tile`, assign
    :class:`PortType` to each :class:`Port` and the corresponding
    :class:`Vertex` corners, and return a new :class:`BoardTopology`.
    """
    tids: list[TileID] = list(topology.tiles.keys())
    desert_tile: TileID = rng.choice(tids)
    prod = [t for t in tids if t != desert_tile]
    assert len(_STANDARD_TERRAIN) == 18
    assert len(_STANDARD_NUMBERS) == 18
    assert len(prod) == 18
    terrains = rng.shuffled(list(_STANDARD_TERRAIN))
    numbers = rng.shuffled(list(_STANDARD_NUMBERS))
    new_tiles = dict(topology.tiles)
    for tid, tr, n in zip(prod, terrains, numbers, strict=True):
        old = new_tiles[tid]
        new_tiles[tid] = Tile(tile_id=old.tile_id, resource=tr, dice_number=n)
    d_old = new_tiles[desert_tile]
    new_tiles[desert_tile] = Tile(
        tile_id=d_old.tile_id, resource=Resource.DESERT, dice_number=None
    )
    port_types = rng.shuffled(list(_STANDARD_PORT_TYPES))
    new_ports: list[Port] = []
    vertex_ports: dict[VertexID, PortType] = {}
    for port, ptype in zip(topology.ports, port_types, strict=True):
        v1, v2 = port.vertices
        vertex_ports[v1] = ptype
        vertex_ports[v2] = ptype
        new_ports.append(Port(port_type=ptype, vertices=port.vertices))
    new_vertices: dict[VertexID, Vertex] = {}
    for vid, v in topology.vertices.items():
        if vid in vertex_ports:
            pt = vertex_ports[vid]
            new_vertices[vid] = Vertex(
                vertex_id=v.vertex_id,
                adjacent_vertices=v.adjacent_vertices,
                adjacent_edges=v.adjacent_edges,
                adjacent_tiles=v.adjacent_tiles,
                port=pt,
            )
        else:
            new_vertices[vid] = v
    return BoardTopology(
        tiles=new_tiles,
        vertices=new_vertices,
        edges=topology.edges,
        ports=tuple(new_ports),
    )
