"""
Randomized assignment of production terrain and harbor types, plus
deterministic clockwise-spiral assignment of dice numbers.

:func:`build_standard_board` only fixes the graph: tile shapes, which
vertices and edges exist, and where the nine :class:`Port` vertex-pairs
are. This module shuffles a standard 4E-style scenario onto that skeleton
and writes :attr:`~domain.board.vertex.Vertex.port` for each port corner.

Dice numbers follow the official-rulebook convention: a fixed 18-token
sequence is laid out clockwise along the spiral path returned by
:func:`~domain.board.layout.standard_board_spiral_tile_ids`, skipping the
desert (which is still randomized; the spiral assignment adapts to wherever
the desert lands).

The engine calls :func:`assign_random_scenario` during
:meth:`GameEngine.new_game`.
"""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar

from domain.board.layout import (
    standard_board_outer_ring_tile_ids,
    standard_board_spiral_tile_ids,
)
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

# 18 dice-number tokens, applied in clockwise spiral order to the non-desert
# tiles. This is the canonical sequence used by the official 4-player Catan
# rulebook (the A–R lettered tokens), placed along the spiral that
# :func:`~domain.board.layout.standard_board_spiral_tile_ids` walks.
STANDARD_SPIRAL_NUMBERS: tuple[int, ...] = (
    5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11,
)

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


def _assign_terrain_and_numbers(
    topology: BoardTopology, rng: ScenarioRandom
) -> dict[TileID, Tile]:
    """
    Pick a random desert tile, lay terrain on the other 18 tiles in a random
    permutation, and place the fixed :data:`STANDARD_SPIRAL_NUMBERS` sequence
    along the clockwise spiral.

    The spiral's outer-ring start tile is itself drawn at random (via ``rng``)
    so different scenarios anchor the spiral on different tiles. The desert
    is skipped — never receives a number — and the spiral remains continuous
    regardless of where the desert lands.
    """
    all_tile_ids: list[TileID] = list(topology.tiles.keys())
    desert_tile: TileID = rng.choice(all_tile_ids)
    production_tiles: list[TileID] = [t for t in all_tile_ids if t != desert_tile]

    assert len(_STANDARD_TERRAIN) == 18
    assert len(STANDARD_SPIRAL_NUMBERS) == 18
    assert len(production_tiles) == 18

    terrains = rng.shuffled(list(_STANDARD_TERRAIN))

    spiral_start = rng.choice(standard_board_outer_ring_tile_ids())
    spiral_order = standard_board_spiral_tile_ids(start=spiral_start)
    spiral_non_desert = [t for t in spiral_order if t != desert_tile]
    if len(spiral_non_desert) != len(STANDARD_SPIRAL_NUMBERS):
        raise RuntimeError(
            f"expected {len(STANDARD_SPIRAL_NUMBERS)} non-desert tiles in spiral, "
            f"got {len(spiral_non_desert)}"
        )
    tile_to_number: dict[TileID, int] = dict(
        zip(spiral_non_desert, STANDARD_SPIRAL_NUMBERS, strict=True)
    )

    new_tiles: dict[TileID, Tile] = dict(topology.tiles)
    for tid, terrain in zip(production_tiles, terrains, strict=True):
        new_tiles[tid] = Tile(
            tile_id=tid,
            resource=terrain,
            dice_number=tile_to_number[tid],
        )
    new_tiles[desert_tile] = Tile(
        tile_id=desert_tile,
        resource=Resource.DESERT,
        dice_number=None,
    )
    return new_tiles


def _assign_port_types(
    topology: BoardTopology, rng: ScenarioRandom
) -> tuple[list[Port], dict[VertexID, PortType]]:
    """Shuffle port types onto the fixed nine port slots."""
    port_types = rng.shuffled(list(_STANDARD_PORT_TYPES))
    new_ports: list[Port] = []
    vertex_ports: dict[VertexID, PortType] = {}
    for port, ptype in zip(topology.ports, port_types, strict=True):
        v1, v2 = port.vertices
        vertex_ports[v1] = ptype
        vertex_ports[v2] = ptype
        new_ports.append(Port(port_type=ptype, vertices=port.vertices))
    return new_ports, vertex_ports


def _materialize_vertices(
    topology: BoardTopology, vertex_ports: dict[VertexID, PortType]
) -> dict[VertexID, Vertex]:
    """Stamp each port's :class:`PortType` into the corresponding :class:`Vertex`."""
    new_vertices: dict[VertexID, Vertex] = {}
    for vid, v in topology.vertices.items():
        if vid in vertex_ports:
            new_vertices[vid] = Vertex(
                vertex_id=v.vertex_id,
                adjacent_vertices=v.adjacent_vertices,
                adjacent_edges=v.adjacent_edges,
                adjacent_tiles=v.adjacent_tiles,
                port=vertex_ports[vid],
            )
        else:
            new_vertices[vid] = v
    return new_vertices


def assign_random_scenario(topology: BoardTopology, rng: ScenarioRandom) -> BoardTopology:
    """
    Fill ``resource`` / ``dice_number`` for every :class:`Tile`, assign
    :class:`PortType` to each :class:`Port` and the corresponding
    :class:`Vertex` corners, and return a new :class:`BoardTopology`.

    Terrain types, the desert position, and port types are shuffled by ``rng``.
    Dice numbers are deterministic: the fixed :data:`STANDARD_SPIRAL_NUMBERS`
    sequence is placed clockwise along the spiral path returned by
    :func:`~domain.board.layout.standard_board_spiral_tile_ids`, skipping the
    desert.
    """
    new_tiles = _assign_terrain_and_numbers(topology, rng)
    new_ports, vertex_ports = _assign_port_types(topology, rng)
    new_vertices = _materialize_vertices(topology, vertex_ports)
    return BoardTopology(
        tiles=new_tiles,
        vertices=new_vertices,
        edges=topology.edges,
        ports=tuple(new_ports),
    )
