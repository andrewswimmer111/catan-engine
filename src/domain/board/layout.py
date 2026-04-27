"""
Factory for the standard 4-player Catan board topology.

Design contract
---------------
* ``build_standard_board()`` is the *only* place the standard board is
  defined.  Every other module receives a ``BoardTopology`` and never needs
  to know how it was built.
* Hex geometry is used *internally* during construction to derive adjacency.
  It is invisible to every caller.
* The output is a pure graph with stable integer IDs:
      TileID   0-18   (axial spiral order, outermost ring first)
      VertexID 0-53   (insertion order during geometry sweep)
      EdgeID   0-71   (insertion order during geometry sweep)
* Tile resources and dice numbers are not assigned here: every tile is
  returned with ``resource=None`` and ``dice_number=None`` until the engine
  randomizes a scenario.
* Port *vertex pairs* (where a port exists) are fixed here.  Port *types* are
  assigned during the same randomized setup step as tile resources and numbers,
  not in this factory.

Geometry
--------
Ponty-top hex layout is used internally.  Axial coordinates (q, r).
Each tile's 6 corners are computed from its Cartesian center using the
standard 60°-step formula.  Shared corners between adjacent tiles are
identified by quantizing floating-point coordinates to 6 decimal places
before deduplication — this is safe for the board sizes involved.
"""

from __future__ import annotations

import math
from collections import defaultdict
from domain.board.edge import Edge
from domain.board.port import Port
from domain.board.tile import Tile
from domain.board.topology import BoardTopology
from domain.board.vertex import Vertex
from domain.ids import EdgeID, TileID, VertexID


# Internal geometry constants
_RADIUS = 2          # hex grid radius → 19 tiles
_SIZE   = 1.0        # circumradius of each hex (unit hex)
_PREC   = 6          # decimal places used to quantise corner coordinates


# Internal geometry helpers
def _axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
    """Convert axial hex coordinates to pointy-top Cartesian (x, y)."""
    x = _SIZE * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = _SIZE * (3 / 2 * r)
    return x, y


def _hex_corners(cx: float, cy: float) -> list[tuple[float, float]]:
    """
    Return the 6 corners of a  pointy-top hex centred at (cx, cy), ordered
    at angles 30°, 90°, 150°, 210°, 270°, 330° from the positive x-axis.
    The ordering defines which corner is at index i for all tiles uniformly.
    """
    corners = []
    for i in range(6):
        angle_rad = math.pi / 6 + math.pi / 3 * i
        corners.append((
            cx + _SIZE * math.cos(angle_rad),
            cy + _SIZE * math.sin(angle_rad),
        ))
    return corners


def _quantize(pt: tuple[float, float]) -> tuple[float, float]:
    """Round a 2-D point to _PREC decimal places for deduplication."""
    return (round(pt[0], _PREC), round(pt[1], _PREC))


def _axial_tiles(radius: int) -> list[tuple[int, int]]:
    """
    Return all axial (q, r) coordinates within the given hex radius,
    in a consistent sweep order (q outer, r inner).
    This order is used to assign stable TileIDs.
    """
    coords: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            coords.append((q, r))
    return coords


# Public factory
def build_standard_board() -> BoardTopology:
    """
    Build and return the immutable topology of the standard 4-player Catan
    board.

    Returns a ``BoardTopology`` where:
    * Every tile has ``resource=None`` and ``dice_number=None``.
    * All adjacency lists on Vertex and Edge objects are complete and correct.
    * Port *locations* (nine :class:`Port` records with vertex pairs) are set;
      ``Port.port_type`` and ``Vertex.port`` stay ``None`` until the engine
      assigns a randomized board (together with tile resources and numbers).
    * All IDs are stable integers in the ranges documented in this module.
    """

    # Step 1 — enumerate tiles, assign TileIDs

    axial_coords: list[tuple[int, int]] = _axial_tiles(_RADIUS)
    # axial_coord -> TileID
    coord_to_tile_id: dict[tuple[int, int], TileID] = {
        coord: TileID(i) for i, coord in enumerate(axial_coords)
    }

    # Step 2 — compute all tile corners; deduplicate into Vertex objects

    # quantised_coord -> VertexID  (assigned in first-seen order)
    coord_to_vertex_id: dict[tuple[float, float], VertexID] = {}

    # TileID -> list[VertexID] (length 6, corner indices 0-5)
    tile_corner_ids: dict[TileID, list[VertexID]] = {}

    # VertexID -> set of TileIDs that share this vertex
    vertex_tile_ids: dict[VertexID, set[TileID]] = defaultdict(set)

    for coord in axial_coords:
        tile_id = coord_to_tile_id[coord]
        cx, cy = _axial_to_cartesian(*coord)
        corners = [_quantize(c) for c in _hex_corners(cx, cy)]
        corner_ids: list[VertexID] = []
        for raw_coord in corners:
            if raw_coord not in coord_to_vertex_id:
                coord_to_vertex_id[raw_coord] = VertexID(len(coord_to_vertex_id))
            vid = coord_to_vertex_id[raw_coord]
            corner_ids.append(vid)
            vertex_tile_ids[vid].add(tile_id)
        tile_corner_ids[tile_id] = corner_ids

    n_vertices = len(coord_to_vertex_id)  # should be 54

    # Step 3 — build edges from consecutive corner pairs of every tile

    # frozenset{v1, v2} -> EdgeID  (assigned in first-seen order)
    pair_to_edge_id: dict[frozenset[VertexID], EdgeID] = {}

    # EdgeID -> (VertexID, VertexID)  (raw, pre-canonicalization)
    edge_raw_verts: dict[EdgeID, tuple[VertexID, VertexID]] = {}

    # VertexID -> set[EdgeID]  (edges incident to each vertex)
    vertex_edge_ids: dict[VertexID, set[EdgeID]] = defaultdict(set)

    for coord in axial_coords:
        tile_id = coord_to_tile_id[coord]
        corners = tile_corner_ids[tile_id]
        for i in range(6):
            v1 = corners[i]
            v2 = corners[(i + 1) % 6]
            key: frozenset[VertexID] = frozenset({v1, v2})
            if key not in pair_to_edge_id:
                eid = EdgeID(len(pair_to_edge_id))
                pair_to_edge_id[key] = eid
                edge_raw_verts[eid] = (v1, v2)
                vertex_edge_ids[v1].add(eid)
                vertex_edge_ids[v2].add(eid)

    n_edges = len(pair_to_edge_id)  # should be 72

    # Step 4 — derive vertex-to-vertex adjacency from shared edges

    # VertexID -> set[VertexID]
    vertex_adj_vertices: dict[VertexID, set[VertexID]] = defaultdict(set)
    for eid, (v1, v2) in edge_raw_verts.items():
        vertex_adj_vertices[v1].add(v2)
        vertex_adj_vertices[v2].add(v1)

    # Step 5 — derive edge-to-edge adjacency (edges sharing a vertex)

    # EdgeID -> set[EdgeID]
    edge_adj_edges: dict[EdgeID, set[EdgeID]] = defaultdict(set)
    for vid in range(n_vertices):
        incident = vertex_edge_ids[VertexID(vid)]
        for e1 in incident:
            for e2 in incident:
                if e1 != e2:
                    edge_adj_edges[e1].add(e2)

    # Step 6 — determine the coastal ring for port assignment

    # Coastal vertices are those adjacent to fewer than 3 tiles.
    coastal_vertex_ids: set[VertexID] = {
        VertexID(vid)
        for vid in range(n_vertices)
        if len(vertex_tile_ids[VertexID(vid)]) < 3
    }

    # Walk the coastal ring in order so we can assign port pairs by
    # ring-index position (matching the official board layout).
    vid_to_coord: dict[VertexID, tuple[float, float]] = {
        v: c for c, v in coord_to_vertex_id.items()
    }
    coastal_ring: list[VertexID] = _walk_coastal_ring(
        coastal_vertex_ids, vertex_adj_vertices, vid_to_coord
    )

    # Step 7 — record port vertex pairs (types assigned at game init with tile setup)
    # this assignment is counterclockwise with index 0 at max(y), min(x)
    # where max(y) is up and min(x) is left
    # the port layout is referenced from Colonist.io
    _PORT_SLOT_INDICES: list[tuple[int, int]] = [
        # Vertex-skip gap pattern [1,1,2]
        (0, 1),
        (3, 4),
        (7, 8),
        (10, 11),
        (13, 14),
        (17, 18),
        (20, 21),
        (23, 24),
        (27, 28)
    ]

    ports: list[Port] = []

    for ring_a, ring_b in _PORT_SLOT_INDICES:
        va = coastal_ring[ring_a]
        vb = coastal_ring[ring_b]
        ports.append(Port(port_type=None, vertices=(va, vb)))

    # Step 8 — assemble immutable domain objects

    tiles: dict[TileID, Tile] = {
        TileID(i): Tile(
            tile_id=TileID(i),
            resource=None,
            dice_number=None,
        )
        for i in range(len(axial_coords))
    }

    vertices: dict[VertexID, Vertex] = {
        VertexID(vid): Vertex(
            vertex_id=VertexID(vid),
            adjacent_vertices=frozenset(vertex_adj_vertices[VertexID(vid)]),
            adjacent_edges=frozenset(vertex_edge_ids[VertexID(vid)]),
            adjacent_tiles=frozenset(vertex_tile_ids[VertexID(vid)]),
            port=None,
        )
        for vid in range(n_vertices)
    }

    edges: dict[EdgeID, Edge] = {
        EdgeID(eid): Edge(
            edge_id=EdgeID(eid),
            vertices=edge_raw_verts[EdgeID(eid)],   # __post_init__ canonicalises
            adjacent_edges=frozenset(edge_adj_edges[EdgeID(eid)]),
        )
        for eid in range(n_edges)
    }

    return BoardTopology(
        tiles=tiles,
        vertices=vertices,
        edges=edges,
        ports=tuple(ports),
    )


# Public helpers — geometric spiral order over the standard board

def _standard_board_tile_coord() -> dict[TileID, tuple[int, int]]:
    """Map every standard-board ``TileID`` to its axial ``(q, r)`` coordinate."""
    return {TileID(i): coord for i, coord in enumerate(_axial_tiles(_RADIUS))}


def _hex_distance(q: int, r: int) -> int:
    """Hex grid distance from the origin to ``(q, r)`` in axial coordinates."""
    return (abs(q) + abs(r) + abs(q + r)) // 2


def _classify_rings_by_distance() -> dict[int, list[TileID]]:
    """Group standard-board tile IDs by hex distance from the centre (0, 1, 2)."""
    rings: dict[int, list[TileID]] = {0: [], 1: [], 2: []}
    for tid, (q, r) in _standard_board_tile_coord().items():
        rings[_hex_distance(q, r)].append(tid)
    return rings


def standard_board_outer_ring_tile_ids() -> list[TileID]:
    """The 12 outer-ring tile IDs of the standard board, in TileID order."""
    return sorted(_classify_rings_by_distance()[2])


def standard_board_spiral_tile_ids(
    *, start: TileID | None = None
) -> list[TileID]:
    """
    Return the 19 tile IDs of the standard board in clockwise spiral order,
    outermost ring first.

    The walk is geometrically continuous: every consecutive pair of tile IDs
    in the returned list shares a hex edge. It visits the outer ring (12
    tiles) clockwise, steps inward to the middle ring (6 tiles), continues
    clockwise, then ends on the centre tile.

    ``start`` selects the outer-ring tile to begin from. When omitted, the
    walk anchors at the outer-ring tile with the smallest polar angle
    relative to the board centre — a deterministic but arbitrary choice.
    Callers that need a randomized start should pick from
    :func:`standard_board_outer_ring_tile_ids` and pass the result here.
    """
    tile_coord = _standard_board_tile_coord()
    rings = _classify_rings_by_distance()

    if start is not None and start not in rings[2]:
        raise ValueError(
            f"spiral start tile {start} is not on the outer ring; "
            f"valid starts: {sorted(rings[2])}"
        )

    def cartesian_angle(tid: TileID) -> float:
        x, y = _axial_to_cartesian(*tile_coord[tid])
        return math.atan2(y, x)

    def hex_adjacent(t1: TileID, t2: TileID) -> bool:
        q1, r1 = tile_coord[t1]
        q2, r2 = tile_coord[t2]
        return _hex_distance(q1 - q2, r1 - r2) == 1

    def cw_delta(from_tid: TileID, to_tid: TileID) -> float:
        """Angular distance from ``from_tid`` to ``to_tid`` in the clockwise direction.

        Hex axial geometry uses screen coordinates (y points down), so an
        increasing :func:`math.atan2` value corresponds to clockwise rotation
        about the board centre.
        """
        return (cartesian_angle(to_tid) - cartesian_angle(from_tid)) % (2 * math.pi)

    def walk_ring_clockwise(ring: set[TileID], begin: TileID) -> list[TileID]:
        out: list[TileID] = [begin]
        remaining = ring - {begin}
        cur = begin
        while remaining:
            same_ring = [t for t in remaining if hex_adjacent(cur, t)]
            if not same_ring:
                break
            cur = min(same_ring, key=lambda t: cw_delta(cur, t))
            out.append(cur)
            remaining.discard(cur)
        return out

    outer_ring = set(rings[2])
    middle_ring = set(rings[1])
    centre_ring = set(rings[0])

    spiral: list[TileID] = []
    if outer_ring:
        outer_start = start if start is not None else min(outer_ring, key=cartesian_angle)
        spiral.extend(walk_ring_clockwise(outer_ring, outer_start))

    if middle_ring:
        last = spiral[-1] if spiral else min(middle_ring, key=cartesian_angle)
        adj_in_ring = [t for t in middle_ring if hex_adjacent(last, t)]
        if adj_in_ring:
            inner_start = min(adj_in_ring, key=lambda t: cw_delta(last, t))
        else:
            inner_start = min(middle_ring, key=cartesian_angle)
        spiral.extend(walk_ring_clockwise(middle_ring, inner_start))

    spiral.extend(sorted(centre_ring))

    if len(spiral) != len(tile_coord):
        raise RuntimeError(
            f"spiral order has {len(spiral)} tiles; expected {len(tile_coord)}. "
            "Geometry is corrupt."
        )
    return spiral


# Internal helper — coastal ring walker

def _walk_coastal_ring(
    coastal_ids: set[VertexID],
    adj: dict[VertexID, set[VertexID]],
    vid_to_coord: dict[VertexID, tuple[float, float]],
) -> list[VertexID]:
  """
  Return the coastal vertices in counterclockwise order, anchored geometrically.

  The start vertex is always the topmost-then-leftmost coastal vertex
  (highest y, most-negative x as tiebreak).  From there the walk proceeds
  counterclockwise — i.e. the first step goes to the coastal neighbor with the
  smaller x coordinate (further left), which on a y-up flat-top hex grid
  is the counterclockwise direction from the top.

  This gives ring index 0 a stable geometric meaning independent of vertex
  ID assignment order, so port slot indices in ``_PORT_SLOT_INDICES`` are
  reliable across any refactor of the geometry sweep.

  Raises ``RuntimeError`` if the ring cannot be completed (which would
  indicate a geometry bug).
  """
  # Geometric anchor: highest y, leftmost x as tiebreak.
  start: VertexID = max(coastal_ids,
                        key=lambda v: (vid_to_coord[v][1], -vid_to_coord[v][0]))

  # First step: go clockwise = towards smaller x from the top vertex.
  first_candidates = adj[start] & coastal_ids
  first: VertexID = min(first_candidates, key=lambda v: vid_to_coord[v][0])

  ring: list[VertexID] = [start, first]
  prev: VertexID = start
  cur: VertexID = first

  while True:
    candidates = (adj[cur] & coastal_ids) - {prev}
    if not candidates:
      raise RuntimeError(
          f"Coastal ring walk stuck at vertex {cur} "
          f"(prev={prev}); topology may be corrupt."
      )
    if len(candidates) > 1:
      raise RuntimeError(
          f"Coastal ring walk has ambiguous next step at vertex {cur}: "
          f"candidates={candidates}; topology may be corrupt."
      )
    nxt: VertexID = next(iter(candidates))
    if nxt == start:
      break
    ring.append(nxt)
    prev, cur = cur, nxt

  if len(ring) != 30:
    raise RuntimeError(
        f"Coastal ring has {len(ring)} vertices; expected 30. "
        "Topology is corrupt."
    )
  return ring