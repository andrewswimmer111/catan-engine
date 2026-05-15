"""
Longest road score and ``Longest road`` special-award (minimum length 5).
"""

from __future__ import annotations

from domain.board.topology import BoardTopology
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, VertexID


def _passable_for_road(
    state: GameState, vertex_id: VertexID, player: PlayerID
) -> bool:
    """A road path may pass through a vertex with no building or *your* building."""
    b = state.occupancy.buildings.get(vertex_id)
    if b is None:
        return True
    return b[0] == player


def _my_road_ids(state: GameState, player: PlayerID) -> frozenset[EdgeID]:
    return frozenset(
        eid for eid, owner in state.occupancy.roads.items() if owner == player
    )


def _other_vertex(topology: BoardTopology, edge_id: EdgeID, at: VertexID) -> VertexID:
    """Return the opposite endpoint of ``edge_id`` from ``at``."""
    v1, v2 = topology.edges[edge_id].vertices
    if at == v1:
        return v2
    if at == v2:
        return v1
    raise ValueError(f"vertex {int(at)} is not incident to edge {int(edge_id)}")


def _max_extension_from_vertex(
    state: GameState,
    player: PlayerID,
    topology: BoardTopology,
    my_edges: frozenset[EdgeID],
    cur_vertex: VertexID,
    visited: frozenset[EdgeID],
) -> int:
    """Max additional edges from ``cur_vertex`` without reusing road edges."""
    # Opponent buildings block traversal through this vertex. The path may
    # end here, but cannot extend further.
    if not _passable_for_road(state, cur_vertex, player):
        return 0

    best_extra = 0
    for nxt in topology.vertices[cur_vertex].adjacent_edges:
        if nxt not in my_edges or nxt in visited:
            continue
        nxt_vertex = _other_vertex(topology, nxt, cur_vertex)
        extra = 1 + _max_extension_from_vertex(
            state,
            player,
            topology,
            my_edges,
            nxt_vertex,
            frozenset(visited | {nxt}),
        )
        best_extra = max(best_extra, extra)
    return best_extra


def compute_longest_road(state: GameState, player_id: PlayerID) -> int:
    """
    Longest path length (number of connected road edges) for ``player_id``.

    A vertex with an *opponent* building cannot be *passed through*; your own
    settlements/cities and empty corners can.
    """
    my_edges = _my_road_ids(state, player_id)
    if not my_edges:
        return 0
    topo = state.topology
    best = 0
    for start in my_edges:
        v1, v2 = topo.edges[start].vertices
        # Start from both edge orientations so extension is constrained to
        # the current path endpoint instead of "jumping" via shared adjacency.
        best = max(
            best,
            1
            + _max_extension_from_vertex(
                state, player_id, topo, my_edges, v1, frozenset({start})
            ),
            1
            + _max_extension_from_vertex(
                state, player_id, topo, my_edges, v2, frozenset({start})
            ),
        )
    return best


def update_longest_road_award(
    state: GameState,
) -> tuple[PlayerID | None, bool]:
    """
    Recompute the ``Longest road`` holder. Requires length ≥ 5 to hold the
    award. If several tie at the maximum, the previous holder keeps the bonus
    when they are in the lead group; if the old holder is not in the lead
    group, the title goes to the lowest ``PlayerID`` in that group
    (deterministic tiebreak — see module docstring in ``victory`` if needed).
    """
    pids = state.config.player_ids
    lengths: dict[PlayerID, int] = {p: compute_longest_road(state, p) for p in pids}
    m = max(lengths.values()) if lengths else 0
    if m < 5:
        new_holder: PlayerID | None = None
    else:
        leaders = {p for p, ln in lengths.items() if ln == m}
        cur = state.longest_road_holder
        if cur is not None and cur in leaders:
            new_holder = cur
        else:
            new_holder = min(leaders)
    changed = new_holder != state.longest_road_holder
    state.longest_road_holder = new_holder
    return new_holder, changed
