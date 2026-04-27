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


def _shared_vertex(topology: BoardTopology, e1: EdgeID, e2: EdgeID) -> VertexID | None:
    a1, a2 = topology.edges[e1].vertices
    b1, b2 = topology.edges[e2].vertices
    for x in (a1, a2):
        if x == b1 or x == b2:
            return x
    return None


def _dfs(
    state: GameState,
    player: PlayerID,
    topology: BoardTopology,
    my_edges: frozenset[EdgeID],
    cur: EdgeID,
    visited: frozenset[EdgeID],
) -> int:
    """Max path *length in edges* along a path that starts on ``cur`` and only uses new edges."""
    e = topology.edges[cur]
    best_len = 1
    for nxt in e.adjacent_edges:
        if nxt not in my_edges or nxt in visited:
            continue
        v = _shared_vertex(topology, cur, nxt)
        if v is None:
            continue
        if not _passable_for_road(state, v, player):
            continue
        sub = _dfs(
            state, player, topology, my_edges, nxt, frozenset(visited | {nxt})
        )
        best_len = max(best_len, 1 + sub)
    return best_len


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
        l = _dfs(state, player_id, topo, my_edges, start, frozenset({start}))
        best = max(best, l)
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
