"""Frozen layout of the discrete action space.

The :class:`ActionEncoder` is the only consumer; constants live here so layout
changes happen in one place and ``_ACTION_LAYOUT_VERSION`` can guard
checkpoint compatibility (loaders bump this when layout changes).

Index map (size 249):

  | range     | width | meaning                                            |
  |-----------|-------|----------------------------------------------------|
  | 0..71     |    72 | road on EdgeID — Place* in setup, Build* otherwise |
  | 72..125   |    54 | settlement on VertexID — Place*/Build* by phase    |
  | 126..179  |    54 | city on VertexID                                   |
  | 180..198  |    19 | move robber to TileID                              |
  | 199..202  |     4 | steal from seat-index in config.player_ids         |
  | 203..222  |    20 | maritime trade (5 give × 4 receive, give≠receive)  |
  | 223       |     1 | roll dice                                          |
  | 224       |     1 | end turn                                           |
  | 225       |     1 | buy dev card                                       |
  | 226       |     1 | play knight                                        |
  | 227       |     1 | play road building                                 |
  | 228..232  |     5 | play monopoly per resource                         |
  | 233..247  |    15 | play year-of-plenty (unordered pair, incl. doubles)|
  | 248       |     1 | discard trigger (heuristic-delegated, see env)     |
"""

from __future__ import annotations

from typing import Final

from domain.enums import Resource

_ACTION_LAYOUT_VERSION: Final[int] = 1

N_EDGES: Final[int] = 72
N_VERTICES: Final[int] = 54
N_TILES: Final[int] = 19
N_STEAL_SLOTS: Final[int] = 4
N_RESOURCES: Final[int] = 5

# Stable resource ordering for index assignment. Do not reorder without
# bumping _ACTION_LAYOUT_VERSION.
RESOURCES: Final[tuple[Resource, ...]] = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)
RESOURCE_TO_INDEX: Final[dict[Resource, int]] = {r: i for i, r in enumerate(RESOURCES)}

ROAD_START: Final[int] = 0
SETTLEMENT_START: Final[int] = ROAD_START + N_EDGES
CITY_START: Final[int] = SETTLEMENT_START + N_VERTICES
ROBBER_MOVE_START: Final[int] = CITY_START + N_VERTICES
STEAL_START: Final[int] = ROBBER_MOVE_START + N_TILES
MARITIME_TRADE_START: Final[int] = STEAL_START + N_STEAL_SLOTS

N_MARITIME_TRADES: Final[int] = N_RESOURCES * (N_RESOURCES - 1)  # 20

ROLL_INDEX: Final[int] = MARITIME_TRADE_START + N_MARITIME_TRADES
END_TURN_INDEX: Final[int] = ROLL_INDEX + 1
BUY_DEV_INDEX: Final[int] = END_TURN_INDEX + 1
KNIGHT_INDEX: Final[int] = BUY_DEV_INDEX + 1
ROAD_BUILDING_INDEX: Final[int] = KNIGHT_INDEX + 1
MONOPOLY_START: Final[int] = ROAD_BUILDING_INDEX + 1
YEAR_OF_PLENTY_START: Final[int] = MONOPOLY_START + N_RESOURCES

# Unordered pairs over RESOURCES, including doubles: 5 + C(5,2) = 15.
N_YEAR_OF_PLENTY: Final[int] = N_RESOURCES * (N_RESOURCES + 1) // 2

DISCARD_INDEX: Final[int] = YEAR_OF_PLENTY_START + N_YEAR_OF_PLENTY
ACTION_SPACE_SIZE: Final[int] = DISCARD_INDEX + 1


def _build_maritime_pairs() -> tuple[tuple[Resource, Resource], ...]:
    """Ordered (give, receive) pairs with give ≠ receive, in 20-entry index order."""
    pairs: list[tuple[Resource, Resource]] = []
    for give in RESOURCES:
        for receive in RESOURCES:
            if give is receive:
                continue
            pairs.append((give, receive))
    return tuple(pairs)


MARITIME_TRADE_PAIRS: Final[tuple[tuple[Resource, Resource], ...]] = _build_maritime_pairs()
MARITIME_TRADE_PAIR_TO_OFFSET: Final[dict[tuple[Resource, Resource], int]] = {
    p: i for i, p in enumerate(MARITIME_TRADE_PAIRS)
}


def _build_yop_pairs() -> tuple[tuple[Resource, Resource], ...]:
    """Unordered resource pairs for year-of-plenty, in canonical (sorted-index) order.

    Includes doubles (e.g. (WOOD, WOOD)). Result ordering:
    (R0,R0), (R0,R1), ..., (R0,R4), (R1,R1), (R1,R2), ..., (R4,R4).
    """
    pairs: list[tuple[Resource, Resource]] = []
    for i in range(N_RESOURCES):
        for j in range(i, N_RESOURCES):
            pairs.append((RESOURCES[i], RESOURCES[j]))
    return tuple(pairs)


YOP_PAIRS: Final[tuple[tuple[Resource, Resource], ...]] = _build_yop_pairs()


def yop_pair_offset(r1: Resource, r2: Resource) -> int:
    """Offset within the year-of-plenty range for an unordered pair (r1, r2)."""
    a = RESOURCE_TO_INDEX[r1]
    b = RESOURCE_TO_INDEX[r2]
    if a > b:
        a, b = b, a
    # Triangular number layout: rows i contribute (N - i) entries.
    # Offset of (i, j) with i<=j is sum_{k<i}(N-k) + (j - i).
    return sum(N_RESOURCES - k for k in range(a)) + (b - a)
