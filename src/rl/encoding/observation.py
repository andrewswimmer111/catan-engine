"""Flat float32 observation encoding of a :class:`PlayerView`.

Layout
------

The observation is the concatenation of fixed-width blocks in the order
below. Bumping any of the constants here, the widths in :mod:`.features`,
or the normalisation constants requires incrementing
:data:`OBS_LAYOUT_VERSION` and regenerating the snapshot fixture committed
under ``tests/rl/fixtures/``.

Offsets (cumulative, ``OBS_SHAPE = (1324,)``)::

  [0       .. 342)   tiles            — 19 × (resource[6] + chit[11] + robber[1])
  [342     .. 828)   vertices         — 54 × (empty[1] + settlement[4] + city[4])
  [828     .. 1188)  edges            — 72 × (empty[1] + owner[4])
  [1188    .. 1242)  ports            — 9 × (type one-hot[6])
  [1242    .. 1257)  self hand        — 5 resources + 5 dev-in-hand + 5 dev-played
  [1257    .. 1284)  opponents        — 3 × 9 features each (3 opponent slots)
  [1284    .. 1304)  awards/counters  — 20 features (turn, bank, awards, self stats)
  [1304    .. 1324)  phase + pending  — 12 phases + 8 pending one-hot (with "none")

Per-block details live in :mod:`rl.encoding.features`.

Perspective
-----------

All player-keyed slots are filled in *viewer-perspective seat order*: the
viewer occupies slot 0, and other seats follow in seat-rotation distance
(i.e. clockwise from the viewer in the configured ``player_ids`` cycle).
This makes a single shared policy reusable across seats without per-seat
heads.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from domain.engine.player_view import PlayerView
from domain.ids import PlayerID
from rl.encoding.features import (
    AWARDS_WIDTH,
    EDGES_BLOCK,
    OPPONENTS_BLOCK,
    PHASE_PENDING_WIDTH,
    PORTS_BLOCK,
    SELF_HAND_WIDTH,
    TILES_BLOCK,
    VERTICES_BLOCK,
    write_awards,
    write_edges,
    write_opponents,
    write_phase_pending,
    write_ports,
    write_self_hand,
    write_tiles,
    write_vertices,
)

__all__ = [
    "FlatObservationEncoder",
    "OBS_LAYOUT_VERSION",
    "OBS_SHAPE",
    "TILES_OFFSET",
    "VERTICES_OFFSET",
    "EDGES_OFFSET",
    "PORTS_OFFSET",
    "SELF_HAND_OFFSET",
    "OPPONENTS_OFFSET",
    "AWARDS_OFFSET",
    "PHASE_PENDING_OFFSET",
]

OBS_LAYOUT_VERSION: Final[int] = 1

TILES_OFFSET: Final[int] = 0
VERTICES_OFFSET: Final[int] = TILES_OFFSET + TILES_BLOCK
EDGES_OFFSET: Final[int] = VERTICES_OFFSET + VERTICES_BLOCK
PORTS_OFFSET: Final[int] = EDGES_OFFSET + EDGES_BLOCK
SELF_HAND_OFFSET: Final[int] = PORTS_OFFSET + PORTS_BLOCK
OPPONENTS_OFFSET: Final[int] = SELF_HAND_OFFSET + SELF_HAND_WIDTH
AWARDS_OFFSET: Final[int] = OPPONENTS_OFFSET + OPPONENTS_BLOCK
PHASE_PENDING_OFFSET: Final[int] = AWARDS_OFFSET + AWARDS_WIDTH

_OBS_LEN: Final[int] = PHASE_PENDING_OFFSET + PHASE_PENDING_WIDTH
OBS_SHAPE: Final[tuple[int, ...]] = (_OBS_LEN,)


def _seat_order(player_ids: list[PlayerID], viewer: PlayerID) -> list[PlayerID]:
    """Return the configured cycle rotated so ``viewer`` is at index 0."""
    try:
        i = player_ids.index(viewer)
    except ValueError as e:
        raise ValueError(
            f"viewer {viewer} is not in player_ids {player_ids}"
        ) from e
    return player_ids[i:] + player_ids[:i]


class FlatObservationEncoder:
    """Encode a :class:`PlayerView` as a fixed-shape float32 vector.

    The encoder is stateless; instances are cheap to construct. ``out_shape``
    is a class attribute so consumers can size buffers without instantiating.
    """

    out_shape: Final[tuple[int, ...]] = OBS_SHAPE
    layout_version: Final[int] = OBS_LAYOUT_VERSION

    def encode(self, view: PlayerView) -> np.ndarray:
        """Return the flat observation for ``view`` (dtype ``float32``).

        The returned array is freshly allocated; callers may freely mutate it.
        """
        out = np.zeros(OBS_SHAPE, dtype=np.float32)

        viewer = self._viewer_id(view)
        seat_order = _seat_order(list(view.config.player_ids), viewer)
        seat_of_player: dict[PlayerID, int] = {pid: i for i, pid in enumerate(seat_order)}

        write_tiles(out, TILES_OFFSET, view.topology, view.occupancy)
        write_vertices(out, VERTICES_OFFSET, view.occupancy, seat_of_player)
        write_edges(out, EDGES_OFFSET, view.occupancy, seat_of_player)
        write_ports(out, PORTS_OFFSET, view.topology)

        self_state = view.players[viewer]
        write_self_hand(out, SELF_HAND_OFFSET, self_state)

        opponents = [view.players.get(pid) for pid in seat_order[1:]]
        write_opponents(out, OPPONENTS_OFFSET, opponents)

        write_awards(
            out,
            AWARDS_OFFSET,
            view,
            self_state,
            view.bank,
            seat_of_player,
        )
        write_phase_pending(out, PHASE_PENDING_OFFSET, view.phase, view.pending)

        return out

    @staticmethod
    def _viewer_id(view: PlayerView) -> PlayerID:
        """The viewer is the player whose row exposes the full dev hand list."""
        for pid, row in view.players.items():
            if not isinstance(row.dev_cards_in_hand, int):
                return pid
        raise ValueError(
            "no viewer found in PlayerView — every row has integer dev_cards_in_hand"
        )
