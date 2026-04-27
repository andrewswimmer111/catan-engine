"""Cached board topologies for tests."""

from __future__ import annotations

import functools

from domain.board.layout import build_standard_board
from domain.board.topology import BoardTopology


@functools.lru_cache(maxsize=1)
def standard_board() -> BoardTopology:
    """The standard 19-hex Catan map graph (``build_standard_board``)."""
    return build_standard_board()


def minimal_board() -> BoardTopology:
    """
    Smallest valid board for fast tests.

    The engine only implements the standard variant, so this is an alias to
    :func:`standard_board` until a smaller build graph exists.
    """
    return standard_board()
