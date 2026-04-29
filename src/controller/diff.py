"""
State diffing between successive game steps.

Compares two :class:`~domain.game.state.GameState` snapshots and produces a
structured description of what changed — resources gained, buildings placed,
phase transitions — so callers can react to only the deltas rather than
re-inspecting the entire state on every tick.
"""

from __future__ import annotations

from typing import Any

from domain.game.state import GameState
from serialization.codec import encode_state

__all__ = ["changed_paths"]


def changed_paths(prev: GameState, curr: GameState) -> set[tuple[str, ...]]:
    """Return the set of dotted paths whose leaf values differ.

    e.g. {('players','0','resources','WOOD'), ('occupancy','robber_tile')}
    """
    prev_dict = encode_state(prev)
    curr_dict = encode_state(curr)
    result: set[tuple[str, ...]] = set()
    _diff(prev_dict, curr_dict, (), result)
    return result


def _diff(a: Any, b: Any, path: tuple[str, ...], out: set[tuple[str, ...]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            _diff(a.get(key), b.get(key), path + (key,), out)
    elif a != b:
        out.add(path)
