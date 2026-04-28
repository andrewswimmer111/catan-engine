"""
State diffing between successive game steps.

Compares two :class:`~domain.game.state.GameState` snapshots and produces a
structured description of what changed — resources gained, buildings placed,
phase transitions — so callers can react to only the deltas rather than
re-inspecting the entire state on every tick.
"""

__all__ = []
