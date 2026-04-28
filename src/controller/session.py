"""
Lifecycle management for a single game session.

Owns the :class:`~domain.engine.game_engine.GameEngine` instance and acts as the
single entry point through which external callers advance the game, keeping
engine construction, seeding, and teardown in one place.
"""

__all__ = []
