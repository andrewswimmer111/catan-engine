"""
Agent dispatch and action selection for AI-controlled players.

Maps each AI player seat to a policy (random, heuristic, or model-backed) and
delegates to that policy to choose a legal action on each turn, keeping
strategy logic decoupled from session management.
"""

__all__ = []
