"""
Typed identifiers for every addressable entity in the game.

Implementation choice — typing.NewType:
  NewType creates a distinct type alias that static type checkers (mypy,
  pyright) treat as incompatible with its base type and with each other.
  At runtime every NewType constructor is the identity function, so
  PlayerID(1) is literally the integer 1 and carries zero overhead.

  Consequence: `PlayerID(1) == VertexID(1)` evaluates to True at runtime
  because both are just the integer 1.  Type safety is enforced by the
  checker, not by the interpreter.  If runtime enforcement were required
  a frozen dataclass would be the correct alternative; it was explicitly
  ruled out in favour of zero runtime cost.

Contract:
  - All other modules import IDs from here.
  - No other module defines its own int aliases for these concepts.
  - IDs are opaque integers; their numeric values carry no semantic meaning
    beyond identity (do not rely on ordering or contiguity).
"""

from typing import NewType

# Identifies a player in the game (0-indexed seat number by convention,
# but callers must not depend on that ordering).
PlayerID = NewType("PlayerID", int)

# Identifies a vertex on the board graph (intersection of tile edges).
# Vertex IDs are assigned by the board builder and are stable for the
# lifetime of a game instance.
VertexID = NewType("VertexID", int)

# Identifies an edge on the board graph (shared border between two vertices).
# Edge IDs are assigned by the board builder alongside vertex IDs.
EdgeID = NewType("EdgeID", int)

# Identifies a hex tile on the board.
TileID = NewType("TileID", int)