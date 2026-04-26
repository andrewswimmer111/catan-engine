"""
Every named constant used across the codebase lives here.
No other module should redefine these values.

Contract:
  - All other modules import from this file; they never redeclare these names.
  - Resource.DESERT labels desert hex tiles. It is never banked, traded, or
    drawn. Use :func:`tradeable_resources` when you need the five tradable
    resource kinds only.
"""

from enum import Enum
from typing import Final


class Resource(Enum):
    """All resource-like tile and card kinds, including the desert label."""
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


# Subset of ``Resource`` used by bank, trades, and player hands.
_TRADEABLE: Final[tuple[Resource, ...]] = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


def tradeable_resources() -> tuple[Resource, ...]:
    """The five resource types that can be held, traded, and banked."""
    return _TRADEABLE


class DevCardType(Enum):
    """Development card varieties."""
    KNIGHT = "knight"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"
    VICTORY_POINT = "victory_point"


class BuildingType(Enum):
    """Structures a player can place on a vertex."""
    SETTLEMENT = "settlement"
    CITY = "city"


class PortType(Enum):
    """Trade port varieties on the board edge."""
    THREE_TO_ONE = "3:1"
    WOOD_TWO = "2:1_wood"
    BRICK_TWO = "2:1_brick"
    SHEEP_TWO = "2:1_sheep"
    WHEAT_TWO = "2:1_wheat"
    ORE_TWO = "2:1_ore"


class TurnPhase(Enum):
    """
    Finite states of a single player's turn (and game-level terminal state).

    Design notes:
      - INITIAL_SETTLEMENT and INITIAL_ROAD are shared across both placement
        rounds (forward and reverse). The game engine, not the phase, tracks
        which round is in progress.
      - BUILD_ROADS signals "Road Building card is active"; the pending-effect
        object carries the remaining road count. Encoding the count in the
        phase enum (e.g. BUILD_ROAD_1 / BUILD_ROAD_2) would be wrong because
        it mixes state into what should be a pure control-flow signal.
      - YEAR_OF_PLENTY_SELECT and MONOPOLY_SELECT are separate phases because
        they require player input before the engine can resolve the card.
    """
    INITIAL_SETTLEMENT = "initial_settlement"
    INITIAL_ROAD = "initial_road"
    ROLL = "roll"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    STEAL = "steal"
    MAIN = "main"
    BUILD_ROADS = "build_roads"
    YEAR_OF_PLENTY_SELECT = "year_of_plenty_select"
    MONOPOLY_SELECT = "monopoly_select"
    GAME_OVER = "game_over"


class DomesticTradeState(Enum):
    """
    Transient negotiation state for a domestic (player-to-player) trade offer.
    These values are never persisted to completed trade records; they exist
    only while a trade offer is live in the pending-effects queue.
    """
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"