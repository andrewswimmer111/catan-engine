"""
All concrete event types the transition layer emits; each satisfies
:class:`GameEvent` in :mod:`domain.events.base`. The union
``AnyGameEvent`` is the contract for serialization round-trips in Task 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from domain.enums import DevCardType, Resource
from domain.ids import EdgeID, PlayerID, TileID, VertexID


@dataclass(frozen=True)
class DiceRolled:
    turn_number: int
    player_id: PlayerID
    die1: int
    die2: int
    total: int


@dataclass(frozen=True)
class ResourcesDistributed:
    turn_number: int
    distributions: dict[PlayerID, dict[Resource, int]]


@dataclass(frozen=True)
class BankShortfall:
    turn_number: int
    resource: Resource
    requested: int
    given: int


@dataclass(frozen=True)
class RoadBuilt:
    turn_number: int
    player_id: PlayerID
    edge_id: EdgeID


@dataclass(frozen=True)
class SettlementBuilt:
    turn_number: int
    player_id: PlayerID
    vertex_id: VertexID


@dataclass(frozen=True)
class CityBuilt:
    turn_number: int
    player_id: PlayerID
    vertex_id: VertexID


@dataclass(frozen=True)
class DevCardBought:
    turn_number: int
    player_id: PlayerID
    card_type: DevCardType


@dataclass(frozen=True)
class DevCardPlayed:
    turn_number: int
    player_id: PlayerID
    card_type: DevCardType


@dataclass(frozen=True)
class PlayerDiscarded:
    turn_number: int
    player_id: PlayerID
    resources: dict[Resource, int]


@dataclass(frozen=True)
class RobberMoved:
    turn_number: int
    player_id: PlayerID
    tile_id: TileID


@dataclass(frozen=True)
class ResourceStolen:
    turn_number: int
    by_player_id: PlayerID
    from_player_id: PlayerID
    resource: Resource


@dataclass(frozen=True)
class TradeCompleted:
    turn_number: int
    player1_id: PlayerID
    player2_id: PlayerID
    player1_gives: dict[Resource, int]
    player2_gives: dict[Resource, int]


@dataclass(frozen=True)
class MaritimeTradeCompleted:
    turn_number: int
    player_id: PlayerID
    gave: Resource
    received: Resource


@dataclass(frozen=True)
class LongestRoadAwarded:
    turn_number: int
    player_id: PlayerID
    length: int


@dataclass(frozen=True)
class LargestArmyAwarded:
    turn_number: int
    player_id: PlayerID
    count: int


@dataclass(frozen=True)
class TurnEnded:
    turn_number: int
    player_id: PlayerID


@dataclass(frozen=True)
class GameWon:
    turn_number: int
    player_id: PlayerID
    victory_points: int


AnyGameEvent = Union[
    DiceRolled,
    ResourcesDistributed,
    BankShortfall,
    RoadBuilt,
    SettlementBuilt,
    CityBuilt,
    DevCardBought,
    DevCardPlayed,
    PlayerDiscarded,
    RobberMoved,
    ResourceStolen,
    TradeCompleted,
    MaritimeTradeCompleted,
    LongestRoadAwarded,
    LargestArmyAwarded,
    TurnEnded,
    GameWon,
]
