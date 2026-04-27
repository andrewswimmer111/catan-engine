"""
JSON-friendly encoding of game state, actions, and events.

* Enums use :attr:`enum.Enum.name` (not ``.value``).
* :class:`typing.NewType` IDs (``PlayerID``, etc.) are plain ``int`` in payloads.
* JSON object keys for ID-keyed dicts are decimal strings, e.g. ``\"5\"`` → ``5``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Type, TypeVar, cast

import domain.events.all_events as E
from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.board.edge import Edge
from domain.board.occupancy import BoardOccupancy
from domain.board.port import Port
from domain.board.tile import Tile
from domain.board.topology import BoardTopology
from domain.board.vertex import Vertex
from domain.enums import (
    BuildingType,
    DevCardType,
    DomesticTradeState,
    EndReason,
    PortType,
    Resource,
    TurnPhase,
)
from domain.game.bank import Bank
from domain.game.config import GameConfig
from domain.game.dev_deck import DevelopmentDeck
from domain.game.player_state import PlayerState
from domain.game.state import GameState
from domain.ids import EdgeID, PlayerID, TileID, VertexID
from domain.turn.pending import (
    DiscardPending,
    DomesticTradePending,
    MonopolyPending,
    PendingEffect,
    RoadBuildingPending,
    RobberMovePending,
    StealPending,
    YearOfPlentyPending,
)

# --- state snapshot version (bump if wire format changes) ---

_STATE_VERSION = 2

# --- small enum helpers ---


def _enum_name(x: Enum) -> str:
    return x.name


E_ = TypeVar("E_", bound=Enum)


def _enum_by_name(cl: Type[E_], s: str) -> E_:
    return cl[s]


# --- id-keyed dicts (JSON string keys) ---

_K = TypeVar("_K")


def _id_map_enc(id_encoder: Callable[[_K], int], d: dict[_K, Any], enc_v: Callable[[Any], Any]) -> dict[str, Any]:
    return {str(id_encoder(k)): enc_v(v) for k, v in sorted(d.items(), key=lambda kv: id_encoder(kv[0]))}


def _resource_map_enc(m: dict[Resource, int]) -> dict[str, int]:
    return { _enum_name(r): c for r, c in sorted(m.items(), key=lambda x: x[0].name)}


def _resource_map_dec(d: dict[str, Any]) -> dict[Resource, int]:
    return {Resource[k]: int(v) for k, v in sorted(d.items())}


def _player_res_map_enc(m: dict[PlayerID, dict[Resource, int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for p in sorted(m.keys(), key=int):
        out[str(p)] = _resource_map_enc(m[p])
    return out


def _player_res_map_dec(d: dict[str, Any]) -> dict[PlayerID, dict[Resource, int]]:
    return {PlayerID(int(k)): _resource_map_dec(v) for k, v in sorted(d.items(), key=lambda kv: int(cast(str, kv[0])))}


def _fset_enc(tname: str, s: frozenset[Any], item_enc: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    if item_enc:
        items: list[Any] = sorted(item_enc(x) for x in s)
    else:
        items = sorted(s)
    return {"__fset__": tname, "items": items}


def _fset_dec(
    tname: str, d: dict[str, Any], item_dec: Callable[[Any], _K] | None = None
) -> frozenset[_K]:
    if d.get("__fset__") != tname:
        raise ValueError(f"expected fset {tname!r}, got {d!r}")
    items = d["items"]
    if item_dec is None:
        return frozenset(items)
    return frozenset(item_dec(x) for x in items)


# --- public config (also used by :mod:`serialization.replay`) ---


def encode_config(cfg: GameConfig) -> dict[str, Any]:
    return {
        "player_ids": [int(p) for p in cfg.player_ids],
        "seed": cfg.seed,
        "board_variant": cfg.board_variant,
    }


def decode_config(d: dict[str, Any]) -> GameConfig:
    return GameConfig(
        player_ids=[PlayerID(p) for p in d["player_ids"]],
        seed=int(d["seed"]),
        board_variant=str(d["board_variant"]),
    )


# --- topology ---


def _encode_tile(t: Tile) -> dict[str, Any]:
    return {
        "tile_id": int(t.tile_id),
        "resource": _enum_name(t.resource) if t.resource is not None else None,
        "dice_number": t.dice_number,
    }


def _decode_tile(d: dict[str, Any]) -> Tile:
    return Tile(
        tile_id=TileID(int(d["tile_id"])),
        resource=_enum_by_name(Resource, d["resource"]) if d.get("resource") is not None else None,
        dice_number=None if d.get("dice_number") is None else int(d["dice_number"]),
    )


def _encode_vertex(v: Vertex) -> dict[str, Any]:
    return {
        "vertex_id": int(v.vertex_id),
        "adjacent_vertices": _fset_enc("VertexID", v.adjacent_vertices, int),
        "adjacent_edges": _fset_enc("EdgeID", v.adjacent_edges, int),
        "adjacent_tiles": _fset_enc("TileID", v.adjacent_tiles, int),
        "port": _enum_name(v.port) if v.port is not None else None,
    }


def _decode_vertex(d: dict[str, Any]) -> Vertex:
    return Vertex(
        vertex_id=VertexID(int(d["vertex_id"])),
        adjacent_vertices=_fset_dec("VertexID", d["adjacent_vertices"], lambda x: VertexID(int(x))),
        adjacent_edges=_fset_dec("EdgeID", d["adjacent_edges"], lambda x: EdgeID(int(x))),
        adjacent_tiles=_fset_dec("TileID", d["adjacent_tiles"], lambda x: TileID(int(x))),
        port=_enum_by_name(PortType, d["port"])
        if d.get("port") is not None
        else None,
    )


def _encode_edge(e: Edge) -> dict[str, Any]:
    return {
        "edge_id": int(e.edge_id),
        "vertices": [int(e.v1), int(e.v2)],
        "adjacent_edges": _fset_enc("EdgeID", e.adjacent_edges, int),
    }


def _decode_edge(d: dict[str, Any]) -> Edge:
    a, b = d["vertices"]
    return Edge(
        edge_id=EdgeID(int(d["edge_id"])),
        vertices=(VertexID(int(a)), VertexID(int(b))),
        adjacent_edges=_fset_dec("EdgeID", d["adjacent_edges"], lambda x: EdgeID(int(x))),
    )


def _encode_port(p: Port) -> dict[str, Any]:
    return {
        "port_type": _enum_name(p.port_type) if p.port_type is not None else None,
        "vertices": [int(p.vertices[0]), int(p.vertices[1])],
    }


def _decode_port(d: dict[str, Any]) -> Port:
    a, b = d["vertices"]
    return Port(
        port_type=_enum_by_name(PortType, d["port_type"]) if d.get("port_type") is not None else None,
        vertices=(VertexID(int(a)), VertexID(int(b))),
    )


def _encode_topology(top: BoardTopology) -> dict[str, Any]:
    return {
        "tiles": _id_map_enc(int, top.tiles, _encode_tile),  # type: ignore[arg-type]
        "vertices": _id_map_enc(int, top.vertices, _encode_vertex),
        "edges": _id_map_enc(int, top.edges, _encode_edge),
        "ports": [_encode_port(p) for p in top.ports],
    }


def _decode_topology(d: dict[str, Any]) -> BoardTopology:
    tiles = {TileID(int(k)): _decode_tile(v) for k, v in d["tiles"].items()}
    vertices = {VertexID(int(k)): _decode_vertex(v) for k, v in d["vertices"].items()}
    edges = {EdgeID(int(k)): _decode_edge(v) for k, v in d["edges"].items()}
    ports = tuple(_decode_port(p) for p in d["ports"])
    return BoardTopology(tiles=tiles, vertices=vertices, edges=edges, ports=ports)


# --- bank, occupancy, player, dev deck, pending ---


def _encode_bank_all(b: Bank) -> dict[str, int]:
    from domain.enums import tradeable_resources

    return { _enum_name(r): b.resources.get(r, 0) for r in tradeable_resources()}


def _decode_bank(d: dict[str, Any]) -> Bank:
    from domain.enums import tradeable_resources

    b = Bank()
    b.resources = {r: int(d.get(_enum_name(r), 0)) for r in tradeable_resources()}
    return b


def _encode_occupancy(o: BoardOccupancy) -> dict[str, Any]:
    return {
        "roads": {str(int(e)): int(p) for e, p in sorted(o.roads.items(), key=lambda x: int(x[0]))},
        "buildings": {
            str(int(vid)): [int(own), _enum_name(bt)]
            for vid, (own, bt) in sorted(o.buildings.items(), key=lambda x: int(x[0]))
        },
        "robber_tile": int(o.robber_tile),
    }


def _decode_occupancy(d: dict[str, Any]) -> BoardOccupancy:
    roads = {EdgeID(int(e)): PlayerID(p) for e, p in d["roads"].items()}
    buildings: dict[VertexID, tuple[PlayerID, BuildingType]] = {}
    for vks, (pid, bname) in d["buildings"].items():
        buildings[VertexID(int(vks))] = (PlayerID(int(pid)), _enum_by_name(BuildingType, bname))  # type: ignore[assignment]
    return BoardOccupancy(roads=roads, buildings=buildings, robber_tile=TileID(int(d["robber_tile"])))


def _encode_player(p: PlayerState) -> dict[str, Any]:
    return {
        "player_id": int(p.player_id),
        "resources": _resource_map_enc(p.resources),
        "dev_cards_in_hand": [[_enum_name(c), t] for c, t in p.dev_cards_in_hand],
        "dev_cards_played": [_enum_name(c) for c in p.dev_cards_played],
        "roads_built": p.roads_built,
        "settlements_built": p.settlements_built,
        "cities_built": p.cities_built,
        "knights_played": p.knights_played,
        "has_played_dev_card_this_turn": p.has_played_dev_card_this_turn,
        "victory_points_public": p.victory_points_public,
    }


def _decode_player(d: dict[str, Any]) -> PlayerState:
    ps = PlayerState(
        player_id=PlayerID(int(d["player_id"])),
        resources=_resource_map_dec(d.get("resources") or {}),
    )
    ps.dev_cards_in_hand = [(_enum_by_name(DevCardType, c), int(t)) for c, t in d["dev_cards_in_hand"]]
    ps.dev_cards_played = [_enum_by_name(DevCardType, c) for c in d["dev_cards_played"]]
    ps.roads_built = int(d["roads_built"])
    ps.settlements_built = int(d["settlements_built"])
    ps.cities_built = int(d["cities_built"])
    ps.knights_played = int(d["knights_played"])
    ps.has_played_dev_card_this_turn = bool(d["has_played_dev_card_this_turn"])
    ps.victory_points_public = int(d["victory_points_public"])
    return ps


def _encode_dev_deck(dd: DevelopmentDeck) -> list[str]:
    return [_enum_name(c) for c in dd.cards]


def _decode_dev_deck(d: list[str]) -> DevelopmentDeck:
    return DevelopmentDeck(cards=[_enum_by_name(DevCardType, c) for c in d])


def _encode_pending(p: PendingEffect | None) -> dict[str, Any] | None:
    if p is None:
        return None
    if isinstance(p, DiscardPending):
        return {
            "type": "DiscardPending",
            "cards_to_discard": {str(int(pid)): c for pid, c in sorted(p.cards_to_discard.items(), key=lambda x: int(x[0]))},
        }
    if isinstance(p, RobberMovePending):
        return {"type": "RobberMovePending", "return_phase": _enum_name(p.return_phase)}
    if isinstance(p, StealPending):
        return {
            "type": "StealPending",
            "valid_targets": sorted(int(x) for x in p.valid_targets),
            "return_phase": _enum_name(p.return_phase),
        }
    if isinstance(p, DomesticTradePending):
        return {
            "type": "DomesticTradePending",
            "offer": _resource_map_enc(p.offer),
            "request": _resource_map_enc(p.request),
            "responses": {str(int(pid)): _enum_name(s) for pid, s in sorted(p.responses.items(), key=lambda x: int(x[0]))},
        }
    if isinstance(p, RoadBuildingPending):
        return {"type": "RoadBuildingPending", "roads_remaining": p.roads_remaining}
    if isinstance(p, YearOfPlentyPending):
        return {"type": "YearOfPlentyPending", "resources_remaining": p.resources_remaining}
    if isinstance(p, MonopolyPending):
        return {"type": "MonopolyPending"}
    raise TypeError(f"unhandled pending {type(p)}")


def _decode_pending(d: dict[str, Any] | None) -> PendingEffect | None:
    if d is None:
        return None
    t = d["type"]
    if t == "DiscardPending":
        return DiscardPending(
            cards_to_discard={PlayerID(int(k)): int(v) for k, v in d["cards_to_discard"].items()}
        )
    if t == "RobberMovePending":
        return RobberMovePending(return_phase=_enum_by_name(TurnPhase, d["return_phase"]))
    if t == "StealPending":
        return StealPending(
            valid_targets=frozenset(PlayerID(x) for x in d["valid_targets"]),
            return_phase=_enum_by_name(TurnPhase, d["return_phase"]),
        )
    if t == "DomesticTradePending":
        return DomesticTradePending(
            offer=_resource_map_dec(d["offer"]),
            request=_resource_map_dec(d["request"]),
            responses={PlayerID(int(k)): _enum_by_name(DomesticTradeState, v) for k, v in d["responses"].items()},
        )
    if t == "RoadBuildingPending":
        return RoadBuildingPending(roads_remaining=int(d["roads_remaining"]))
    if t == "YearOfPlentyPending":
        return YearOfPlentyPending(resources_remaining=int(d["resources_remaining"]))
    if t == "MonopolyPending":
        return MonopolyPending()
    raise ValueError(f"unknown pending {t}")


def encode_state(state: GameState) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "config": encode_config(state.config),
        "topology": _encode_topology(state.topology),
        "occupancy": _encode_occupancy(state.occupancy),
        "players": {str(int(pid)): _encode_player(p) for pid, p in sorted(state.players.items(), key=lambda x: int(x[0]))},
        "bank": _encode_bank_all(state.bank),
        "dev_deck": _encode_dev_deck(state.dev_deck),
        "current_player": int(state.current_player),
        "phase": _enum_name(state.phase),
        "turn_number": state.turn_number,
        "pending": _encode_pending(state.pending),
        "setup_order": [int(p) for p in state.setup_order],
        "setup_index": state.setup_index,
        "last_settlement_vertex": None
        if state.last_settlement_vertex is None
        else int(state.last_settlement_vertex),
        "longest_road_holder": None if state.longest_road_holder is None else int(state.longest_road_holder),
        "largest_army_holder": None if state.largest_army_holder is None else int(state.largest_army_holder),
        "winner": None if state.winner is None else int(state.winner),
        "end_reason": None if state.end_reason is None else _enum_name(state.end_reason),
        "turns_since_vp_change": int(state.turns_since_vp_change),
    }


def decode_state(data: dict[str, Any]) -> GameState:
    if int(data.get("version", 0)) != _STATE_VERSION:
        raise ValueError("unsupported or missing state version")
    pls = {PlayerID(int(k)): _decode_player(v) for k, v in data["players"].items()}
    st = GameState(
        config=decode_config(data["config"]),
        topology=_decode_topology(data["topology"]),
        occupancy=_decode_occupancy(data["occupancy"]),
        players=pls,
        bank=_decode_bank(data["bank"]),
        dev_deck=_decode_dev_deck(data["dev_deck"]),
        current_player=PlayerID(int(data["current_player"])),
        phase=_enum_by_name(TurnPhase, data["phase"]),
        turn_number=int(data["turn_number"]),
        pending=_decode_pending(data.get("pending")),
        setup_order=[PlayerID(p) for p in data["setup_order"]],
        setup_index=int(data["setup_index"]),
        last_settlement_vertex=None
        if data.get("last_settlement_vertex") is None
        else VertexID(int(data["last_settlement_vertex"])),
        longest_road_holder=None
        if data.get("longest_road_holder") is None
        else PlayerID(int(data["longest_road_holder"])),
        largest_army_holder=None
        if data.get("largest_army_holder") is None
        else PlayerID(int(data["largest_army_holder"])),
        winner=None if data.get("winner") is None else PlayerID(int(data["winner"])),
        end_reason=None
        if data.get("end_reason") is None
        else _enum_by_name(EndReason, data["end_reason"]),
        turns_since_vp_change=int(data.get("turns_since_vp_change", 0)),
    )
    return st


# --- actions ---


def encode_action(action: Action) -> dict[str, Any]:
    t = type(action).__name__
    pid = int(action.player_id)
    if isinstance(action, A.PlaceSettlementAction):
        return {"type": t, "player_id": pid, "vertex_id": int(action.vertex_id)}
    if isinstance(action, A.PlaceRoadAction):
        return {"type": t, "player_id": pid, "edge_id": int(action.edge_id)}
    if isinstance(action, A.BuildSettlementAction):
        return {"type": t, "player_id": pid, "vertex_id": int(action.vertex_id)}
    if isinstance(action, A.BuildRoadAction):
        return {"type": t, "player_id": pid, "edge_id": int(action.edge_id)}
    if isinstance(action, A.BuildCityAction):
        return {"type": t, "player_id": pid, "vertex_id": int(action.vertex_id)}
    if isinstance(action, (A.RollDiceAction, A.EndTurnAction, A.BuyDevCardAction, A.PlayKnightAction, A.PlayRoadBuildingAction, A.CancelDomesticTradeAction)):
        return {"type": t, "player_id": pid}
    if isinstance(action, A.DiscardResourcesAction):
        return {"type": t, "player_id": pid, "resources": _resource_map_enc(action.resources)}
    if isinstance(action, A.MoveRobberAction):
        return {"type": t, "player_id": pid, "tile_id": int(action.tile_id)}
    if isinstance(action, A.StealResourceAction):
        return {"type": t, "player_id": pid, "target_player_id": int(action.target_player_id)}
    if isinstance(action, A.PlayYearOfPlentyAction):
        return {
            "type": t,
            "player_id": pid,
            "resource1": _enum_name(action.resource1),
            "resource2": _enum_name(action.resource2),
        }
    if isinstance(action, A.PlayMonopolyAction):
        return {"type": t, "player_id": pid, "resource": _enum_name(action.resource)}
    if isinstance(action, A.MaritimeTradeAction):
        return {
            "type": t,
            "player_id": pid,
            "give": _enum_name(action.give),
            "give_count": int(action.give_count),
            "receive": _enum_name(action.receive),
        }
    if isinstance(action, A.ProposeDomesticTradeAction):
        return {
            "type": t,
            "player_id": pid,
            "offer": _resource_map_enc(action.offer),
            "request": _resource_map_enc(action.request),
        }
    if isinstance(action, A.RespondDomesticTradeAction):
        return {"type": t, "player_id": pid, "response": _enum_name(action.response)}
    if isinstance(action, A.ConfirmDomesticTradeAction):
        return {"type": t, "player_id": pid, "trade_with": int(action.trade_with)}
    raise TypeError(f"unhandled action {type(action)}")


def decode_action(data: dict[str, Any]) -> Action:
    t = data["type"]
    pid = PlayerID(int(data["player_id"]))
    if t == "PlaceSettlementAction":
        return A.PlaceSettlementAction(player_id=pid, vertex_id=VertexID(int(data["vertex_id"])))
    if t == "PlaceRoadAction":
        return A.PlaceRoadAction(player_id=pid, edge_id=EdgeID(int(data["edge_id"])))
    if t == "BuildSettlementAction":
        return A.BuildSettlementAction(player_id=pid, vertex_id=VertexID(int(data["vertex_id"])))
    if t == "BuildRoadAction":
        return A.BuildRoadAction(player_id=pid, edge_id=EdgeID(int(data["edge_id"])))
    if t == "BuildCityAction":
        return A.BuildCityAction(player_id=pid, vertex_id=VertexID(int(data["vertex_id"])))
    if t == "RollDiceAction":
        return A.RollDiceAction(player_id=pid)
    if t == "EndTurnAction":
        return A.EndTurnAction(player_id=pid)
    if t == "BuyDevCardAction":
        return A.BuyDevCardAction(player_id=pid)
    if t == "PlayKnightAction":
        return A.PlayKnightAction(player_id=pid)
    if t == "PlayRoadBuildingAction":
        return A.PlayRoadBuildingAction(player_id=pid)
    if t == "CancelDomesticTradeAction":
        return A.CancelDomesticTradeAction(player_id=pid)
    if t == "DiscardResourcesAction":
        return A.DiscardResourcesAction(player_id=pid, resources=_resource_map_dec(data["resources"]))
    if t == "MoveRobberAction":
        return A.MoveRobberAction(player_id=pid, tile_id=TileID(int(data["tile_id"])))
    if t == "StealResourceAction":
        return A.StealResourceAction(player_id=pid, target_player_id=PlayerID(int(data["target_player_id"])))
    if t == "PlayYearOfPlentyAction":
        return A.PlayYearOfPlentyAction(
            player_id=pid,
            resource1=_enum_by_name(Resource, data["resource1"]),
            resource2=_enum_by_name(Resource, data["resource2"]),
        )
    if t == "PlayMonopolyAction":
        return A.PlayMonopolyAction(player_id=pid, resource=_enum_by_name(Resource, data["resource"]))
    if t == "MaritimeTradeAction":
        return A.MaritimeTradeAction(
            player_id=pid,
            give=_enum_by_name(Resource, data["give"]),
            give_count=int(data["give_count"]),
            receive=_enum_by_name(Resource, data["receive"]),
        )
    if t == "ProposeDomesticTradeAction":
        return A.ProposeDomesticTradeAction(
            player_id=pid,
            offer=_resource_map_dec(data["offer"]),
            request=_resource_map_dec(data["request"]),
        )
    if t == "RespondDomesticTradeAction":
        return A.RespondDomesticTradeAction(
            player_id=pid, response=_enum_by_name(DomesticTradeState, data["response"])
        )
    if t == "ConfirmDomesticTradeAction":
        return A.ConfirmDomesticTradeAction(
            player_id=pid, trade_with=PlayerID(int(data["trade_with"]))
        )
    raise ValueError(f"unknown action type {t!r}")


# --- events ---


def encode_event(event: E.AnyGameEvent) -> dict[str, Any]:
    t = type(event).__name__
    turn = int(event.turn_number)  # type: ignore[arg-type]
    if isinstance(event, E.DiceRolled):
        return {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),
            "die1": int(event.die1),
            "die2": int(event.die2),
            "total": int(event.total),
        }
    if isinstance(event, E.ResourcesDistributed):
        return {
            "type": t,
            "turn_number": turn,
            "distributions": _player_res_map_enc(event.distributions),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.BankShortfall):
        return {
            "type": t,
            "turn_number": turn,
            "resource": _enum_name(event.resource),  # type: ignore[attr-defined]
            "requested": int(event.requested),  # type: ignore[attr-defined]
            "given": int(event.given),  # type: ignore[attr-defined]
        }
    if isinstance(event, (E.RoadBuilt, E.SettlementBuilt, E.CityBuilt, E.DevCardBought, E.DevCardPlayed)):
        common = {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
        }
        if isinstance(event, (E.DevCardBought, E.DevCardPlayed)):
            return {**common, "card_type": _enum_name(event.card_type)}  # type: ignore[attr-defined]
        if isinstance(event, E.RoadBuilt):
            return {**common, "edge_id": int(event.edge_id)}  # type: ignore[attr-defined]
        return {**common, "vertex_id": int(event.vertex_id)}  # type: ignore[attr-defined]
    if isinstance(event, E.PlayerDiscarded):
        return {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
            "resources": _resource_map_enc(event.resources),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.RobberMoved):
        return {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
            "tile_id": int(event.tile_id),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.ResourceStolen):
        return {
            "type": t,
            "turn_number": turn,
            "by_player_id": int(event.by_player_id),  # type: ignore[attr-defined]
            "from_player_id": int(event.from_player_id),  # type: ignore[attr-defined]
            "resource": _enum_name(event.resource),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.TradeCompleted):
        return {
            "type": t,
            "turn_number": turn,
            "player1_id": int(event.player1_id),  # type: ignore[attr-defined]
            "player2_id": int(event.player2_id),  # type: ignore[attr-defined]
            "player1_gives": _resource_map_enc(event.player1_gives),  # type: ignore[attr-defined]
            "player2_gives": _resource_map_enc(event.player2_gives),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.MaritimeTradeCompleted):
        return {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
            "gave": _enum_name(event.gave),  # type: ignore[attr-defined]
            "received": _enum_name(event.received),  # type: ignore[attr-defined]
        }
    if isinstance(event, (E.LongestRoadAwarded, E.LargestArmyAwarded, E.GameWon)):
        base = {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
        }
        if isinstance(event, E.GameWon):
            return {**base, "victory_points": int(event.victory_points)}  # type: ignore[attr-defined]
        if isinstance(event, E.LargestArmyAwarded):
            return {**base, "count": int(event.count)}  # type: ignore[attr-defined]
        return {**base, "length": int(event.length)}  # type: ignore[attr-defined]
    if isinstance(event, E.TurnEnded):
        return {
            "type": t,
            "turn_number": turn,
            "player_id": int(event.player_id),  # type: ignore[attr-defined]
        }
    if isinstance(event, E.GameStalled):
        return {
            "type": t,
            "turn_number": turn,
            "reason": _enum_name(event.reason),  # type: ignore[attr-defined]
        }
    raise TypeError(f"unhandled event {type(event)}")


def decode_event(data: dict[str, Any]) -> E.AnyGameEvent:
    t = data["type"]
    turn = int(data["turn_number"])
    if t == "DiceRolled":
        return E.DiceRolled(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            die1=int(data["die1"]),
            die2=int(data["die2"]),
            total=int(data["total"]),
        )
    if t == "ResourcesDistributed":
        return E.ResourcesDistributed(turn_number=turn, distributions=_player_res_map_dec(data["distributions"]))
    if t == "BankShortfall":
        return E.BankShortfall(
            turn_number=turn,
            resource=_enum_by_name(Resource, data["resource"]),
            requested=int(data["requested"]),
            given=int(data["given"]),
        )
    if t == "RoadBuilt":
        return E.RoadBuilt(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            edge_id=EdgeID(int(data["edge_id"])),
        )
    if t == "SettlementBuilt":
        return E.SettlementBuilt(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            vertex_id=VertexID(int(data["vertex_id"])),
        )
    if t == "CityBuilt":
        return E.CityBuilt(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            vertex_id=VertexID(int(data["vertex_id"])),
        )
    if t == "DevCardBought":
        return E.DevCardBought(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            card_type=_enum_by_name(DevCardType, data["card_type"]),
        )
    if t == "DevCardPlayed":
        return E.DevCardPlayed(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            card_type=_enum_by_name(DevCardType, data["card_type"]),
        )
    if t == "PlayerDiscarded":
        return E.PlayerDiscarded(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            resources=_resource_map_dec(data["resources"]),
        )
    if t == "RobberMoved":
        return E.RobberMoved(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            tile_id=TileID(int(data["tile_id"])),
        )
    if t == "ResourceStolen":
        return E.ResourceStolen(
            turn_number=turn,
            by_player_id=PlayerID(int(data["by_player_id"])),
            from_player_id=PlayerID(int(data["from_player_id"])),
            resource=_enum_by_name(Resource, data["resource"]),
        )
    if t == "TradeCompleted":
        return E.TradeCompleted(
            turn_number=turn,
            player1_id=PlayerID(int(data["player1_id"])),
            player2_id=PlayerID(int(data["player2_id"])),
            player1_gives=_resource_map_dec(data["player1_gives"]),
            player2_gives=_resource_map_dec(data["player2_gives"]),
        )
    if t == "MaritimeTradeCompleted":
        return E.MaritimeTradeCompleted(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            gave=_enum_by_name(Resource, data["gave"]),
            received=_enum_by_name(Resource, data["received"]),
        )
    if t == "LongestRoadAwarded":
        return E.LongestRoadAwarded(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            length=int(data["length"]),
        )
    if t == "LargestArmyAwarded":
        return E.LargestArmyAwarded(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            count=int(data["count"]),
        )
    if t == "TurnEnded":
        return E.TurnEnded(turn_number=turn, player_id=PlayerID(int(data["player_id"])))
    if t == "GameWon":
        return E.GameWon(
            turn_number=turn,
            player_id=PlayerID(int(data["player_id"])),
            victory_points=int(data["victory_points"]),
        )
    if t == "GameStalled":
        return E.GameStalled(
            turn_number=turn,
            reason=_enum_by_name(EndReason, data["reason"]),
        )
    raise ValueError(f"unknown event type {t!r}")
