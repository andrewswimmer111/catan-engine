import domain.actions.all_actions as A
from controller.selectors import (
    ACTION_GROUPS,
    edge_targets,
    grouped,
    player_steal_targets,
    tile_targets,
    vertex_targets,
)
from domain.ids import EdgeID, PlayerID, TileID, VertexID


P0 = PlayerID(0)


def _legal():
    return [
        A.PlaceSettlementAction(player_id=P0, vertex_id=VertexID(1)),
        A.PlaceSettlementAction(player_id=P0, vertex_id=VertexID(2)),
        A.PlaceRoadAction(player_id=P0, edge_id=EdgeID(10)),
        A.BuildSettlementAction(player_id=P0, vertex_id=VertexID(3)),
        A.BuildCityAction(player_id=P0, vertex_id=VertexID(4)),
        A.BuildRoadAction(player_id=P0, edge_id=EdgeID(11)),
        A.MoveRobberAction(player_id=P0, tile_id=TileID(5)),
        A.MoveRobberAction(player_id=P0, tile_id=TileID(6)),
        A.StealResourceAction(player_id=P0, target_player_id=PlayerID(0)),
        A.StealResourceAction(player_id=P0, target_player_id=PlayerID(2)),
        A.RollDiceAction(player_id=P0),
        A.EndTurnAction(player_id=P0),
    ]


def test_vertex_targets_groups_by_class():
    result = vertex_targets(_legal())
    assert result[A.PlaceSettlementAction] == {VertexID(1), VertexID(2)}
    assert result[A.BuildSettlementAction] == {VertexID(3)}
    assert result[A.BuildCityAction] == {VertexID(4)}
    assert A.BuildRoadAction not in result


def test_vertex_targets_empty():
    assert vertex_targets([A.RollDiceAction(player_id=P0)]) == {}


def test_edge_targets_groups_by_class():
    result = edge_targets(_legal())
    assert result[A.PlaceRoadAction] == {EdgeID(10)}
    assert result[A.BuildRoadAction] == {EdgeID(11)}
    assert A.PlaceSettlementAction not in result


def test_edge_targets_empty():
    assert edge_targets([]) == {}


def test_tile_targets():
    assert tile_targets(_legal()) == {TileID(5), TileID(6)}


def test_tile_targets_empty():
    assert tile_targets([A.RollDiceAction(player_id=P0)]) == set()


def test_player_steal_targets():
    assert player_steal_targets(_legal()) == {PlayerID(0), PlayerID(2)}


def test_player_steal_targets_empty():
    assert player_steal_targets([]) == set()


def test_grouped_partitions_correctly():
    result = grouped(_legal())
    assert A.PlaceSettlementAction(player_id=P0, vertex_id=VertexID(1)) in result["Setup"]
    assert A.PlaceRoadAction(player_id=P0, edge_id=EdgeID(10)) in result["Setup"]
    assert A.BuildSettlementAction(player_id=P0, vertex_id=VertexID(3)) in result["Build"]
    assert A.RollDiceAction(player_id=P0) in result["Turn"]
    assert A.EndTurnAction(player_id=P0) in result["Turn"]
    assert A.MoveRobberAction(player_id=P0, tile_id=TileID(5)) in result["Robber"]
    assert "DevCard" not in result
    assert "Trade" not in result


def test_grouped_empty():
    assert grouped([]) == {}


def test_action_groups_contains_expected_keys():
    assert set(ACTION_GROUPS) == {"Build", "Setup", "Turn", "Robber", "DevCard", "Trade"}
