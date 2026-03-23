from board_buildables import Building, Settlement, Road

from math import sqrt
from typing import Optional


class Vertex:
  """
  Represents a vertex on the gameboard

  Attributes:
      x (int): The x-coordinate
      y (int): The y-coordinate
      ids (List<Tuple<Int>>): A list of ids, where an ID is a tile + offset
  """

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.ids = []
    self.tiles: list[Tile] = []
    self.edges: list[Edge] = []
    self.building: Optional[Building] = None

  def add_id(self, id):
    self.ids.append(id)

  def get_first_id(self):
    if len(self.ids) > 0:
      return self.ids[0]
    return None

  def add_tile(self, tile):
    self.tiles.append(tile)

  def add_edge(self, edge):
    self.edges.append(edge)

  def has_building(self):
    return self.building is not None

  def get_building_player(self):
    if self.has_building():
      return self.building.player
    else:
      return None

  def set_building(self, building: Building):
    self.building = building

  def has_settlement(self):
    return isinstance(self.building, Settlement)


class Edge:
  """
  Represents an edge between two vertices
  """


class Tile:
  """
  Represents a resource tile in the gameboard
  """
