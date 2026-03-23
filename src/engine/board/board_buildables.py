class Building:
  collection_amount: int

  def __init__(self, player):
    self.player = player

  def get_collection_amount(self):
    return self.collection_amount


class Settlement(Building):
  collection_amount = 1


class City(Building):
  collection_amount = 2


class Road:
  def __init__(self, player, edge):
    self.player = player
