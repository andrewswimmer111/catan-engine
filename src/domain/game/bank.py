"""
Resource bank: finite supply of the five tradable resources, with default stock
(19 each), affordability checks, and deposit/withdraw operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.enums import Resource, tradeable_resources


def default_bank_stock() -> dict[Resource, int]:
    """Initial bank: 19 of each of the five tradable resource types."""
    return {r: 19 for r in tradeable_resources()}


@dataclass
class Bank:
    """
    Finite resource supply. Only the five tradable resources exist in the bank;
    :class:`~domain.enums.Resource` ``DESERT`` is never present.
    """

    resources: dict[Resource, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resources:
            self.resources = default_bank_stock()

    def can_afford(self, cost: dict[Resource, int]) -> bool:
        return all(self.resources.get(r, 0) >= c for r, c in cost.items())

    def withdraw(self, cost: dict[Resource, int]) -> None:
        if not self.can_afford(cost):
            short = {r: c for r, c in cost.items() if self.resources.get(r, 0) < c}
            raise ValueError(f"insufficient bank resources: {short!r}, bank={self.resources!r}")
        for r, c in cost.items():
            self.resources[r] -= c

    def deposit(self, resources: dict[Resource, int]) -> None:
        for r, c in resources.items():
            self.resources[r] = self.resources.get(r, 0) + c
