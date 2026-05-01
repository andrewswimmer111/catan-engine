from __future__ import annotations

from collections import Counter

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from domain.engine.player_view import PlayerPerspectiveState
from domain.game.player_state import PlayerState
from gui import theme

_RES_ABBR: dict[str, str] = {
    "WOOD": "Wd", "BRICK": "Br", "SHEEP": "Sh",
    "WHEAT": "Wh", "ORE": "Or", "DESERT": "—",
}
_DEV_ABBR: dict[str, str] = {
    "KNIGHT": "Kn", "ROAD_BUILDING": "RB",
    "YEAR_OF_PLENTY": "YP", "MONOPOLY": "Mo", "VICTORY_POINT": "VP",
}


def _awards_str(longest_road: bool, largest_army: bool) -> str:
    tags = (["[LR]"] if longest_road else []) + (["[LA]"] if largest_army else [])
    return ("  " + " ".join(tags)) if tags else ""


def _fmt_res_full(resources: dict) -> str:
    parts = [
        f"{_RES_ABBR.get(r.name, r.name)}×{n}"
        for r, n in sorted(resources.items(), key=lambda x: x[0].name)
        if n > 0 and r.name != "DESERT"
    ]
    return "  ".join(parts) if parts else "(empty)"


def _fmt_dev_list(cards: list) -> str:
    counts = Counter(card for card, _ in cards)
    parts = [f"{_DEV_ABBR.get(ct.name, ct.name)}×{n}" for ct, n in counts.items()]
    return "  ".join(parts) if parts else "(none)"


def _fmt_stats(vp: int, knights: int, roads: int, settlements: int, cities: int) -> str:
    return (
        f"VP:{vp}  Kn:{knights}"
        f"  Rds:{roads}  Stl:{settlements}  Cty:{cities}"
    )


class PlayerPanel(QFrame):
    def __init__(self, player_id: int) -> None:
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        color = theme.PLAYER_COLORS[player_id % len(theme.PLAYER_COLORS)]
        self.setStyleSheet(f"QFrame {{ border-left: 4px solid {color}; }}")

        self._header = QLabel(f"P{player_id}")
        self._header.setStyleSheet(f"font-weight: bold; color: {color};")
        self._res_label = QLabel()
        self._res_label.setWordWrap(True)
        self._dev_label = QLabel()
        self._dev_label.setWordWrap(True)
        self._stats_label = QLabel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        layout.addWidget(self._header)
        layout.addWidget(self._res_label)
        layout.addWidget(self._dev_label)
        layout.addWidget(self._stats_label)

    def render_full(
        self,
        ps: PlayerState,
        *,
        longest_road: bool = False,
        largest_army: bool = False,
    ) -> None:
        self._header.setText(f"P{int(ps.player_id)}{_awards_str(longest_road, largest_army)}")
        self._res_label.setText("Res: " + _fmt_res_full(ps.resources))
        self._dev_label.setText("Dev: " + _fmt_dev_list(ps.dev_cards_in_hand))
        self._stats_label.setText(
            _fmt_stats(
                ps.victory_points_public, ps.knights_played,
                ps.roads_built, ps.settlements_built, ps.cities_built,
            )
        )

    def render_perspective(
        self,
        row: PlayerPerspectiveState,
        *,
        longest_road: bool = False,
        largest_army: bool = False,
    ) -> None:
        self._header.setText(f"P{int(row.player_id)}{_awards_str(longest_road, largest_army)}")

        if isinstance(row.dev_cards_in_hand, list):
            # This is the viewer's own row — show full detail
            self._res_label.setText("Res: " + _fmt_res_full(row.resources))
            self._dev_label.setText("Dev: " + _fmt_dev_list(row.dev_cards_in_hand))
        else:
            # Opponent row — show totals only
            total = sum(v for v in row.resources.values() if v > 0)
            self._res_label.setText(f"Res: {total} cards")
            self._dev_label.setText(f"Dev: {row.dev_cards_in_hand} cards")

        self._stats_label.setText(
            _fmt_stats(
                row.victory_points_public, row.knights_played,
                row.roads_built, row.settlements_built, row.cities_built,
            )
        )
