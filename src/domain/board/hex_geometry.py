from __future__ import annotations

import math

RADIUS = 2
SIZE = 1.0
PREC = 6


def axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
    """Convert axial hex coordinates to pointy-top Cartesian (x, y)."""
    x = SIZE * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = SIZE * (3 / 2 * r)
    return x, y


def hex_corners(cx: float, cy: float) -> list[tuple[float, float]]:
    """Return the 6 corners of a pointy-top hex centred at (cx, cy)."""
    corners = []
    for i in range(6):
        angle_rad = math.pi / 6 + math.pi / 3 * i
        corners.append((
            cx + SIZE * math.cos(angle_rad),
            cy + SIZE * math.sin(angle_rad),
        ))
    return corners


def quantize(pt: tuple[float, float]) -> tuple[float, float]:
    """Round a 2-D point to PREC decimal places for deduplication."""
    return (round(pt[0], PREC), round(pt[1], PREC))


def axial_tiles(radius: int = RADIUS) -> list[tuple[int, int]]:
    """Return all axial (q, r) coordinates within the given hex radius."""
    coords: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            coords.append((q, r))
    return coords


def hex_distance(q: int, r: int) -> int:
    """Hex grid distance from the origin to (q, r) in axial coordinates."""
    return (abs(q) + abs(r) + abs(q + r)) // 2
