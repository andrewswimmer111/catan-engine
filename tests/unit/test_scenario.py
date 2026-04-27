"""Unit tests for the spiral tile order and scenario number assignment."""

from __future__ import annotations

from collections import Counter

import pytest

from domain.board.layout import (
    build_standard_board,
    standard_board_outer_ring_tile_ids,
    standard_board_spiral_tile_ids,
)
from domain.board.scenario import (
    STANDARD_SPIRAL_NUMBERS,
    assign_random_scenario,
)
from domain.board.topology import BoardTopology
from domain.engine.randomizer import SeededRandomizer
from domain.ids import TileID


def _hex_neighbors(topology: BoardTopology, tid: TileID) -> set[TileID]:
    """Return tile IDs that share a hex edge (≥ 2 vertices) with ``tid``."""
    own_vertices = {
        vid for vid, vx in topology.vertices.items() if tid in vx.adjacent_tiles
    }
    counts: Counter[TileID] = Counter()
    for vid in own_vertices:
        for other in topology.vertices[vid].adjacent_tiles:
            if other != tid:
                counts[other] += 1
    return {other for other, n in counts.items() if n >= 2}


def _spiral_continuity_holds(topology: BoardTopology, spiral: list[TileID]) -> bool:
    return all(b in _hex_neighbors(topology, a) for a, b in zip(spiral, spiral[1:]))


def _find_spiral_start_in_topology(topology: BoardTopology) -> TileID:
    """
    Return the unique outer-ring tile from which a clockwise spiral walk over
    the (already-assigned) topology produces :data:`STANDARD_SPIRAL_NUMBERS`
    along its non-desert tiles.
    """
    for candidate in standard_board_outer_ring_tile_ids():
        spiral = standard_board_spiral_tile_ids(start=candidate)
        placed = tuple(
            topology.tiles[tid].dice_number
            for tid in spiral
            if not topology.tiles[tid].is_desert()
        )
        if placed == STANDARD_SPIRAL_NUMBERS:
            return candidate
    raise AssertionError(
        "no outer-ring start reproduces STANDARD_SPIRAL_NUMBERS along the spiral"
    )


def test_standard_board_spiral_returns_19_unique_tile_ids() -> None:
    spiral = standard_board_spiral_tile_ids()
    assert len(spiral) == 19
    assert len(set(spiral)) == 19


def test_standard_board_outer_ring_returns_12_tile_ids() -> None:
    outer = standard_board_outer_ring_tile_ids()
    assert len(outer) == 12
    assert len(set(outer)) == 12


@pytest.mark.parametrize(
    "start", standard_board_outer_ring_tile_ids()
)
def test_spiral_path_is_geometrically_continuous_for_any_outer_start(
    start: TileID,
) -> None:
    """Every consecutive pair of tile IDs in the spiral must share a hex edge,
    regardless of which outer-ring tile we start on."""
    topology = build_standard_board()
    spiral = standard_board_spiral_tile_ids(start=start)
    assert spiral[0] == start
    assert _spiral_continuity_holds(topology, spiral)


def test_standard_board_spiral_visits_outer_then_middle_then_centre() -> None:
    """First 12 tiles are on the outer ring, next 6 are middle-ring, last is centre."""
    topology = build_standard_board()
    spiral = standard_board_spiral_tile_ids()

    outer = spiral[:12]
    middle = spiral[12:18]
    centre = spiral[18]

    for tid in outer:
        assert len(_hex_neighbors(topology, tid)) in (3, 4)
    for tid in middle:
        assert len(_hex_neighbors(topology, tid)) == 6
    assert _hex_neighbors(topology, centre) == set(middle)


def test_spiral_rejects_start_off_outer_ring() -> None:
    centre = next(
        tid
        for tid in build_standard_board().tiles
        if tid not in standard_board_outer_ring_tile_ids()
    )
    with pytest.raises(ValueError, match="not on the outer ring"):
        standard_board_spiral_tile_ids(start=centre)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 99])
def test_dice_numbers_match_spec_sequence_along_spiral(seed: int) -> None:
    """For any seed there exists an outer-ring start such that walking the
    spiral skipping the desert produces the spec sequence exactly."""
    raw = build_standard_board()
    topology = assign_random_scenario(raw, SeededRandomizer(seed))
    # Raises AssertionError if no start matches.
    _find_spiral_start_in_topology(topology)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 99])
def test_desert_tile_has_no_dice_number(seed: int) -> None:
    raw = build_standard_board()
    topology = assign_random_scenario(raw, SeededRandomizer(seed))
    desert = next(t for t in topology.tiles.values() if t.is_desert())
    assert desert.dice_number is None


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 99])
def test_eighteen_non_desert_tiles_carry_full_number_multiset(seed: int) -> None:
    raw = build_standard_board()
    topology = assign_random_scenario(raw, SeededRandomizer(seed))
    numbers = [
        t.dice_number for t in topology.tiles.values() if not t.is_desert()
    ]
    assert len(numbers) == 18
    assert Counter(numbers) == Counter(STANDARD_SPIRAL_NUMBERS)


def test_dice_numbers_are_deterministic_across_runs_with_same_seed() -> None:
    """Two scenarios with identical seeds produce identical numbers and starts."""
    raw = build_standard_board()
    a = assign_random_scenario(raw, SeededRandomizer(0))
    b = assign_random_scenario(raw, SeededRandomizer(0))
    for tid in a.tiles:
        assert a.tiles[tid].dice_number == b.tiles[tid].dice_number


def test_spiral_start_varies_across_seeds() -> None:
    """Different seeds anchor the spiral on different outer-ring tiles."""
    raw = build_standard_board()
    starts: set[TileID] = set()
    for seed in range(40):
        topology = assign_random_scenario(raw, SeededRandomizer(seed))
        starts.add(_find_spiral_start_in_topology(topology))
    # 40 seeds across a 12-element start space — virtually impossible to
    # collapse to one bucket unless randomization is broken.
    assert len(starts) > 1


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 99])
def test_assigned_scenario_numbers_lie_on_a_continuous_spiral(seed: int) -> None:
    """For any seed, the start used by scenario.py yields a continuous spiral."""
    raw = build_standard_board()
    topology = assign_random_scenario(raw, SeededRandomizer(seed))
    start = _find_spiral_start_in_topology(topology)
    spiral = standard_board_spiral_tile_ids(start=start)
    assert _spiral_continuity_holds(topology, spiral)
