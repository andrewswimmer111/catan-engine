from domain.board.hex_geometry import axial_tiles, hex_corners, quantize


def test_axial_tiles_radius_2_has_19():
    assert len(axial_tiles(2)) == 19


def test_hex_corners_returns_6_distinct_points():
    corners = hex_corners(0.0, 0.0)
    assert len(corners) == 6
    assert len(set(corners)) == 6


def test_quantize_is_deterministic():
    pt = (1.123456789, -2.987654321)
    assert quantize(pt) == quantize(pt)


def test_quantize_rounds_to_6_decimal_places():
    result = quantize((1.1234567, 2.9876543))
    assert result == (round(1.1234567, 6), round(2.9876543, 6))
