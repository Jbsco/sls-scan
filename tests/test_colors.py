import numpy as np
import pytest

from slscan import colors, geometry


def fields(reflectance, ambient=25.0, gain=200.0, shape=(4, 6)):
    r, g, b = reflectance
    black = np.full(shape, ambient, dtype=np.uint8)
    return {"black": black, "red": black + np.uint8(gain * r), "green": black + np.uint8(gain * g),
            "blue": black + np.uint8(gain * b)}


def test_responses_remove_room_light():
    view = fields((0.5, 0.25, 1.0))
    assert np.allclose(colors.responses(view)[0, 0], [100.0, 50.0, 200.0])


def test_white_balance_is_the_median_response_of_a_lit_card():
    view = fields((1.0, 1.0, 1.0))
    view["red"][0, :] = view["black"][0, :]
    assert np.allclose(colors.white_balance(view), [200.0, 200.0, 200.0])
    with pytest.raises(ValueError):
        colors.white_balance(fields((0.0, 0.0, 0.0)))


def test_point_colors_scale_by_the_white_balance():
    view = fields((0.5, 0.25, 1.0))
    pixels = np.array([[1.2, 2.8], [5.0, 0.0]])
    assert np.array_equal(colors.point_colors(view, pixels, [200.0, 200.0, 200.0]),
                          [[128, 64, 255], [128, 64, 255]])
    assert np.array_equal(colors.point_colors(view, pixels), [[100, 50, 200], [100, 50, 200]])


def test_colored_ply_round_trip(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 1.0]])
    rgb = np.array([[1, 2, 3], [250, 251, 252]], dtype=np.uint8)
    path = str(tmp_path / "colored.ply")
    geometry.write_ply(path, points, rgb)
    assert np.allclose(geometry.read_ply(path), points, atol=1e-7)
    assert np.array_equal(geometry.read_ply_colors(path), rgb)
    geometry.write_ply(path, points)
    assert geometry.read_ply_colors(path) is None
