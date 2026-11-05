import numpy as np

from slscan import geometry


def test_colored_ply_round_trip(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 1.0]])
    rgb = np.array([[1, 2, 3], [250, 251, 252]], dtype=np.uint8)
    path = str(tmp_path / "colored.ply")
    geometry.write_ply(path, points, rgb)
    assert np.allclose(geometry.read_ply(path), points, atol=1e-7)
    assert np.array_equal(geometry.read_ply_colors(path), rgb)
    geometry.write_ply(path, points)
    assert geometry.read_ply_colors(path) is None
