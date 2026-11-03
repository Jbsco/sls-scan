import numpy as np

from slscan import register, synthetic, track


def test_fpfh_moves_with_the_points():
    r, t = synthetic.orbit_poses([20.0])[0]
    view = synthetic.point_view(synthetic.static_object(), r, t, np.random.default_rng(1))
    points = track.voxel_average(view, 0.005)
    normals = register.facing_normals(points)
    turn = synthetic.rotation(10, 20, 30)
    moved = register.fpfh(points @ turn.T + [0.1, 0.0, 0.0], normals @ turn.T, 0.025)
    still = register.fpfh(points, normals, 0.025)
    assert np.quantile(np.abs(moved - still).max(axis=1), 0.99) < 1e-9
