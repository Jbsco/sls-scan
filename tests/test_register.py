import numpy as np

from slscan import geometry, register, synthetic, track


def angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def test_batch_rigid_fit_matches_rigid_fit():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((4, 5, 3))
    target = source @ synthetic.rotation(20, -35, 50).T + [0.1, -0.2, 0.3]
    target += 1e-3 * rng.standard_normal(source.shape)
    r, t = register.batch_rigid_fit(source, target)
    for k in range(4):
        rk, tk = geometry.rigid_fit(source[k], target[k])
        assert np.allclose(r[k], rk) and np.allclose(t[k], tk)


def test_fpfh_moves_with_the_points():
    r, t = synthetic.orbit_poses([20.0])[0]
    view = synthetic.point_view(synthetic.static_object(), r, t, np.random.default_rng(1))
    points = track.voxel_average(view, 0.005)
    normals = register.facing_normals(points)
    turn = synthetic.rotation(10, 20, 30)
    moved = register.fpfh(points @ turn.T + [0.1, 0.0, 0.0], normals @ turn.T, 0.025)
    still = register.fpfh(points, normals, 0.025)
    assert np.quantile(np.abs(moved - still).max(axis=1), 0.99) < 1e-9


def test_views_far_apart_register_from_geometry():
    scene = synthetic.static_object()
    rng = np.random.default_rng(2)
    (ra, ta), (rb, tb) = synthetic.orbit_poses([0.0, 120.0])
    a = synthetic.point_view(scene, ra, ta, rng)
    b = synthetic.point_view(scene, rb, tb, rng)
    r, t, _ = register.register(b, a, 0.005)
    r_true = ra @ rb.T
    assert angle_deg(r.T @ r_true) < 0.01
    assert np.linalg.norm(t - (ta - r_true @ tb)) < 0.0001
