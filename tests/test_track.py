import numpy as np

from slscan import stereo, synthetic, track


def test_rodrigues_matches_axis_angle():
    r = track.rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(track.rodrigues(np.zeros(3)), np.eye(3))


def test_voxel_average_merges_points_in_a_voxel():
    points = np.array([[0.0001, 0.0001, 0.0001], [0.0003, 0.0003, 0.0003], [0.0021, 0.0, 0.0]])
    merged = track.voxel_average(points, 0.0005)
    assert len(merged) == 2
    assert np.allclose(sorted(merged[:, 0]), [0.0002, 0.0021])


def scene_points(rng):
    u, v = rng.uniform(-0.05, 0.05, (2, 4000))
    plane = np.column_stack([u, v, 0.52 + 0.1 * u])
    d = rng.normal(size=(3000, 3))
    d[:, 2] = -np.abs(d[:, 2])
    sphere = np.array([0.01, -0.01, 0.5]) + 0.0125 * d / np.linalg.norm(d, axis=1)[:, None]
    return np.vstack([plane, sphere])


def test_icp_recovers_a_rigid_motion():
    rng = np.random.default_rng(0)
    target = scene_points(rng)
    true_r = track.rodrigues(np.array([0.002, -0.003, 0.001]))
    true_t = np.array([0.001, -0.0005, 0.0008])
    source = (target - true_t) @ true_r
    r, t, rms = track.icp(source, target, track.normals(target), np.eye(3), np.zeros(3))
    assert np.allclose(r, true_r, atol=1e-6) and np.allclose(t, true_t, atol=1e-6)
    assert rms < 1e-6


def test_short_sweep_keeps_ball_bar_spacing():
    rig = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(rig, synthetic.SMALL_BOARD)
    pattern = synthetic.random_dots(rig.light.width, rig.light.height, 3.0, 1.0,
                                    np.random.default_rng(0))
    scene = synthetic.sweep_scene()
    poses = synthetic.sweep_poses(5, yaw_deg=5.0, pitch_deg=2.0)
    frames = [stereo.depth_points(cal, im["left"], im["right"], 0.4, 0.6, step=16)[0]
              for im in synthetic.sweep_captures(rig, scene, poses, pattern)]
    result = track.sweep_bench(rig, cal, scene, poses, frames)
    assert abs(result["ball_bar_error_m"]) < 0.00002
    assert result["max_translation_error_m"] < 0.0005
    assert result["max_rotation_error_deg"] < 0.2
