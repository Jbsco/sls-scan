import math

import numpy as np
import pytest

from slscan import calibrate, geometry, stereo, synthetic

BOARD = synthetic.SMALL_BOARD


@pytest.fixture(scope="module")
def head():
    return synthetic.stereo_head()


@pytest.fixture(scope="module")
def stereo_calibration(head):
    return calibrate.calibrate_stereo(synthetic.stereo_calibration_views(head, BOARD), BOARD)


@pytest.fixture(scope="module")
def pattern(head):
    return synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0,
                                 np.random.default_rng(0))


def capture(head, scene, pattern):
    return synthetic.stereo_capture(head, scene, pattern, np.random.default_rng(1))


def test_stereo_calibration_matches_the_head(head, stereo_calibration):
    r, t = synthetic.stereo_extrinsics(head)
    for side in (stereo_calibration.left, stereo_calibration.right):
        assert abs(side.matrix[0, 0] / head.left.matrix[0, 0] - 1) < 0.003
        assert side.rms_px < 0.3
    assert abs(np.linalg.norm(stereo_calibration.stereo.translation) - np.linalg.norm(t)) < 0.0005
    d = stereo_calibration.stereo.rotation @ r.T
    assert math.degrees(math.acos(min(1.0, (np.trace(d) - 1) / 2))) < 0.2


def test_stereo_calibration_yaml_round_trip(stereo_calibration, tmp_path):
    path = str(tmp_path / "stereo.yaml")
    calibrate.save_stereo(path, stereo_calibration)
    back = calibrate.load_stereo(path)
    assert back.board == stereo_calibration.board and back.views == stereo_calibration.views
    assert np.allclose(back.right.matrix, stereo_calibration.right.matrix)
    assert np.allclose(back.stereo.translation, stereo_calibration.stereo.translation)


def test_plain_plate_is_flat(head, stereo_calibration, pattern):
    images = capture(head, synthetic.stereo_scenes()["plate"], pattern)
    points, _ = stereo.depth_points(stereo_calibration, images["left"], images["right"], 0.4, 0.6)
    assert len(points) > 3000
    assert geometry.fit_plane(points)[2] < 0.00002


def test_ball_bar_spacing(head, stereo_calibration, pattern):
    bar = synthetic.stereo_scenes()["ball-bar"]
    images = capture(head, bar, pattern)
    points, _ = stereo.depth_points(stereo_calibration, images["left"], images["right"], 0.4, 0.6)
    labels = geometry.split_two(points)
    centres = [geometry.fit_sphere(points[labels == i])[0] for i in (0, 1)]
    truth = np.linalg.norm(bar[0].centre - bar[1].centre)
    assert abs(np.linalg.norm(centres[0] - centres[1]) - truth) < 0.00002


def test_window_curvature_bias_shrinks_with_window(head, pattern):
    truth = synthetic.true_stereo_calibration(head, BOARD)
    ball = synthetic.stereo_scenes()["sphere"][0]
    centre = synthetic.to_left(head, ball.centre[None])[0]
    images = capture(head, [ball], pattern)
    offsets = []
    for window in (21, 11):
        points, _ = stereo.depth_points(truth, images["left"], images["right"], 0.4, 0.6,
                                        step=6, window=window)
        offsets.append(np.mean(np.linalg.norm(points - centre, axis=1) - ball.radius))
    assert -0.00006 < offsets[0] < -0.00002
    assert offsets[0] < offsets[1] < 0.0


def test_unlit_scene_gives_no_points(head, stereo_calibration):
    dark = np.zeros((head.light.height, head.light.width), dtype=np.float32)
    images = capture(head, synthetic.stereo_scenes()["plate"], dark)
    points, _ = stereo.depth_points(stereo_calibration, images["left"], images["right"], 0.4, 0.6)
    assert len(points) == 0


def test_second_order_windows_remove_curvature_bias(head, pattern):
    truth = synthetic.true_stereo_calibration(head, BOARD)
    ball = synthetic.stereo_scenes()["sphere"][0]
    centre = synthetic.to_left(head, ball.centre[None])[0]
    images = capture(head, [ball], pattern)
    points, _ = stereo.depth_points(truth, images["left"], images["right"], 0.4, 0.6, step=6,
                                    window=25, order=2)
    offset = np.mean(np.linalg.norm(points - centre, axis=1) - ball.radius)
    _, radius, _ = geometry.fit_sphere(points)
    assert len(points) > 400
    assert abs(offset) < 0.000005
    assert abs(radius - ball.radius) < 0.00002


def test_second_order_matches_flag_the_kept_samples(head, pattern):
    truth = synthetic.true_stereo_calibration(head, BOARD)
    images = capture(head, synthetic.stereo_scenes()["sphere"], pattern)
    _, rl, rr, left, right, warps = stereo.affine_matches(truth, images["left"], images["right"],
                                                          0.4, 0.6, step=12, window=25)
    refined, kept = stereo.second_order_matches(rl, rr, left, right, warps, 25)
    kept_left, kept_right = stereo.refine_second_order(rl, rr, left, right, warps, 25)
    assert refined.shape == right.shape and kept.shape == (len(left),)
    assert kept.sum() > 0.9 * len(left)
    assert np.array_equal(kept_left, left[kept])
    assert np.array_equal(kept_right, refined[kept])


def test_rectification_params_reproduce_the_maps(stereo_calibration):
    rect = stereo.rectify(stereo_calibration)
    params = stereo.rectification_params(stereo_calibration, rect)
    w, h = rect.size
    j, i = np.meshgrid(np.arange(0, w, 37.0), np.arange(0, h, 29.0))
    for side, maps in (("left", rect.left_maps), ("right", rect.right_maps)):
        p = params[side]
        ir = np.array(p["inverse"]).reshape(3, 3)
        k1, k2, p1, p2, k3 = p["distortion"]
        hx, hy, hw = (ir[r, 0] * j + ir[r, 1] * i + ir[r, 2] for r in range(3))
        x, y = hx / hw, hy / hw
        r2 = x * x + y * y
        radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
        u = p["focal"][0] * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + p["centre"][0]
        v = p["focal"][1] * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + p["centre"][1]
        rows, cols = i.astype(int), j.astype(int)
        assert np.max(np.abs(u - maps[0][rows, cols])) < 0.001
        assert np.max(np.abs(v - maps[1][rows, cols])) < 0.001
