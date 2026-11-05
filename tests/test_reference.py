import numpy as np
import pytest

from slscan import decode, geometry, patterns, synthetic, triangulate

PITCH = 16


def scan(rig, cal, scene, seed=0):
    captures, view = synthetic.scan_captures(rig, scene, pitch=PITCH, seed=seed)
    columns, mask = decode.decode_columns(captures["gray"], captures["phase"],
                                          rig.projector.width, PITCH)
    points, _ = triangulate.triangulate(columns, mask, cal)
    return points, columns, mask, view


def plate():
    return synthetic.acceptance_scenes()["plate"][0]


def test_unwrap_with_columns_tolerates_gray_code_error():
    true = np.linspace(0.0, 255.0, 1000)
    phase = np.mod(2 * np.pi * true / PITCH, 2 * np.pi)
    gray = np.rint(true) + np.where(np.arange(1000) % 2, 3, -3)
    assert np.allclose(patterns.unwrap_with_columns(phase, gray, PITCH), true)


def test_decode_recovers_projector_columns(rig):
    captures, view = synthetic.scan_captures(rig, [plate()], pitch=PITCH)
    columns, mask = decode.decode_columns(captures["gray"], captures["phase"],
                                          rig.projector.width, PITCH)
    lit = view.lit.reshape(view.shape)
    assert not (mask & ~lit).any()
    assert mask.sum() > 0.99 * lit.sum()
    error = np.abs(columns - view.u.reshape(view.shape))[mask]
    assert np.median(error) < 0.03 and error.max() < 0.3


def test_decode_drops_saturated_phase_samples(rig):
    captures, view = synthetic.scan_captures(rig, [plate()], pitch=PITCH, exposure=4.0)
    columns, mask = decode.decode_columns(captures["gray"], captures["phase"],
                                          rig.projector.width, PITCH)
    saturated = captures["phase"].max(axis=0) >= decode.SATURATED
    lit = view.lit.reshape(view.shape)
    assert np.count_nonzero(saturated & lit) > 0.5 * np.count_nonzero(lit)
    assert not (mask & saturated).any()


def test_true_calibration_places_points_on_the_plate(rig, spec):
    p = plate()
    points, *_ = scan(rig, synthetic.true_calibration(rig, spec), [p])
    distance = (points - p.centre) @ p.axes[:, 2]
    assert np.sqrt(np.mean(distance ** 2)) < 0.0001


def test_flat_plate_acceptance(rig, calibration):
    points, *_ = scan(rig, calibration, [plate()])
    _, _, rms = geometry.fit_plane(points)
    assert rms < 0.0002


def test_sphere_probing(rig, calibration):
    ball = synthetic.acceptance_scenes()["sphere"][0]
    points, *_ = scan(rig, calibration, [ball])
    _, radius, rms = geometry.fit_sphere(points)
    assert abs(radius - ball.radius) < 0.0001
    assert rms < 0.0001


def test_ball_bar_spacing(rig, calibration):
    balls = synthetic.acceptance_scenes()["ball-bar"]
    points, *_ = scan(rig, calibration, balls)
    labels = geometry.split_two(points)
    centres = [geometry.fit_sphere(points[labels == i])[0] for i in range(2)]
    true = np.linalg.norm(balls[0].centre - balls[1].centre)
    assert abs(np.linalg.norm(centres[0] - centres[1]) - true) < 0.0002


def test_ply_round_trip(tmp_path):
    points = np.random.default_rng(0).normal(size=(100, 3))
    path = str(tmp_path / "cloud.ply")
    geometry.write_ply(path, points)
    assert np.allclose(geometry.read_ply(path), points.astype(np.float32))


def test_sphere_fit_recovers_known_sphere():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(2000, 3))
    directions = directions[directions[:, 2] < 0]
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    points = np.array([0.1, 0.2, 1.0]) + 0.03 * directions
    centre, radius, rms = geometry.fit_sphere(points)
    assert np.allclose(centre, [0.1, 0.2, 1.0]) and radius == pytest.approx(0.03) and rms < 1e-9
