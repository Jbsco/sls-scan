import copy
import math

import numpy as np
import pytest

from slscan import calibrate, synthetic

from conftest import SPEC


@pytest.fixture
def result(calibration):
    return calibration


def angle_deg(r):
    return math.degrees(math.acos(min(1.0, (np.trace(r) - 1.0) / 2.0)))


def test_camera_intrinsics_match_the_rig(rig, result):
    k, truth = result.camera.matrix, rig.camera.matrix
    assert abs(k[0, 0] / truth[0, 0] - 1) < 0.003
    assert abs(k[1, 1] / truth[1, 1] - 1) < 0.003
    assert abs(k[0, 2] - truth[0, 2]) < 5 and abs(k[1, 2] - truth[1, 2]) < 5
    assert result.camera.rms_px < 0.3


def test_wide_lenses_fit_k2(result):
    camera = result.camera
    assert calibrate.half_field((camera.width, camera.height), camera.matrix) >= calibrate.NARROW_FIELD
    assert camera.distortion[1] != 0.0


def test_projector_intrinsics_match_the_rig(rig, result):
    k, truth = result.projector.matrix, rig.projector.matrix
    assert abs(k[0, 0] / truth[0, 0] - 1) < 0.003
    assert abs(k[1, 1] / truth[1, 1] - 1) < 0.003
    assert abs(k[0, 2] - truth[0, 2]) < 5 and abs(k[1, 2] - truth[1, 2]) < 5
    assert result.projector.rms_px < 0.5


def test_stereo_extrinsics_match_the_rig(rig, result):
    t, truth = result.stereo.translation, rig.translation
    assert abs(np.linalg.norm(t) / np.linalg.norm(truth) - 1) < 0.005
    cosine = np.dot(t, truth) / (np.linalg.norm(t) * np.linalg.norm(truth))
    assert math.degrees(math.acos(min(1.0, cosine))) < 0.3
    assert angle_deg(result.stereo.rotation @ rig.rotation.T) < 0.25


def test_projector_corners_match_the_projection(rig):
    r, t = synthetic.board_poses(SPEC)[5]
    v = synthetic.calibration_view(rig, SPEC, r, t, np.random.default_rng(1))
    ids, corners = calibrate.detect_corners(v["white"], SPEC)
    columns, column_mask = calibrate.decode_axis(v["columns"], rig.projector.width, 10.0)
    rows, row_mask = calibrate.decode_axis(v["rows"], rig.projector.height, 10.0)
    keep, projected = calibrate.projector_corners(corners, columns, rows, column_mask & row_mask)
    board = calibrate.object_points(SPEC, ids[keep])
    truth = (v["view"].projector_homography @ np.column_stack(
        [board[:, 0], board[:, 1], np.ones(len(board))]).T).T
    error = np.linalg.norm(projected[keep] - truth[:, :2] / truth[:, 2:], axis=1)
    assert keep.sum() >= 30
    assert error.max() < 0.3 and error.mean() < 0.1


def test_calibration_yaml_round_trip(result, tmp_path):
    path = str(tmp_path / "calibration.yaml")
    calibrate.save(path, result)
    back = calibrate.load(path)
    assert back.board == result.board and back.views == result.views
    assert np.allclose(back.camera.matrix, result.camera.matrix)
    assert np.allclose(back.projector.distortion, result.projector.distortion)
    assert np.allclose(back.stereo.translation, result.stereo.translation)
    assert all(v == 0 for v in calibrate.compare(result, back).values())


def test_compare_reports_relative_changes(result):
    changed = copy.deepcopy(result)
    changed.camera.matrix[0, 0] *= 1.002
    direction = changed.stereo.translation / np.linalg.norm(changed.stereo.translation)
    changed.stereo.translation = changed.stereo.translation + 0.0002 * direction
    d = calibrate.compare(result, changed)
    assert d["camera_fx"] == pytest.approx(0.002, rel=1e-6)
    assert d["camera_fy"] == 0
    assert d["baseline_m"] == pytest.approx(0.0002, rel=1e-6)
