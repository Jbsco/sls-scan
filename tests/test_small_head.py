import numpy as np

from slscan import synthetic


def field_width(pinhole, distance):
    return pinhole.width / pinhole.matrix[0, 0] * distance


def test_small_volume_head_geometry():
    rig = synthetic.small_volume_rig(0.15)
    assert abs(field_width(rig.camera, 0.5) - 0.1257) < 0.001
    assert abs(field_width(rig.projector, 0.5) - 0.15) < 1e-9
    preview = synthetic.small_volume_preview_rig()
    assert field_width(preview.camera, 0.5) > field_width(rig.camera, 0.5)
    assert np.linalg.norm(rig.translation) == np.linalg.norm(preview.translation)


def test_board_poses_scale_with_board_and_distance():
    for r, t in synthetic.board_poses(synthetic.SMALL_BOARD, 0.5):
        centre = t + r @ np.array([0.045, 0.03, 0.0])
        assert 0.44 < centre[2] < 0.58
        assert np.all(np.abs(centre[:2]) < 0.011)


def test_moving_plate_error_grows_with_axial_speed(rig, spec):
    cal = synthetic.true_calibration(rig, spec)
    plate = synthetic.acceptance_scenes()["plate"][0]
    still = synthetic.moving_plate_error(rig, cal, plate, np.zeros(3))
    moving = synthetic.moving_plate_error(rig, cal, plate, np.array([0.0, 0.0, 0.02]))
    assert still < 0.0001
    assert moving > 1.5 * still
