import numpy as np

from slscan import occlusion, synthetic


def test_a_flat_plate_is_measured_by_the_pair():
    rig = synthetic.stereo_head()
    plate = [synthetic.facing_plate([0.0, 0.0, 0.5], [0, 0, -1], [1, 0, 0], 0.03, 0.03)]
    xs, ys = np.meshgrid(np.linspace(-0.025, 0.025, 11), np.linspace(-0.025, 0.025, 11))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.5)])
    normals = np.tile([0.0, 0.0, -1.0], (len(points), 1))
    assert occlusion.coverage(plate, points, normals, rig)["any"] == 1.0
    assert occlusion.coverage(plate, points, -normals, rig)["any"] == 0.0


def test_a_third_camera_sees_a_slot_floor_the_pair_misses():
    rig = synthetic.stereo_head()
    scene, points, normals = occlusion.channel(0.02, 0.02)
    floor = (normals[:, 2] < -0.5) & (np.abs(points[:, 0]) < 0.01)
    pair = occlusion.coverage(scene, points[floor], normals[floor], rig)
    above = occlusion.coverage(scene, points[floor], normals[floor], rig,
                               occlusion.third_cameras()["above"])
    assert 0.6 < pair["any"] < 0.7
    assert above["any"] == 1.0
