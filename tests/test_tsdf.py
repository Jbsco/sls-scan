from types import SimpleNamespace

import numpy as np

from slscan import geometry, track, tsdf

STEP, WINDOW, SIZE = 4, 8, (320, 240)


def camera():
    f, cx, cy = 4000.0, 160.0, 120.0
    projection = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]])
    return SimpleNamespace(left_rotation=np.eye(3), left_projection=projection, size=SIZE)


def rays(rect):
    """Unit-depth rays through every node of the sample grid."""
    p = rect.left_projection
    ys, xs = np.mgrid[WINDOW:SIZE[1] - WINDOW:STEP, WINDOW:SIZE[0] - WINDOW:STEP]
    return np.column_stack([(xs.ravel() - p[0, 2]) / p[0, 0], (ys.ravel() - p[1, 2]) / p[1, 1],
                            np.ones(xs.size)])


def plane_frame(rect, rotation, translation, point, normal):
    """Grid samples of a world plane seen from a camera posed into the world."""
    d = rays(rect)
    n = rotation.T @ normal
    c = rotation.T @ (point - translation)
    return d * ((c @ n) / (d @ n))[:, None]


def sphere_frame(rect, rotation, translation, centre, radius):
    """Grid samples of the near side of a world sphere seen from a posed camera."""
    d = rays(rect)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    c = rotation.T @ (centre - translation)
    b = d @ c
    disc = b ** 2 - (c @ c - radius ** 2)
    hit = disc > 0
    return d[hit] * (b[hit] - np.sqrt(disc[hit]))[:, None]


def poses():
    """The first camera and two more turned about a target 0.5 m ahead of it."""
    target = np.array([0.0, 0.0, 0.5])
    turned = [track.rodrigues(np.radians(w)) for w in ([4.0, -6.0, 0.0], [-3.0, 5.0, 2.0])]
    return [(np.eye(3), np.zeros(3))] + [(r, target - r @ target) for r in turned]


def test_depth_grid_places_each_sample_at_its_node():
    rect = camera()
    points = plane_frame(rect, np.eye(3), np.zeros(3), np.array([0, 0, 0.5]), np.array([0, 0, 1]))
    grid = tsdf.depth_grid(rect, points, STEP, WINDOW, SIZE)
    assert grid.shape == (len(range(WINDOW, SIZE[1] - WINDOW, STEP)),
                          len(range(WINDOW, SIZE[0] - WINDOW, STEP)))
    assert np.allclose(grid, 0.5)


def test_tilted_plane_fuses_flat_from_two_views():
    rect = camera()
    normal = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
    point = np.array([0.0, 0.0, 0.5])
    frames = [plane_frame(rect, r, t, point, normal) for r, t in poses()]
    _, surface = tsdf.fuse(frames, poses(), rect, STEP, WINDOW, 0.0005, 0.001)
    assert len(surface) > 4000
    assert np.max(np.abs((surface - point) @ normal)) < 0.000003


def test_sphere_keeps_its_radius():
    rect = camera()
    centre, radius = np.array([0.0, 0.0, 0.5]), 0.0125
    frames = [sphere_frame(rect, r, t, centre, radius) for r, t in poses()]
    _, surface = tsdf.fuse(frames, poses(), rect, STEP, WINDOW, 0.0005, 0.001)
    fitted_centre, fitted_radius, rms = geometry.fit_sphere(surface)
    assert len(surface) > 2000 and rms < 0.000005
    assert abs(fitted_radius - radius) < 0.00003
    assert np.linalg.norm(fitted_centre - centre) < 0.00002
