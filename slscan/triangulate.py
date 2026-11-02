"""Triangulate: each camera pixel's ray meets its projector column's plane.

Projector distortion bends the column planes, so the plane comes from the undistorted
projector point at the row where the previous estimate lands, refined over a few
iterations.
"""

import cv2
import numpy as np

from .calibrate import Calibration


def triangulate(columns: np.ndarray, mask: np.ndarray, cal: Calibration,
                iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return points (N, 3) in metres in the camera frame and their pixels (N, 2) as x, y."""
    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    if len(pixels) == 0:
        return np.empty((0, 3)), pixels
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), cal.camera.matrix,
                                     cal.camera.distortion).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    r, t = cal.stereo.rotation, cal.stereo.translation
    rvec, _ = cv2.Rodrigues(r)
    rotated = rays @ r.T
    u = columns[ys, xs].astype(np.float64)
    v = np.full_like(u, cal.projector.matrix[1, 2])
    points = np.zeros((len(u), 3))
    for _ in range(iterations):
        projector = cv2.undistortPoints(np.column_stack([u, v]).reshape(-1, 1, 2),
                                        cal.projector.matrix,
                                        cal.projector.distortion).reshape(-1, 2)
        normals = np.column_stack([np.ones(len(u)), np.zeros(len(u)), -projector[:, 0]])
        scale = -(normals @ t) / np.einsum("ij,ij->i", normals, rotated)
        points = rays * scale[:, np.newaxis]
        image, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, t, cal.projector.matrix,
                                     cal.projector.distortion)
        v = image.reshape(-1, 2)[:, 1]
    keep = points[:, 2] > 0
    return points[keep], pixels[keep]
