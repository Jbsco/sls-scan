"""Triangulate: each camera pixel's ray meets its projector column's plane.

Projector distortion bends the column planes, so the plane comes from the undistorted
projector point at the row where the previous estimate lands, refined over a few
iterations.
"""

import cv2
import numpy as np

from . import patterns
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


def scan_params(cal: Calibration, pitch: int, steps: int, exposures=None, color: bool = False,
                white=None) -> dict:
    """Return a calibration and the decode settings of decode.decode_columns as the
    application takes them: each device's size, row-major matrix and five distortion
    coefficients, the camera-to-projector rotation (row-major) and translation, and the
    decode settings, with exposures relative to the nominal one when there are several.
    With color, scans end with the projector's fields, and white is a white card's
    response per channel for point colors."""
    def device(intrinsics):
        distortion = np.zeros(5)
        coefficients = np.ravel(intrinsics.distortion)
        if np.any(coefficients[5:]) or intrinsics.matrix[0, 1] != 0:
            raise ValueError("the application takes five distortion terms and no skew")
        distortion[:min(5, len(coefficients))] = coefficients[:5]
        return {"width": int(intrinsics.width), "height": int(intrinsics.height),
                "matrix": np.asarray(intrinsics.matrix, dtype=np.float64).ravel().tolist(),
                "distortion": distortion.tolist()}
    return {"camera": device(cal.camera), "projector": device(cal.projector),
            "rotation": np.asarray(cal.stereo.rotation, dtype=np.float64).ravel().tolist(),
            "translation": np.asarray(cal.stereo.translation, dtype=np.float64).ravel().tolist(),
            "decode": {"planes": patterns.bit_planes(cal.projector.width), "steps": int(steps),
                       "projector_width": int(cal.projector.width), "pitch": float(pitch),
                       "min_contrast": 10.0, "min_modulation": 10.0, "max_disagreement": 1.0,
                       "exposures": [float(x) for x in exposures or []], "color": bool(color),
                       "white": [float(x) for x in white] if white is not None else []}}
