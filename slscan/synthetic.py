"""Synthetic views through a virtual camera and projector with known parameters.

A planar board in a known pose is rendered as the camera sees it, and projector patterns
are rendered onto the same plane as the camera sees them. The calibration and decode
stages are tested against these views before any hardware exists.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import calibrate, patterns


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    matrix: np.ndarray


@dataclass(frozen=True)
class Rig:
    """A camera and a projector; rotation and translation take camera to projector frame."""

    camera: Pinhole
    projector: Pinhole
    rotation: np.ndarray
    translation: np.ndarray


def rotation(rx_deg: float, ry_deg: float, rz_deg: float = 0.0) -> np.ndarray:
    """Return the rotation Rz @ Ry @ Rx for angles in degrees."""
    ax, ay, az = (math.radians(a) for a in (rx_deg, ry_deg, rz_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def pinhole(width: int, height: int, f: float, cx: float, cy: float) -> Pinhole:
    return Pinhole(width, height, np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]))


def reference_rig(baseline_m: float = 0.35, crossing_m: float = 1.0) -> Rig:
    """A camera and a projector toed in so their axes cross at crossing_m."""
    camera = pinhole(1280, 960, 1500.0, 643.0, 478.0)
    projector = pinhole(1024, 768, 1400.0, 512.0, 480.0)
    toe = math.degrees(math.atan2(-baseline_m, crossing_m))
    r_projector_in_camera = rotation(0.0, toe)
    r = r_projector_in_camera.T
    t = -r @ np.array([baseline_m, 0.0, 0.0])
    return Rig(camera, projector, r, t)


def board_poses(spec: calibrate.BoardSpec) -> list:
    """Return board poses (rotation, translation) around the crossing point, tilted in turn."""
    size = np.array([spec.squares_x * spec.square_m, spec.squares_y * spec.square_m, 0.0])
    table = [(0, 0, 0, 0.0, 0.0, 1.0), (20, 0, 0, 0.0, 0.03, 1.0), (-20, 0, 0, 0.0, -0.03, 1.0),
             (0, 20, 0, 0.03, 0.0, 1.0), (0, -20, 0, -0.03, 0.0, 1.0),
             (15, 15, 5, 0.04, 0.03, 0.9), (-15, 15, -5, 0.04, -0.03, 0.95),
             (15, -15, 10, -0.04, 0.03, 1.05), (-15, -15, -10, -0.04, -0.03, 1.1),
             (25, 10, 0, 0.0, 0.0, 1.1), (-10, 25, 0, 0.0, 0.0, 0.9), (10, -25, 15, 0.02, 0.02, 1.0)]
    poses = []
    for ax, ay, az, x, y, z in table:
        r = rotation(ax, ay, az)
        poses.append((r, np.array([x, y, z]) - r @ (size / 2.0)))
    return poses


def plane_homography(matrix: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the homography from board-plane metres to image pixels."""
    return matrix @ np.column_stack([r[:, 0], r[:, 1], t])


@dataclass
class View:
    """One board pose as the camera sees it: albedo and the camera-to-projector map."""

    albedo: np.ndarray
    camera_to_projector: np.ndarray
    camera_homography: np.ndarray
    projector_homography: np.ndarray


def view(rig: Rig, spec: calibrate.BoardSpec, r: np.ndarray, t: np.ndarray,
         px_per_square: int = 64, black: float = 0.12, background: float = 0.1) -> View:
    """Render the board print's albedo in the camera and the plane's projector mapping."""
    margin = px_per_square
    printed = calibrate.board_image(spec, px_per_square, margin).astype(np.float32) / 255.0
    printed = black + (1.0 - black) * printed
    k = spec.square_m / px_per_square
    print_to_board = np.array([[k, 0.0, (0.5 - margin) * k], [0.0, k, (0.5 - margin) * k],
                               [0.0, 0.0, 1.0]])
    h_cam = plane_homography(rig.camera.matrix, r, t)
    albedo = cv2.warpPerspective(printed, h_cam @ print_to_board,
                                 (rig.camera.width, rig.camera.height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    h_proj = plane_homography(rig.projector.matrix, rig.rotation @ r, rig.rotation @ t + rig.translation)
    return View(albedo, h_proj @ np.linalg.inv(h_cam), h_cam, h_proj)


def capture(rig: Rig, v: View, pattern: np.ndarray, rng: np.random.Generator,
            ambient: float = 25.0, gain: float = 200.0, noise: float = 1.0) -> np.ndarray:
    """Return the camera image of one projector pattern on the board, as uint8."""
    seen = cv2.warpPerspective(pattern, v.camera_to_projector,
                               (rig.camera.width, rig.camera.height),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    image = v.albedo * (np.float32(ambient) + np.float32(gain / 255.0) * seen)
    image += np.float32(noise) * rng.standard_normal(seen.shape, dtype=np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def calibration_view(rig: Rig, spec: calibrate.BoardSpec, r: np.ndarray, t: np.ndarray,
                     rng: np.random.Generator) -> dict:
    """Render one board pose under white light and under both Gray code stacks."""
    w, h = rig.projector.width, rig.projector.height
    v = view(rig, spec, r, t)
    white = np.full((h, w), 255, dtype=np.uint8)
    return {"white": capture(rig, v, white, rng),
            "columns": np.stack([capture(rig, v, p, rng)
                                 for p in patterns.gray_code_images(w, h, "columns")]),
            "rows": np.stack([capture(rig, v, p, rng)
                              for p in patterns.gray_code_images(w, h, "rows")]),
            "view": v}


def calibration_views(rig: Rig, spec: calibrate.BoardSpec, seed: int = 0):
    """Yield every board pose of board_poses as a calibration view, one at a time."""
    rng = np.random.default_rng(seed)
    for r, t in board_poses(spec):
        yield calibration_view(rig, spec, r, t, rng)
