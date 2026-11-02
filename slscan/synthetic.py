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


def toed_in_rig(camera: Pinhole, projector: Pinhole, baseline_m: float, crossing_m: float) -> Rig:
    """A camera and a projector toed in so their axes cross at crossing_m."""
    toe = math.degrees(math.atan2(-baseline_m, crossing_m))
    r = rotation(0.0, toe).T
    t = -r @ np.array([baseline_m, 0.0, 0.0])
    return Rig(camera, projector, r, t)


def reference_rig(baseline_m: float = 0.35, crossing_m: float = 1.0) -> Rig:
    """The 1 m rig: a 1280 × 960 camera at f = 1500 px and a 1024 × 768 projector."""
    return toed_in_rig(pinhole(1280, 960, 1500.0, 643.0, 478.0),
                       pinhole(1024, 768, 1400.0, 512.0, 480.0), baseline_m, crossing_m)


def small_volume_rig(projector_width_m: float = 0.15) -> Rig:
    """The small-volume head: the HQ camera in its 2 × 2 binned mode (2028 × 1520, 3.1 µm)
    behind a 25 mm lens, f = 8,065 px, and a 1280 × 800 projector whose image is
    projector_width_m wide at 0.5 m, on a 0.22 m baseline crossing at 0.5 m."""
    f_projector = 1280 * 0.5 / projector_width_m
    return toed_in_rig(pinhole(2028, 1520, 25.0 / 0.0031, 1016.0, 758.0),
                       pinhole(1280, 800, f_projector, 640.0, 400.0), 0.22, 0.5)


def small_volume_preview_rig(projector_width_m: float = 0.15) -> Rig:
    """The small-volume head's preview: the Global Shutter camera (1456 × 1088, 3.45 µm)
    behind a 16 mm lens, f = 4,638 px, with the projector of small_volume_rig."""
    f_projector = 1280 * 0.5 / projector_width_m
    return toed_in_rig(pinhole(1456, 1088, 16.0 / 0.00345, 728.0, 544.0),
                       pinhole(1280, 800, f_projector, 640.0, 400.0), 0.22, 0.5)


SMALL_BOARD = calibrate.BoardSpec(9, 6, 0.010, 0.0075)


def board_poses(spec: calibrate.BoardSpec, crossing_m: float = 1.0) -> list:
    """Return board poses (rotation, translation) around the crossing point, tilted in turn.

    Offsets scale with the board width against a 0.36 m board, and distances with crossing_m.
    """
    size = np.array([spec.squares_x * spec.square_m, spec.squares_y * spec.square_m, 0.0])
    k = size[0] / 0.36
    table = [(0, 0, 0, 0.0, 0.0, 1.0), (20, 0, 0, 0.0, 0.03, 1.0), (-20, 0, 0, 0.0, -0.03, 1.0),
             (0, 20, 0, 0.03, 0.0, 1.0), (0, -20, 0, -0.03, 0.0, 1.0),
             (15, 15, 5, 0.04, 0.03, 0.9), (-15, 15, -5, 0.04, -0.03, 0.95),
             (15, -15, 10, -0.04, 0.03, 1.05), (-15, -15, -10, -0.04, -0.03, 1.1),
             (25, 10, 0, 0.0, 0.0, 1.1), (-10, 25, 0, 0.0, 0.0, 0.9), (10, -25, 15, 0.02, 0.02, 1.0)]
    poses = []
    for ax, ay, az, x, y, z in table:
        r = rotation(ax, ay, az)
        poses.append((r, np.array([k * x, k * y, crossing_m * z]) - r @ (size / 2.0)))
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
         px_per_square: int = 0, black: float = 0.12, background: float = 0.1) -> View:
    """Render the board print's albedo in the camera and the plane's projector mapping.

    The print is rendered at px_per_square, or when it is 0 at about the camera's sampling
    of a square at the board's distance, so the warp neither aliases nor blurs it much.
    """
    if px_per_square <= 0:
        seen = rig.camera.matrix[0, 0] * spec.square_m / max(float(t[2]), 1e-6)
        px_per_square = int(min(256, max(64, round(seen))))
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


def calibration_views(rig: Rig, spec: calibrate.BoardSpec, seed: int = 0,
                      crossing_m: float = 1.0):
    """Yield every board pose of board_poses as a calibration view, one at a time."""
    rng = np.random.default_rng(seed)
    for r, t in board_poses(spec, crossing_m):
        yield calibration_view(rig, spec, r, t, rng)


@dataclass(frozen=True)
class Plate:
    """A flat rectangle: centre, axes as rotation columns (u, v, normal), half sizes."""

    centre: np.ndarray
    axes: np.ndarray
    half_u: float
    half_v: float
    albedo: float = 0.8

    def hit(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        normal = self.axes[:, 2]
        denominator = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.centre - origin) @ normal) / denominator
        local = (origin + t[:, np.newaxis] * directions - self.centre) @ self.axes[:, :2]
        inside = (np.abs(local[:, 0]) <= self.half_u) & (np.abs(local[:, 1]) <= self.half_v)
        return np.where(np.isfinite(t) & (t > 0) & inside, t, np.inf)


@dataclass(frozen=True)
class Ball:
    centre: np.ndarray
    radius: float
    albedo: float = 0.8

    def hit(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        offset = origin - self.centre
        a = np.einsum("ij,ij->i", directions, directions)
        b = 2.0 * directions @ offset
        c = offset @ offset - self.radius ** 2
        disc = b * b - 4.0 * a * c
        with np.errstate(invalid="ignore"):
            t = (-b - np.sqrt(disc)) / (2.0 * a)
        return np.where((disc >= 0) & (t > 0), t, np.inf)


def nearest_hit(scene: list, origin: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ray parameter of the nearest hit (inf for none) and its albedo."""
    t = np.full(len(directions), np.inf)
    albedo = np.zeros(len(directions))
    for shape in scene:
        ts = shape.hit(origin, directions)
        closer = ts < t
        t[closer] = ts[closer]
        albedo[closer] = shape.albedo
    return t, albedo


@dataclass
class SceneView:
    """A scene as the camera sees it: points, albedo, projector coordinates and lighting."""

    points: np.ndarray
    albedo: np.ndarray
    u: np.ndarray
    v: np.ndarray
    lit: np.ndarray
    shape: tuple


def scene_view(rig: Rig, scene: list) -> SceneView:
    """Ray-cast a scene from the camera and find where the projector lights it."""
    h, w = rig.camera.height, rig.camera.width
    ys, xs = np.mgrid[0:h, 0:w]
    pixels = np.column_stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    rays = pixels @ np.linalg.inv(rig.camera.matrix).T
    t, albedo = nearest_hit(scene, np.zeros(3), rays)
    hit = np.isfinite(t)
    points = np.where(hit[:, np.newaxis], rays * np.where(hit, t, 0.0)[:, np.newaxis], 0.0)
    in_projector = points @ rig.rotation.T + rig.translation
    with np.errstate(divide="ignore", invalid="ignore"):
        image = in_projector @ rig.projector.matrix.T
        u = image[:, 0] / image[:, 2]
        v = image[:, 1] / image[:, 2]
    inside = hit & (in_projector[:, 2] > 0) & (u >= -0.5) & (u <= rig.projector.width - 0.5) \
        & (v >= -0.5) & (v <= rig.projector.height - 0.5)
    centre = -rig.rotation.T @ rig.translation
    shadow_t, _ = nearest_hit(scene, centre, points - centre)
    lit = inside & (shadow_t >= 1.0 - 1e-9)
    return SceneView(points, albedo, u, v, lit, (h, w))


def capture_scene(sv: SceneView, values: np.ndarray, rng: np.random.Generator,
                  ambient: float = 25.0, gain: float = 200.0, noise: float = 1.0) -> np.ndarray:
    """Return the camera image, as uint8, of projector intensities (0 to 255) per pixel."""
    lighting = ambient + gain * np.where(sv.lit, values, 0.0) / 255.0
    image = sv.albedo * lighting + noise * rng.standard_normal(len(values))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(sv.shape)


def scan_captures(rig: Rig, scene: list, pitch: int = 16, steps: int = 4,
                  seed: int = 0) -> tuple[dict, SceneView]:
    """Render the column Gray code stack and the phase-shift stack of one static view.

    Gray code stripes are sampled at the nearest projector pixel, so their edges stay
    sharp. Sinusoids are evaluated at the continuous projector column, as a slightly
    defocused projector shows them.
    """
    rng = np.random.default_rng(seed)
    sv = scene_view(rig, scene)
    w, h = rig.projector.width, rig.projector.height
    column = np.clip(np.rint(sv.u), 0, w - 1).astype(np.int64)
    column[~sv.lit] = 0
    gray = np.stack([capture_scene(sv, p[0][column].astype(np.float64), rng)
                     for p in patterns.gray_code_images(w, 1, "columns")])
    phase = np.stack([capture_scene(sv, 127.5 + 127.5 * np.cos(
        2.0 * math.pi * (np.where(sv.lit, sv.u, 0.0) / pitch + i / steps)), rng)
        for i in range(steps)])
    return {"gray": gray, "phase": phase}, sv


def true_calibration(rig: Rig, spec: calibrate.BoardSpec) -> calibrate.Calibration:
    """Return the rig's own parameters as a calibration with zero distortion."""
    def intrinsics(p):
        return calibrate.Intrinsics(p.width, p.height, p.matrix.copy(), np.zeros(5), 0.0)
    return calibrate.Calibration(spec, intrinsics(rig.camera), intrinsics(rig.projector),
                                 calibrate.Stereo(rig.rotation.copy(), rig.translation.copy(), 0.0), 0)


def acceptance_scenes() -> dict:
    """Return the virtual bench scenes: a flat plate, a sphere and a ball bar."""
    return {"plate": [Plate(np.array([0.0, 0.0, 1.0]), rotation(10, -15), 0.25, 0.25)],
            "sphere": [Ball(np.array([0.02, -0.01, 1.0]), 0.05)],
            "ball-bar": [Ball(np.array([-0.15, 0.02, 1.0]), 0.025),
                         Ball(np.array([0.15, -0.02, 1.02]), 0.025)]}


def small_volume_scenes() -> dict:
    """Return the virtual bench scenes scaled to the small-volume head at 0.5 m: an 80 mm
    plate, a 25 mm sphere and a ball bar of two 12 mm spheres about 80 mm apart."""
    return {"plate": [Plate(np.array([0.0, 0.0, 0.5]), rotation(10, -15), 0.04, 0.04)],
            "sphere": [Ball(np.array([0.003, -0.002, 0.5]), 0.0125)],
            "ball-bar": [Ball(np.array([-0.04, 0.004, 0.5]), 0.006),
                         Ball(np.array([0.04, -0.004, 0.502]), 0.006)]}

