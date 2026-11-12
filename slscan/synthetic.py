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


def board_poses(spec: calibrate.BoardSpec, crossing_m: float = 1.0,
                rng: np.random.Generator | None = None, jitter_deg: float = 0.0,
                jitter_m: float = 0.0) -> list:
    """Return board poses (rotation, translation) around the crossing point, tilted in turn.

    Offsets scale with the board width against a 0.36 m board, and distances with crossing_m.
    With rng, each pose's three angles and its position move by uniform amounts up to
    jitter_deg and jitter_m, as a person holding the board would place it.
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
        offset = np.zeros(3)
        if rng is not None:
            ax, ay, az = np.array([ax, ay, az]) + rng.uniform(-jitter_deg, jitter_deg, 3)
            offset = rng.uniform(-jitter_m, jitter_m, 3)
        r = rotation(ax, ay, az)
        poses.append((r, np.array([k * x, k * y, crossing_m * z]) + offset - r @ (size / 2.0)))
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
    dots: tuple = ()
    dot_radius: float = 0.004
    dot_albedo: float = 0.05
    tint: tuple = (1.0, 1.0, 1.0)

    def shade(self, origin: np.ndarray, directions: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Albedo at each hit: dot_albedo within dot_radius of a dot centre (u, v in the
        plate's axes), albedo elsewhere."""
        albedo = np.full(len(t), self.albedo)
        if not self.dots:
            return albedo
        from scipy.spatial import cKDTree

        hit = np.isfinite(t)
        local = (origin + t[hit, np.newaxis] * directions[hit] - self.centre) @ self.axes[:, :2]
        distance, _ = cKDTree(np.asarray(self.dots)).query(local,
                                                           distance_upper_bound=self.dot_radius)
        albedo[np.nonzero(hit)[0][np.isfinite(distance)]] = self.dot_albedo
        return albedo

    def hit(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        normal = self.axes[:, 2]
        denominator = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.centre - origin) @ normal) / denominator
        with np.errstate(invalid="ignore"):
            local = (origin + t[:, np.newaxis] * directions - self.centre) @ self.axes[:, :2]
        inside = (np.abs(local[:, 0]) <= self.half_u) & (np.abs(local[:, 1]) <= self.half_v)
        return np.where(np.isfinite(t) & (t > 0) & inside, t, np.inf)


@dataclass(frozen=True)
class Ball:
    centre: np.ndarray
    radius: float
    albedo: float = 0.8
    tint: tuple = (1.0, 1.0, 1.0)

    def shade(self, origin: np.ndarray, directions: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.full(len(t), self.albedo)

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
        albedo[closer] = shape.shade(origin, directions[closer], ts[closer])
    return t, albedo


def hit_tints(scene: list, origin: np.ndarray, directions: np.ndarray,
              t: np.ndarray) -> np.ndarray:
    """Return the red, green and blue tint (N, 3) of the shape each ray hits at t, white
    where it hits none."""
    tints = np.ones((len(directions), 3))
    for shape in scene:
        tints[shape.hit(origin, directions) == t] = shape.tint
    return tints


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
                  ambient: float = 25.0, gain: float = 200.0, noise: float = 1.0,
                  exposure: float = 1.0) -> np.ndarray:
    """Return the camera image, as uint8, of projector intensities (0 to 255) per pixel,
    with the light scaled by exposure before the sensor clips it."""
    lighting = ambient + gain * np.where(sv.lit, values, 0.0) / 255.0
    image = exposure * sv.albedo * lighting + noise * rng.standard_normal(len(values))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(sv.shape)


def scan_captures(rig: Rig, scene: list, pitch: int = 16, steps: int = 4,
                  seed: int = 0, exposure: float = 1.0) -> tuple[dict, SceneView]:
    """Render the column Gray code stack and the phase-shift stack of one static view at
    an exposure relative to the nominal one.

    Gray code stripes are sampled at the nearest projector pixel, so their edges stay
    sharp. Sinusoids are evaluated at the continuous projector column, as a slightly
    defocused projector shows them.
    """
    rng = np.random.default_rng(seed)
    sv = scene_view(rig, scene)
    w, h = rig.projector.width, rig.projector.height
    column = np.clip(np.rint(sv.u), 0, w - 1).astype(np.int64)
    column[~sv.lit] = 0
    gray = np.stack([capture_scene(sv, p[0][column].astype(np.float64), rng, exposure=exposure)
                     for p in patterns.gray_code_images(w, 1, "columns")])
    phase = np.stack([capture_scene(sv, 127.5 + 127.5 * np.cos(
        2.0 * math.pi * (np.where(sv.lit, sv.u, 0.0) / pitch + i / steps)), rng,
        exposure=exposure) for i in range(steps)])
    return {"gray": gray, "phase": phase}, sv


def color_captures(rig: Rig, scene: list, seed: int = 0, ambient: float = 25.0,
                   gain: float = 200.0, noise: float = 1.0) -> dict:
    """Render the camera under a black projector field and under full red, green and blue
    fields, as "black", "red", "green" and "blue". Room light reaches the mono camera at a
    surface's mean tint, and each field at the tint of its own channel."""
    rng = np.random.default_rng(seed)
    sv = scene_view(rig, scene)
    h, w = sv.shape
    ys, xs = np.mgrid[0:h, 0:w]
    rays = np.column_stack([xs.ravel(), ys.ravel(), np.ones(h * w)]) @ \
        np.linalg.inv(rig.camera.matrix).T
    t, _ = nearest_hit(scene, np.zeros(3), rays)
    tint = hit_tints(scene, np.zeros(3), rays, t)
    room = sv.albedo * ambient * tint.mean(axis=1)

    def capture(field):
        image = room + field + noise * rng.standard_normal(h * w)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(h, w)
    images = {"black": capture(0.0)}
    for k, name in enumerate(("red", "green", "blue")):
        images[name] = capture(np.where(sv.lit, sv.albedo * gain * tint[:, k], 0.0))
    return images


def exposure_captures(rig: Rig, scene: list, exposures, pitch: int = 16, steps: int = 4,
                      seed: int = 0) -> list:
    """Render scan_captures at each exposure, with noise seeded per exposure; the first
    exposure keeps seed, so one exposure renders as scan_captures does."""
    return [scan_captures(rig, scene, pitch, steps, seed + k, x)[0]
            for k, x in enumerate(exposures)]


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


def strip_scenes() -> dict:
    """Return a scene for exposure tests: the acceptance plate as three strips side by
    side in one plane, dark (albedo 0.04), middle (0.8) and bright (2.5), whose dark strip
    lacks modulation and whose bright strip saturates at the nominal exposure."""
    centre, axes = np.array([0.0, 0.0, 1.0]), rotation(10, -15)
    third = 0.5 / 3.0
    return {"strips": [Plate(centre + axes[:, 0] * k * third, axes, third / 2.0, 0.25, albedo)
                       for k, albedo in ((-1, 0.04), (0, 0.8), (1, 2.5))]}


def color_scenes() -> dict:
    """Return scenes for color tests: the acceptance plate as red, green and blue strips of
    albedo 0.8, each with a quarter of the other channels, and a white card of albedo 1."""
    centre, axes = np.array([0.0, 0.0, 1.0]), rotation(10, -15)
    third = 0.5 / 3.0
    tints = ((1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.25, 1.0))
    return {"colors": [Plate(centre + axes[:, 0] * k * third, axes, third / 2.0, 0.25, 0.8,
                             tint=tint) for k, tint in zip((-1, 0, 1), tints)],
            "white": [Plate(centre, axes, 0.25, 0.25, 1.0)]}


def small_volume_scenes() -> dict:
    """Return the virtual bench scenes scaled to the small-volume head at 0.5 m: an 80 mm
    plate, a 25 mm sphere and a ball bar of two 12 mm spheres about 80 mm apart."""
    return {"plate": [Plate(np.array([0.0, 0.0, 0.5]), rotation(10, -15), 0.04, 0.04)],
            "sphere": [Ball(np.array([0.003, -0.002, 0.5]), 0.0125)],
            "ball-bar": [Ball(np.array([-0.04, 0.004, 0.5]), 0.006),
                         Ball(np.array([0.04, -0.004, 0.502]), 0.006)]}


def moving_plate_error(rig: Rig, cal: calibrate.Calibration, plate: Plate, velocity: np.ndarray,
                       rate_hz: float = 60.0, steps: int = 4, pitch: int = 16,
                       seed: int = 0) -> float:
    """Return the RMS depth error in metres of one phase-shift cycle on a moving plate.

    Frame i is rendered with the plate moved by velocity × i / rate_hz. The wrapped phase is
    unwrapped against the true column at mid-cycle, as the preview path unwraps against its
    model, and the points are compared with the plate at mid-cycle.
    """
    rng = np.random.default_rng(seed)

    def moved(t):
        return Plate(plate.centre + velocity * t, plate.axes, plate.half_u, plate.half_v,
                     plate.albedo)

    frames, lit = [], None
    for i in range(steps):
        sv = scene_view(rig, [moved(i / rate_hz)])
        values = 127.5 + 127.5 * np.cos(2.0 * math.pi * (np.where(sv.lit, sv.u, 0.0) / pitch
                                                         + i / steps))
        frames.append(capture_scene(sv, values, rng))
        lit = sv.lit if lit is None else lit & sv.lit
    middle = moved((steps - 1) / 2.0 / rate_hz)
    sv = scene_view(rig, [middle])
    wrapped = patterns.wrapped_phase(np.stack(frames).astype(np.float64)).ravel()
    columns = patterns.unwrap_with_columns(wrapped, sv.u, pitch)
    mask = (lit & sv.lit & (np.abs(columns - sv.u) < pitch / 2.0)).reshape(sv.shape)
    from . import triangulate
    points, _ = triangulate.triangulate(columns.reshape(sv.shape), mask, cal)
    distance = (points - middle.centre) @ middle.axes[:, 2]
    return float(np.sqrt(np.mean(distance ** 2)))


def transform_scene(scene: list, r: np.ndarray, t: np.ndarray) -> list:
    """Return the scene's shapes moved into another frame, x' = r x + t."""
    moved = []
    for s in scene:
        if isinstance(s, Plate):
            moved.append(Plate(r @ s.centre + t, r @ s.axes, s.half_u, s.half_v, s.albedo,
                               s.dots, s.dot_radius, s.dot_albedo, s.tint))
        else:
            moved.append(Ball(r @ s.centre + t, s.radius, s.albedo, s.tint))
    return moved


@dataclass(frozen=True)
class StereoRig:
    """Two cameras and an uncalibrated pattern light. The world is the rig's centre, where
    the light sits looking along +z; each rotation and translation takes world to device."""

    left: Pinhole
    left_rotation: np.ndarray
    left_translation: np.ndarray
    right: Pinhole
    right_rotation: np.ndarray
    right_translation: np.ndarray
    light: Pinhole


def stereo_head(lens_mm: float = 16.0, baseline_m: float = 0.2, crossing_m: float = 0.5,
                light_width_m: float = 0.3) -> StereoRig:
    """Two Global Shutter cameras (1456 × 1088, 3.45 µm) toed in to cross at crossing_m, and
    a 1024 × 768 pattern light between them whose image is light_width_m wide there."""
    f = lens_mm / 0.00345
    camera = pinhole(1456, 1088, f, 727.5, 543.5)
    half = math.degrees(math.atan2(baseline_m / 2.0, crossing_m))
    r_left = rotation(0.0, half).T
    r_right = rotation(0.0, -half).T
    light = pinhole(1024, 768, 1024 * crossing_m / light_width_m, 511.5, 383.5)
    return StereoRig(camera, r_left, -r_left @ np.array([-baseline_m / 2.0, 0.0, 0.0]),
                     camera, r_right, -r_right @ np.array([baseline_m / 2.0, 0.0, 0.0]), light)


def metre_head() -> StereoRig:
    """The metre head of docs/hardware.md: 6 mm lenses on a 0.35 m baseline crossing at 1 m,
    and a pattern light 0.9 m wide there."""
    return stereo_head(6.0, 0.35, 1.0, light_width_m=0.9)


METRE_BOARD = calibrate.BoardSpec()


def metre_scenes(crossing_m: float = 1.0) -> dict:
    """Uncoated Phase 2 artifacts at crossing_m in the rig's frame: a 0.5 m plate, and a
    scale bar of two 50.8 mm spheres about 400 mm apart."""
    z = crossing_m
    return {"plate": [Plate(np.array([0.0, 0.0, z]), rotation(10, -15), 0.25, 0.25)],
            "scale-bar": [Ball(np.array([-0.2, 0.01, z]), 0.0254),
                          Ball(np.array([0.2, -0.01, z + 0.01]), 0.0254)]}


def stereo_extrinsics(rig: StereoRig) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation from the left camera's frame to the right camera's."""
    r = rig.right_rotation @ rig.left_rotation.T
    return r, rig.right_translation - r @ rig.left_translation


def true_stereo_calibration(rig: StereoRig, spec: calibrate.BoardSpec) -> "calibrate.StereoCalibration":
    """Return the rig's own parameters as a stereo calibration with zero distortion."""
    def intrinsics(p):
        return calibrate.Intrinsics(p.width, p.height, p.matrix.copy(), np.zeros(5), 0.0)
    r, t = stereo_extrinsics(rig)
    return calibrate.StereoCalibration(spec, intrinsics(rig.left), intrinsics(rig.right),
                                       calibrate.Stereo(r, t, 0.0), 0)


def to_left(rig: StereoRig, points: np.ndarray) -> np.ndarray:
    """Move world points (N, 3) into the left camera's frame."""
    return points @ rig.left_rotation.T + rig.left_translation


def random_dots(width: int, height: int, blob_px: float, defocus_px: float,
                rng: np.random.Generator) -> np.ndarray:
    """Return a binary random-dot pattern, softened by defocus, as float32 in 0 to 1."""
    noise = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), blob_px / 2.0)
    dots = (noise > np.median(noise)).astype(np.float32)
    return cv2.GaussianBlur(dots, (0, 0), defocus_px) if defocus_px > 0 else dots


def stereo_capture(rig: StereoRig, scene: list, pattern: np.ndarray, rng: np.random.Generator,
                   ambient: float = 15.0, gain: float = 200.0, noise: float = 1.0) -> dict:
    """Render the scene lit by the pattern in both cameras at the same instant."""
    def render(camera, r_cam, t_cam):
        shapes = transform_scene(scene, r_cam, t_cam)
        sv = scene_view(Rig(camera, rig.light, r_cam.T, -r_cam.T @ t_cam), shapes)
        u = np.where(sv.lit, sv.u, -1.0).astype(np.float32).reshape(sv.shape)
        v = np.where(sv.lit, sv.v, -1.0).astype(np.float32).reshape(sv.shape)
        lit = cv2.remap(pattern, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0).ravel()
        image = sv.albedo * (ambient + gain * np.where(sv.lit, lit, 0.0))
        image = image + noise * rng.standard_normal(image.shape)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(sv.shape)
    return {"left": render(rig.left, rig.left_rotation, rig.left_translation),
            "right": render(rig.right, rig.right_rotation, rig.right_translation)}


def stereo_room_view(rig: StereoRig, scene: list, rng: np.random.Generator, room: float = 120.0,
                     noise: float = 1.0) -> dict:
    """Render the scene in both cameras under room light alone, with the pattern light
    off."""
    def render(camera, r_cam, t_cam):
        shapes = transform_scene(scene, r_cam, t_cam)
        sv = scene_view(Rig(camera, rig.light, r_cam.T, -r_cam.T @ t_cam), shapes)
        image = sv.albedo * room + noise * rng.standard_normal(sv.albedo.shape)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(sv.shape)
    return {"left": render(rig.left, rig.left_rotation, rig.left_translation),
            "right": render(rig.right, rig.right_rotation, rig.right_translation)}


def stereo_board_view(rig: StereoRig, spec: calibrate.BoardSpec, r: np.ndarray, t: np.ndarray,
                      rng: np.random.Generator) -> dict:
    """Render one board pose (board to world) in both cameras under room light."""
    images = {}
    for side, camera, r_cam, t_cam in (("left", rig.left, rig.left_rotation, rig.left_translation),
                                       ("right", rig.right, rig.right_rotation,
                                        rig.right_translation)):
        v = view(Rig(camera, rig.light, np.eye(3), np.zeros(3)), spec, r_cam @ r, r_cam @ t + t_cam)
        image = v.albedo * 200.0 + rng.standard_normal(v.albedo.shape).astype(np.float32)
        images[side] = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return images


def stereo_calibration_views(rig: StereoRig, spec: calibrate.BoardSpec, seed: int = 0,
                             crossing_m: float = 0.5, jitter_deg: float = 0.0,
                             jitter_m: float = 0.0, passes: int = 1):
    """Yield each board pose of board_poses seen by both cameras under room light, passes
    times over. With a jitter, the poses move by seeded random amounts, so each seed is its
    own session and each pass its own set of poses."""
    rng = np.random.default_rng(seed)
    jittered = jitter_deg > 0.0 or jitter_m > 0.0
    for _ in range(passes):
        poses = board_poses(spec, crossing_m, rng if jittered else None, jitter_deg, jitter_m)
        for r, t in poses:
            yield stereo_board_view(rig, spec, r, t, rng)


def stereo_scenes(crossing_m: float = 0.5) -> dict:
    """Uncoated acceptance artifacts at crossing_m in the rig's frame: an 80 mm plate, a
    25 mm sphere and a ball bar of two 25 mm spheres about 80 mm apart."""
    z = crossing_m
    return {"plate": [Plate(np.array([0.0, 0.0, z]), rotation(10, -15), 0.04, 0.04)],
            "sphere": [Ball(np.array([0.003, -0.002, z]), 0.0125)],
            "ball-bar": [Ball(np.array([-0.04, 0.004, z]), 0.0125),
                         Ball(np.array([0.04, -0.004, z + 0.002]), 0.0125)]}


def sweep_pose(s: float, yaw_deg: float = 25.0, pitch_deg: float = 8.0,
               distance_m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Head pose (world to head) at fraction s of an arc around a target at distance_m
    ahead of its first pose, yawing from -yaw_deg to +yaw_deg and pitching through
    ±pitch_deg."""
    target = np.array([0.0, 0.0, distance_m])
    yaw = -yaw_deg + 2.0 * yaw_deg * s
    pitch = pitch_deg * math.sin(2.0 * math.pi * s)
    orient = rotation(pitch, yaw)
    centre = target - orient @ np.array([0.0, 0.0, distance_m])
    r = orient.T
    return r, -r @ centre


def sweep_poses(frames: int, yaw_deg: float = 25.0, pitch_deg: float = 8.0,
                distance_m: float = 0.5) -> list:
    """Head poses (world to head) at equal steps along the arc of sweep_pose."""
    return [sweep_pose(i / max(frames - 1, 1), yaw_deg, pitch_deg, distance_m)
            for i in range(frames)]


def loop_poses(frames: int, yaw_deg: float = 25.0, pitch_deg: float = 8.0,
               distance_m: float = 0.5) -> list:
    """Head poses (world to head) around a closed loop about a target at distance_m ahead
    of the first pose: yaw follows -yaw_deg cos and pitch pitch_deg sin of one turn, and a
    last pose repeats the first, so frames + 1 poses in all."""
    target = np.array([0.0, 0.0, distance_m])
    poses = []
    for i in range(frames + 1):
        s = (i % frames) / frames
        orient = rotation(pitch_deg * math.sin(2.0 * math.pi * s),
                          -yaw_deg * math.cos(2.0 * math.pi * s))
        centre = target - orient @ np.array([0.0, 0.0, distance_m])
        r = orient.T
        poses.append((r, -r @ centre))
    return poses


def sweep_scene(distance_m: float = 0.5) -> dict:
    """A part on a mat for sweep tests, in the world frame of sweep_poses: a 140 mm mat, a
    ball bar of two 25 mm spheres about 80 mm apart, and a 16 mm sphere."""
    z = distance_m
    return {"mat": Plate(np.array([0.0, 0.0, z + 0.02]), np.eye(3), 0.07, 0.07),
            "ball-bar": [Ball(np.array([-0.04, 0.005, z]), 0.0125),
                         Ball(np.array([0.04, -0.005, z + 0.002]), 0.0125)],
            "sphere": Ball(np.array([0.0, -0.04, z + 0.008]), 0.008)}


def panel_scene(distance_m: float = 0.5, dot_pitch: float = 0.035) -> dict:
    """A smooth panel for light-off sweeps, in the world frame of slide_poses: a flat
    0.6 × 0.3 m plate tilted 8° at distance_m, with reference dots of 2 mm radius about
    dot_pitch apart, each moved up to 3 mm from a grid as hand-placed dots would be."""
    rng = np.random.default_rng(7)
    grid = [(u, v) for u in np.arange(-0.28, 0.281, dot_pitch)
            for v in np.arange(-0.13, 0.131, dot_pitch)]
    dots = tuple((u + rng.uniform(-0.003, 0.003), v + rng.uniform(-0.003, 0.003))
                 for u, v in grid)
    return {"panel": Plate(np.array([0.0, 0.0, distance_m]), rotation(8.0, 0.0), 0.3, 0.15, 0.8,
                           dots, 0.002)}


def slide_poses(frames: int, step_m: float = 0.0025) -> list:
    """Head poses (world to head) sliding step_m per frame along the head's x axis."""
    return [(np.eye(3), np.array([-step_m * k, 0.0, 0.0])) for k in range(frames)]


def sweep_capture(rig: StereoRig, scene: dict, pose: tuple, pattern: np.ndarray,
                  rng: np.random.Generator, dark: bool = False) -> dict:
    """Return the stereo capture at one head pose (world to head); the pattern light moves
    with the head. With dark it is taken under room light alone, with the pattern light
    off."""
    shapes = [shape for value in scene.values()
              for shape in (value if isinstance(value, list) else [value])]
    moved = transform_scene(shapes, *pose)
    if dark:
        return stereo_room_view(rig, moved, rng)
    return stereo_capture(rig, moved, pattern, rng)


def sweep_captures(rig: StereoRig, scene: dict, poses: list, pattern: np.ndarray, seed: int = 0,
                   dark_every: int = 0):
    """Yield the stereo capture at each head pose. With dark_every, every dark_every-th
    pair from the first is taken with the pattern light off."""
    rng = np.random.default_rng(seed)
    for i, pose in enumerate(poses):
        yield sweep_capture(rig, scene, pose, pattern, rng,
                            dark_every > 0 and i % dark_every == 0)


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Return the pose (world to camera) of a camera at eye looking at target, with its y
    axis pointing down, away from up."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, np.asarray(up, dtype=float))
    x /= np.linalg.norm(x)
    r = np.vstack([x, np.cross(z, x), z])
    return r, -r @ eye


def orbit_poses(azimuths_deg, distance_m: float = 0.5, height_m: float = 0.4,
                target=(0.0, 0.0, 0.03)) -> list:
    """Return camera poses (world to camera) on a circle about the vertical axis, each at
    distance_m out and height_m up, looking at target."""
    target = np.asarray(target, dtype=float)
    poses = []
    for a in np.radians(np.asarray(azimuths_deg, dtype=float)):
        eye = np.array([distance_m * math.sin(a), -distance_m * math.cos(a), height_m])
        poses.append(look_at(eye, target))
    return poses


def point_view(scene: list, r: np.ndarray, t: np.ndarray, rng: np.random.Generator,
               half_field: float = 0.3, rays: int = 320, noise_m: float = 30e-6) -> np.ndarray:
    """Return the points (N, 3) a camera at pose (world to camera) sees of the scene, in its
    frame: the nearest hits of a rays x 3/4 rays grid over tangents to half_field, with
    noise_m of noise on each coordinate."""
    shapes = transform_scene(scene, r, t)
    gx, gy = np.meshgrid(np.linspace(-half_field, half_field, rays),
                         np.linspace(-0.75 * half_field, 0.75 * half_field, rays * 3 // 4))
    directions = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    hit, _ = nearest_hit(shapes, np.zeros(3), directions)
    ok = np.isfinite(hit)
    points = directions[ok] * hit[ok, np.newaxis]
    return points + noise_m * rng.standard_normal(points.shape)


def facing_plate(centre, normal, u, half_u: float, half_v: float) -> Plate:
    """Return a plate at centre with the given normal, its u axis along u made square to
    the normal."""
    n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    u = np.asarray(u, dtype=float) - (np.asarray(u, dtype=float) @ n) * n
    u /= np.linalg.norm(u)
    return Plate(np.asarray(centre, dtype=float), np.column_stack([u, np.cross(n, u), n]),
                 half_u, half_v)


def static_object() -> list:
    """A part about 0.3 m across for registration tests: a base plate, two walls and four
    balls of different sizes, standing on the world's xy plane."""
    return [facing_plate([0.0, 0.0, 0.0], [0, 0, 1], [1, 0, 0], 0.15, 0.10),
            facing_plate([0.0, 0.10, 0.06], [0, -1, 0], [1, 0, 0], 0.15, 0.06),
            facing_plate([-0.15, 0.0, 0.04], [1, 0, 0.3], [0, 1, 0], 0.10, 0.04),
            Ball(np.array([0.05, 0.03, 0.04]), 0.04),
            Ball(np.array([-0.08, -0.04, 0.025]), 0.025),
            Ball(np.array([0.09, -0.06, 0.015]), 0.015),
            Ball(np.array([-0.05, 0.05, 0.03]), 0.03)]


def metre_object() -> dict:
    """A part about a metre across for registration tests, standing on the world's xy
    plane: a base, a back wall, a slanted side, a raised step and four balls, with two
    scale bars of 50.8 mm spheres about 400 mm apart, one level and clear of the base and
    one rising across it."""
    part = [facing_plate([0.0, 0.0, 0.0], [0, 0, 1], [1, 0, 0], 0.5, 0.3),
            facing_plate([0.0, 0.3, 0.15], [0, -1, 0], [1, 0, 0], 0.5, 0.15),
            facing_plate([-0.5, 0.0, 0.1], [1, 0, 0.4], [0, 1, 0], 0.3, 0.1),
            facing_plate([0.3, -0.1, 0.1], [0, 0, 1], [1, 0.3, 0], 0.12, 0.08),
            facing_plate([0.3, -0.18, 0.05], [0, -1, 0], [1, 0, 0], 0.12, 0.05),
            Ball(np.array([0.3, 0.15, 0.08]), 0.08),
            Ball(np.array([-0.25, -0.15, 0.05]), 0.05),
            Ball(np.array([0.05, 0.2, 0.06]), 0.06),
            Ball(np.array([-0.35, 0.12, 0.1]), 0.1)]
    bars = [[Ball(np.array([-0.2, -0.22, 0.06]), 0.0254),
             Ball(np.array([0.2, -0.24, 0.06]), 0.0254)],
            [Ball(np.array([-0.15, 0.02, 0.12]), 0.0254),
             Ball(np.array([0.12, 0.12, 0.38]), 0.0254)]]
    return {"part": part, "scale-bars": bars}

def panel_frames(rng: np.random.Generator, frames: int = 24, slide: float = 0.24,
                 dot_pitch: float = 0.1) -> tuple[list, list, list]:
    """Return point frames, reference dot frames and true translations (into the first
    frame) of a camera sliding slide metres at one speed along the panel of
    panel_frames_at."""
    return panel_frames_at(rng, [slide * (i / (frames - 1) - 0.5) for i in range(frames)],
                           dot_pitch)


def eased_slide(frames: int, slide: float, back: bool = False) -> list:
    """Camera positions of a slide of slide metres about 0 that starts and ends at rest,
    with its highest speed halfway. With back, an odd count of frames slides there by its
    middle frame and returns the same way."""
    if back:
        there = eased_slide(frames // 2 + 1, slide)
        return there + there[-2::-1]
    return [slide * ((1.0 - math.cos(math.pi * i / (frames - 1))) / 2.0 - 0.5)
            for i in range(frames)]


def panel_frames_at(rng: np.random.Generator, positions: list, dot_pitch: float = 0.1,
                    dot_jitter: float = 0.0) -> tuple[list, list, list]:
    """Return point frames, reference dot frames and true translations (into the first
    frame) of a camera at each of positions, in metres along a cylindrical panel of 1.5 m
    radius 1 m away, with dots every dot_pitch metres, each moved up to dot_jitter metres
    from that grid along the panel as hand-placed dots would be."""
    def z(x, y):
        return 1.0 + 1.5 - np.sqrt(1.5 ** 2 - y ** 2)
    x, y = rng.uniform(-0.6, 0.6, 15000), rng.uniform(-0.3, 0.3, 15000)
    world = np.column_stack([x, y, z(x, y)])
    gx, gy = np.meshgrid(np.arange(-0.6, 0.6001, dot_pitch), np.arange(-0.25, 0.2501, dot_pitch))
    gx, gy = gx.ravel(), gy.ravel()
    if dot_jitter > 0:
        gx = gx + rng.uniform(-dot_jitter, dot_jitter, gx.size)
        gy = gy + rng.uniform(-dot_jitter, dot_jitter, gy.size)
    dots = np.column_stack([gx, gy, z(gx, gy)])
    clouds, dot_frames, truth = [], [], []
    for position in positions:
        t = np.array([position, 0.0, 0.0])
        for points, out, noise in ((world, clouds, 3e-5), (dots, dot_frames, 5e-5)):
            seen = points - t
            inside = (np.abs(seen[:, 0]) < 0.4 * seen[:, 2]) & (np.abs(seen[:, 1]) < 0.3 * seen[:, 2])
            seen = seen[inside]
            out.append(seen + rng.normal(scale=noise, size=seen.shape))
        truth.append(t - np.array([positions[0], 0.0, 0.0]))
    return clouds, dot_frames, truth
