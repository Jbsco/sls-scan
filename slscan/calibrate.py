"""Calibration: camera intrinsics from a ChArUco board, the projector as an inverse
camera through projected Gray codes, and the stereo extrinsics between the two.

A projector corner comes from the camera pixels around a detected board corner: their
decoded projector columns and rows give a local homography, which maps the corner into
the projector image at a fraction of a projector pixel.
"""

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import yaml

from . import patterns


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board: squares across and down, square and marker size in metres."""

    squares_x: int = 11
    squares_y: int = 8
    square_m: float = 0.050
    marker_m: float = 0.037
    dictionary: str = "DICT_5X5_100"

    def board(self) -> "cv2.aruco.CharucoBoard":
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m,
                                      self.marker_m, dictionary)


@dataclass
class Intrinsics:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float


@dataclass
class Stereo:
    """Rotation and translation from the camera frame to the projector frame."""

    rotation: np.ndarray
    translation: np.ndarray
    rms_px: float


@dataclass
class Calibration:
    board: BoardSpec
    camera: Intrinsics
    projector: Intrinsics
    stereo: Stereo
    views: int


def board_image(spec: BoardSpec, px_per_square: int, margin_px: int) -> np.ndarray:
    """Return the printable board as a uint8 image with a white margin."""
    size = (spec.squares_x * px_per_square + 2 * margin_px,
            spec.squares_y * px_per_square + 2 * margin_px)
    return spec.board().generateImage(size, marginSize=margin_px)


def detect_corners(image: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return the ids (N,) and sub-pixel image positions (N, 2) of the board corners seen."""
    detector = cv2.aruco.CharucoDetector(spec.board())
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64)
    return ids.reshape(-1).astype(np.int64), corners.reshape(-1, 2).astype(np.float64)


def object_points(spec: BoardSpec, ids: np.ndarray) -> np.ndarray:
    """Return the board-frame positions (N, 3) in metres of the given corner ids."""
    return spec.board().getChessboardCorners()[ids].astype(np.float64)


def decode_axis(captured: np.ndarray, length: int, min_contrast: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode a captured Gray code stack of one axis into projector indices and a mask."""
    bits, mask = patterns.binarize_pairs(captured, min_contrast)
    values = patterns.decode_gray_bits(bits)
    return values, mask & (values < length)


def projector_corners(corners: np.ndarray, columns: np.ndarray, rows: np.ndarray,
                      mask: np.ndarray, half_window: int = 20,
                      min_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Map camera corners into the projector image through local homographies.

    Returns a keep mask (N,) and projector positions (N, 2); a corner whose window has
    fewer than min_points decoded pixels is not kept.
    """
    height, width = mask.shape
    keep = np.zeros(len(corners), dtype=bool)
    out = np.zeros((len(corners), 2), dtype=np.float64)
    for i, (x, y) in enumerate(corners):
        x0, x1 = max(0, int(round(x)) - half_window), min(width, int(round(x)) + half_window + 1)
        y0, y1 = max(0, int(round(y)) - half_window), min(height, int(round(y)) + half_window + 1)
        valid = mask[y0:y1, x0:x1]
        if np.count_nonzero(valid) < min_points:
            continue
        ys, xs = np.nonzero(valid)
        src = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
        dst = np.column_stack([columns[y0:y1, x0:x1][valid],
                               rows[y0:y1, x0:x1][valid]]).astype(np.float64)
        homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
        if homography is None:
            continue
        p = homography @ np.array([x, y, 1.0])
        out[i] = p[:2] / p[2]
        keep[i] = True
    return keep, out


NARROW_FIELD = 0.3


def half_field(size: tuple[int, int], matrix: np.ndarray) -> float:
    """Return the tangent of a device's half-diagonal field of view."""
    return math.hypot(*size) / 2.0 / matrix[0, 0]


def calibrate_intrinsics(objects: list, images: list, size: tuple[int, int]) -> Intrinsics:
    """Calibrate one pinhole device with radial and tangential distortion, k3 fixed. A
    device whose half_field is under NARROW_FIELD keeps k2 at 0 as well (D80)."""
    objects = [o.astype(np.float32) for o in objects]
    images = [i.astype(np.float32) for i in images]
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        objects, images, size, None, None, flags=cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2)
    if half_field(size, matrix) >= NARROW_FIELD:
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            objects, images, size, None, None, flags=cv2.CALIB_FIX_K3)
    return Intrinsics(size[0], size[1], matrix, distortion.reshape(-1), float(rms))


def calibrate(views, spec: BoardSpec, projector_size: tuple[int, int],
              min_contrast: float = 10.0, half_window: int = 20,
              min_corners: int = 6) -> Calibration:
    """Calibrate the camera, the projector and the pair from captured views.

    Each view is a mapping with "white", the board under white light, and "columns" and
    "rows", the captured Gray code stacks in the order of patterns.gray_code_images.
    """
    cam_obj, cam_img, proj_obj, proj_img, both_cam = [], [], [], [], []
    size = None
    for view in views:
        white = view["white"]
        size = (white.shape[1], white.shape[0])
        ids, corners = detect_corners(white, spec)
        if len(ids) < min_corners:
            continue
        cam_obj.append(object_points(spec, ids))
        cam_img.append(corners)
        columns, col_mask = decode_axis(view["columns"], projector_size[0], min_contrast)
        rows, row_mask = decode_axis(view["rows"], projector_size[1], min_contrast)
        keep, proj = projector_corners(corners, columns, rows, col_mask & row_mask, half_window)
        if np.count_nonzero(keep) < min_corners:
            continue
        proj_obj.append(object_points(spec, ids[keep]))
        proj_img.append(proj[keep])
        both_cam.append(corners[keep])
    if len(cam_obj) < 3 or len(proj_obj) < 3:
        raise ValueError("calibration needs at least three views with the board detected and decoded")
    camera = calibrate_intrinsics(cam_obj, cam_img, size)
    projector = calibrate_intrinsics(proj_obj, proj_img, projector_size)
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        [o.astype(np.float32) for o in proj_obj], [c.astype(np.float32) for c in both_cam],
        [p.astype(np.float32) for p in proj_img], camera.matrix, camera.distortion,
        projector.matrix, projector.distortion, size, flags=cv2.CALIB_FIX_INTRINSIC)
    stereo = Stereo(rotation, translation.reshape(-1), float(rms))
    return Calibration(spec, camera, projector, stereo, len(proj_obj))


@dataclass
class StereoCalibration:
    """Two cameras: left and right intrinsics, and the left-to-right rotation and translation."""

    board: BoardSpec
    left: Intrinsics
    right: Intrinsics
    stereo: Stereo
    views: int


def calibrate_stereo(views, spec: BoardSpec, min_corners: int = 6) -> StereoCalibration:
    """Calibrate two cameras and their extrinsics from views of the board.

    Each view is a mapping with "left" and "right" images of the board. Each camera is
    calibrated from every view in which it sees the board, and the pair from the corners
    both cameras see in the same view.
    """
    sides = {"left": ([], [], None), "right": ([], [], None)}
    pair_obj, pair_left, pair_right = [], [], []
    for view in views:
        found = {}
        for side in ("left", "right"):
            image = view[side]
            ids, corners = detect_corners(image, spec)
            objs, imgs, _ = sides[side]
            sides[side] = (objs, imgs, (image.shape[1], image.shape[0]))
            if len(ids) >= min_corners:
                objs.append(object_points(spec, ids))
                imgs.append(corners)
                found[side] = dict(zip(ids.tolist(), corners))
        if len(found) == 2:
            common = sorted(set(found["left"]) & set(found["right"]))
            if len(common) >= min_corners:
                pair_obj.append(object_points(spec, np.array(common)))
                pair_left.append(np.array([found["left"][i] for i in common]))
                pair_right.append(np.array([found["right"][i] for i in common]))
    if len(pair_obj) < 3:
        raise ValueError("stereo calibration needs at least three views with the board seen by both cameras")
    left = calibrate_intrinsics(sides["left"][0], sides["left"][1], sides["left"][2])
    right = calibrate_intrinsics(sides["right"][0], sides["right"][1], sides["right"][2])
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        [o.astype(np.float32) for o in pair_obj], [p.astype(np.float32) for p in pair_left],
        [p.astype(np.float32) for p in pair_right], left.matrix, left.distortion,
        right.matrix, right.distortion, sides["left"][2], flags=cv2.CALIB_FIX_INTRINSIC)
    return StereoCalibration(spec, left, right, Stereo(rotation, translation.reshape(-1), float(rms)),
                             len(pair_obj))


def save_stereo(path: str, cal: StereoCalibration) -> None:
    """Write a stereo calibration as YAML."""
    data = {"board": asdict(cal.board), "left": _intrinsics_dict(cal.left),
            "right": _intrinsics_dict(cal.right),
            "stereo": {"rotation": cal.stereo.rotation.tolist(),
                       "translation_m": cal.stereo.translation.tolist(),
                       "rms_px": cal.stereo.rms_px},
            "views": cal.views}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_stereo(path: str) -> StereoCalibration:
    """Read a stereo calibration written by save_stereo."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    stereo = data["stereo"]
    return StereoCalibration(BoardSpec(**data["board"]), _intrinsics(data["left"]),
                             _intrinsics(data["right"]),
                             Stereo(np.array(stereo["rotation"], dtype=np.float64),
                                    np.array(stereo["translation_m"], dtype=np.float64),
                                    float(stereo["rms_px"])),
                             int(data["views"]))


def _intrinsics_dict(i: Intrinsics) -> dict:
    return {"width": i.width, "height": i.height, "matrix": i.matrix.tolist(),
            "distortion": i.distortion.tolist(), "rms_px": i.rms_px}


def _intrinsics(d: dict) -> Intrinsics:
    return Intrinsics(int(d["width"]), int(d["height"]), np.array(d["matrix"], dtype=np.float64),
                      np.array(d["distortion"], dtype=np.float64), float(d["rms_px"]))


def save(path: str, cal: Calibration) -> None:
    """Write a calibration as YAML."""
    data = {"board": asdict(cal.board), "camera": _intrinsics_dict(cal.camera),
            "projector": _intrinsics_dict(cal.projector),
            "stereo": {"rotation": cal.stereo.rotation.tolist(),
                       "translation_m": cal.stereo.translation.tolist(),
                       "rms_px": cal.stereo.rms_px},
            "views": cal.views}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load(path: str) -> Calibration:
    """Read a calibration written by save."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    stereo = data["stereo"]
    return Calibration(BoardSpec(**data["board"]), _intrinsics(data["camera"]),
                       _intrinsics(data["projector"]),
                       Stereo(np.array(stereo["rotation"], dtype=np.float64),
                              np.array(stereo["translation_m"], dtype=np.float64),
                              float(stereo["rms_px"])),
                       int(data["views"]))


def compare_stereo(a: "StereoCalibration", b: "StereoCalibration") -> dict:
    """Return relative focal length changes of both cameras and the baseline change in
    metres from a to b."""
    def rel(x, y):
        return abs(y - x) / abs(x)
    return {"left_fx": rel(a.left.matrix[0, 0], b.left.matrix[0, 0]),
            "left_fy": rel(a.left.matrix[1, 1], b.left.matrix[1, 1]),
            "right_fx": rel(a.right.matrix[0, 0], b.right.matrix[0, 0]),
            "right_fy": rel(a.right.matrix[1, 1], b.right.matrix[1, 1]),
            "baseline_m": abs(float(np.linalg.norm(b.stereo.translation))
                              - float(np.linalg.norm(a.stereo.translation)))}


def compare(a: Calibration, b: Calibration) -> dict:
    """Return relative focal length changes and the baseline change in metres from a to b."""
    def rel(x, y):
        return abs(y - x) / abs(x)
    return {"camera_fx": rel(a.camera.matrix[0, 0], b.camera.matrix[0, 0]),
            "camera_fy": rel(a.camera.matrix[1, 1], b.camera.matrix[1, 1]),
            "projector_fx": rel(a.projector.matrix[0, 0], b.projector.matrix[0, 0]),
            "projector_fy": rel(a.projector.matrix[1, 1], b.projector.matrix[1, 1]),
            "baseline_m": abs(float(np.linalg.norm(b.stereo.translation))
                              - float(np.linalg.norm(a.stereo.translation)))}
