"""Active stereo: depth from two calibrated cameras matching a projected random texture.

Matching runs on rectified images. Semi-global matching at quarter resolution gives a
coarse disparity. Each sample point is then tracked by Lucas-Kanade and refined by an
affine window fitted with ECC, which follows the different foreshortening the two cameras
see on a slanted surface.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .calibrate import StereoCalibration


@dataclass
class Rectification:
    left_rotation: np.ndarray
    right_rotation: np.ndarray
    left_projection: np.ndarray
    right_projection: np.ndarray
    left_maps: tuple
    right_maps: tuple
    size: tuple


def rectify(cal: StereoCalibration) -> Rectification:
    """Rectify the pair so matching points share a row."""
    size = (cal.left.width, cal.left.height)
    rl, rr, pl, pr, _, _, _ = cv2.stereoRectify(
        cal.left.matrix, cal.left.distortion, cal.right.matrix, cal.right.distortion, size,
        cal.stereo.rotation, cal.stereo.translation.reshape(3, 1), flags=0, alpha=1)
    left_maps = cv2.initUndistortRectifyMap(cal.left.matrix, cal.left.distortion, rl, pl, size,
                                            cv2.CV_32FC1)
    right_maps = cv2.initUndistortRectifyMap(cal.right.matrix, cal.right.distortion, rr, pr, size,
                                             cv2.CV_32FC1)
    return Rectification(rl, rr, pl, pr, left_maps, right_maps, size)


def disparity_at(rect: Rectification, depth_m: float) -> float:
    """Return the rectified disparity of a point at depth_m in the rectified frame."""
    pl, pr = rect.left_projection, rect.right_projection
    return pl[0, 2] - pr[0, 2] - pr[0, 3] / depth_m


def coarse_disparity(left: np.ndarray, right: np.ndarray, d_min: float, d_max: float,
                     scale: int = 4, block: int = 5) -> np.ndarray:
    """Semi-global matching at 1/scale resolution, returned at full resolution (NaN where
    no match)."""
    small = (left.shape[1] // scale, left.shape[0] // scale)
    ls = cv2.resize(left, small, interpolation=cv2.INTER_AREA)
    rs = cv2.resize(right, small, interpolation=cv2.INTER_AREA)
    lo = int(np.floor(d_min / scale)) - 2
    span = int(np.ceil((d_max - d_min) / scale)) + 4
    span = (span + 15) // 16 * 16
    sgbm = cv2.StereoSGBM_create(minDisparity=lo, numDisparities=span, blockSize=block,
                                 P1=8 * block * block, P2=32 * block * block,
                                 uniquenessRatio=5, speckleWindowSize=0, speckleRange=0,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(ls, rs).astype(np.float32) / 16.0
    d[d < lo] = np.nan
    d = cv2.resize(d, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    return d * scale


def refine(left: np.ndarray, right: np.ndarray, points: np.ndarray, guesses: np.ndarray,
           window: int = 21, min_correlation: float = 0.95,
           max_shift: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Refine matches of left points near their right guesses. Returns the kept left and
    right points (N, 2). A match is dropped when its affine window correlates below
    min_correlation or lands more than max_shift pixels from its guess."""
    if len(points) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    params = dict(winSize=(window, window), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-4))
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(left, right, points.astype(np.float32),
                                                  guesses.astype(np.float32), **params)
    ok = (status.ravel() == 1) & (np.abs(tracked[:, 1] - points[:, 1]) < 1.0)
    hw = window // 2
    margin = 2 * hw
    h, w = left.shape
    lf, rf = left.astype(np.float32), right.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    kept_left, kept_right = [], []
    for (x, y), (u, v) in zip(points[ok], tracked[ok]):
        xi, yi, ui, vi = int(round(x)), int(round(y)), int(round(u)), int(round(v))
        x0, y0 = ui - hw - margin, vi - hw - margin
        if (xi - hw < 0 or yi - hw < 0 or xi + hw >= w or yi + hw >= h or x0 < 0 or y0 < 0
                or x0 + window + 2 * margin > w or y0 + window + 2 * margin > h):
            continue
        template = lf[yi - hw:yi + hw + 1, xi - hw:xi + hw + 1]
        if template.std() < 3.0:
            continue
        crop = rf[y0:y0 + window + 2 * margin, x0:x0 + window + 2 * margin]
        warp = np.array([[1, 0, u - (x - xi) - hw - x0], [0, 1, v - hw - y0]], dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(template, crop, warp, cv2.MOTION_AFFINE,
                                                     criteria, None, 1)
        except cv2.error:
            continue
        if correlation < min_correlation:
            continue
        cx, cy = hw + (x - xi), hw + (y - yi)
        matched = (warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2] + x0,
                   warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2] + y0)
        if abs(matched[0] - u) > max_shift or abs(matched[1] - y) > 1.0:
            continue
        kept_left.append((x, y))
        kept_right.append(matched)
    return np.array(kept_left).reshape(-1, 2), np.array(kept_right).reshape(-1, 2)


def depth_points(cal: StereoCalibration, left: np.ndarray, right: np.ndarray,
                 near_m: float, far_m: float, step: int = 8, window: int = 21,
                 min_level: float = 6.0) -> tuple[np.ndarray, Rectification]:
    """Return points (N, 3) in metres in the left camera's frame, matched on a grid of
    rectified left pixels every step pixels, for surfaces between near_m and far_m.

    A sample is matched only when its whole window lies on lit surface (local mean at or
    above min_level in the left image) that semi-global matching also matched, so no window
    straddles an edge. The right side of each match is judged by its correlation."""
    rect = rectify(cal)
    rl = cv2.remap(left, *rect.left_maps, cv2.INTER_LINEAR)
    rr = cv2.remap(right, *rect.right_maps, cv2.INTER_LINEAR)
    coarse = coarse_disparity(rl, rr, disparity_at(rect, far_m), disparity_at(rect, near_m))
    h, w = rl.shape
    ys, xs = np.mgrid[window:h - window:step, window:w - window:step]
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    lit = cv2.blur(rl, (5, 5)) >= min_level
    matched = (np.isfinite(coarse) & lit).astype(np.uint8)
    inside = cv2.erode(matched, np.ones((window + 8, window + 8), np.uint8))
    d = coarse[points[:, 1].astype(int), points[:, 0].astype(int)]
    valid = inside[points[:, 1].astype(int), points[:, 0].astype(int)] > 0
    points, d = points[valid], d[valid]
    guesses = points.copy()
    guesses[:, 0] -= d
    left_pts, right_pts = refine(rl, rr, points, guesses, window)
    if len(left_pts) == 0:
        return np.empty((0, 3)), rect
    right_pts[:, 1] = left_pts[:, 1]
    x = cv2.triangulatePoints(rect.left_projection, rect.right_projection, left_pts.T, right_pts.T)
    x = (x[:3] / x[3]).T
    depth_ok = (x[:, 2] > near_m * 0.8) & (x[:, 2] < far_m * 1.2)
    return x[depth_ok] @ rect.left_rotation, rect
