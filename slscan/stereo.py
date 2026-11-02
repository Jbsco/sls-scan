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
           max_shift: float = 3.0, return_warps: bool = False):
    """Refine matches of left points near their right guesses. Returns the kept left and
    right points (N, 2), and with return_warps each kept window's affine part (N, 2, 2).
    A match is dropped when its affine window correlates below min_correlation or lands
    more than max_shift pixels from its guess."""
    if len(points) == 0:
        empty = (np.empty((0, 2)), np.empty((0, 2)))
        return (*empty, np.empty((0, 2, 2))) if return_warps else empty
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
    kept_left, kept_right, kept_warps = [], [], []
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
        kept_warps.append(warp[:, :2].copy())
    left_out, right_out = np.array(kept_left).reshape(-1, 2), np.array(kept_right).reshape(-1, 2)
    if return_warps:
        return left_out, right_out, np.array(kept_warps).reshape(-1, 2, 2)
    return left_out, right_out


def refine_second_order(left: np.ndarray, right: np.ndarray, left_points: np.ndarray,
                        right_points: np.ndarray, affine: np.ndarray, window: int = 21,
                        iterations: int = 20, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return the left points that second_order_matches keeps, with their refined right
    points."""
    refined, kept = second_order_matches(left, right, left_points, right_points, affine,
                                         window, iterations, batch)
    return left_points[kept], refined[kept]


def second_order_matches(left: np.ndarray, right: np.ndarray, left_points: np.ndarray,
                         right_points: np.ndarray, affine: np.ndarray, window: int = 21,
                         iterations: int = 20,
                         batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Refine every match with a second-order window by Gauss-Newton.

    Each window maps to the right image through a quadratic displacement field, 12
    parameters, with an intensity gain and offset, starting from the affine match. Returns
    the refined right points (N, 2) and a kept flag (N,). A match is not kept when its
    solve fails or its column moves 1 px or more from the affine match.
    """
    hw = window // 2
    dy, dx = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(np.float32)
    dx, dy = dx.ravel(), dy.ravel()
    shape = np.stack([np.ones_like(dx), dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy], axis=1)
    lf, rf = left.astype(np.float32), right.astype(np.float32)
    gx = cv2.Scharr(rf, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(rf, cv2.CV_32F, 0, 1) / 32.0
    out_right, out_kept = [], []
    for start in range(0, len(left_points), batch):
        lp = left_points[start:start + batch]
        rp = right_points[start:start + batch]
        a = affine[start:start + batch]
        n = len(lp)
        f = cv2.remap(lf, (lp[:, :1] + dx).astype(np.float32), (lp[:, 1:2] + dy).astype(np.float32),
                      cv2.INTER_LINEAR)
        p = np.zeros((n, 2, 6), dtype=np.float64)
        p[:, 0, 0] = rp[:, 0] - lp[:, 0]
        p[:, 1, 0] = rp[:, 1] - lp[:, 1]
        p[:, 0, 1], p[:, 0, 2] = a[:, 0, 0] - 1.0, a[:, 0, 1]
        p[:, 1, 1], p[:, 1, 2] = a[:, 1, 0], a[:, 1, 1] - 1.0
        gain = np.ones(n)
        offset = np.zeros(n)
        ok = np.ones(n, dtype=bool)
        active = np.arange(n)
        for _ in range(iterations):
            if len(active) == 0:
                break
            pa = p[active]
            u = pa[:, 0] @ shape.T
            v = pa[:, 1] @ shape.T
            mx = (lp[active, :1] + dx + u).astype(np.float32)
            my = (lp[active, 1:2] + dy + v).astype(np.float32)
            g = cv2.remap(rf, mx, my, cv2.INTER_CUBIC)
            gxs = cv2.remap(gx, mx, my, cv2.INTER_CUBIC)
            gys = cv2.remap(gy, mx, my, cv2.INTER_CUBIC)
            ga = gain[active][:, None]
            residual = f[active] - (ga * g + offset[active][:, None])
            jac = np.concatenate([(ga * gxs)[:, :, None] * shape[None],
                                  (ga * gys)[:, :, None] * shape[None],
                                  g[:, :, None], np.ones_like(g)[:, :, None]], axis=2)
            hessian = np.einsum("nki,nkj->nij", jac, jac).astype(np.float64)
            rhs = np.einsum("nki,nk->ni", jac, residual).astype(np.float64)
            try:
                step = np.linalg.solve(hessian + 1e-9 * np.eye(14), rhs[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                ok[active] = False
                break
            p[active, 0] += step[:, 0:6]
            p[active, 1] += step[:, 6:12]
            gain[active] += step[:, 12]
            offset[active] += step[:, 13]
            moving = np.max(np.abs(step[:, [0, 6]]), axis=1) >= 1e-5
            active = active[moving]
        ok &= np.all(np.isfinite(p.reshape(n, -1)), axis=1)
        ok &= np.abs(p[:, 0, 0] - (rp[:, 0] - lp[:, 0])) < 1.0
        out_kept.append(ok)
        out_right.append(np.column_stack([lp[:, 0] + p[:, 0, 0], lp[:, 1] + p[:, 1, 0]]))
    if not out_right:
        return np.empty((0, 2)), np.empty(0, dtype=bool)
    return np.vstack(out_right), np.concatenate(out_kept)


def affine_matches(cal: StereoCalibration, left: np.ndarray, right: np.ndarray,
                   near_m: float, far_m: float, step: int = 8, window: int = 21,
                   min_level: float = 6.0):
    """Rectify a view and match a grid of left pixels every step pixels with affine windows,
    for surfaces between near_m and far_m. Returns the rectification, both rectified
    images, the kept left and right points (N, 2) and each window's affine part (N, 2, 2).

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
    left_pts, right_pts, warps = refine(rl, rr, points, guesses, window, return_warps=True)
    return rect, rl, rr, left_pts, right_pts, warps


def depth_points(cal: StereoCalibration, left: np.ndarray, right: np.ndarray,
                 near_m: float, far_m: float, step: int = 8, window: int = 21,
                 min_level: float = 6.0, order: int = 1) -> tuple[np.ndarray, Rectification]:
    """Return points (N, 3) in metres in the left camera's frame from affine_matches.
    Order 2 refines each affine match with a second-order window."""
    rect, rl, rr, left_pts, right_pts, warps = affine_matches(cal, left, right, near_m, far_m,
                                                              step, window, min_level)
    if order == 2:
        left_pts, right_pts = refine_second_order(rl, rr, left_pts, right_pts, warps, window)
    if len(left_pts) == 0:
        return np.empty((0, 3)), rect
    right_pts[:, 1] = left_pts[:, 1]
    x = cv2.triangulatePoints(rect.left_projection, rect.right_projection, left_pts.T, right_pts.T)
    x = (x[:3] / x[3]).T
    depth_ok = (x[:, 2] > near_m * 0.8) & (x[:, 2] < far_m * 1.2)
    return x[depth_ok] @ rect.left_rotation, rect
