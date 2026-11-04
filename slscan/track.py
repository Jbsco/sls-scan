"""Tracking and fusion for continuous sweeps.

Each depth frame is aligned to the model by point-to-plane ICP, starting from the previous
frame's pose, and its points join the model, which is kept as voxel averages.
"""

import numpy as np


def rodrigues(w: np.ndarray) -> np.ndarray:
    """Return the rotation of the rotation vector w."""
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * kx @ kx


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the rotation vector of a rotation, as rodrigues inverts it."""
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    w = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                  rotation[1, 0] - rotation[0, 1]])
    return w * angle / (2.0 * np.sin(angle))


def voxel_average(points: np.ndarray, voxel: float) -> np.ndarray:
    """Return the mean point of every occupied voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse.ravel(), points)
    return sums / counts[:, np.newaxis]


def normals(points: np.ndarray, neighbours: int = 12) -> np.ndarray:
    """Return a unit normal per point from the plane of its nearest neighbours."""
    from scipy.spatial import cKDTree

    _, idx = cKDTree(points).query(points, k=neighbours)
    local = points[idx] - points[idx].mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(local, full_matrices=False)
    return vt[:, 2, :]


def icp(source: np.ndarray, target: np.ndarray, target_normals: np.ndarray, rotation: np.ndarray,
        translation: np.ndarray, max_distances=(0.004, 0.002, 0.001, 0.0005),
        iterations: int = 10, pairs=None,
        hold_below: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-plane ICP of source onto target from an initial pose.

    Pairs, when given, are matched points (source (M, 3), target (M, 3)), such as
    reference dots, each adding a point-to-point term to every step, which fixes the
    motion a smooth surface leaves free. With hold_below above zero, each step holds the
    pose at the initial pose along directions of motion whose normal-equation eigenvalue
    lies below hold_below times the largest, which the surface leaves nearly free, and
    solves the others from the surface. Returns the rotation and translation that take
    source into the target's frame, and the RMS point-to-plane residual of the last
    matched pairs.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    r, t = rotation.copy(), translation.copy()
    r0, t0 = rotation.copy(), translation.copy()
    rms = float("inf")
    for limit in max_distances:
        for _ in range(iterations):
            moved = source @ r.T + t
            distance, idx = tree.query(moved, distance_upper_bound=limit)
            ok = np.isfinite(distance)
            if np.count_nonzero(ok) < 50:
                break
            p, q, n = moved[ok], target[idx[ok]], target_normals[idx[ok]]
            residual = np.einsum("ij,ij->i", p - q, n)
            a = np.column_stack([np.cross(p, n), n])
            b = -residual
            if pairs is not None and len(pairs[0]):
                s = pairs[0] @ r.T + t
                zeros = np.zeros(len(s))
                skew = [np.column_stack([zeros, s[:, 2], -s[:, 1]]),
                        np.column_stack([-s[:, 2], zeros, s[:, 0]]),
                        np.column_stack([s[:, 1], -s[:, 0], zeros])]
                rows = [np.column_stack([skew[k], np.tile(np.eye(3)[k], (len(s), 1))])
                        for k in range(3)]
                a = np.vstack([a, *rows])
                b = np.concatenate([b, *[-(s[:, k] - pairs[1][:, k]) for k in range(3)]])
            if hold_below > 0:
                moved_by = r @ r0.T
                away = np.concatenate([rotation_vector(moved_by), t - moved_by @ t0])
                values, vectors = np.linalg.eigh(a.T @ a)
                along = vectors.T @ (a.T @ b)
                free = values < hold_below * values[-1]
                along[~free] /= values[~free]
                along[free] = -(vectors.T @ away)[free]
                step = vectors @ along
            else:
                step, *_ = np.linalg.lstsq(a, b, rcond=None)
            dr = rodrigues(step[:3])
            r, t = dr @ r, dr @ t + step[3:]
            rms = float(np.sqrt(np.mean(residual ** 2)))
            if np.linalg.norm(step[:3]) < 1e-7 and np.linalg.norm(step[3:]) < 1e-8:
                break
    return r, t, rms


def match_dots(dots: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
               dot_map: np.ndarray, radius: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Pair each dot (in its frame) with the nearest dot of the map within radius, after
    moving it by a pose. Returns the indices of paired dots and of their map dots."""
    if len(dots) == 0 or len(dot_map) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    from scipy.spatial import cKDTree

    distance, idx = cKDTree(dot_map).query(dots @ rotation.T + translation,
                                           distance_upper_bound=radius)
    ok = np.isfinite(distance)
    return np.nonzero(ok)[0], idx[ok]


def fuse(frames, voxel: float = 0.0005, dots=None,
         dot_radius: float = 0.03) -> tuple[list, np.ndarray]:
    """Track and fuse a sequence of point sets (each in its own camera frame).

    The world is the first frame's camera frame, and each frame's ICP starts from the
    previous frame's pose. With dots, one array of reference dot positions per frame in
    its camera frame, each frame's dots pair with a map of the dots seen so far and add
    point-to-point terms to its ICP, and dots without a pair join the map. A dot pairs
    with a map dot within dot_radius, which must stay under half the dots' spacing.
    Returns each frame's pose (rotation, translation into the world) and the fused model
    as voxel averages.
    """
    poses, model, raw = [], None, []
    r, t = np.eye(3), np.zeros(3)
    dot_map, dot_counts = np.empty((0, 3)), np.empty(0)
    for k, points in enumerate(frames):
        if len(points) == 0:
            poses.append((r, t))
            continue
        seen = dots[k] if dots is not None else np.empty((0, 3))
        if model is not None:
            mine, theirs = match_dots(seen, r, t, dot_map, dot_radius)
            pairs = (seen[mine], dot_map[theirs]) if len(mine) >= 3 else None
            r, t, _ = icp(points, model, model_normals, r, t, pairs=pairs)
        if len(seen):
            mine, theirs = match_dots(seen, r, t, dot_map, dot_radius)
            placed = seen @ r.T + t
            for i, j in zip(mine, theirs):
                dot_map[j] = (dot_map[j] * dot_counts[j] + placed[i]) / (dot_counts[j] + 1)
                dot_counts[j] += 1
            fresh = np.setdiff1d(np.arange(len(seen)), mine)
            dot_map = np.vstack([dot_map, placed[fresh]])
            dot_counts = np.concatenate([dot_counts, np.ones(len(fresh))])
        poses.append((r, t))
        raw.append(points @ r.T + t)
        model = voxel_average(np.vstack(raw), voxel)
        model_normals = normals(model)
    return poses, (model if model is not None else np.empty((0, 3)))


def left_camera_pose(rig, head_rotation: np.ndarray, head_translation: np.ndarray):
    """Return the left camera's pose (world to camera) for a head pose (world to head)."""
    return rig.left_rotation @ head_rotation, rig.left_rotation @ head_translation + rig.left_translation


def sweep_bench(rig, cal, scene: dict, poses: list, frames: list, voxel: float = 0.0005,
                surface=None) -> dict:
    """Track and fuse sweep frames and compare poses and artifacts with the truth.

    The tracked world is the first frame's left camera frame, so the truth is moved there.
    With surface, the model measured is what surface returns for the tracked poses in place
    of the fused points.
    """
    from . import geometry

    tracked, model = fuse(frames, voxel)
    if surface is not None:
        model = surface(tracked)
    r0, t0 = left_camera_pose(rig, *poses[0])
    pose_errors = []
    for (r, t), head in zip(tracked, poses):
        rk, tk = left_camera_pose(rig, *head)
        true_r = r0 @ rk.T
        true_t = t0 - true_r @ tk
        d = r @ true_r.T
        angle = np.degrees(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0)))
        pose_errors.append((float(np.linalg.norm(t - true_t)), float(angle)))

    def to_first(p):
        return r0 @ p + t0

    centres, radii = [], []
    for ball in scene["ball-bar"]:
        c = to_first(ball.centre)
        near = model[np.linalg.norm(model - c, axis=1) < ball.radius + 0.0015]
        centre, radius, _ = geometry.fit_sphere(near)
        centres.append(centre)
        radii.append(radius - ball.radius)
    truth = np.linalg.norm(scene["ball-bar"][0].centre - scene["ball-bar"][1].centre)
    mat = scene["mat"]
    normal = r0 @ mat.axes[:, 2]
    on_mat = np.abs((model - to_first(mat.centre)) @ normal) < 0.001
    clear = np.all([np.linalg.norm(model - to_first(b.centre), axis=1) > b.radius + 0.004
                    for b in [*scene["ball-bar"], scene["sphere"]]], axis=0)
    _, _, flatness = geometry.fit_plane(model[on_mat & clear])
    return {"frames": len(frames), "points": len(model),
            "max_translation_error_m": max(e[0] for e in pose_errors),
            "max_rotation_error_deg": max(e[1] for e in pose_errors),
            "last_translation_error_m": pose_errors[-1][0],
            "last_rotation_error_deg": pose_errors[-1][1],
            "ball_bar_error_m": float(np.linalg.norm(centres[0] - centres[1]) - truth),
            "ball_radius_errors_m": radii, "mat_flatness_m": float(flatness)}
