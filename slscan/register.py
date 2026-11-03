"""Registration of static views from their geometry.

A coarse match pairs points of two views by fast point feature histograms (FPFH) and
keeps the rigid motion most pairs agree on, found by RANSAC. Point-to-plane ICP refines
it.
"""

import numpy as np

from . import geometry, track


def facing_normals(points: np.ndarray, neighbours: int = 12) -> np.ndarray:
    """Return unit normals turned toward the camera at the origin of the points' frame."""
    n = track.normals(points, neighbours)
    n[np.einsum("ij,ij->i", n, points) > 0] *= -1.0
    return n


def fpfh(points: np.ndarray, normals: np.ndarray, radius: float, bins: int = 11) -> np.ndarray:
    """Return each point's fast point feature histogram (N, 3 bins) over its neighbours
    within radius: three histograms of the angles between its normal and theirs, each
    summing to one."""
    from scipy.spatial import cKDTree

    n = len(points)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    d = points[j] - points[i]
    distance = np.linalg.norm(d, axis=1)
    d /= distance[:, np.newaxis]
    u = normals[i]
    v = np.cross(u, d)
    length = np.linalg.norm(v, axis=1)
    keep = length > 1e-9
    i, j, d, distance, u, v = i[keep], j[keep], d[keep], distance[keep], u[keep], v[keep]
    v /= length[keep, np.newaxis]
    w = np.cross(u, v)
    nj = normals[j]
    alpha = np.einsum("ij,ij->i", v, nj)
    phi = np.einsum("ij,ij->i", u, d)
    theta = np.arctan2(np.einsum("ij,ij->i", w, nj), np.einsum("ij,ij->i", u, nj))
    index = [np.clip(((alpha + 1.0) * 0.5 * bins).astype(int), 0, bins - 1),
             bins + np.clip(((phi + 1.0) * 0.5 * bins).astype(int), 0, bins - 1),
             2 * bins + np.clip(((theta + np.pi) / (2.0 * np.pi) * bins).astype(int), 0, bins - 1)]
    spfh = np.zeros((n, 3 * bins))
    for k in index:
        spfh += np.bincount(i * 3 * bins + k, minlength=n * 3 * bins).reshape(n, 3 * bins)
    counts = np.maximum(np.bincount(i, minlength=n), 1)[:, np.newaxis]
    spfh /= counts
    inverse = 1.0 / distance
    spread = np.column_stack([np.bincount(i, weights=spfh[j, c] * inverse, minlength=n)
                              for c in range(3 * bins)])
    weights = np.bincount(i, weights=inverse, minlength=n)
    result = spfh + spread / np.maximum(weights, 1e-12)[:, np.newaxis]
    for k in range(3):
        part = result[:, k * bins:(k + 1) * bins]
        part /= np.maximum(part.sum(axis=1, keepdims=True), 1e-12)
    return result


def feature_pairs(source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    """Return index pairs (M, 2) of source and target points that are each other's nearest
    feature."""
    from scipy.spatial import cKDTree

    _, forward = cKDTree(target_features).query(source_features)
    _, backward = cKDTree(source_features).query(target_features)
    mutual = backward[forward] == np.arange(len(source_features))
    return np.column_stack([np.nonzero(mutual)[0], forward[mutual]])


def batch_rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotations (K, 3, 3) and translations (K, 3) taking each set of source points
    (K, P, 3) onto its target points, by the method of geometry.rigid_fit."""
    cs, ct = source.mean(axis=1), target.mean(axis=1)
    h = np.einsum("kpi,kpj->kij", source - cs[:, np.newaxis], target - ct[:, np.newaxis])
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("kji,kjl->kil", vt, np.transpose(u, (0, 2, 1)))))
    fix = np.repeat(np.eye(3)[np.newaxis], len(h), axis=0)
    fix[:, 2, 2] = d
    r = np.einsum("kji,kjl,klm->kim", vt, fix, np.transpose(u, (0, 2, 1)))
    return r, ct - np.einsum("kij,kj->ki", r, cs)


def ransac(source: np.ndarray, target: np.ndarray, pairs: np.ndarray, inlier: float,
           rng: np.random.Generator, trials: int = 20000, edge_ratio: float = 0.9,
           shortlist: int = 100, chunk: int = 500) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the rigid motion that best overlays source on target, and the share of
    source points it brings within inlier of a target point. Trials take three pairs whose
    edge lengths agree to edge_ratio. The shortlist that takes the most paired points
    within inlier of their targets is judged by overlap of all points, and the winner is
    refit on its pairs."""
    from scipy.spatial import cKDTree

    s, t = source[pairs[:, 0]], target[pairs[:, 1]]
    picks = rng.integers(0, len(pairs), (trials, 3))
    ps, pt = s[picks], t[picks]
    ls = np.linalg.norm(ps - np.roll(ps, 1, axis=1), axis=2)
    lt = np.linalg.norm(pt - np.roll(pt, 1, axis=1), axis=2)
    agree = np.all(np.minimum(ls, lt) >= edge_ratio * np.maximum(ls, lt), axis=1)
    agree &= np.all(ls > inlier, axis=1)
    ps, pt = ps[agree], pt[agree]
    rotations, translations, counts = [], [], []
    for k in range(0, len(ps), chunk):
        r, tr = batch_rigid_fit(ps[k:k + chunk], pt[k:k + chunk])
        moved = np.einsum("kij,mj->kmi", r, s) + tr[:, np.newaxis]
        rotations.append(r)
        translations.append(tr)
        counts.append(np.count_nonzero(np.linalg.norm(moved - t, axis=2) < inlier, axis=1))
    if not counts:
        return np.eye(3), np.zeros(3), 0.0
    rotations, translations = np.concatenate(rotations), np.concatenate(translations)
    best = np.argsort(np.concatenate(counts))[::-1][:shortlist]
    tree = cKDTree(target)
    overlap = [np.mean(tree.query(source @ rotations[k].T + translations[k],
                                  distance_upper_bound=inlier)[0] < inlier) for k in best]
    k = best[int(np.argmax(overlap))]
    r, tr = rotations[k], translations[k]
    ok = np.linalg.norm(s @ r.T + tr - t, axis=1) < inlier
    r, tr = geometry.rigid_fit(s[ok], t[ok])
    return r, tr, float(max(overlap))


def describe(points: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a view's points (in its camera frame) thinned to voxel spacing, and their
    FPFH over 5 voxels."""
    thin = track.voxel_average(points, voxel)
    return thin, fpfh(thin, facing_normals(thin), 5.0 * voxel)


def register(source: np.ndarray, target: np.ndarray, voxel: float,
             rng: np.random.Generator | None = None,
             described=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the rotation and translation that take one view's points (each in its own
    camera frame) onto another's, by a coarse match at voxel spacing and ICP, and a record
    of the match. Described, when given, is describe of source and of target."""
    rng = np.random.default_rng(0) if rng is None else rng
    (s, fs), (t, ft) = described or (describe(source, voxel), describe(target, voxel))
    pairs = feature_pairs(fs, ft)
    r, tr, overlap = ransac(s, t, pairs, 1.5 * voxel, rng)
    fine = track.voxel_average(target, voxel / 4.0)
    r, tr, rms = track.icp(source, fine, track.normals(fine), r, tr,
                           max_distances=(3.0 * voxel, voxel, voxel / 2.0, voxel / 4.0))
    return r, tr, {"pairs": len(pairs), "overlap": overlap, "rms": rms}
