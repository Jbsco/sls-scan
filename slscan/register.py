"""Registration of static views from their geometry.

Points of two views pair by fast point feature histograms (FPFH).
"""

import numpy as np

from . import track


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
        np.add.at(spfh, (i, k), 1.0)
    counts = np.maximum(np.bincount(i, minlength=n), 1)[:, np.newaxis]
    spfh /= counts
    spread = np.zeros_like(spfh)
    np.add.at(spread, i, spfh[j] / distance[:, np.newaxis])
    weights = np.zeros(n)
    np.add.at(weights, i, 1.0 / distance)
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
