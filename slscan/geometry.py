"""Fits of the acceptance artifacts and point cloud output."""

import numpy as np


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the centroid, unit normal and RMS residual of the least-squares plane."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    residual = (points - centroid) @ normal
    return centroid, normal, float(np.sqrt(np.mean(residual ** 2)))


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return the centre, radius and RMS radial residual of the least-squares sphere."""
    a = np.column_stack([2.0 * points, np.ones(len(points))])
    b = np.sum(points ** 2, axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = solution[:3]
    radius = float(np.sqrt(solution[3] + centre @ centre))
    residual = np.linalg.norm(points - centre, axis=1) - radius
    for _ in range(3):
        directions = (points - centre) / np.linalg.norm(points - centre, axis=1)[:, np.newaxis]
        jacobian = np.column_stack([-directions, -np.ones(len(points))])
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        centre = centre + step[:3]
        radius += float(step[3])
        residual = np.linalg.norm(points - centre, axis=1) - radius
    return centre, radius, float(np.sqrt(np.mean(residual ** 2)))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation that take source points (N, 3) onto matching
    target points with least squared distance, by the SVD method of Kabsch."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, ct - rotation @ cs


def split_two(points: np.ndarray) -> np.ndarray:
    """Return a 0 or 1 label per point for two clusters along the main axis of the cloud."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    s = centred @ vt[0]
    threshold = 0.0
    for _ in range(20):
        low, high = s[s <= threshold], s[s > threshold]
        if len(low) == 0 or len(high) == 0:
            break
        threshold = 0.5 * (low.mean() + high.mean())
    return (s > threshold).astype(np.int64)


def write_ply(path: str, points: np.ndarray) -> None:
    """Write points (N, 3) as a binary little-endian PLY file of float vertices."""
    data = np.ascontiguousarray(points, dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(data)}\nproperty float x\nproperty float y\n"
              "property float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: str) -> np.ndarray:
    """Read the points of a PLY file written by write_ply."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    count = int(next(line.split()[2] for line in data[:end].decode("ascii").split("\n")
                     if line.startswith("element vertex")))
    return np.frombuffer(data[end:end + 12 * count], dtype="<f4").reshape(count, 3).astype(np.float64)


def read_ply_faces(path: str) -> np.ndarray:
    """Read the triangles (F, 3) of a PLY mesh written by mesh.write_mesh, or none from a
    PLY of points."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").split("\n")
    counts = {line.split()[1]: int(line.split()[2]) for line in header
              if line.startswith("element ")}
    faces = counts.get("face", 0)
    start = end + 12 * counts["vertex"]
    rows = np.frombuffer(data[start:start + 13 * faces], dtype=[("n", "u1"), ("v", "<i4", 3)])
    return rows["v"].astype(np.int64)
