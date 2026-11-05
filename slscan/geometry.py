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


def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write points (N, 3) as a binary little-endian PLY file of float vertices, with red,
    green and blue bytes per vertex when colors (N, 3) are given."""
    if colors is None:
        data = np.ascontiguousarray(points, dtype="<f4")
        extra = ""
    else:
        data = np.zeros(len(points), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        data["xyz"] = points
        data["rgb"] = colors
        extra = "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(data)}\nproperty float x\nproperty float y\n"
              f"property float z\n{extra}end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def _ply_vertices(path: str) -> tuple[bytes, int, np.ndarray]:
    """Return a PLY file written here, the end of its header and its vertex records."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    lines = data[:end].decode("ascii").split("\n")
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex")))
    start = next(i for i, line in enumerate(lines) if line.startswith("element vertex"))
    fields = []
    for line in lines[start + 1:]:
        if not line.startswith("property "):
            break
        kind, name = line.split()[1:3]
        fields.append((name, {"float": "<f4", "uchar": "u1"}[kind]))
    vertices = np.frombuffer(data[end:end + np.dtype(fields).itemsize * count], dtype=fields)
    return data, end, vertices


def read_ply(path: str) -> np.ndarray:
    """Read the points of a PLY file written by write_ply or mesh.write_mesh."""
    _, _, v = _ply_vertices(path)
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def read_ply_colors(path: str) -> np.ndarray | None:
    """Read the red, green and blue bytes (N, 3) of a PLY file written by write_ply, or none
    from one without colors."""
    _, _, v = _ply_vertices(path)
    if "red" not in v.dtype.names:
        return None
    return np.column_stack([v["red"], v["green"], v["blue"]])


def read_ply_faces(path: str) -> np.ndarray:
    """Read the triangles (F, 3) of a PLY mesh written by mesh.write_mesh, or none from a
    PLY of points."""
    data, end, vertices = _ply_vertices(path)
    header = data[:end].decode("ascii").split("\n")
    counts = {line.split()[1]: int(line.split()[2]) for line in header
              if line.startswith("element ")}
    faces = counts.get("face", 0)
    start = end + vertices.nbytes
    rows = np.frombuffer(data[start:start + 13 * faces], dtype=[("n", "u1"), ("v", "<i4", 3)])
    return rows["v"].astype(np.int64)
