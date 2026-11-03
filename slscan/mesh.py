"""Triangle meshes from a TSDF volume by surface nets. Each cell between eight observed
voxels whose distances change sign gets one vertex, the mean of its edges' zero
crossings, and each crossing edge joins the four cells around it in a quad facing the
side where the distance is positive, split along the diagonal that leaves the larger
smallest triangle."""

import numpy as np

from .tsdf import Volume


def surface_nets(volume: Volume) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh vertices (V, 3) in the world and triangles (F, 3) of vertex indices."""
    d = volume.tsdf
    observed = (volume.weight > 0) & (np.abs(d) < 1.0)
    cells = tuple(n - 1 for n in volume.shape)
    sums = np.zeros(cells + (3,))
    counts = np.zeros(cells)
    edges = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        da, db = d[tuple(a)], d[tuple(b)]
        cross = observed[tuple(a)] & observed[tuple(b)] & ((da > 0) != (db > 0))
        idx = np.argwhere(cross)
        t = da[cross] / (da[cross] - db[cross])
        pos = idx.astype(np.float64)
        pos[:, axis] += t
        others = [k for k in range(3) if k != axis]
        for du in (0, 1):
            for dv in (0, 1):
                cell = idx.copy()
                cell[:, others[0]] -= du
                cell[:, others[1]] -= dv
                inside = np.all((cell >= 0) & (cell < np.array(cells)), axis=1)
                np.add.at(sums, tuple(cell[inside].T), pos[inside])
                np.add.at(counts, tuple(cell[inside].T), 1)
        edges.append((axis, idx, da[cross] > 0))
    corners = np.ones(cells, dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corners &= observed[dx:dx + cells[0], dy:dy + cells[1], dz:dz + cells[2]]
    has_vertex = corners & (counts > 0)
    index = np.full(cells, -1, dtype=np.int64)
    index[has_vertex] = np.arange(np.count_nonzero(has_vertex))
    vertices = volume.lower + volume.voxel * (sums[has_vertex] / counts[has_vertex][:, None])
    quads = []
    for axis, idx, positive in edges:
        u, v = [k for k in range(3) if k != axis]
        quad = []
        for du, dv in ((1, 1), (0, 1), (0, 0), (1, 0)):
            cell = idx.copy()
            cell[:, u] -= du
            cell[:, v] -= dv
            inside = np.all((cell >= 0) & (cell < np.array(cells)), axis=1)
            found = np.full(len(idx), -1, dtype=np.int64)
            found[inside] = index[tuple(cell[inside].T)]
            quad.append(found)
        quad = np.stack(quad, axis=1)
        whole = np.all(quad >= 0, axis=1)
        quad, positive = quad[whole], positive[whole]
        flip = ~positive if axis == 1 else positive
        quad[flip] = quad[flip][:, ::-1]
        quads.append(quad)
    if not quads:
        return vertices, np.empty((0, 3), dtype=np.int64)
    quad = np.vstack(quads)
    p = vertices[quad]

    def area(i, j, k):
        return np.linalg.norm(np.cross(p[:, j] - p[:, i], p[:, k] - p[:, i]), axis=1)

    first = np.minimum(area(0, 1, 2), area(0, 2, 3)) >= np.minimum(area(0, 1, 3), area(1, 2, 3))
    faces = np.vstack([np.where(first[:, None], quad[:, [0, 1, 2]], quad[:, [0, 1, 3]]),
                       np.where(first[:, None], quad[:, [0, 2, 3]], quad[:, [1, 2, 3]])])
    return vertices, faces


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal of each triangle by its winding."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 0, length, 1.0)
