"""Triangle meshes from a TSDF volume by surface nets, their decimation by quadric error,
and their export as PLY, OBJ and STL. Each cell between eight observed voxels whose
distances change sign gets one vertex, the mean of its edges' zero crossings, and each
crossing edge joins the four cells around it in a quad facing the side where the distance
is positive, split along the diagonal that leaves the larger smallest triangle."""

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


def decimate(vertices: np.ndarray, faces: np.ndarray, target: int,
             boundary_weight: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Collapse edges by least quadric error (Garland and Heckbert) until at most target
    triangles remain. Each vertex sums the planes of its triangles; boundary edges add a
    plane through them perpendicular to their triangle, weighted by boundary_weight, so
    open borders keep their shape. A collapse that would flip a triangle is skipped.
    Returns the remaining vertices and triangles."""
    import heapq

    v = vertices.astype(np.float64).copy()
    f = faces.astype(np.int64).copy()
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    n = n / np.where(area > 0, area, 1.0)[:, None]
    planes = np.column_stack([n, -np.einsum("ij,ij->i", n, a)])
    q = np.zeros((len(v), 4, 4))
    kp = np.einsum("fi,fj->fij", planes, planes) * (area / 2)[:, None, None]
    for k in range(3):
        np.add.at(q, f[:, k], kp)
    edge_faces = {}
    for i, tri in enumerate(f):
        for k in range(3):
            e = (min(tri[k], tri[(k + 1) % 3]), max(tri[k], tri[(k + 1) % 3]))
            edge_faces.setdefault(e, []).append(i)
    for (p0, p1), owners in edge_faces.items():
        if len(owners) == 1:
            d = v[p1] - v[p0]
            side = np.cross(d, n[owners[0]])
            norm = np.linalg.norm(side)
            if norm > 0:
                side /= norm
                plane = np.append(side, -side @ v[p0])
                kb = np.outer(plane, plane) * boundary_weight * np.linalg.norm(d) ** 2
                q[p0] += kb
                q[p1] += kb
    vertex_faces = [set() for _ in range(len(v))]
    for i, tri in enumerate(f):
        for k in tri:
            vertex_faces[k].add(i)
    alive = np.ones(len(f), dtype=bool)
    version = np.zeros(len(v), dtype=np.int64)
    removed = np.zeros(len(v), dtype=bool)

    def place(p0, p1):
        m = q[p0] + q[p1]
        a3 = m[:3, :3]
        try:
            x = np.linalg.solve(a3, -m[:3, 3])
            if np.linalg.norm(x - (v[p0] + v[p1]) / 2) > 4 * np.linalg.norm(v[p0] - v[p1]):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            x = (v[p0] + v[p1]) / 2
        h = np.append(x, 1.0)
        return float(h @ m @ h), x

    heap = []
    for p0, p1 in edge_faces:
        heap.append((place(p0, p1)[0], p0, p1, 0, 0))
    heapq.heapify(heap)
    count = len(f)
    while count > target and heap:
        cost, p0, p1, s0, s1 = heapq.heappop(heap)
        if removed[p0] or removed[p1] or version[p0] != s0 or version[p1] != s1:
            continue
        _, x = place(p0, p1)
        shared = vertex_faces[p0] & vertex_faces[p1]
        changed = (vertex_faces[p0] | vertex_faces[p1]) - shared
        flips = False
        for i in changed:
            tri = f[i]
            old = v[tri]
            new = old.copy()
            new[(tri == p0) | (tri == p1)] = x
            n_old = np.cross(old[1] - old[0], old[2] - old[0])
            n_new = np.cross(new[1] - new[0], new[2] - new[0])
            if n_old @ n_new <= 0:
                flips = True
                break
        if flips:
            continue
        v[p0] = x
        q[p0] += q[p1]
        removed[p1] = True
        for i in shared:
            alive[i] = False
            count -= 1
            for k in f[i]:
                vertex_faces[k].discard(i)
        for i in vertex_faces[p1]:
            f[i][f[i] == p1] = p0
            vertex_faces[p0].add(i)
        vertex_faces[p1] = set()
        version[p0] += 1
        neighbours = {k for i in vertex_faces[p0] for k in f[i]} - {p0}
        for k in neighbours:
            cost, _ = place(p0, k)
            lo, hi = (p0, k) if p0 < k else (k, p0)
            heapq.heappush(heap, (cost, lo, hi, version[lo], version[hi]))
    kept = f[alive]
    used = np.unique(kept)
    index = np.full(len(v), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    return v[used], index[kept]


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal of each triangle by its winding."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 0, length, 1.0)


def write_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a mesh as binary PLY, ASCII OBJ or binary STL by the path's extension."""
    if path.endswith(".obj"):
        with open(path, "w", encoding="ascii") as f:
            for p in vertices:
                f.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            for t in faces + 1:
                f.write(f"f {t[0]} {t[1]} {t[2]}\n")
    elif path.endswith(".stl"):
        records = np.zeros(len(faces), dtype=[("normal", "<f4", 3), ("corners", "<f4", (3, 3)),
                                              ("attribute", "<u2")])
        records["normal"] = face_normals(vertices, faces)
        records["corners"] = vertices[faces]
        with open(path, "wb") as f:
            f.write(b"\0" * 80)
            f.write(np.uint32(len(faces)).tobytes())
            f.write(records.tobytes())
    elif path.endswith(".ply"):
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(vertices)}\nproperty float x\nproperty float y\n"
                  f"property float z\nelement face {len(faces)}\n"
                  "property list uchar int vertex_indices\nend_header\n")
        rows = np.zeros(len(faces), dtype=[("n", "u1"), ("v", "<i4", 3)])
        rows["n"] = 3
        rows["v"] = faces
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
            f.write(rows.tobytes())
    else:
        raise ValueError(f"{path}: mesh files end in .ply, .obj or .stl")
