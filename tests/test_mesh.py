import numpy as np

from slscan import geometry, mesh, tsdf


def sphere_volume(centre, radius, voxel=0.0005, truncation=0.001):
    half = radius + 3 * truncation
    offset = np.array([0.37, 0.21, 0.13]) * voxel
    volume = tsdf.Volume(centre - half - offset, centre + half, voxel, truncation)
    axes = [volume.lower[i] + voxel * np.arange(n) for i, n in enumerate(volume.shape)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1) - radius
    volume.tsdf[:] = np.clip(distance / truncation, -1.0, 1.0)
    volume.weight[:] = 1.0
    return volume


def test_sphere_mesh_is_closed_and_faces_out():
    centre, radius = np.array([0.01, -0.02, 0.5]), 0.0125
    vertices, faces = mesh.surface_nets(sphere_volume(centre, radius))
    assert len(faces) > 1000
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    assert np.all(uses == 2)
    centroids = vertices[faces].mean(axis=1)
    outward = np.einsum("ij,ij->i", mesh.face_normals(vertices, faces), centroids - centre)
    assert np.all(outward > 0)
    _, fitted, rms = geometry.fit_sphere(vertices)
    assert abs(fitted - radius) < 0.00001 and rms < 0.000005


def test_mesh_files_round_trip(tmp_path):
    vertices, faces = mesh.surface_nets(sphere_volume(np.array([0.0, 0.0, 0.5]), 0.004))
    for name in ("m.ply", "m.obj", "m.stl"):
        mesh.write_mesh(str(tmp_path / name), vertices, faces)
    assert np.allclose(geometry.read_ply(str(tmp_path / "m.ply")), vertices, atol=1e-7)
    lines = (tmp_path / "m.obj").read_text().splitlines()
    assert sum(line.startswith("v ") for line in lines) == len(vertices)
    assert sum(line.startswith("f ") for line in lines) == len(faces)
    stl = (tmp_path / "m.stl").read_bytes()
    assert int(np.frombuffer(stl[80:84], "<u4")[0]) == len(faces)
    assert len(stl) == 84 + 50 * len(faces)


def test_decimated_sphere_stays_closed_and_round():
    centre, radius = np.array([0.01, -0.02, 0.5]), 0.0125
    vertices, faces = mesh.surface_nets(sphere_volume(centre, radius))
    fewer, kept = mesh.decimate(vertices, faces, 4000)
    assert len(kept) <= 4000 and len(fewer) < len(vertices) / 4
    edges = np.sort(np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]]), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    assert np.all(uses == 2)
    assert np.max(np.abs(np.linalg.norm(fewer - centre, axis=1) - radius)) < 0.00002
