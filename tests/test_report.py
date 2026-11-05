import json

import numpy as np
import pytest

from slscan import cli, geometry, report, track


def sphere_points(centre, radius, rng, n=800):
    d = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * d / np.linalg.norm(d, axis=1)[:, None]


def test_rigid_fit_recovers_a_motion():
    rng = np.random.default_rng(0)
    source = rng.uniform(-0.1, 0.1, (12, 3))
    rotation = track.rodrigues(np.array([0.3, -0.2, 0.5]))
    translation = np.array([0.01, -0.4, 0.2])
    r, t = geometry.rigid_fit(source, source @ rotation.T + translation)
    assert np.allclose(r, rotation) and np.allclose(t, translation)


def test_report_aligns_datums_and_measures_bars():
    rng = np.random.default_rng(1)
    measured = {"a": [0.0, 0.0, 0.0], "b": [0.08, 0.0, 0.0], "c": [0.0, 0.06, 0.01]}
    rotation = track.rodrigues(np.array([0.1, 0.2, -0.3]))
    translation = np.array([0.02, 0.01, 0.5])
    shifts = {"a": [0.0, 0.0, 0.0], "b": [0.00005, 0.0, 0.0], "c": [0.0, 0.0, 0.0]}
    points, spheres = [], {}
    for name, centre in measured.items():
        scan_centre = rotation @ (np.array(centre) + shifts[name]) + translation
        points.append(sphere_points(scan_centre, 0.0125, rng))
        spheres[name] = {"centre": centre, "radius": 0.0125,
                         "seed": (scan_centre + 0.0003).tolist()}
    result = report.measure(np.vstack(points), {"spheres": spheres, "scale_bars": [["a", "b"]]})
    assert abs(result["scale_bars_m"]["a-b"] - 0.00005) < 1e-9
    assert all(abs(d["radius_error_m"]) < 1e-9 for d in result["datums"].values())
    assert max(d["residual_m"] for d in result["datums"].values()) < 0.00005


def cap_points(centre, radius, rng, n=1500, noise=2e-5):
    """Points on the half of a sphere that faces the origin."""
    centre = np.asarray(centre, dtype=np.float64)
    d = rng.normal(size=(4 * n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    d = d[d @ centre < 0][:n]
    return centre + radius * d + rng.normal(scale=noise, size=d.shape)


def scan_of(measured, rotation, translation, rng, extra=()):
    parts = [cap_points(rotation @ np.asarray(c) + translation, 0.0125, rng)
             for c in measured.values()]
    parts += [cap_points(c, r, rng, 4000) for c, r in extra]
    xy = rng.uniform(-0.15, 0.15, (6000, 2))
    parts.append(np.column_stack([xy, 0.56 + 0.1 * xy[:, 0]]))
    return np.vstack(parts)


def test_spheres_are_found_among_other_shapes():
    rng = np.random.default_rng(2)
    centres = np.array([[0.0, 0.0, 0.5], [0.08, 0.01, 0.52], [-0.05, 0.06, 0.49]])
    points = scan_of({str(i): c for i, c in enumerate(centres)}, np.eye(3), np.zeros(3), rng,
                     extra=[([0.1, -0.08, 0.5], 0.03)])
    found = report.find_spheres(points, 0.0125)
    assert len(found) == 3
    assert max(np.min(np.linalg.norm(found - c, axis=1)) for c in centres) < 0.0002


def test_unseeded_datums_match_their_spheres():
    rng = np.random.default_rng(3)
    measured = {"a": [0.0, 0.0, 0.0], "b": [0.08, 0.0, 0.0], "c": [0.0, 0.06, 0.01],
                "d": [0.05, 0.07, -0.02]}
    rotation = track.rodrigues(np.array([0.1, 0.2, -0.3]))
    translation = np.array([0.02, 0.01, 0.5])
    points = scan_of(measured, rotation, translation, rng, extra=[([-0.09, -0.07, 0.48], 0.0125)])
    spheres = {n: {"centre": c, "radius": 0.0125} for n, c in measured.items()}
    found = report.measure(points, {"spheres": spheres, "scale_bars": [["a", "b"]]})
    seeded = report.measure(points, {"spheres": {
        n: dict(s, seed=(rotation @ np.array(s["centre"]) + translation).tolist())
        for n, s in spheres.items()}, "scale_bars": [["a", "b"]]})
    assert found["datums"] == seeded["datums"]
    assert np.allclose(found["rotation"], rotation.T, atol=1e-4)
    assert max(d["residual_m"] for d in found["datums"].values()) < 0.0001


def test_symmetric_datum_layouts_are_refused():
    rng = np.random.default_rng(4)
    side = 0.08
    measured = {"a": [0.0, 0.0, 0.0], "b": [side, 0.0, 0.0], "c": [side / 2, side * 0.866025, 0.0]}
    points = scan_of(measured, np.eye(3), np.array([0.0, 0.0, 0.5]), rng)
    spheres = {n: {"centre": c, "radius": 0.0125} for n, c in measured.items()}
    with pytest.raises(ValueError, match="symmetric"):
        report.measure(points, {"spheres": spheres})


def test_report_writes_the_cloud_in_the_object_frame(tmp_path):
    rng = np.random.default_rng(5)
    measured = {"a": [0.0, 0.0, 0.0], "b": [0.08, 0.0, 0.0], "c": [0.0, 0.06, 0.01]}
    rotation = track.rodrigues(np.array([0.05, -0.1, 0.2]))
    translation = np.array([-0.01, 0.02, 0.5])
    geometry.write_ply(str(tmp_path / "scan.ply"), scan_of(measured, rotation, translation, rng))
    with open(tmp_path / "datums.json", "w", encoding="utf-8") as f:
        json.dump({"spheres": {n: {"centre": c, "radius": 0.0125} for n, c in measured.items()}}, f)
    assert cli.main(["report", str(tmp_path / "scan.ply"), "--datums", str(tmp_path / "datums.json"),
                     "--aligned", str(tmp_path / "aligned.ply")]) == 0
    aligned = geometry.read_ply(str(tmp_path / "aligned.ply"))
    near = aligned[np.linalg.norm(aligned - measured["b"], axis=1) < 0.014]
    assert np.linalg.norm(geometry.fit_sphere(near)[0] - measured["b"]) < 0.0001


def test_mesh_faces_read_back(tmp_path):
    from slscan import mesh

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    mesh.write_mesh(str(tmp_path / "mesh.ply"), vertices, faces)
    geometry.write_ply(str(tmp_path / "points.ply"), vertices)
    assert np.array_equal(geometry.read_ply_faces(str(tmp_path / "mesh.ply")), faces)
    assert len(geometry.read_ply_faces(str(tmp_path / "points.ply"))) == 0
    assert np.allclose(geometry.read_ply(str(tmp_path / "mesh.ply")), vertices)
