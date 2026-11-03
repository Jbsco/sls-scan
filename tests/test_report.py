import numpy as np

from slscan import geometry, report, track


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
