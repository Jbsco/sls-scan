"""Datum residuals and scale bar lengths of a scan. A datum file names spheres on the
object, each with its measured centre in the object's frame, its radius and a seed near
its centre in the scan's frame. Each sphere is fitted to the scan's points within its
radius and a margin of its seed, the fitted centres are aligned to the measured ones by
a rigid fit, and each datum's residual is its distance after that alignment."""

import json

import numpy as np

from . import geometry


def load_datums(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def measure(points: np.ndarray, datums: dict, margin: float = 0.0015) -> dict:
    """Fit each datum sphere near its seed, align the fits to the measured centres, and
    return fitted radii, residuals and scale bar lengths, all in metres."""
    names, fitted, measured, radii = [], [], [], []
    for name, sphere in datums["spheres"].items():
        seed = np.asarray(sphere["seed"], dtype=np.float64)
        near = points[np.linalg.norm(points - seed, axis=1) < sphere["radius"] + margin]
        if len(near) < 10:
            raise ValueError(f"datum {name}: {len(near)} points near its seed")
        centre, radius, _ = geometry.fit_sphere(near)
        names.append(name)
        fitted.append(centre)
        measured.append(np.asarray(sphere["centre"], dtype=np.float64))
        radii.append(radius - sphere["radius"])
    fitted, measured = np.array(fitted), np.array(measured)
    rotation, translation = geometry.rigid_fit(fitted, measured)
    aligned = fitted @ rotation.T + translation
    residuals = np.linalg.norm(aligned - measured, axis=1)
    bars = {}
    for a, b in datums.get("scale_bars", []):
        i, j = names.index(a), names.index(b)
        bars[f"{a}-{b}"] = float(np.linalg.norm(fitted[i] - fitted[j])
                                 - np.linalg.norm(measured[i] - measured[j]))
    return {"datums": {n: {"residual_m": float(r), "radius_error_m": float(e)}
                       for n, r, e in zip(names, residuals, radii)},
            "scale_bars_m": bars,
            "rotation": rotation.tolist(), "translation_m": translation.tolist()}
