"""Datum residuals and scale bar lengths of a scan. A datum file names spheres on the
object, each with its measured centre in the object's frame, its radius and optionally a
seed near its centre in the scan's frame. Without seeds, spheres of each datum radius
are found in the scan and matched to the datums by their distances. Each sphere is
fitted to the scan's points within its radius and a margin of its seed, the fitted
centres are aligned to the measured ones by a rigid fit, and each datum's residual is
its distance after that alignment."""

import itertools
import json

import numpy as np

from . import geometry

RADIUS_TOLERANCE = 0.1
FORM_LIMIT = 0.05
MIN_SPREAD = 0.3
MIN_POINTS = 30


def load_datums(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vote_peaks(points: np.ndarray, normals: np.ndarray, radius: float, count: int) -> np.ndarray:
    """Return up to count centres (K, 3), most votes first, at least 2 radius apart. Each
    point votes for the cells of radius / 4 at radius along both directions of its normal,
    and a cell's votes are summed over its neighbours."""
    votes = np.vstack([points + radius * normals, points - radius * normals])
    cell = radius / 4.0
    cells, counts = np.unique(np.floor(votes / cell).astype(np.int64), axis=0,
                              return_counts=True)
    low = cells.min(axis=0) - 1
    span = cells.max(axis=0) - low + 2

    def code(keys):
        k = keys - low
        return (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]

    codes = code(cells)
    total = np.zeros(len(cells))
    for offset in itertools.product((-1, 0, 1), repeat=3):
        c = code(cells + np.array(offset))
        i = np.minimum(np.searchsorted(codes, c), len(codes) - 1)
        hit = codes[i] == c
        total[hit] += counts[i[hit]]
    peaks = []
    for k in np.argsort(-total, kind="stable"):
        centre = (cells[k] + 0.5) * cell
        if all(np.linalg.norm(centre - p) >= 2.0 * radius for p in peaks):
            peaks.append(centre)
            if len(peaks) == count:
                break
    return np.array(peaks).reshape(-1, 3)


def find_spheres(points: np.ndarray, radius: float, candidates: int = 40,
                 neighbours: int = 12) -> np.ndarray:
    """Return centres (K, 3) of spheres in points (N, 3) whose fitted radius lies within
    RADIUS_TOLERANCE of radius, most votes first. Voting runs on points averaged over
    voxels of radius / 10. A fit is kept when its form RMS is under FORM_LIMIT of radius
    and the directions from its centre to its points spread by MIN_SPREAD or more, so a
    flat patch is not taken for a cap."""
    from scipy.spatial import cKDTree

    from . import track

    coarse = track.voxel_average(points, radius / 10.0)
    if len(coarse) < neighbours:
        return np.empty((0, 3))
    peaks = vote_peaks(coarse, track.normals(coarse, neighbours), radius, candidates)
    tree = cKDTree(points)
    found = []
    for centre in peaks:
        fit = None
        for band in (0.3, 0.15, 0.05):
            near = points[tree.query_ball_point(centre, radius * (1.0 + band))]
            shell = near[np.abs(np.linalg.norm(near - centre, axis=1) - radius) < band * radius]
            if len(shell) < MIN_POINTS:
                fit = None
                break
            centre, fitted, rms = geometry.fit_sphere(shell)
            fit = (fitted, rms, shell)
        if fit is None:
            continue
        fitted, rms, shell = fit
        directions = (shell - centre) / np.linalg.norm(shell - centre, axis=1)[:, np.newaxis]
        spread = np.sqrt(np.mean(np.sum((directions - directions.mean(axis=0)) ** 2, axis=1)))
        if (abs(fitted - radius) < RADIUS_TOLERANCE * radius and rms < FORM_LIMIT * radius
                and spread >= MIN_SPREAD
                and all(np.linalg.norm(centre - f) >= radius for f in found)):
            found.append(centre)
    return np.array(found).reshape(-1, 3)


def match_datums(found: np.ndarray, found_radii: np.ndarray, measured: np.ndarray,
                 radii: np.ndarray, tolerance: float = 0.003) -> np.ndarray:
    """Return for each measured centre (M, 3) the index of its found centre (K, 3). The
    rigid motion that takes three found spheres onto the measured triangle of largest area,
    with distances and radii that agree, is kept when it places every datum within
    tolerance of a found sphere of its radius. A layout that two such motions fit is
    refused as symmetric."""
    if len(measured) < 3:
        raise ValueError("datum alignment needs three or more spheres")
    triples = list(itertools.combinations(range(len(measured)), 3))
    areas = [np.linalg.norm(np.cross(measured[j] - measured[i], measured[k] - measured[i]))
             for i, j, k in triples]
    anchor = triples[int(np.argmax(areas))]
    sides = [np.linalg.norm(measured[a] - measured[b]) for a, b in
             ((anchor[0], anchor[1]), (anchor[1], anchor[2]), (anchor[0], anchor[2]))]
    matches = []
    for picked in itertools.permutations(range(len(found)), 3):
        if any(found_radii[p] != radii[a] for p, a in zip(picked, anchor)):
            continue
        lengths = [np.linalg.norm(found[picked[0]] - found[picked[1]]),
                   np.linalg.norm(found[picked[1]] - found[picked[2]]),
                   np.linalg.norm(found[picked[0]] - found[picked[2]])]
        if max(abs(a - b) for a, b in zip(lengths, sides)) > tolerance:
            continue
        rotation, translation = geometry.rigid_fit(found[list(picked)], measured[list(anchor)])
        placed = found @ rotation.T + translation
        assignment = np.full(len(measured), -1)
        for d in range(len(measured)):
            gaps = np.linalg.norm(placed - measured[d], axis=1)
            gaps[found_radii != radii[d]] = np.inf
            gaps[assignment[assignment >= 0]] = np.inf
            best = int(np.argmin(gaps))
            if gaps[best] < tolerance:
                assignment[d] = best
        if np.all(assignment >= 0) and not any(np.array_equal(assignment, m) for m in matches):
            matches.append(assignment)
    if not matches:
        raise ValueError("no set of spheres in the scan matches the datum layout")
    if len(matches) > 1:
        raise ValueError("the datum layout is symmetric: more than one set of spheres fits it")
    return matches[0]


def find_seeds(points: np.ndarray, datums: dict, tolerance: float = 0.003) -> dict:
    """Return the datum file with each sphere's seed set to the centre of the sphere found
    for it in points."""
    names = list(datums["spheres"])
    radii = np.array([datums["spheres"][n]["radius"] for n in names], dtype=np.float64)
    found, found_radii = [], []
    for radius in np.unique(radii):
        centres = find_spheres(points, float(radius))
        found.extend(centres)
        found_radii.extend([radius] * len(centres))
    measured = np.array([datums["spheres"][n]["centre"] for n in names], dtype=np.float64)
    assignment = match_datums(np.array(found).reshape(-1, 3), np.array(found_radii), measured,
                              radii, tolerance)
    seeded = {"spheres": {n: dict(datums["spheres"][n], seed=found[i].tolist())
                          for n, i in zip(names, assignment)}}
    if "scale_bars" in datums:
        seeded["scale_bars"] = datums["scale_bars"]
    return seeded


def measure(points: np.ndarray, datums: dict, margin: float = 0.0015) -> dict:
    """Fit each datum sphere near its seed, align the fits to the measured centres, and
    return fitted radii, residuals and scale bar lengths, all in metres. Spheres are found
    first when any datum has no seed."""
    if any("seed" not in s for s in datums["spheres"].values()):
        datums = find_seeds(points, datums)
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
