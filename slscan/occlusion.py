"""Which surface points a stereo head can measure: a point counts for a pair of cameras
when both cameras and the pattern light see it, within their fields and at an angle
under a grazing limit. Used to weigh a third camera against occlusion before any
hardware is built."""

import numpy as np

from . import synthetic


def camera_centre(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return a camera's centre in the world from its pose (world to camera)."""
    return -rotation.T @ translation


def sees(points: np.ndarray, normals: np.ndarray, scene: list, rotation: np.ndarray,
         translation: np.ndarray, device: "synthetic.Pinhole",
         grazing_deg: float = 75.0) -> np.ndarray:
    """Return which points (N, 3) with outward normals (N, 3) a device at pose (world to
    device) sees: in front of it, inside its image, facing it within grazing_deg of the
    normal, and with nothing of the scene between."""
    centre = camera_centre(rotation, translation)
    local = points @ rotation.T + translation
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = local @ device.matrix.T
        u, v = pixels[:, 0] / pixels[:, 2], pixels[:, 1] / pixels[:, 2]
    inside = (local[:, 2] > 0) & (u >= 0) & (u <= device.width - 1) & (v >= 0) \
        & (v <= device.height - 1)
    towards = centre - points
    distance = np.linalg.norm(towards, axis=1)
    facing = np.einsum("ij,ij->i", normals, towards) > distance * np.cos(np.radians(grazing_deg))
    t, _ = synthetic.nearest_hit(scene, centre, points - centre)
    clear = t >= 1.0 - 1e-6
    return inside & facing & clear


def channel(width: float = 0.02, depth: float = 0.02, length: float = 0.08,
            distance: float = 0.5, step: float = 0.0005) -> tuple[list, np.ndarray, np.ndarray]:
    """Return a slot along the head's y axis, width across and depth deep, cut into a
    flat top at distance: its shapes, and sample points with their outward normals on
    the floor, both walls and the top beside the slot."""
    half_w, half_l = width / 2.0, length / 2.0
    top, floor = distance, distance + depth
    shapes = [synthetic.facing_plate([0.0, 0.0, floor], [0, 0, -1], [1, 0, 0], half_w, half_l),
              synthetic.facing_plate([-half_w, 0.0, floor - depth / 2], [1, 0, 0], [0, 0, 1],
                                     depth / 2, half_l),
              synthetic.facing_plate([half_w, 0.0, floor - depth / 2], [-1, 0, 0], [0, 0, 1],
                                     depth / 2, half_l),
              synthetic.facing_plate([-half_w - 0.02, 0.0, top], [0, 0, -1], [1, 0, 0], 0.02,
                                     half_l),
              synthetic.facing_plate([half_w + 0.02, 0.0, top], [0, 0, -1], [1, 0, 0], 0.02,
                                     half_l)]
    ys = np.arange(-half_l + step / 2, half_l, step)
    across = np.arange(-half_w + step / 2, half_w, step)
    down = np.arange(top + step / 2, floor, step)
    beside = np.arange(half_w + step / 2, half_w + 0.04, step)
    parts = [(np.stack(np.meshgrid(across, ys, [floor], indexing="ij"), -1), [0, 0, -1]),
             (np.stack(np.meshgrid([-half_w], ys, down, indexing="ij"), -1), [1, 0, 0]),
             (np.stack(np.meshgrid([half_w], ys, down, indexing="ij"), -1), [-1, 0, 0]),
             (np.stack(np.meshgrid(-beside, ys, [top], indexing="ij"), -1), [0, 0, -1]),
             (np.stack(np.meshgrid(beside, ys, [top], indexing="ij"), -1), [0, 0, -1])]
    points = np.vstack([p.reshape(-1, 3) for p, _ in parts])
    normals = np.vstack([np.tile(n, (p.reshape(-1, 3).shape[0], 1)) for p, n in parts])
    return shapes, points, normals.astype(np.float64)


def coverage(scene: list, points: np.ndarray, normals: np.ndarray, rig, third=None,
             grazing_deg: float = 75.0) -> dict:
    """Return the share of points each pair measures and the share any pair measures:
    the left and right cameras of rig, and with third, a pose (world to camera) for a
    third camera like the others, the pairs it makes with each of them."""
    light = sees(points, normals, scene, np.eye(3), np.zeros(3), rig.light, grazing_deg)
    seen = {"left": sees(points, normals, scene, rig.left_rotation, rig.left_translation,
                         rig.left, grazing_deg),
            "right": sees(points, normals, scene, rig.right_rotation, rig.right_translation,
                          rig.right, grazing_deg)}
    pairs = [("left", "right")]
    if third is not None:
        seen["third"] = sees(points, normals, scene, third[0], third[1], rig.left, grazing_deg)
        pairs += [("left", "third"), ("third", "right")]
    measured = {f"{a}-{b}": light & seen[a] & seen[b] for a, b in pairs}
    union = np.logical_or.reduce(list(measured.values()))
    return {**{k: float(np.mean(v)) for k, v in measured.items()}, "any": float(np.mean(union))}


def third_cameras(distance: float = 0.5) -> dict:
    """Candidate third cameras for the near head, as poses (world to camera) looking at the
    crossing point: 0.1 m above and below the pattern light and 0.05 m beside it."""
    target = np.array([0.0, 0.0, distance])
    return {name: synthetic.look_at(np.array(eye), target, (0.0, -1.0, 0.0))
            for name, eye in (("above", [0.0, -0.1, 0.0]), ("below", [0.0, 0.1, 0.0]),
                              ("beside", [0.05, 0.0, 0.0]))}
