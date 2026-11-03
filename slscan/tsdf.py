"""Truncated signed distance fusion of tracked depth frames, the reference for a coarse
TSDF kernel. Each depth frame is resampled into a depth image at its sample grid, and
every voxel takes the running mean of its projective distance to each frame's surface."""

import numpy as np


class Volume:
    """A box of voxels from lower towards upper, each with a truncated signed distance in
    units of the truncation and a weight."""

    def __init__(self, lower, upper, voxel: float, truncation: float):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.voxel = voxel
        self.truncation = truncation
        self.shape = tuple(int(n) for n in
                           np.ceil((np.asarray(upper) - self.lower) / voxel).astype(int) + 1)
        self.tsdf = np.ones(self.shape, np.float32)
        self.weight = np.zeros(self.shape, np.float32)

    def slab(self, k: int) -> np.ndarray:
        """Centres (nx, ny, 3) in the world of the voxels in layer k along z."""
        xs = self.lower[0] + self.voxel * np.arange(self.shape[0])
        ys = self.lower[1] + self.voxel * np.arange(self.shape[1])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([gx, gy, np.full_like(gx, self.lower[2] + self.voxel * k)], axis=-1)


def depth_grid(rect, points: np.ndarray, step: int, window: int,
               size: tuple[int, int]) -> np.ndarray:
    """Rectified depth of each sample of a depth frame at its node of the sample grid, NaN
    where the frame has no point. Points are in the left camera's frame, as triangulate
    returns them, and the grid starts at window px with a node every step px."""
    width, height = size
    grid = np.full((len(range(window, height - window, step)),
                    len(range(window, width - window, step))), np.nan)
    if len(points) == 0:
        return grid
    x = points @ rect.left_rotation.T
    p = rect.left_projection
    u = p[0, 0] * x[:, 0] / x[:, 2] + p[0, 2]
    v = p[1, 1] * x[:, 1] / x[:, 2] + p[1, 2]
    gx = np.rint((u - window) / step).astype(int)
    gy = np.rint((v - window) / step).astype(int)
    inside = (gx >= 0) & (gx < grid.shape[1]) & (gy >= 0) & (gy < grid.shape[0])
    grid[gy[inside], gx[inside]] = x[inside, 2]
    return grid


def integrate(volume: Volume, grid: np.ndarray, rect, rotation: np.ndarray,
              translation: np.ndarray, step: int, window: int) -> None:
    """Fold one depth grid into the volume. The frame's pose maps its left camera frame
    into the world. A voxel reads the grid's depth by bilinear interpolation where all
    four nodes have depth within one truncation of each other, and keeps its distance
    when it lies no further than one truncation behind the surface."""
    p = rect.left_projection
    to_rect = rect.left_rotation @ rotation.T
    for k in range(volume.shape[2]):
        world = volume.slab(k)
        x = (world - translation) @ to_rect.T
        z = x[..., 2]
        ahead = z > 1e-6
        u = p[0, 0] * x[..., 0] / np.where(ahead, z, 1.0) + p[0, 2]
        v = p[1, 1] * x[..., 1] / np.where(ahead, z, 1.0) + p[1, 2]
        gx = (u - window) / step
        gy = (v - window) / step
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        inside = (ahead & (ix >= 0) & (ix < grid.shape[1] - 1)
                  & (iy >= 0) & (iy < grid.shape[0] - 1))
        ix, iy = np.where(inside, ix, 0), np.where(inside, iy, 0)
        corners = np.stack([grid[iy, ix], grid[iy, ix + 1], grid[iy + 1, ix],
                            grid[iy + 1, ix + 1]])
        valid = inside & np.all(np.isfinite(corners), axis=0)
        span = np.where(valid, corners.max(axis=0) - corners.min(axis=0), np.inf)
        valid &= span <= volume.truncation
        fx, fy = gx - ix, gy - iy
        depth = ((corners[0] * (1 - fx) + corners[1] * fx) * (1 - fy)
                 + (corners[2] * (1 - fx) + corners[3] * fx) * fy)
        sdf = np.where(valid, depth - z, -np.inf)
        keep = sdf > -volume.truncation
        value = np.minimum(1.0, sdf[keep] / volume.truncation)
        tsdf = volume.tsdf[:, :, k]
        weight = volume.weight[:, :, k]
        tsdf[keep] = (tsdf[keep] * weight[keep] + value) / (weight[keep] + 1.0)
        weight[keep] += 1.0


def surface(volume: Volume, return_keys: bool = False):
    """Zero crossings of the distance between neighbouring observed voxels along each
    axis, both inside the truncation band, as points in the world. With return_keys, also
    return each crossing's key: three times the flat index of its lower voxel plus its
    axis."""
    points, keys = [], []
    observed = (volume.weight > 0) & (np.abs(volume.tsdf) < 1.0)
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        da, db = volume.tsdf[tuple(a)], volume.tsdf[tuple(b)]
        cross = observed[tuple(a)] & observed[tuple(b)] & ((da > 0) != (db > 0))
        idx = np.argwhere(cross)
        t = da[cross] / (da[cross] - db[cross])
        pos = volume.lower + volume.voxel * idx.astype(np.float64)
        pos[:, axis] += volume.voxel * t
        points.append(pos)
        keys.append(np.ravel_multi_index(idx.T, volume.shape) * 3 + axis)
    if return_keys:
        return np.vstack(points), np.concatenate(keys)
    return np.vstack(points)


def fuse(frames: list, tracked: list, rect, step: int, window: int, voxel: float = 0.0005,
         truncation: float = 0.001) -> tuple[Volume, np.ndarray]:
    """Fold tracked depth frames into a volume around all their points. Frames are points
    in each frame's left camera frame and tracked their poses into the world, as
    track.fuse returns them. Returns the volume and its surface points."""
    placed = [f @ r.T + t for f, (r, t) in zip(frames, tracked) if len(f)]
    if not placed:
        return Volume(np.zeros(3), np.zeros(3), voxel, truncation), np.empty((0, 3))
    every = np.vstack(placed)
    volume = Volume(every.min(axis=0) - 2 * truncation, every.max(axis=0) + 2 * truncation,
                    voxel, truncation)
    for f, (r, t) in zip(frames, tracked):
        integrate(volume, depth_grid(rect, f, step, window, rect.size), rect, r, t, step, window)
    return volume, surface(volume)
