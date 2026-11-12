//! Tracking and fusion of depth frames: point-to-plane ICP against a model kept as voxel
//! averages, following `track.fuse` in `slscan/track.py`.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::thread;

pub type Mat3 = [[f64; 3]; 3];
pub type Vec3 = [f64; 3];

const NEIGHBOURS: usize = 12;
const LIMITS: [f64; 4] = [0.004, 0.002, 0.001, 0.0005];
const ITERATIONS: usize = 10;
const MIN_PAIRS: usize = 50;
/// A frame's dot pairs with the nearest map dot within 0.4 of the map's median dot
/// spacing, at most this, as `track.fuse` takes it.
const DOT_RADIUS: f64 = 0.03;
/// Fewest dot pairs that join a frame's ICP.
const MIN_DOT_PAIRS: usize = 3;
/// Once pairs have been taken with the pattern light off, a lit frame's ICP holds its
/// starting pose along directions whose normal-equation eigenvalue lies below this share
/// of the largest, as `track.fuse` takes it.
const HOLD_BELOW: f64 = 1e-4;
/// Fewest points a thread takes, so that small frames stay on one thread.
const CHUNK: usize = 2048;
/// A frame's ICP runs on model points within this of the box around the frame, in metres.
const NEAR_MARGIN: f64 = 0.02;
/// Cell of the search for changed points near a model point, in voxels.
const RENEW_CELL_VOXELS: f64 = 8.0;
/// Normals are computed again from model points within this of those that need one, in
/// voxels. A normal that reaches farther is computed from the whole model.
const RENEW_MARGIN_VOXELS: f64 = 16.0;

/// Multiply-and-rotate hash for cell keys, much cheaper than the default for short
/// integer keys that need no protection from chosen inputs.
#[derive(Default)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(word))
                .wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
}

type CellMap<V> = HashMap<[i64; 3], V, BuildHasherDefault<CellHasher>>;

/// Run `work` on consecutive chunks of `0..n` on scoped threads and return the results in
/// chunk order.
pub(crate) fn chunked<T: Send>(
    n: usize,
    work: impl Fn(std::ops::Range<usize>) -> T + Sync,
) -> Vec<T> {
    let threads = thread::available_parallelism().map_or(1, |p| p.get());
    let chunks = n.div_ceil(CHUNK).clamp(1, threads);
    let size = n.div_ceil(chunks);
    if chunks == 1 {
        return vec![work(0..n)];
    }
    thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = (0..chunks)
            .map(|c| scope.spawn(move || work(c * size..((c + 1) * size).min(n))))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// A rigid motion taking points from a frame into the model: `rotation · p + translation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub rotation: Mat3,
    pub translation: Vec3,
}

impl Pose {
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    pub fn apply(&self, p: &Vec3) -> Vec3 {
        add(&mul_vec(&self.rotation, p), &self.translation)
    }
}

pub(crate) fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub(crate) fn mul_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    [dot(&a[0], v), dot(&a[1], v), dot(&a[2], v)]
}

pub(crate) fn transpose(a: &Mat3) -> Mat3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

/// Rotation of the rotation vector `w`.
pub fn rodrigues(w: &Vec3) -> Mat3 {
    let angle = dot(w, w).sqrt();
    if angle < 1e-12 {
        return Pose::IDENTITY.rotation;
    }
    let k = [w[0] / angle, w[1] / angle, w[2] / angle];
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let kx2 = mul(&kx, &kx);
    let (s, c) = (angle.sin(), 1.0 - angle.cos());
    let mut out = Pose::IDENTITY.rotation;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += s * kx[i][j] + c * kx2[i][j];
        }
    }
    out
}

/// Rotation vector of a rotation, as `rodrigues` inverts it.
pub fn rotation_vector(m: &Mat3) -> Vec3 {
    let angle = ((m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0)
        .clamp(-1.0, 1.0)
        .acos();
    if angle < 1e-12 {
        return [0.0; 3];
    }
    let w = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]];
    w.map(|v| v * angle / (2.0 * angle.sin()))
}

/// Unit eigenvector of the smallest eigenvalue of a symmetric 3 × 3 matrix, by Jacobi
/// rotations.
fn smallest_eigenvector(m: &Mat3) -> Vec3 {
    let mut a = *m;
    let mut v = Pose::IDENTITY.rotation;
    for _ in 0..50 {
        let (mut p, mut q, mut largest) = (0, 1, 0.0);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if a[i][j].abs() > largest {
                (p, q, largest) = (i, j, a[i][j].abs());
            }
        }
        let scale = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if largest <= 1e-15 * scale.max(f64::MIN_POSITIVE) {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let mut rotation = Pose::IDENTITY.rotation;
        rotation[p][p] = c;
        rotation[q][q] = c;
        rotation[p][q] = s;
        rotation[q][p] = -s;
        a = mul(&transpose(&rotation), &mul(&a, &rotation));
        v = mul(&v, &rotation);
    }
    let smallest = (0..3)
        .min_by(|&i, &j| a[i][i].total_cmp(&a[j][j]))
        .unwrap_or(2);
    [v[0][smallest], v[1][smallest], v[2][smallest]]
}

/// Eigenvalues and unit eigenvectors (as columns) of a symmetric 6 × 6 matrix, by cyclic
/// Jacobi rotations.
fn symmetric_eigen6(m: [[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    let mut a = m;
    let mut v = [[0.0; 6]; 6];
    (0..6).for_each(|i| v[i][i] = 1.0);
    for _ in 0..50 {
        let off: f64 = (0..6)
            .flat_map(|i| (i + 1..6).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..6).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-30 * scale.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..6 {
            for q in p + 1..6 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut a {
                    let (ap, aq) = (row[p], row[q]);
                    row[p] = c * ap - s * aq;
                    row[q] = s * ap + c * aq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for (k, (apk, aqk)) in row_p.iter().zip(&row_q).enumerate() {
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in &mut v {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    (std::array::from_fn(|i| a[i][i]), v)
}

/// The step that solves `ata · step = atb` along directions whose eigenvalue reaches
/// `hold_below` times the largest, and that returns the motion `away` from the starting
/// pose to zero along the others.
fn held_step(ata: [[f64; 6]; 6], atb: [f64; 6], away: &[f64; 6], hold_below: f64) -> [f64; 6] {
    let (values, vectors) = symmetric_eigen6(ata);
    let largest = values.iter().copied().fold(f64::MIN, f64::max);
    let mut step = [0.0; 6];
    for (i, &value) in values.iter().enumerate() {
        let column: [f64; 6] = std::array::from_fn(|r| vectors[r][i]);
        let along = if value < hold_below * largest {
            -(0..6).map(|r| column[r] * away[r]).sum::<f64>()
        } else {
            (0..6).map(|r| column[r] * atb[r]).sum::<f64>() / value
        };
        for r in 0..6 {
            step[r] += along * column[r];
        }
    }
    step
}

/// Solve a 6 × 6 linear system by Gaussian elimination with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let pivot = (c..6).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[pivot][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, pivot);
        b.swap(c, pivot);
        let pivot_row = a[c];
        for r in c + 1..6 {
            let f = a[r][c] / pivot_row[c];
            for (x, p) in a[r][c..].iter_mut().zip(&pivot_row[c..]) {
                *x -= f * p;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 6];
    for r in (0..6).rev() {
        let s: f64 = (r + 1..6).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Points bucketed in cubic cells for neighbour searches.
pub(crate) struct Grid {
    cell: f64,
    cells: CellMap<Vec<u32>>,
    lo: [i64; 3],
    hi: [i64; 3],
}

impl Grid {
    pub(crate) fn new(points: &[Vec3], cell: f64) -> Self {
        let mut cells: CellMap<Vec<u32>> = CellMap::default();
        let (mut lo, mut hi) = ([i64::MAX; 3], [i64::MIN; 3]);
        for (i, p) in points.iter().enumerate() {
            let key = Self::key_of(p, cell);
            for d in 0..3 {
                lo[d] = lo[d].min(key[d]);
                hi[d] = hi[d].max(key[d]);
            }
            cells.entry(key).or_default().push(i as u32);
        }
        Self {
            cell,
            cells,
            lo,
            hi,
        }
    }

    pub(crate) fn key_of(p: &Vec3, cell: f64) -> [i64; 3] {
        [0, 1, 2].map(|d| (p[d] / cell).floor() as i64)
    }

    /// Visit every point in cells whose largest key offset from `centre` is `shell`.
    pub(crate) fn shell(&self, centre: [i64; 3], shell: i64, mut visit: impl FnMut(u32)) {
        for dx in -shell..=shell {
            for dy in -shell..=shell {
                for dz in -shell..=shell {
                    if dx.abs().max(dy.abs()).max(dz.abs()) != shell {
                        continue;
                    }
                    let key = [centre[0] + dx, centre[1] + dy, centre[2] + dz];
                    if let Some(list) = self.cells.get(&key) {
                        list.iter().for_each(|&i| visit(i));
                    }
                }
            }
        }
    }

    /// Nearest point closer than `limit`, which must not exceed the cell size. Cells are
    /// visited in the order of `shell`, and one is passed over when every point in it
    /// would lie farther than the limit or the nearest found so far.
    pub(crate) fn nearest(&self, points: &[Vec3], q: &Vec3, limit: f64) -> Option<usize> {
        let centre = Self::key_of(q, self.cell);
        let slack = 1e-9 * self.cell;
        let gap = |d: usize, offset: i64| -> f64 {
            let to_face = match offset {
                0 => return 0.0,
                1 => (centre[d] + 1) as f64 * self.cell - q[d],
                _ => q[d] - centre[d] as f64 * self.cell,
            };
            (to_face - slack).max(0.0)
        };
        let mut bound = limit * limit;
        let mut best = None;
        for shell in 0..=1i64 {
            for dx in -shell..=shell {
                for dy in -shell..=shell {
                    for dz in -shell..=shell {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != shell {
                            continue;
                        }
                        let g = [gap(0, dx), gap(1, dy), gap(2, dz)];
                        if dot(&g, &g) >= bound {
                            continue;
                        }
                        let key = [centre[0] + dx, centre[1] + dy, centre[2] + dz];
                        let Some(list) = self.cells.get(&key) else {
                            continue;
                        };
                        for &i in list {
                            let d = sub(&points[i as usize], q);
                            let d2 = dot(&d, &d);
                            if d2 < bound {
                                bound = d2;
                                best = Some(i as usize);
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// The `k` nearest points, nearest first.
    fn knn(&self, points: &[Vec3], q: &Vec3, k: usize) -> Vec<usize> {
        let centre = Self::key_of(q, self.cell);
        let reach = (0..3)
            .map(|d| {
                (centre[d] - self.lo[d])
                    .abs()
                    .max((self.hi[d] - centre[d]).abs())
            })
            .max()
            .unwrap_or(0);
        let mut found: Vec<(f64, usize)> = Vec::new();
        for shell in 0..=reach {
            self.shell(centre, shell, |i| {
                let d = sub(&points[i as usize], q);
                found.push((dot(&d, &d), i as usize));
            });
            if found.len() >= k {
                found.sort_by(|a, b| a.0.total_cmp(&b.0));
                let bound = shell as f64 * self.cell;
                if found[k - 1].0 <= bound * bound {
                    break;
                }
            }
        }
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found.into_iter().take(k).map(|(_, i)| i).collect()
    }
}

/// Unit normal per point from the plane of its nearest neighbours.
pub(crate) fn normals(points: &[Vec3], cell: f64) -> Vec<Vec3> {
    let grid = Grid::new(points, cell);
    let parts = chunked(points.len(), |range| {
        points[range]
            .iter()
            .map(|p| normal(&grid, points, p))
            .collect::<Vec<_>>()
    });
    parts.into_iter().flatten().collect()
}

/// Unit normal of one point from the plane of its nearest neighbours.
fn normal(grid: &Grid, points: &[Vec3], p: &Vec3) -> Vec3 {
    normal_reach(grid, points, p).0
}

/// Unit normal of one point from the plane of its nearest neighbours, with the squared
/// distance to the farthest of them, or infinity when fewer than `NEIGHBOURS` exist.
fn normal_reach(grid: &Grid, points: &[Vec3], p: &Vec3) -> (Vec3, f64) {
    let idx = grid.knn(points, p, NEIGHBOURS);
    let reach = match idx.last() {
        Some(&last) if idx.len() == NEIGHBOURS => {
            let d = sub(&points[last], p);
            dot(&d, &d)
        }
        _ => f64::INFINITY,
    };
    let n = idx.len() as f64;
    let mut mean = [0.0; 3];
    for &i in &idx {
        mean = add(&mean, &points[i]);
    }
    mean = mean.map(|v| v / n);
    let mut cov = [[0.0; 3]; 3];
    for &i in &idx {
        let d = sub(&points[i], &mean);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    (smallest_eigenvector(&cov), reach)
}

/// The box around points, as its lowest and highest corner.
fn bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    (lo, hi)
}

/// The box around a box's corners after a pose.
fn posed_bounds(pose: &Pose, lo: &Vec3, hi: &Vec3) -> (Vec3, Vec3) {
    let corners: Vec<Vec3> = (0..8)
        .map(|c| pose.apply(&[0, 1, 2].map(|d| if c >> d & 1 == 0 { lo[d] } else { hi[d] })))
        .collect();
    bounds(&corners)
}

/// Point-to-plane ICP of `source` onto `target` from `start`, with match distances
/// `limits`, coarse to fine. Each of `fixed`, a source point and its target such as a
/// reference dot, adds point-to-point terms to every step, which fixes the motion a
/// smooth surface leaves free. With `hold_below` above zero, each step holds the pose at
/// `start` along directions whose normal-equation eigenvalue lies below `hold_below` times
/// the largest, which the surface leaves nearly free, and solves the others from the
/// surface. Those motions turn about the centroid of `source` at `start`, and a turn
/// counts by the shift it gives a point at `source`'s RMS distance from that centroid. A
/// tilt the surface corrects then leaves the centroid's place along the surface as
/// `start` gave it. Returns the pose and the RMS point-to-plane residual of the last
/// matched pairs.
pub(crate) fn icp_with(
    source: &[Vec3],
    target: &[Vec3],
    target_normals: &[Vec3],
    start: Pose,
    limits: &[f64],
    fixed: &[(Vec3, Vec3)],
    hold_below: f64,
) -> (Pose, f64) {
    icp_within(
        source,
        target,
        target_normals,
        start,
        limits,
        fixed,
        hold_below,
        None,
    )
    .unwrap_or((start, f64::INFINITY))
}

/// ICP as `icp_with`, on a target that holds every point of a larger one inside the box
/// `region`. Returns nothing when a step's source points, by the box around them, could
/// match a point outside that box, where the larger target could give another result.
#[allow(clippy::too_many_arguments)]
fn icp_within(
    source: &[Vec3],
    target: &[Vec3],
    target_normals: &[Vec3],
    start: Pose,
    limits: &[f64],
    fixed: &[(Vec3, Vec3)],
    hold_below: f64,
    region: Option<&(Vec3, Vec3)>,
) -> Option<(Pose, f64)> {
    let mut pose = start;
    let mut rms = f64::INFINITY;
    let (source_lo, source_hi) = bounds(source);
    let (centre, reach) = if hold_below > 0.0 && !source.is_empty() {
        let count = source.len() as f64;
        let mut middle = [0.0; 3];
        for s in source {
            middle = add(&middle, s);
        }
        let middle = [middle[0] / count, middle[1] / count, middle[2] / count];
        let squares: f64 = source
            .iter()
            .map(|s| {
                let d = sub(s, &middle);
                dot(&d, &d)
            })
            .sum();
        (start.apply(&middle), (squares / count).sqrt())
    } else {
        ([0.0; 3], 1.0)
    };
    let grids: Vec<Grid> = thread::scope(|scope| {
        let handles: Vec<_> = limits
            .iter()
            .map(|&limit| scope.spawn(move || Grid::new(target, limit)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    for (&limit, grid) in limits.iter().zip(&grids) {
        for _ in 0..ITERATIONS {
            if let Some((lo, hi)) = region {
                let (a, b) = posed_bounds(&pose, &source_lo, &source_hi);
                if (0..3).any(|d| a[d] - limit < lo[d] || b[d] + limit > hi[d]) {
                    return None;
                }
            }
            let parts = chunked(source.len(), |range| {
                let mut ata = [[0.0; 6]; 6];
                let mut atb = [0.0; 6];
                let (mut pairs, mut squares) = (0usize, 0.0);
                for s in &source[range] {
                    let p = pose.apply(s);
                    let Some(j) = grid.nearest(target, &p, limit) else {
                        continue;
                    };
                    let n = &target_normals[j];
                    let residual = dot(&sub(&p, &target[j]), n);
                    let c = cross(&sub(&p, &centre), n);
                    let a = [c[0] / reach, c[1] / reach, c[2] / reach, n[0], n[1], n[2]];
                    for r in 0..6 {
                        for k in 0..6 {
                            ata[r][k] += a[r] * a[k];
                        }
                        atb[r] -= a[r] * residual;
                    }
                    pairs += 1;
                    squares += residual * residual;
                }
                (ata, atb, pairs, squares)
            });
            let mut ata = [[0.0; 6]; 6];
            let mut atb = [0.0; 6];
            let (mut pairs, mut squares) = (0usize, 0.0);
            for (part_ata, part_atb, part_pairs, part_squares) in parts {
                for r in 0..6 {
                    for k in 0..6 {
                        ata[r][k] += part_ata[r][k];
                    }
                    atb[r] += part_atb[r];
                }
                pairs += part_pairs;
                squares += part_squares;
            }
            if pairs < MIN_PAIRS {
                break;
            }
            for (s, q) in fixed {
                let p = pose.apply(s);
                let arm = sub(&p, &centre);
                let arm = [arm[0] / reach, arm[1] / reach, arm[2] / reach];
                for k in 0..3 {
                    let mut a = [0.0; 6];
                    a[..3].copy_from_slice(&match k {
                        0 => [0.0, arm[2], -arm[1]],
                        1 => [-arm[2], 0.0, arm[0]],
                        _ => [arm[1], -arm[0], 0.0],
                    });
                    a[3 + k] = 1.0;
                    let b = q[k] - p[k];
                    for r in 0..6 {
                        for c in 0..6 {
                            ata[r][c] += a[r] * a[c];
                        }
                        atb[r] += a[r] * b;
                    }
                }
            }
            let step = if hold_below > 0.0 {
                let moved_by = mul(&pose.rotation, &transpose(&start.rotation));
                let w = rotation_vector(&moved_by);
                let shift = sub(
                    &sub(&pose.translation, &centre),
                    &mul_vec(&moved_by, &sub(&start.translation, &centre)),
                );
                let away = [
                    reach * w[0],
                    reach * w[1],
                    reach * w[2],
                    shift[0],
                    shift[1],
                    shift[2],
                ];
                held_step(ata, atb, &away, hold_below)
            } else {
                let Some(step) = solve6(ata, atb) else {
                    break;
                };
                step
            };
            let turn = [step[0] / reach, step[1] / reach, step[2] / reach];
            let dr = rodrigues(&turn);
            pose = Pose {
                rotation: mul(&dr, &pose.rotation),
                translation: add(
                    &add(&mul_vec(&dr, &sub(&pose.translation, &centre)), &centre),
                    &[step[3], step[4], step[5]],
                ),
            };
            rms = (squares / pairs as f64).sqrt();
            let rotation_step = dot(&turn, &turn);
            let translation_step = dot(&[step[3], step[4], step[5]], &[step[3], step[4], step[5]]);
            if rotation_step.sqrt() < 1e-7 && translation_step.sqrt() < 1e-8 {
                break;
            }
        }
    }
    Some((pose, rms))
}

/// A model built from depth frames, each aligned to it before it joins.
/// One frame of a sequence for `Fusion::track`: its capture time, its points, or none for
/// a light-off pair, and its reference dots.
pub struct Frame<'a> {
    pub time: f64,
    pub points: Option<&'a [Vec3]>,
    pub dots: &'a [Vec3],
}

pub struct Fusion {
    voxel: f64,
    sums: CellMap<(Vec3, u32)>,
    /// Voxels in order, with each one's average, its normal and that normal's reach.
    keys: Vec<[i64; 3]>,
    model: Vec<Vec3>,
    normals: Vec<Vec3>,
    reach: Vec<f64>,
    pose: Pose,
    pub poses: Vec<Pose>,
    pub last_rms: Option<f64>,
    dot_map: Vec<Vec3>,
    dot_counts: Vec<f64>,
    /// Frame time and pose of each pair taken with the pattern light off.
    fixes: Vec<(f64, Pose)>,
}

impl Fusion {
    pub fn new(voxel: f64) -> Self {
        Self {
            voxel,
            sums: CellMap::default(),
            keys: Vec::new(),
            model: Vec::new(),
            normals: Vec::new(),
            reach: Vec::new(),
            pose: Pose::IDENTITY,
            poses: Vec::new(),
            last_rms: None,
            dot_map: Vec::new(),
            dot_counts: Vec::new(),
            fixes: Vec::new(),
        }
    }

    /// The light-off poses a frame at `time` starts from, as a range of `fixes`, and
    /// whether a later light-off pair could still change that choice: the two before
    /// `time` and the one after; with one before, that one and the two after; and with
    /// none after, the last three.
    fn fixes_around(&self, time: f64) -> (std::ops::Range<usize>, bool) {
        let before = self.fixes.iter().filter(|(t, _)| *t < time).count();
        let after = self.fixes.iter().filter(|(t, _)| *t > time).count();
        let first_after = self.fixes.len() - after;
        if before >= 2 && after >= 1 {
            (before - 2..first_after + 1, false)
        } else if before == 1 && after >= 1 {
            (0..first_after + after.min(2), after < 2)
        } else {
            (before.saturating_sub(3)..before, true)
        }
    }

    /// Whether a lit frame captured at `time` should wait for a later light-off pair
    /// before it is added: one has been posed, and a later one would change the poses
    /// that frame starts from.
    pub fn waits(&self, time: f64) -> bool {
        !self.fixes.is_empty() && self.fixes_around(time).1
    }

    /// Move to the starting pose of a frame at `time`, on the curve through the light-off
    /// poses of `fixes_around`: a parabola through three and a line through two. With
    /// fewer the pose stays.
    fn predict(&mut self, time: f64) {
        let used = &self.fixes[self.fixes_around(time).0];
        let Some((_, reference)) = used.last().filter(|_| used.len() >= 2) else {
            return;
        };
        let (mut turn, mut translation) = ([0.0; 3], [0.0; 3]);
        for (i, (ti, pose)) in used.iter().enumerate() {
            let mut weight = 1.0;
            for (j, (tj, _)) in used.iter().enumerate() {
                if j != i {
                    weight *= (time - tj) / (ti - tj);
                }
            }
            let w = rotation_vector(&mul(&pose.rotation, &transpose(&reference.rotation)));
            for d in 0..3 {
                turn[d] += weight * w[d];
                translation[d] += weight * pose.translation[d];
            }
        }
        self.pose = Pose {
            rotation: mul(&rodrigues(&turn), &reference.rotation),
            translation,
        };
    }

    /// Refine paired map dots by averaging with a frame's dots placed at the current pose,
    /// and add its unpaired dots to the map.
    fn place_dots(&mut self, dots: &[Vec3]) {
        let mut paired = vec![false; dots.len()];
        for (i, j) in self.match_dots(dots, &self.pose) {
            let placed = self.pose.apply(&dots[i]);
            let n = self.dot_counts[j];
            self.dot_map[j] = [0, 1, 2].map(|k| (self.dot_map[j][k] * n + placed[k]) / (n + 1.0));
            self.dot_counts[j] = n + 1.0;
            paired[i] = true;
        }
        for (d, _) in dots.iter().zip(&paired).filter(|(_, &p)| !p) {
            self.dot_map.push(self.pose.apply(d));
            self.dot_counts.push(1.0);
        }
    }

    /// Pose a pair taken with the pattern light off, which has dots and no points, by the
    /// rigid fit of its dots to their map dots when three or more pair, and add its dots
    /// to the map. Returns its pose.
    pub fn add_dark(&mut self, dots: &[Vec3]) -> Pose {
        self.add_dark_at(self.poses.len() as f64, dots)
    }

    /// Pose a light-off pair as `add_dark` does, captured at `time`. Times are in one unit
    /// for every frame of a fusion: capture times here and in `add_at`, or the frame
    /// counts that `add`, `add_with_dots` and `add_dark` give, which are right only when
    /// every frame is added in capture order and none is left out.
    pub fn add_dark_at(&mut self, time: f64, dots: &[Vec3]) -> Pose {
        self.predict(time);
        let matched = self.match_dots(dots, &self.pose);
        if matched.len() >= MIN_DOT_PAIRS {
            let source: Vec<Vec3> = matched.iter().map(|&(i, _)| dots[i]).collect();
            let target: Vec<Vec3> = matched.iter().map(|&(_, j)| self.dot_map[j]).collect();
            self.pose = crate::register::rigid_fit(&source, &target);
        }
        self.place_dots(dots);
        self.fixes.push((time, self.pose));
        self.poses.push(self.pose);
        self.pose
    }

    /// Reference dots seen so far, in the model's frame.
    pub fn dots(&self) -> &[Vec3] {
        &self.dot_map
    }

    /// Match radius: 0.4 of the map's median nearest-neighbour spacing, at most
    /// `DOT_RADIUS`.
    fn dot_radius(&self) -> f64 {
        if self.dot_map.len() < 2 {
            return DOT_RADIUS;
        }
        let mut nearest: Vec<f64> = self
            .dot_map
            .iter()
            .enumerate()
            .map(|(i, a)| {
                self.dot_map
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, b)| {
                        let d = sub(a, b);
                        dot(&d, &d).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        nearest.sort_by(f64::total_cmp);
        let n = nearest.len();
        let median = if n % 2 == 1 {
            nearest[n / 2]
        } else {
            0.5 * (nearest[n / 2 - 1] + nearest[n / 2])
        };
        DOT_RADIUS.min(0.4 * median)
    }

    /// Pairs of a frame dot and the nearest map dot within `dot_radius` after `pose`.
    fn match_dots(&self, dots: &[Vec3], pose: &Pose) -> Vec<(usize, usize)> {
        let radius = self.dot_radius();
        dots.iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let q = pose.apply(d);
                self.dot_map
                    .iter()
                    .enumerate()
                    .map(|(j, m)| {
                        let e = sub(m, &q);
                        (dot(&e, &e).sqrt(), j)
                    })
                    .filter(|&(distance, _)| distance < radius)
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, j)| (i, j))
            })
            .collect()
    }

    /// Voxel averages, ordered by voxel.
    pub fn model(&self) -> &[Vec3] {
        &self.model
    }

    /// Add a whole sequence in the order `track.fuse` adds it, and return each frame's
    /// pose in capture order. A lit frame for which `waits` is true is added once a later
    /// light-off pair settles the poses it starts from, or at the end.
    pub fn track(&mut self, frames: &[Frame]) -> Vec<Pose> {
        let mut poses = vec![Pose::IDENTITY; frames.len()];
        let mut waiting = std::collections::VecDeque::new();
        for (k, frame) in frames.iter().enumerate() {
            match frame.points {
                Some(_) if self.waits(frame.time) => {
                    waiting.push_back(k);
                    continue;
                }
                Some(points) => poses[k] = self.add_at(frame.time, points, frame.dots),
                None => poses[k] = self.add_dark_at(frame.time, frame.dots),
            }
            while let Some(&j) = waiting.front().filter(|&&j| !self.waits(frames[j].time)) {
                let held = &frames[j];
                poses[j] = self.add_at(held.time, held.points.unwrap_or(&[]), held.dots);
                waiting.pop_front();
            }
        }
        for j in waiting {
            let held = &frames[j];
            poses[j] = self.add_at(held.time, held.points.unwrap_or(&[]), held.dots);
        }
        poses
    }

    /// Align a frame's points (in its own camera frame) to the model, starting from the
    /// last pose, and add them. Returns the frame's pose.
    pub fn add(&mut self, points: &[Vec3]) -> Pose {
        self.add_with_dots(points, &[])
    }

    /// Align a frame as `add` does, with its reference dots (in its camera frame) paired
    /// to the map of dots seen so far as point-to-point terms of its ICP when three or more
    /// pair. Paired dots then refine their map dot by averaging, and unpaired dots join the
    /// map, following `track.fuse`.
    pub fn add_with_dots(&mut self, points: &[Vec3], dots: &[Vec3]) -> Pose {
        self.add_at(self.poses.len() as f64, points, dots)
    }

    /// Align and add a frame as `add_with_dots` does, captured at `time`. A lit frame for
    /// which `waits` is true belongs after the next light-off pair, as `track.fuse` adds
    /// it, so that it starts between light-off poses.
    pub fn add_at(&mut self, time: f64, points: &[Vec3], dots: &[Vec3]) -> Pose {
        self.predict(time);
        if points.is_empty() {
            self.poses.push(self.pose);
            return self.pose;
        }
        if !self.model.is_empty() {
            let matched = self.match_dots(dots, &self.pose);
            let fixed: Vec<(Vec3, Vec3)> = if matched.len() >= MIN_DOT_PAIRS {
                matched
                    .iter()
                    .map(|&(i, j)| (dots[i], self.dot_map[j]))
                    .collect()
            } else {
                Vec::new()
            };
            let hold_below = if self.fixes.is_empty() {
                0.0
            } else {
                HOLD_BELOW
            };
            let (new, rms) = self.align(points, &fixed, hold_below);
            self.pose = new;
            self.last_rms = Some(rms);
        }
        if !dots.is_empty() {
            self.place_dots(dots);
        }
        self.poses.push(self.pose);
        let mut touched = Vec::with_capacity(points.len());
        for p in points {
            let q = self.pose.apply(p);
            let key = Grid::key_of(&q, self.voxel);
            let entry = self.sums.entry(key).or_insert(([0.0; 3], 0));
            entry.0 = add(&entry.0, &q);
            entry.1 += 1;
            touched.push(key);
        }
        touched.sort_unstable();
        touched.dedup();
        let (changed, moved) = self.merge(&touched);
        self.renew_normals(changed, &moved);
        self.pose
    }

    /// ICP of a frame's points onto the model from the current pose. It runs on the model
    /// points within `NEAR_MARGIN` of the box around the frame, and on the whole model
    /// when a step could reach a point outside that box, so its result is that of the
    /// whole model.
    fn align(&self, points: &[Vec3], fixed: &[(Vec3, Vec3)], hold_below: f64) -> (Pose, f64) {
        let (source_lo, source_hi) = bounds(points);
        let (lo, hi) = posed_bounds(&self.pose, &source_lo, &source_hi);
        let region = (lo.map(|v| v - NEAR_MARGIN), hi.map(|v| v + NEAR_MARGIN));
        let inside = |p: &Vec3| (0..3).all(|d| p[d] >= region.0[d] && p[d] <= region.1[d]);
        let near: Vec<usize> = (0..self.model.len())
            .filter(|&i| inside(&self.model[i]))
            .collect();
        if near.len() < self.model.len() {
            let target: Vec<Vec3> = near.iter().map(|&i| self.model[i]).collect();
            let target_normals: Vec<Vec3> = near.iter().map(|&i| self.normals[i]).collect();
            let found = icp_within(
                points,
                &target,
                &target_normals,
                self.pose,
                &LIMITS,
                fixed,
                hold_below,
                Some(&region),
            );
            if let Some(result) = found {
                return result;
            }
        }
        icp_with(
            points,
            &self.model,
            &self.normals,
            self.pose,
            &LIMITS,
            fixed,
            hold_below,
        )
    }

    /// Bring the ordered model up to date for voxels a frame touched, given in order.
    /// Returns which model points changed or are new, and every position such a point
    /// left or took.
    fn merge(&mut self, touched: &[[i64; 3]]) -> (Vec<bool>, Vec<Vec3>) {
        let size = self.keys.len() + touched.len();
        let mut keys = Vec::with_capacity(size);
        let mut model = Vec::with_capacity(size);
        let mut normals = Vec::with_capacity(size);
        let mut reach = Vec::with_capacity(size);
        let mut changed = Vec::with_capacity(size);
        let mut moved = Vec::with_capacity(2 * touched.len());
        let (mut i, mut j) = (0, 0);
        while i < self.keys.len() || j < touched.len() {
            if j == touched.len() || (i < self.keys.len() && self.keys[i] < touched[j]) {
                keys.push(self.keys[i]);
                model.push(self.model[i]);
                normals.push(self.normals[i]);
                reach.push(self.reach[i]);
                changed.push(false);
                i += 1;
                continue;
            }
            let key = touched[j];
            let (sum, count) = self.sums[&key];
            let average = sum.map(|v| v / f64::from(count));
            if i < self.keys.len() && self.keys[i] == key {
                moved.push(self.model[i]);
                i += 1;
            }
            moved.push(average);
            keys.push(key);
            model.push(average);
            normals.push([0.0; 3]);
            reach.push(f64::INFINITY);
            changed.push(true);
            j += 1;
        }
        self.keys = keys;
        self.model = model;
        self.normals = normals;
        self.reach = reach;
        (changed, moved)
    }

    /// Compute normals again for the model points that changed and for those whose
    /// neighbours could have: a point within its normal's reach of a position in `moved`.
    /// Other points keep their normal, which the whole model would give them again.
    fn renew_normals(&mut self, changed: Vec<bool>, moved: &[Vec3]) {
        let cell = RENEW_CELL_VOXELS * self.voxel;
        let (lo, hi) = bounds(moved);
        let moved_grid = Grid::new(moved, cell);
        let (model, reach) = (&self.model, &self.reach);
        let affected = |i: usize| {
            if changed[i] || reach[i].is_infinite() {
                return true;
            }
            let (p, r) = (&model[i], reach[i].sqrt());
            if (0..3).any(|d| p[d] < lo[d] - r || p[d] > hi[d] + r) {
                return false;
            }
            let mut near = false;
            let centre = Grid::key_of(p, cell);
            for shell in 0..=(r / cell).ceil() as i64 {
                moved_grid.shell(centre, shell, |m| {
                    let d = sub(&moved[m as usize], p);
                    near |= dot(&d, &d) <= reach[i];
                });
            }
            near
        };
        let flags = chunked(model.len(), |range| range.map(affected).collect::<Vec<_>>());
        let indices: Vec<usize> = flags
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(i, renew)| renew.then_some(i))
            .collect();
        let renewed: Vec<Vec3> = indices.iter().map(|&i| model[i]).collect();
        let margin = RENEW_MARGIN_VOXELS * self.voxel;
        let (lo, hi) = bounds(&renewed);
        let inside = |p: &Vec3| (0..3).all(|d| p[d] >= lo[d] - margin && p[d] <= hi[d] + margin);
        let near: Vec<Vec3> = model.iter().filter(|p| inside(p)).copied().collect();
        let grid = Grid::new(&near, 2.0 * self.voxel);
        let parts = chunked(indices.len(), |range| {
            indices[range]
                .iter()
                .map(|&i| normal_reach(&grid, &near, &model[i]))
                .collect::<Vec<_>>()
        });
        let mut results: Vec<(Vec3, f64)> = parts.into_iter().flatten().collect();
        let beyond: Vec<usize> = (0..results.len())
            .filter(|&k| results[k].1 > margin * margin)
            .collect();
        if !beyond.is_empty() && near.len() < model.len() {
            let whole = Grid::new(model, 2.0 * self.voxel);
            let parts = chunked(beyond.len(), |range| {
                beyond[range]
                    .iter()
                    .map(|&k| normal_reach(&whole, model, &model[indices[k]]))
                    .collect::<Vec<_>>()
            });
            for (&k, result) in beyond.iter().zip(parts.into_iter().flatten()) {
                results[k] = result;
            }
        }
        for (&i, (normal, reach)) in indices.iter().zip(results) {
            self.normals[i] = normal;
            self.reach[i] = reach;
        }
    }
}

/// Sample spacings within which a point needs a neighbour, as `stereo.depth_points` uses.
pub const ISOLATED_SPACINGS: f64 = 3.0;

/// The points that have another point within `spacings` sample spacings, where one
/// spacing at depth z is `step` times z over `focal`, as `drop_isolated` in
/// `slscan/stereo.py` keeps them.
pub fn drop_isolated(points: &[Vec3], step: f64, focal: f64, spacings: f64) -> Vec<Vec3> {
    let radius = |p: &Vec3| spacings * step * p[2] / focal;
    let cell = points.iter().map(radius).fold(0.0, f64::max);
    if points.len() < 2 || cell <= 0.0 {
        return Vec::new();
    }
    let grid = Grid::new(points, cell);
    points
        .iter()
        .enumerate()
        .filter(|&(i, p)| {
            let r = radius(p);
            let centre = Grid::key_of(p, cell);
            let mut found = false;
            for shell in 0..=1 {
                grid.shell(centre, shell, |j| {
                    let d = sub(&points[j as usize], p);
                    found |= j as usize != i && dot(&d, &d) <= r * r;
                });
            }
            found
        })
        .map(|(_, p)| *p)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dots_hold_a_slide_along_a_flat_plate() {
        let plate: Vec<Vec3> = (0..6000)
            .map(|k| {
                [
                    f64::from(k % 100) * 0.006 - 0.3,
                    f64::from(k / 100) * 0.006 - 0.18,
                    1.0,
                ]
            })
            .collect();
        let dots: Vec<Vec3> = (0..35)
            .map(|k| {
                [
                    f64::from(k % 7) * 0.1 - 0.3,
                    f64::from(k / 7) * 0.08 - 0.16,
                    1.0,
                ]
            })
            .collect();
        let view = |points: &[Vec3], shift: f64| -> Vec<Vec3> {
            points
                .iter()
                .map(|p| [p[0] - shift, p[1], p[2]])
                .filter(|p| p[0].abs() < 0.2)
                .collect()
        };
        let mut fusion = Fusion::new(0.004);
        for i in 0..6 {
            let shift = 0.01 * f64::from(i);
            fusion.add_with_dots(&view(&plate, shift), &view(&dots, shift));
        }
        let last = fusion.poses[5].translation;
        assert!((last[0] - 0.05).abs() < 1e-6, "slide tracked to {last:?}");
    }

    #[test]
    fn light_off_dots_hold_a_slide_along_a_flat_plate() {
        let plate: Vec<Vec3> = (0..6000)
            .map(|k| {
                [
                    f64::from(k % 100) * 0.006 - 0.3,
                    f64::from(k / 100) * 0.006 - 0.18,
                    1.0,
                ]
            })
            .collect();
        let dots: Vec<Vec3> = (0..35)
            .map(|k| {
                [
                    f64::from(k % 7) * 0.1 - 0.3,
                    f64::from(k / 7) * 0.08 - 0.16,
                    1.0,
                ]
            })
            .collect();
        let view = |points: &[Vec3], shift: f64| -> Vec<Vec3> {
            points
                .iter()
                .map(|p| [p[0] - shift, p[1], p[2]])
                .filter(|p| p[0].abs() < 0.2)
                .collect()
        };
        let slide = |shifts: &[f64]| -> Vec<Pose> {
            let views: Vec<Vec<Vec3>> = shifts
                .iter()
                .enumerate()
                .map(|(i, &shift)| view(if i % 4 == 0 { &dots } else { &plate }, shift))
                .collect();
            let frames: Vec<Frame> = views
                .iter()
                .enumerate()
                .map(|(i, seen)| Frame {
                    time: i as f64,
                    points: (i % 4 != 0).then_some(seen.as_slice()),
                    dots: if i % 4 == 0 { seen } else { &[] },
                })
                .collect();
            Fusion::new(0.004).track(&frames)
        };
        let steady: Vec<f64> = (0..12).map(|i| 0.005 * f64::from(i)).collect();
        for (i, pose) in slide(&steady).iter().enumerate() {
            let x = pose.translation[0];
            assert!((x - steady[i]).abs() < 1e-6, "frame {i} at {x}");
        }
        let speeding: Vec<f64> = (0..17).map(|i| 0.0005 * f64::from(i * i)).collect();
        for (i, pose) in slide(&speeding).iter().enumerate() {
            let x = pose.translation[0];
            assert!((x - speeding[i]).abs() < 1e-6, "frame {i} at {x}");
        }
    }

    #[test]
    fn held_alignment_turns_a_tilted_start_about_its_points() {
        let plate: Vec<Vec3> = (0..6000)
            .map(|k| {
                [
                    f64::from(k % 100) * 0.006 - 0.3,
                    f64::from(k / 100) * 0.006 - 0.18,
                    1.0,
                ]
            })
            .collect();
        let plate_normals = vec![[0.0, 0.0, 1.0]; plate.len()];
        let source: Vec<Vec3> = plate
            .iter()
            .map(|p| [p[0] - 0.05, p[1], p[2]])
            .filter(|p| p[0].abs() < 0.2)
            .collect();
        let count = source.len() as f64;
        let sum = source.iter().fold([0.0; 3], |a, p| add(&a, p));
        let middle = [sum[0] / count, sum[1] / count, sum[2] / count];
        let tilt = rodrigues(&[0.0, 0.002, 0.0]);
        let start = Pose {
            rotation: tilt,
            translation: add(&sub(&middle, &mul_vec(&tilt, &middle)), &[0.05, 0.0, 0.0]),
        };
        let (pose, _) = icp_with(
            &source,
            &plate,
            &plate_normals,
            start,
            &LIMITS,
            &[],
            HOLD_BELOW,
        );
        let off = sub(&pose.apply(&middle), &add(&middle, &[0.05, 0.0, 0.0]));
        assert!(dot(&off, &off).sqrt() < 1e-7, "centroid moved {off:?}");
        let turn = rotation_vector(&pose.rotation);
        assert!(dot(&turn, &turn).sqrt() < 1e-6, "tilt left {turn:?}");
    }

    #[test]
    fn capture_times_hold_a_slide_when_lit_frames_are_left_out() {
        let plate: Vec<Vec3> = (0..6000)
            .map(|k| {
                [
                    f64::from(k % 100) * 0.006 - 0.3,
                    f64::from(k / 100) * 0.006 - 0.18,
                    1.0,
                ]
            })
            .collect();
        let dots: Vec<Vec3> = (0..35)
            .map(|k| {
                [
                    f64::from(k % 7) * 0.1 - 0.3,
                    f64::from(k / 7) * 0.08 - 0.16,
                    1.0,
                ]
            })
            .collect();
        let view = |points: &[Vec3], shift: f64| -> Vec<Vec3> {
            points
                .iter()
                .map(|p| [p[0] - shift, p[1], p[2]])
                .filter(|p| p[0].abs() < 0.2)
                .collect()
        };
        let kept: Vec<u32> = (0..16).filter(|i| i % 4 != 1).collect();
        let worst = |timed: bool| {
            let views: Vec<Vec<Vec3>> = kept
                .iter()
                .map(|&i| {
                    view(
                        if i % 4 == 0 { &dots } else { &plate },
                        0.005 * f64::from(i),
                    )
                })
                .collect();
            let frames: Vec<Frame> = kept
                .iter()
                .zip(&views)
                .enumerate()
                .map(|(count, (&i, seen))| Frame {
                    time: if timed {
                        f64::from(i) / 30.0
                    } else {
                        count as f64
                    },
                    points: (i % 4 != 0).then_some(seen.as_slice()),
                    dots: if i % 4 == 0 { seen } else { &[] },
                })
                .collect();
            kept.iter()
                .zip(Fusion::new(0.004).track(&frames))
                .map(|(&i, pose)| (pose.translation[0] - 0.005 * f64::from(i)).abs())
                .fold(0.0, f64::max)
        };
        assert!(worst(true) < 1e-6, "by capture time: {}", worst(true));
        assert!(worst(false) > 1e-3, "by frame count: {}", worst(false));
    }

    #[test]
    fn dots_between_map_dots_do_not_pair() {
        let grid: Vec<Vec3> = (0..45)
            .map(|k| {
                [
                    f64::from(k % 9) * 0.035 - 0.14,
                    f64::from(k / 9) * 0.035 - 0.07,
                    0.5,
                ]
            })
            .collect();
        let shifted: Vec<Vec3> = grid.iter().map(|p| [p[0] + 0.018, p[1], p[2]]).collect();
        let mut fusion = Fusion::new(0.001);
        fusion.add_dark(&grid);
        let pose = fusion.add_dark(&shifted);
        assert_eq!(pose.translation, [0.0; 3]);
    }

    #[test]
    fn drop_isolated_keeps_a_surface_and_drops_strays() {
        let mut points: Vec<Vec3> = (0..400)
            .map(|k| [f64::from(k % 20) * 0.001, f64::from(k / 20) * 0.001, 0.5])
            .collect();
        let surface = points.clone();
        points.push([0.01, 0.01, 0.45]);
        points.push([0.05, 0.0, 0.5]);
        assert_eq!(drop_isolated(&points, 8.0, 4000.0, 3.0), surface);
    }

    fn surface(pose: &Pose) -> Vec<Vec3> {
        let inverse = Pose {
            rotation: transpose(&pose.rotation),
            translation: mul_vec(&transpose(&pose.rotation), &pose.translation).map(|v| -v),
        };
        let mut points = Vec::new();
        for i in -60..=60 {
            for j in -60..=60 {
                let (x, y) = (f64::from(i) * 0.0005, f64::from(j) * 0.0005);
                let z = 0.5 + 0.004 * (40.0 * x).sin() * (40.0 * y).cos();
                points.push(inverse.apply(&[x, y, z]));
            }
        }
        points
    }

    #[test]
    fn rodrigues_rotates_about_the_axis() {
        let r = rodrigues(&[0.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let p = mul_vec(&r, &[1.0, 0.0, 0.0]);
        assert!((p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normals_of_a_plane_are_its_axis() {
        let points: Vec<Vec3> = (0..400)
            .map(|i| [f64::from(i % 20) * 0.001, f64::from(i / 20) * 0.001, 0.5])
            .collect();
        for n in normals(&points, 0.001) {
            assert!((n[2].abs() - 1.0).abs() < 1e-9, "normal {n:?}");
        }
    }

    /// A window of a wavy sheet that starts at `x0`, sampled every 1.3 mm, in the frame
    /// of a camera at `pose`.
    fn window(pose: &Pose, x0: f64) -> Vec<Vec3> {
        let inverse = Pose {
            rotation: transpose(&pose.rotation),
            translation: mul_vec(&transpose(&pose.rotation), &pose.translation).map(|v| -v),
        };
        let mut points = Vec::new();
        for i in 0..=120 {
            for j in -40..=40 {
                let (x, y) = (x0 + f64::from(i) * 0.0013, f64::from(j) * 0.0013);
                let z = 0.5 + 0.004 * (40.0 * x).sin() * (40.0 * y).cos();
                points.push(inverse.apply(&[x, y, z]));
            }
        }
        points
    }

    /// Frame `k` of a sweep along the sheet, 12 mm further each frame under a small motion.
    fn sweep_frame(k: u32) -> Vec<Vec3> {
        let s = f64::from(k);
        let pose = Pose {
            rotation: rodrigues(&[0.0005 * s, -0.0004 * s, 0.0003 * s]),
            translation: [0.0003 * s, -0.0002 * s, 0.0001 * s],
        };
        window(&pose, 0.012 * s)
    }

    /// The model and normals of a fusion from all of its voxels.
    fn rebuilt(fusion: &Fusion) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut keys: Vec<&[i64; 3]> = fusion.sums.keys().collect();
        keys.sort();
        let model: Vec<Vec3> = keys
            .into_iter()
            .map(|k| {
                let (s, n) = fusion.sums[k];
                s.map(|v| v / f64::from(n))
            })
            .collect();
        let normals = normals(&model, 2.0 * fusion.voxel);
        (model, normals)
    }

    #[test]
    fn model_and_normals_equal_a_rebuild_from_every_voxel() {
        for (voxel, frames) in [(0.0005, 8), (0.0001, 3)] {
            let mut fusion = Fusion::new(voxel);
            for k in 0..frames {
                fusion.add(&sweep_frame(k));
                let (model, normals) = rebuilt(&fusion);
                assert_eq!(fusion.model, model, "model after frame {k}");
                assert_eq!(fusion.normals, normals, "normals after frame {k}");
            }
        }
    }

    #[test]
    fn normals_far_from_a_frame_are_kept() {
        let mut fusion = Fusion::new(0.0005);
        for k in 0..8 {
            fusion.add(&sweep_frame(k));
        }
        let lowest = |points: &[Vec3]| points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let far = (0..fusion.model.len())
            .find(|&i| {
                let p = fusion.model[i];
                p[0] > 0.01 && p[0] < 0.02 && p[1].abs() < 0.01
            })
            .expect("a model point behind the sweep");
        let marked = fusion.keys[far];
        fusion.normals[far] = [9.0; 3];
        let frame = sweep_frame(8);
        assert!(lowest(&frame) > 0.08);
        fusion.add(&frame);
        let at = fusion
            .keys
            .binary_search(&marked)
            .expect("the marked voxel");
        assert_eq!(fusion.normals[at], [9.0; 3]);
    }

    #[test]
    fn alignment_to_nearby_points_equals_alignment_to_the_whole_model() {
        let mut fusion = Fusion::new(0.0005);
        for k in 0..8 {
            fusion.add(&sweep_frame(k));
        }
        let frame = sweep_frame(8);
        let (lo, hi) = bounds(&frame);
        let (lo, hi) = posed_bounds(&fusion.pose, &lo, &hi);
        let near = fusion
            .model
            .iter()
            .filter(|p| (0..3).all(|d| p[d] >= lo[d] - NEAR_MARGIN && p[d] <= hi[d] + NEAR_MARGIN))
            .count();
        assert!(near > 0 && near < fusion.model.len());
        let whole = icp_with(
            &frame,
            &fusion.model,
            &fusion.normals,
            fusion.pose,
            &LIMITS,
            &[],
            0.0,
        );
        assert_eq!(fusion.align(&frame, &[], 0.0), whole);
    }

    #[test]
    fn alignment_stops_when_a_step_could_match_outside_its_box() {
        let frame = window(&Pose::IDENTITY, 0.0);
        let target_normals = normals(&frame, 0.004);
        let region = bounds(&frame);
        let found = icp_within(
            &frame,
            &frame,
            &target_normals,
            Pose::IDENTITY,
            &LIMITS,
            &[],
            0.0,
            Some(&region),
        );
        assert!(found.is_none());
    }

    #[test]
    fn second_frame_recovers_a_small_motion() {
        let mut fusion = Fusion::new(0.0005);
        let motion = Pose {
            rotation: rodrigues(&[0.004, -0.006, 0.003]),
            translation: [0.0008, -0.0005, 0.0006],
        };
        fusion.add(&surface(&Pose::IDENTITY));
        let pose = fusion.add(&surface(&motion));
        let r = mul(&pose.rotation, &transpose(&motion.rotation));
        let angle = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos();
        let offset = sub(&pose.translation, &motion.translation);
        assert!(angle < 1e-5, "rotation error {angle}");
        assert!(
            dot(&offset, &offset).sqrt() < 2e-6,
            "translation error {offset:?}"
        );
    }
}
