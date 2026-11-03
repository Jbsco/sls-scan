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
/// Fewest points a thread takes, so that small frames stay on one thread.
const CHUNK: usize = 2048;

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
fn chunked<T: Send>(n: usize, work: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
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

fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mul_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    [dot(&a[0], v), dot(&a[1], v), dot(&a[2], v)]
}

fn transpose(a: &Mat3) -> Mat3 {
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
struct Grid {
    cell: f64,
    cells: CellMap<Vec<u32>>,
    lo: [i64; 3],
    hi: [i64; 3],
}

impl Grid {
    fn new(points: &[Vec3], cell: f64) -> Self {
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

    fn key_of(p: &Vec3, cell: f64) -> [i64; 3] {
        [0, 1, 2].map(|d| (p[d] / cell).floor() as i64)
    }

    /// Visit every point in cells whose largest key offset from `centre` is `shell`.
    fn shell(&self, centre: [i64; 3], shell: i64, mut visit: impl FnMut(u32)) {
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

    /// Nearest point closer than `limit`, which must not exceed the cell size.
    fn nearest(&self, points: &[Vec3], q: &Vec3, limit: f64) -> Option<usize> {
        let centre = Self::key_of(q, self.cell);
        let mut best: Option<(f64, usize)> = None;
        for shell in 0..=1 {
            self.shell(centre, shell, |i| {
                let d = sub(&points[i as usize], q);
                let d2 = dot(&d, &d);
                if d2 < limit * limit && best.is_none_or(|(b, _)| d2 < b) {
                    best = Some((d2, i as usize));
                }
            });
        }
        best.map(|(_, i)| i)
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
fn normals(points: &[Vec3], cell: f64) -> Vec<Vec3> {
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
    let idx = grid.knn(points, p, NEIGHBOURS);
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
    smallest_eigenvector(&cov)
}

/// Point-to-plane ICP of `source` onto `target` from `start`. Returns the pose and the RMS
/// point-to-plane residual of the last matched pairs.
fn icp(source: &[Vec3], target: &[Vec3], target_normals: &[Vec3], start: Pose) -> (Pose, f64) {
    let mut pose = start;
    let mut rms = f64::INFINITY;
    for limit in LIMITS {
        let grid = Grid::new(target, limit);
        for _ in 0..ITERATIONS {
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
                    let c = cross(&p, n);
                    let a = [c[0], c[1], c[2], n[0], n[1], n[2]];
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
            let Some(step) = solve6(ata, atb) else {
                break;
            };
            let dr = rodrigues(&[step[0], step[1], step[2]]);
            pose = Pose {
                rotation: mul(&dr, &pose.rotation),
                translation: add(
                    &mul_vec(&dr, &pose.translation),
                    &[step[3], step[4], step[5]],
                ),
            };
            rms = (squares / pairs as f64).sqrt();
            let rotation_step = dot(&[step[0], step[1], step[2]], &[step[0], step[1], step[2]]);
            let translation_step = dot(&[step[3], step[4], step[5]], &[step[3], step[4], step[5]]);
            if rotation_step.sqrt() < 1e-7 && translation_step.sqrt() < 1e-8 {
                break;
            }
        }
    }
    (pose, rms)
}

/// A model built from depth frames, each aligned to it before it joins.
pub struct Fusion {
    voxel: f64,
    sums: CellMap<(Vec3, u32)>,
    model: Vec<Vec3>,
    normals: Vec<Vec3>,
    pose: Pose,
    pub poses: Vec<Pose>,
    pub last_rms: Option<f64>,
}

impl Fusion {
    pub fn new(voxel: f64) -> Self {
        Self {
            voxel,
            sums: CellMap::default(),
            model: Vec::new(),
            normals: Vec::new(),
            pose: Pose::IDENTITY,
            poses: Vec::new(),
            last_rms: None,
        }
    }

    /// Voxel averages, ordered by voxel.
    pub fn model(&self) -> &[Vec3] {
        &self.model
    }

    /// Align a frame's points (in its own camera frame) to the model, starting from the
    /// last pose, and add them. Returns the frame's pose.
    pub fn add(&mut self, points: &[Vec3]) -> Pose {
        if points.is_empty() {
            self.poses.push(self.pose);
            return self.pose;
        }
        if !self.model.is_empty() {
            let (new, rms) = icp(points, &self.model, &self.normals, self.pose);
            self.pose = new;
            self.last_rms = Some(rms);
        }
        self.poses.push(self.pose);
        for p in points {
            let q = self.pose.apply(p);
            let key = Grid::key_of(&q, self.voxel);
            let entry = self.sums.entry(key).or_insert(([0.0; 3], 0));
            entry.0 = add(&entry.0, &q);
            entry.1 += 1;
        }
        let mut keys: Vec<&[i64; 3]> = self.sums.keys().collect();
        keys.sort();
        self.model = keys
            .into_iter()
            .map(|k| {
                let (s, n) = self.sums[k];
                s.map(|v| v / f64::from(n))
            })
            .collect();
        self.normals = normals(&self.model, 2.0 * self.voxel);
        self.pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
