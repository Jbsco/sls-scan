//! Registration of static views from their geometry, following `slscan/register.py`.
//! Points of two views pair by fast point feature histograms (FPFH), RANSAC shortlists the
//! rigid motions most pairs agree on, overlap of all points picks one, and point-to-plane
//! ICP refines it.

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

use crate::fuse::{self, add, chunked, cross, dot, mul, mul_vec, sub, Grid, Pose, Vec3};

const BINS: usize = 11;
/// Length of a feature: three histograms of `BINS` bins.
pub const FEATURES: usize = 3 * BINS;
const TRIALS: usize = 20_000;
const EDGE_RATIO: f64 = 0.9;
const SHORTLIST: usize = 100;
const NEIGHBOURS_SPACING: f64 = 2.0;

pub type Feature = [f64; FEATURES];

/// Mean point of every occupied voxel, ordered by voxel.
pub fn voxel_average(points: &[Vec3], voxel: f64) -> Vec<Vec3> {
    let mut sums: HashMap<[i64; 3], (Vec3, u32)> = HashMap::new();
    for p in points {
        let entry = sums.entry(Grid::key_of(p, voxel)).or_insert(([0.0; 3], 0));
        entry.0 = add(&entry.0, p);
        entry.1 += 1;
    }
    let mut keys: Vec<&[i64; 3]> = sums.keys().collect();
    keys.sort();
    keys.into_iter()
        .map(|k| {
            let (s, n) = sums[k];
            s.map(|v| v / f64::from(n))
        })
        .collect()
}

/// Unit normals turned toward the camera at the origin of the points' frame.
pub fn facing_normals(points: &[Vec3], spacing: f64) -> Vec<Vec3> {
    fuse::normals(points, NEIGHBOURS_SPACING * spacing)
        .into_iter()
        .zip(points)
        .map(|(n, p)| if dot(&n, p) > 0.0 { n.map(|v| -v) } else { n })
        .collect()
}

/// Indices of the points within `radius` of each point, the point itself left out.
fn neighbours(points: &[Vec3], radius: f64) -> Vec<Vec<u32>> {
    let grid = Grid::new(points, radius);
    let parts = chunked(points.len(), |range| {
        range
            .map(|i| {
                let p = &points[i];
                let centre = Grid::key_of(p, radius);
                let mut found = Vec::new();
                for shell in 0..=1 {
                    grid.shell(centre, shell, |j| {
                        let d = sub(&points[j as usize], p);
                        if j as usize != i && dot(&d, &d) <= radius * radius {
                            found.push(j);
                        }
                    });
                }
                found
            })
            .collect::<Vec<_>>()
    });
    parts.into_iter().flatten().collect()
}

fn bin(value: f64, low: f64, span: f64) -> usize {
    (((value - low) / span * BINS as f64) as usize).min(BINS - 1)
}

/// Each point's fast point feature histogram over its neighbours within `radius`: three
/// histograms of the angles between its normal and theirs, each summing to one.
pub fn fpfh(points: &[Vec3], normals: &[Vec3], radius: f64) -> Vec<Feature> {
    let near = neighbours(points, radius);
    let pair = |i: usize, j: usize| -> Option<([usize; 3], f64)> {
        let mut d = sub(&points[j], &points[i]);
        let distance = dot(&d, &d).sqrt();
        d = d.map(|v| v / distance);
        let u = normals[i];
        let v = cross(&u, &d);
        let length = dot(&v, &v).sqrt();
        if length <= 1e-9 {
            return None;
        }
        let v = v.map(|x| x / length);
        let w = cross(&u, &v);
        let nj = normals[j];
        let alpha = dot(&v, &nj);
        let phi = dot(&u, &d);
        let theta = dot(&w, &nj).atan2(dot(&u, &nj));
        Some((
            [
                bin(alpha, -1.0, 2.0),
                BINS + bin(phi, -1.0, 2.0),
                2 * BINS + bin(theta, -std::f64::consts::PI, 2.0 * std::f64::consts::PI),
            ],
            distance,
        ))
    };
    let spfh: Vec<Feature> = chunked(points.len(), |range| {
        range
            .map(|i| {
                let mut h = [0.0; FEATURES];
                let mut count = 0usize;
                for &j in &near[i] {
                    if let Some((index, _)) = pair(i, j as usize) {
                        index.iter().for_each(|&k| h[k] += 1.0);
                        count += 1;
                    }
                }
                let scale = 1.0 / count.max(1) as f64;
                h.map(|v| v * scale)
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    chunked(points.len(), |range| {
        range
            .map(|i| {
                let mut spread = [0.0; FEATURES];
                let mut weights = 0.0;
                for &j in &near[i] {
                    if let Some((_, distance)) = pair(i, j as usize) {
                        let inverse = 1.0 / distance;
                        for (s, v) in spread.iter_mut().zip(&spfh[j as usize]) {
                            *s += v * inverse;
                        }
                        weights += inverse;
                    }
                }
                let mut h = spfh[i];
                for (x, s) in h.iter_mut().zip(&spread) {
                    *x += s / weights.max(1e-12);
                }
                for part in h.chunks_mut(BINS) {
                    let total: f64 = part.iter().sum();
                    part.iter_mut().for_each(|x| *x /= total.max(1e-12));
                }
                h
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect()
}

fn nearest_feature(features: &[Feature], q: &Feature) -> usize {
    let mut best = (f64::INFINITY, 0);
    for (i, f) in features.iter().enumerate() {
        let d: f64 = f.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

/// Index pairs of source and target points that are each other's nearest feature.
pub fn feature_pairs(source: &[Feature], target: &[Feature]) -> Vec<(usize, usize)> {
    if source.is_empty() || target.is_empty() {
        return Vec::new();
    }
    let forward: Vec<usize> = chunked(source.len(), |range| {
        range
            .map(|i| nearest_feature(target, &source[i]))
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let backward: Vec<usize> = chunked(target.len(), |range| {
        range
            .map(|j| nearest_feature(source, &target[j]))
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    forward
        .iter()
        .enumerate()
        .filter(|&(i, &j)| backward[j] == i)
        .map(|(i, &j)| (i, j))
        .collect()
}

/// Unit eigenvector of the largest eigenvalue of a symmetric 4 × 4 matrix, by Jacobi
/// rotations.
fn largest_eigenvector4(m: [[f64; 4]; 4]) -> [f64; 4] {
    let mut a = m;
    let mut v = [[0.0; 4]; 4];
    (0..4).for_each(|i| v[i][i] = 1.0);
    for _ in 0..100 {
        let (mut p, mut q, mut largest) = (0, 1, 0.0);
        for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            if a[i][j].abs() > largest {
                (p, q, largest) = (i, j, a[i][j].abs());
            }
        }
        let scale: f64 = (0..4).map(|i| a[i][i].abs()).sum();
        if largest <= 1e-15 * scale.max(f64::MIN_POSITIVE) {
            break;
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
            let (akp, akq) = (row[p], row[q]);
            row[p] = c * akp - s * akq;
            row[q] = s * akp + c * akq;
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
    let top = (0..4)
        .max_by(|&i, &j| a[i][i].total_cmp(&a[j][j]))
        .unwrap_or(0);
    [v[0][top], v[1][top], v[2][top], v[3][top]]
}

/// The rigid motion that takes source points onto matching target points with least
/// squared distance, by the unit quaternion method of Horn.
pub fn rigid_fit(source: &[Vec3], target: &[Vec3]) -> Pose {
    let n = source.len().max(1) as f64;
    let mean = |pts: &[Vec3]| pts.iter().fold([0.0; 3], |a, p| add(&a, p)).map(|v| v / n);
    let (cs, ct) = (mean(source), mean(target));
    let mut s = [[0.0; 3]; 3];
    for (a, b) in source.iter().zip(target) {
        let (a, b) = (sub(a, &cs), sub(b, &ct));
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += a[r] * b[c];
            }
        }
    }
    let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
    let q = largest_eigenvector4([
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ]);
    let [w, x, y, z] = q;
    let rotation = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    Pose {
        rotation,
        translation: sub(&ct, &mul_vec(&rotation, &cs)),
    }
}

/// Small seeded generator (xorshift64*) for RANSAC draws.
struct Draws(u64);

impl Draws {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % n as u64) as usize
    }
}

fn distance(a: &Vec3, b: &Vec3) -> f64 {
    let d = sub(a, b);
    dot(&d, &d).sqrt()
}

/// The rigid motion that best overlays `source` on `target`, and the share of source
/// points it brings within `inlier` of a target point. Trials take three pairs whose edge
/// lengths agree to `EDGE_RATIO`. The shortlist that takes the most paired points within
/// `inlier` of their targets is judged by overlap of all points, and the winner is refit
/// on its pairs.
pub fn ransac(
    source: &[Vec3],
    target: &[Vec3],
    pairs: &[(usize, usize)],
    inlier: f64,
    seed: u64,
) -> (Pose, f64) {
    if pairs.len() < 3 {
        return (Pose::IDENTITY, 0.0);
    }
    let s: Vec<Vec3> = pairs.iter().map(|&(i, _)| source[i]).collect();
    let t: Vec<Vec3> = pairs.iter().map(|&(_, j)| target[j]).collect();
    let mut draws = Draws(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut picks = Vec::new();
    for _ in 0..TRIALS {
        let pick = [0; 3].map(|_| draws.below(pairs.len()));
        let agree = (0..3).all(|k| {
            let l = (k + 2) % 3;
            let ls = distance(&s[pick[k]], &s[pick[l]]);
            let lt = distance(&t[pick[k]], &t[pick[l]]);
            ls.min(lt) >= EDGE_RATIO * ls.max(lt) && ls > inlier
        });
        if agree {
            picks.push(pick);
        }
    }
    if picks.is_empty() {
        return (Pose::IDENTITY, 0.0);
    }
    let count = |pose: &Pose| {
        s.iter()
            .zip(&t)
            .filter(|(a, b)| distance(&pose.apply(a), b) < inlier)
            .count()
    };
    let mut scored: Vec<(usize, Pose)> = chunked(picks.len(), |range| {
        range
            .map(|k| {
                let pick = picks[k];
                let pose = rigid_fit(&pick.map(|i| s[i]), &pick.map(|i| t[i]));
                (count(&pose), pose)
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.truncate(SHORTLIST);
    let grid = Grid::new(target, inlier);
    let overlaps: Vec<f64> = chunked(scored.len(), |range| {
        range
            .map(|k| {
                let pose = &scored[k].1;
                let near = source
                    .iter()
                    .filter(|p| grid.nearest(target, &pose.apply(p), inlier).is_some())
                    .count();
                near as f64 / source.len().max(1) as f64
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let best = (0..scored.len())
        .max_by(|&a, &b| overlaps[a].total_cmp(&overlaps[b]).then(b.cmp(&a)))
        .unwrap_or(0);
    let pose = scored[best].1;
    let (inside_s, inside_t): (Vec<Vec3>, Vec<Vec3>) = s
        .iter()
        .zip(&t)
        .filter(|(a, b)| distance(&pose.apply(a), b) < inlier)
        .map(|(a, b)| (*a, *b))
        .unzip();
    (rigid_fit(&inside_s, &inside_t), overlaps[best])
}

/// A view thinned to voxel spacing and its points' FPFH over 5 voxels.
pub struct Described {
    pub points: Vec<Vec3>,
    pub features: Vec<Feature>,
}

pub fn describe(points: &[Vec3], voxel: f64) -> Described {
    let thin = voxel_average(points, voxel);
    let normals = facing_normals(&thin, voxel);
    let features = fpfh(&thin, &normals, 5.0 * voxel);
    Described {
        points: thin,
        features,
    }
}

/// How a view's coarse match went.
#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub pairs: usize,
    pub overlap: f64,
    pub rms: f64,
}

/// The rigid motion that takes one view's points (each in its own camera frame) onto
/// another's, by a coarse match at `voxel` spacing and ICP, and a record of the match.
pub fn register(
    source: &[Vec3],
    target: &[Vec3],
    voxel: f64,
    described: (&Described, &Described),
) -> (Pose, Record) {
    let (s, t) = described;
    let pairs = feature_pairs(&s.features, &t.features);
    let (coarse, overlap) = ransac(&s.points, &t.points, &pairs, 1.5 * voxel, 0);
    let fine = voxel_average(target, voxel / 4.0);
    let normals = fuse::normals(&fine, NEIGHBOURS_SPACING * voxel / 4.0);
    let limits = [3.0 * voxel, voxel, voxel / 2.0, voxel / 4.0];
    let thinned = voxel_average(source, voxel / 4.0);
    let (pose, rms) = fuse::icp_with(&thinned, &fine, &normals, coarse, &limits, &[]);
    (
        pose,
        Record {
            pairs: pairs.len(),
            overlap,
            rms,
        },
    )
}

/// A pose (into the first view's frame) for each of `views` taken in order around an
/// object, and the record of each view's match to the one before. Each view registers to
/// the one before, then each round refines every view but the first by ICP against all
/// the others.
pub fn register_views(views: &[Vec<Vec3>], voxel: f64, rounds: usize) -> (Vec<Pose>, Vec<Record>) {
    let described: Vec<Described> = views.iter().map(|v| describe(v, voxel)).collect();
    let mut poses = vec![Pose::IDENTITY];
    let mut records = Vec::new();
    for k in 1..views.len() {
        let (step, record) = register(
            &views[k],
            &views[k - 1],
            voxel,
            (&described[k], &described[k - 1]),
        );
        records.push(record);
        let before = poses[k - 1];
        poses.push(Pose {
            rotation: mul(&before.rotation, &step.rotation),
            translation: before.apply(&step.translation),
        });
    }
    let fine = voxel / 4.0;
    let thinned: Vec<Vec<Vec3>> = views.iter().map(|v| voxel_average(v, fine)).collect();
    let normals: Vec<Vec<Vec3>> = thinned
        .iter()
        .map(|v| fuse::normals(v, NEIGHBOURS_SPACING * fine))
        .collect();
    let limits = [voxel, voxel / 2.0, voxel / 4.0, voxel / 8.0];
    for _ in 0..rounds {
        for k in 1..views.len() {
            let (mut target, mut target_normals) = (Vec::new(), Vec::new());
            for j in (0..views.len()).filter(|&j| j != k) {
                target.extend(thinned[j].iter().map(|p| poses[j].apply(p)));
                target_normals.extend(normals[j].iter().map(|n| mul_vec(&poses[j].rotation, n)));
            }
            poses[k] = fuse::icp_with(
                &thinned[k],
                &target,
                &target_normals,
                poses[k],
                &limits,
                &[],
            )
            .0;
        }
    }
    (poses, records)
}

/// Points of a binary little-endian PLY file of float vertices, as `depth::write_ply` and
/// `slscan` write them.
pub fn read_ply(path: &Path) -> io::Result<Vec<Vec3>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    let marker = b"end_header\n";
    let end = data
        .windows(marker.len())
        .position(|w| w == marker)
        .map(|i| i + marker.len())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no PLY header end"))?;
    let header = String::from_utf8_lossy(&data[..end]);
    let count: usize = header
        .lines()
        .find_map(|l| l.strip_prefix("element vertex "))
        .and_then(|n| n.trim().parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no vertex count"))?;
    if !header.contains("format binary_little_endian") || data.len() < end + 12 * count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected binary little-endian float x y z vertices",
        ));
    }
    let floats = |k: usize| {
        let at = end + 4 * k;
        f64::from(f32::from_le_bytes([
            data[at],
            data[at + 1],
            data[at + 2],
            data[at + 3],
        ]))
    };
    Ok((0..count)
        .map(|i| [floats(3 * i), floats(3 * i + 1), floats(3 * i + 2)])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuse::rodrigues;

    fn cloud() -> Vec<Vec3> {
        let mut points = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let (u, v) = (f64::from(i) * 0.005 - 0.1, f64::from(j) * 0.005 - 0.1);
                let bump = 0.03 * (-(u * u + v * v) / 0.002).exp() + 0.01 * (20.0 * u).sin();
                points.push([u, v, 0.5 - bump]);
            }
        }
        points
    }

    fn moved(points: &[Vec3], pose: &Pose) -> Vec<Vec3> {
        points.iter().map(|p| pose.apply(p)).collect()
    }

    #[test]
    fn rigid_fit_recovers_a_motion() {
        let pose = Pose {
            rotation: rodrigues(&[0.3, -0.2, 0.5]),
            translation: [0.1, -0.05, 0.2],
        };
        let source = cloud();
        let fit = rigid_fit(&source, &moved(&source, &pose));
        for r in 0..3 {
            for c in 0..3 {
                assert!((fit.rotation[r][c] - pose.rotation[r][c]).abs() < 1e-12);
            }
            assert!((fit.translation[r] - pose.translation[r]).abs() < 1e-12);
        }
    }

    #[test]
    fn features_move_with_the_points() {
        let points = cloud();
        let normals = facing_normals(&points, 0.005);
        let pose = Pose {
            rotation: rodrigues(&[0.2, 0.4, -0.3]),
            translation: [0.1, 0.0, 0.0],
        };
        let turned: Vec<Vec3> = normals.iter().map(|n| mul_vec(&pose.rotation, n)).collect();
        let still = fpfh(&points, &normals, 0.025);
        let shifted = fpfh(&moved(&points, &pose), &turned, 0.025);
        let differing = still
            .iter()
            .zip(&shifted)
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9))
            .count();
        assert!(
            differing * 100 <= points.len(),
            "{differing} features differ"
        );
    }

    #[test]
    fn views_of_a_surface_register() {
        let surface = cloud();
        let pose = Pose {
            rotation: rodrigues(&[0.0, 0.25, 0.1]),
            translation: [0.02, -0.01, 0.03],
        };
        let other = moved(&surface, &pose);
        let (found, _) = register_views(&[surface, other], 0.005, 0);
        let inverse = found[1];
        let back = mul(&inverse.rotation, &pose.rotation);
        let angle = ((back[0][0] + back[1][1] + back[2][2] - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos();
        assert!(angle < 1e-4, "rotation off by {angle} rad");
    }
}
