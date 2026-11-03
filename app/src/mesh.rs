//! Triangle meshes from a TSDF volume by surface nets, following `surface_nets` in
//! `slscan/mesh.py` vertex for vertex and triangle for triangle, and their export as
//! PLY, OBJ and STL.

use crate::tsdf::VolumeSpec;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// One vertex per cell between eight observed voxels whose distances change sign, at the
/// mean of its edges' zero crossings, and one quad per crossing edge joining the four
/// cells around it, facing the positive side and split along the diagonal that leaves
/// the larger smallest triangle.
pub fn surface_nets(spec: &VolumeSpec, tsdf: &[f32], weight: &[f32]) -> Mesh {
    let [nx, ny, nz] = spec.shape.map(|n| n as usize);
    if nx < 2 || ny < 2 || nz < 2 {
        return Mesh::default();
    }
    let index = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let observed = |p: usize| weight[p] > 0.0 && tsdf[p].abs() < 1.0;
    let cells = [nx - 1, ny - 1, nz - 1];
    let cell = |c: [usize; 3]| (c[0] * cells[1] + c[1]) * cells[2] + c[2];
    let mut sums = vec![[0.0f64; 3]; cells.iter().product()];
    let mut counts = vec![0u32; sums.len()];
    let mut edges: Vec<(usize, [usize; 3], bool)> = Vec::new();
    for axis in 0..3 {
        let mut size = [nx, ny, nz];
        size[axis] -= 1;
        let others: Vec<usize> = (0..3).filter(|&d| d != axis).collect();
        for i in 0..size[0] {
            for j in 0..size[1] {
                for k in 0..size[2] {
                    let a = index(i, j, k);
                    let mut next = [i, j, k];
                    next[axis] += 1;
                    let b = index(next[0], next[1], next[2]);
                    let (da, db) = (tsdf[a], tsdf[b]);
                    if !(observed(a) && observed(b)) || (da > 0.0) == (db > 0.0) {
                        continue;
                    }
                    let t = f64::from(da / (da - db));
                    let mut pos = [i as f64, j as f64, k as f64];
                    pos[axis] += t;
                    for (du, dv) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let mut c = [i as isize, j as isize, k as isize];
                        c[others[0]] -= du;
                        c[others[1]] -= dv;
                        if c.iter()
                            .zip(cells)
                            .all(|(&v, n)| v >= 0 && (v as usize) < n)
                        {
                            let at = cell(c.map(|v| v as usize));
                            for d in 0..3 {
                                sums[at][d] += pos[d];
                            }
                            counts[at] += 1;
                        }
                    }
                    edges.push((axis, [i, j, k], da > 0.0));
                }
            }
        }
    }
    let mut vertex_of = vec![u32::MAX; sums.len()];
    let mut vertices = Vec::new();
    for i in 0..cells[0] {
        for j in 0..cells[1] {
            for k in 0..cells[2] {
                let at = cell([i, j, k]);
                if counts[at] == 0 {
                    continue;
                }
                let corners =
                    (0..8).all(|c| observed(index(i + (c >> 2), j + ((c >> 1) & 1), k + (c & 1))));
                if !corners {
                    continue;
                }
                vertex_of[at] = vertices.len() as u32;
                let n = f64::from(counts[at]);
                vertices.push([0, 1, 2].map(|d| spec.lower[d] + spec.voxel * (sums[at][d] / n)));
            }
        }
    }
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for &(axis, p, positive) in &edges {
        let others: Vec<usize> = (0..3).filter(|&d| d != axis).collect();
        let mut quad = [0u32; 4];
        let mut whole = true;
        for (slot, (du, dv)) in [(1, 1), (0, 1), (0, 0), (1, 0)].into_iter().enumerate() {
            let mut c = p.map(|v| v as isize);
            c[others[0]] -= du;
            c[others[1]] -= dv;
            let inside = c
                .iter()
                .zip(cells)
                .all(|(&v, n)| v >= 0 && (v as usize) < n);
            let v = if inside {
                vertex_of[cell(c.map(|v| v as usize))]
            } else {
                u32::MAX
            };
            if v == u32::MAX {
                whole = false;
                break;
            }
            quad[slot] = v;
        }
        if !whole {
            continue;
        }
        if positive != (axis == 1) {
            quad.reverse();
        }
        quads.push(quad);
    }
    let area = |a: u32, b: u32, c: u32| {
        let (pa, pb, pc) = (
            vertices[a as usize],
            vertices[b as usize],
            vertices[c as usize],
        );
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let w = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    };
    let first: Vec<bool> = quads
        .iter()
        .map(|q| {
            area(q[0], q[1], q[2]).min(area(q[0], q[2], q[3]))
                >= area(q[0], q[1], q[3]).min(area(q[1], q[2], q[3]))
        })
        .collect();
    let mut faces = Vec::with_capacity(2 * quads.len());
    for (q, &f) in quads.iter().zip(&first) {
        faces.push(if f {
            [q[0], q[1], q[2]]
        } else {
            [q[0], q[1], q[3]]
        });
    }
    for (q, &f) in quads.iter().zip(&first) {
        faces.push(if f {
            [q[0], q[2], q[3]]
        } else {
            [q[1], q[2], q[3]]
        });
    }
    Mesh { vertices, faces }
}

/// Close each boundary loop whose perimeter is `max_perimeter` or less with a fan from a
/// new vertex at its centroid, wound to match the triangles along it, as `fill_holes` in
/// `slscan/mesh.py` does. Returns the count of loops closed.
pub fn fill_holes(mesh: &mut Mesh, max_perimeter: f64) -> usize {
    let mut half_edges = std::collections::HashSet::new();
    let mut order = Vec::new();
    for f in &mesh.faces {
        for k in 0..3 {
            let edge = (f[k], f[(k + 1) % 3]);
            if half_edges.insert(edge) {
                order.push(edge);
            }
        }
    }
    let mut after = std::collections::HashMap::new();
    let mut starts = Vec::new();
    for &(a, b) in &order {
        if !half_edges.contains(&(b, a)) && after.insert(b, a).is_none() {
            starts.push(b);
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut closed = 0;
    for start in starts {
        if seen.contains(&start) {
            continue;
        }
        let mut ring = Vec::new();
        let mut k = start;
        while !seen.contains(&k) {
            let Some(&next) = after.get(&k) else { break };
            seen.insert(k);
            ring.push(k);
            k = next;
        }
        if k != start || ring.len() < 3 {
            continue;
        }
        let point = |v: u32| mesh.vertices[v as usize];
        let perimeter: f64 = (0..ring.len())
            .map(|j| distance(&point(ring[j]), &point(ring[(j + 1) % ring.len()])))
            .sum();
        if perimeter > max_perimeter {
            continue;
        }
        let n = ring.len() as f64;
        let centre: [f64; 3] =
            std::array::from_fn(|d| ring.iter().map(|&v| point(v)[d]).sum::<f64>() / n);
        let c = mesh.vertices.len() as u32;
        mesh.vertices.push(centre);
        for j in 0..ring.len() {
            mesh.faces.push([ring[j], ring[(j + 1) % ring.len()], c]);
        }
        closed += 1;
    }
    closed
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn cross(u: &[f64; 3], w: &[f64; 3]) -> [f64; 3] {
    [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ]
}

fn minus(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

type Quadric = [[f64; 4]; 4];

fn outer(p: &[f64; 4], scale: f64) -> Quadric {
    std::array::from_fn(|i| std::array::from_fn(|j| p[i] * p[j] * scale))
}

fn add_quadric(a: &mut Quadric, b: &Quadric) {
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] += b[i][j];
        }
    }
}

/// Collapse edges by least quadric error until at most `target` triangles remain, as
/// `decimate` in `slscan/mesh.py` does: plane quadrics weighted by area, boundary planes
/// weighted by `boundary_weight`, collapses that turn a triangle over skipped.
pub fn decimate(mesh: &Mesh, target: usize, boundary_weight: f64) -> Mesh {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap, HashSet};

    #[derive(PartialEq)]
    struct Cost(f64);
    impl Eq for Cost {}
    impl PartialOrd for Cost {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Cost {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    let mut v = mesh.vertices.clone();
    let mut f = mesh.faces.clone();
    let mut q: Vec<Quadric> = vec![[[0.0; 4]; 4]; v.len()];
    let mut normals = Vec::with_capacity(f.len());
    for face in &f {
        let [a, b, c] = face.map(|i| v[i as usize]);
        let n = cross(&minus(&b, &a), &minus(&c, &a));
        let area = dot(&n, &n).sqrt();
        let n = if area > 0.0 { n.map(|x| x / area) } else { n };
        normals.push(n);
        let plane = [n[0], n[1], n[2], -dot(&n, &a)];
        let k = outer(&plane, area / 2.0);
        for &i in face {
            add_quadric(&mut q[i as usize], &k);
        }
    }
    let mut owners: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut edge_order = Vec::new();
    for (i, face) in f.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (face[k], face[(k + 1) % 3]);
            let e = (a.min(b), a.max(b));
            let list = owners.entry(e).or_default();
            if list.is_empty() {
                edge_order.push(e);
            }
            list.push(i);
        }
    }
    for &(p0, p1) in &edge_order {
        let list = &owners[&(p0, p1)];
        if list.len() != 1 {
            continue;
        }
        let d = minus(&v[p1 as usize], &v[p0 as usize]);
        let side = cross(&d, &normals[list[0]]);
        let norm = dot(&side, &side).sqrt();
        if norm > 0.0 {
            let side = side.map(|x| x / norm);
            let plane = [side[0], side[1], side[2], -dot(&side, &v[p0 as usize])];
            let k = outer(&plane, boundary_weight * dot(&d, &d));
            add_quadric(&mut q[p0 as usize], &k);
            add_quadric(&mut q[p1 as usize], &k);
        }
    }
    let mut vertex_faces: Vec<HashSet<usize>> = vec![HashSet::new(); v.len()];
    for (i, face) in f.iter().enumerate() {
        for &k in face {
            vertex_faces[k as usize].insert(i);
        }
    }
    let mut alive = vec![true; f.len()];
    let mut version = vec![0u64; v.len()];
    let mut removed = vec![false; v.len()];
    let place = |q: &[Quadric], v: &[[f64; 3]], p0: u32, p1: u32| -> (f64, [f64; 3]) {
        let mut m = q[p0 as usize];
        add_quadric(&mut m, &q[p1 as usize]);
        let (a, b) = (v[p0 as usize], v[p1 as usize]);
        let mid = [0, 1, 2].map(|d| (a[d] + b[d]) / 2.0);
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let mut x = mid;
        if det != 0.0 && det.is_finite() {
            let rhs = [-m[0][3], -m[1][3], -m[2][3]];
            let solved: [f64; 3] = std::array::from_fn(|col| {
                let mut c = [
                    [m[0][0], m[0][1], m[0][2]],
                    [m[1][0], m[1][1], m[1][2]],
                    [m[2][0], m[2][1], m[2][2]],
                ];
                for r in 0..3 {
                    c[r][col] = rhs[r];
                }
                (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
                    - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
                    + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]))
                    / det
            });
            if distance(&solved, &mid) <= 4.0 * distance(&a, &b) {
                x = solved;
            }
        }
        let h = [x[0], x[1], x[2], 1.0];
        let cost: f64 = (0..4)
            .map(|i| (0..4).map(|j| h[i] * m[i][j] * h[j]).sum::<f64>())
            .sum();
        (cost, x)
    };
    let mut heap = BinaryHeap::new();
    for &(p0, p1) in &edge_order {
        heap.push(Reverse((Cost(place(&q, &v, p0, p1).0), p0, p1, 0u64, 0u64)));
    }
    let mut count = f.len();
    while count > target {
        let Some(Reverse((_, p0, p1, s0, s1))) = heap.pop() else {
            break;
        };
        let (i0, i1) = (p0 as usize, p1 as usize);
        if removed[i0] || removed[i1] || version[i0] != s0 || version[i1] != s1 {
            continue;
        }
        let (_, x) = place(&q, &v, p0, p1);
        let shared: HashSet<usize> = vertex_faces[i0]
            .intersection(&vertex_faces[i1])
            .copied()
            .collect();
        let changed: Vec<usize> = vertex_faces[i0]
            .union(&vertex_faces[i1])
            .filter(|i| !shared.contains(i))
            .copied()
            .collect();
        let flips = changed.iter().any(|&i| {
            let tri = f[i];
            let old = tri.map(|k| v[k as usize]);
            let new = tri.map(|k| if k == p0 || k == p1 { x } else { v[k as usize] });
            let n_old = cross(&minus(&old[1], &old[0]), &minus(&old[2], &old[0]));
            let n_new = cross(&minus(&new[1], &new[0]), &minus(&new[2], &new[0]));
            dot(&n_old, &n_new) <= 0.0
        });
        if flips {
            continue;
        }
        v[i0] = x;
        let q1 = q[i1];
        add_quadric(&mut q[i0], &q1);
        removed[i1] = true;
        for &i in &shared {
            alive[i] = false;
            count -= 1;
            for k in f[i] {
                vertex_faces[k as usize].remove(&i);
            }
        }
        let moved: Vec<usize> = vertex_faces[i1].iter().copied().collect();
        for i in moved {
            for k in f[i].iter_mut() {
                if *k == p1 {
                    *k = p0;
                }
            }
            vertex_faces[i0].insert(i);
        }
        vertex_faces[i1].clear();
        version[i0] += 1;
        let neighbours: HashSet<u32> = vertex_faces[i0]
            .iter()
            .flat_map(|&i| f[i])
            .filter(|&k| k != p0)
            .collect();
        for k in neighbours {
            let (lo, hi) = if p0 < k { (p0, k) } else { (k, p0) };
            let cost = place(&q, &v, p0, k).0;
            heap.push(Reverse((
                Cost(cost),
                lo,
                hi,
                version[lo as usize],
                version[hi as usize],
            )));
        }
    }
    let kept: Vec<[u32; 3]> = f
        .iter()
        .zip(&alive)
        .filter(|(_, &a)| a)
        .map(|(t, _)| *t)
        .collect();
    let mut used: Vec<u32> = kept.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let mut index = vec![u32::MAX; v.len()];
    for (n, &u) in used.iter().enumerate() {
        index[u as usize] = n as u32;
    }
    Mesh {
        vertices: used.iter().map(|&u| v[u as usize]).collect(),
        faces: kept.iter().map(|t| t.map(|k| index[k as usize])).collect(),
    }
}

/// Unit normal of a triangle by its winding, zero for a degenerate one.
fn normal(mesh: &Mesh, face: &[u32; 3]) -> [f64; 3] {
    let [a, b, c] = face.map(|v| mesh.vertices[v as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if length > 0.0 {
        n.map(|v| v / length)
    } else {
        n
    }
}

/// Write a mesh as binary PLY, ASCII OBJ or binary STL by the path's extension, as
/// `write_mesh` in `slscan/mesh.py` does.
pub fn write_mesh(path: &Path, mesh: &Mesh) -> io::Result<()> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    let mut file = io::BufWriter::new(std::fs::File::create(path)?);
    match extension {
        "ply" => {
            write!(
                file,
                "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
                mesh.vertices.len(),
                mesh.faces.len()
            )?;
            for p in &mesh.vertices {
                for v in p {
                    file.write_all(&(*v as f32).to_le_bytes())?;
                }
            }
            for f in &mesh.faces {
                file.write_all(&[3u8])?;
                for v in f {
                    file.write_all(&(*v as i32).to_le_bytes())?;
                }
            }
        }
        "obj" => {
            for p in &mesh.vertices {
                writeln!(file, "v {} {} {}", p[0], p[1], p[2])?;
            }
            for f in &mesh.faces {
                writeln!(file, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
            }
        }
        "stl" => {
            file.write_all(&[0u8; 80])?;
            let count = u32::try_from(mesh.faces.len()).map_err(io::Error::other)?;
            file.write_all(&count.to_le_bytes())?;
            for f in &mesh.faces {
                for v in normal(mesh, f) {
                    file.write_all(&(v as f32).to_le_bytes())?;
                }
                for &i in f {
                    for v in mesh.vertices[i as usize] {
                        file.write_all(&(v as f32).to_le_bytes())?;
                    }
                }
                file.write_all(&[0u8; 2])?;
            }
        }
        _ => {
            return Err(io::Error::other(format!(
                "{}: mesh files end in .ply, .obj or .stl",
                path.display()
            )))
        }
    }
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere() -> Mesh {
        let spec = VolumeSpec {
            lower: [-0.0071, -0.0073, -0.0069],
            voxel: 0.0005,
            truncation: 0.001,
            shape: [30, 30, 30],
        };
        let mut tsdf = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                for k in 0..30 {
                    let q = [i, j, k].map(|n| spec.voxel * f64::from(n));
                    let p = [0, 1, 2].map(|d| spec.lower[d] + q[d]);
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.005;
                    tsdf.push((r / spec.truncation).clamp(-1.0, 1.0) as f32);
                }
            }
        }
        let weight = vec![1.0f32; tsdf.len()];
        surface_nets(&spec, &tsdf, &weight)
    }

    fn edge_uses(mesh: &Mesh) -> std::collections::HashMap<(u32, u32), u32> {
        let mut uses = std::collections::HashMap::new();
        for f in &mesh.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        uses
    }

    #[test]
    fn decimated_sphere_stays_closed_and_round() {
        let full = sphere();
        let fewer = decimate(&full, 1000, 1000.0);
        assert!(fewer.faces.len() <= 1000 && fewer.vertices.len() < full.vertices.len() / 2);
        assert!(edge_uses(&fewer).values().all(|&n| n == 2));
        for p in &fewer.vertices {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 0.005).abs() < 0.00005);
        }
    }

    #[test]
    fn fill_holes_closes_a_small_hole_only() {
        let mut mesh = sphere();
        let top = [0.0, 0.0, 0.005];
        let bottom = [0.0, 0.0, -0.005];
        mesh.faces.retain(|f| {
            let c: [f64; 3] = std::array::from_fn(|d| {
                f.iter().map(|&v| mesh.vertices[v as usize][d]).sum::<f64>() / 3.0
            });
            distance(&c, &top) >= 0.001 && distance(&c, &bottom) >= 0.003
        });
        let open = mesh.faces.len();
        assert_eq!(fill_holes(&mut mesh, 0.01), 1);
        assert!(mesh.faces.len() > open);
        let uses = edge_uses(&mesh);
        assert!(uses.values().all(|&n| n <= 2) && uses.values().any(|&n| n == 1));
    }

    #[test]
    fn sphere_meshes_closed() {
        let mesh = sphere();
        assert!(mesh.faces.len() > 1000);
        assert!(edge_uses(&mesh).values().all(|&n| n == 2));
        for p in &mesh.vertices {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 0.005).abs() < 0.00003);
        }
    }
}
