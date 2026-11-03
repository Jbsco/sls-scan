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
