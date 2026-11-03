//! Truncated signed distance fusion on the GPU: one WGSL invocation per voxel and depth
//! frame, and one per voxel for the surface, following `depth_grid`, `integrate` and
//! `surface` in `slscan/tsdf.py`.

use crate::gpu::{download, Gpu};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

/// A box of voxels from `lower`, laid out as `tsdf.Volume` lays it out: x slowest, z
/// fastest. Distances are in units of `truncation`.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeSpec {
    pub lower: [f64; 3],
    pub voxel: f64,
    pub truncation: f64,
    pub shape: [u32; 3],
}

impl VolumeSpec {
    pub fn voxels(&self) -> usize {
        self.shape.iter().map(|&n| n as usize).product()
    }
}

/// The left camera's rectifying rotation (3 x 3) and rectified projection (3 x 4), both
/// row-major, as triangulation uses them.
#[derive(Debug, Clone, PartialEq)]
pub struct LeftCamera {
    pub rotation: [f64; 9],
    pub projection: [f64; 12],
}

/// Rectified depth of a depth frame at each node of its sample grid, 0 where it has no
/// point. Nodes start at `window` px and come every `step` px.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthGrid {
    pub width: u32,
    pub height: u32,
    pub step: u32,
    pub window: u32,
    pub depth: Vec<f32>,
}

/// Place each point, in the left camera's frame, at its node of the sample grid.
pub fn depth_grid(
    points: &[[f64; 3]],
    camera: &LeftCamera,
    step: u32,
    window: u32,
    size: (u32, u32),
) -> DepthGrid {
    let nodes = |extent: u32| extent.saturating_sub(2 * window).div_ceil(step);
    let (width, height) = (nodes(size.0), nodes(size.1));
    let mut depth = vec![0.0f32; width as usize * height as usize];
    let (r, p) = (&camera.rotation, &camera.projection);
    for point in points {
        let x = [0, 1, 2].map(|i| (0..3).map(|j| r[i * 3 + j] * point[j]).sum::<f64>());
        let u = p[0] * x[0] / x[2] + p[2];
        let v = p[5] * x[1] / x[2] + p[6];
        let gx = ((u - f64::from(window)) / f64::from(step)).round_ties_even();
        let gy = ((v - f64::from(window)) / f64::from(step)).round_ties_even();
        if gx >= 0.0 && gy >= 0.0 && gx < f64::from(width) && gy < f64::from(height) {
            depth[gy as usize * width as usize + gx as usize] = x[2] as f32;
        }
    }
    DepthGrid {
        width,
        height,
        step,
        window,
        depth,
    }
}

pub struct Integrator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    surface: wgpu::ComputePipeline,
    spec: VolumeSpec,
    tsdf: wgpu::Buffer,
    weight: wgpu::Buffer,
    pub adapter: String,
}

impl Integrator {
    /// Open the first adapter wgpu offers, or a software one when `fallback` is set.
    pub fn new(fallback: bool, spec: VolumeSpec) -> Result<Self, String> {
        Ok(Self::with(&Gpu::open(fallback)?, spec))
    }

    /// Build the kernel and an empty volume on a shared device.
    pub fn with(gpu: &Gpu, spec: VolumeSpec) -> Self {
        let device = gpu.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tsdf"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tsdf.wgsl").into()),
        });
        let entry = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline = entry("integrate");
        let surface = entry("surface");
        let filled = |label: &str, value: f32| {
            let bytes: Vec<u8> = std::iter::repeat_n(value.to_le_bytes(), spec.voxels())
                .flatten()
                .collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let tsdf = filled("tsdf", 1.0);
        let weight = filled("weight", 0.0);
        Self {
            queue: gpu.queue.clone(),
            pipeline,
            surface,
            spec,
            tsdf,
            weight,
            adapter: gpu.adapter.clone(),
            device,
        }
    }

    /// Fold one depth grid into the volume. `rotation` (3 x 3 row-major) and
    /// `translation` map the frame's left camera frame into the world.
    pub fn integrate(
        &self,
        grid: &DepthGrid,
        camera: &LeftCamera,
        rotation: &[f64; 9],
        translation: &[f64; 3],
    ) -> Result<(), String> {
        if grid.width < 2 || grid.height < 2 {
            return Ok(());
        }
        let r = &camera.rotation;
        let m: [f64; 9] = std::array::from_fn(|k| {
            let (i, j) = (k / 3, k % 3);
            (0..3).map(|n| r[i * 3 + n] * rotation[j * 3 + n]).sum()
        });
        let b: [f64; 3] =
            std::array::from_fn(|i| -(0..3).map(|j| m[i * 3 + j] * translation[j]).sum::<f64>());
        let (x, y) = self.groups()?;
        let [nx, ny, nz] = self.spec.shape;
        let mut uniform = Vec::with_capacity(128);
        for v in [nx, ny, nz, x * WORKGROUP, grid.width, grid.height] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        let p = &camera.projection;
        let floats = [
            f64::from(grid.step),
            f64::from(grid.window),
            self.spec.lower[0],
            self.spec.lower[1],
            self.spec.lower[2],
            self.spec.voxel,
            self.spec.truncation,
            p[0],
            p[5],
            p[2],
            p[6],
            0.0,
            0.0,
            0.0,
        ]
        .into_iter()
        .chain(m)
        .chain(b);
        for v in floats {
            uniform.extend_from_slice(&(v as f32).to_le_bytes());
        }
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &uniform,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let depth: Vec<u8> = grid.depth.iter().flat_map(|v| v.to_le_bytes()).collect();
        let depth = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid"),
                contents: &depth,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tsdf"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tsdf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.weight.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tsdf"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tsdf"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Workgroups along x and y for one invocation per voxel.
    fn groups(&self) -> Result<(u32, u32), String> {
        let voxels = self.spec.voxels();
        let groups =
            u32::try_from(voxels.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
        Ok((
            groups.clamp(1, MAX_GROUPS),
            groups.div_ceil(MAX_GROUPS).max(1),
        ))
    }

    /// Zero crossings between observed neighbours along each axis, as `tsdf.surface` finds
    /// them, with their keys, in no fixed order. Fails when there are more than `capacity`.
    pub fn surface(&self, capacity: u32) -> Result<(Vec<[f32; 3]>, Vec<u32>), String> {
        let (x, y) = self.groups()?;
        let [nx, ny, nz] = self.spec.shape;
        let mut uniform = Vec::with_capacity(128);
        for v in [nx, ny, nz, x * WORKGROUP, 0, 0] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        let s = &self.spec;
        let head = [
            0.0,
            0.0,
            s.lower[0],
            s.lower[1],
            s.lower[2],
            s.voxel,
            s.truncation,
        ];
        for v in head.into_iter().chain([0.0; 4]) {
            uniform.extend_from_slice(&(v as f32).to_le_bytes());
        }
        uniform.extend_from_slice(&capacity.to_le_bytes());
        uniform.extend_from_slice(&[0u8; 14 * 4]);
        let init = |label: &str, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &uniform,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let count = init("count", &[0u8; 4]);
        let list = |label: &str, bytes: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let slots = u64::from(capacity.max(1));
        let points = list("points", slots * 12);
        let keys = list("keys", slots * 4);
        let entries: Vec<wgpu::BindGroupEntry> = [
            (0, &params),
            (2, &self.tsdf),
            (3, &self.weight),
            (4, &count),
            (5, &points),
            (6, &keys),
        ]
        .into_iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface"),
            layout: &self.surface.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("surface"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.surface);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit([encoder.finish()]);
        let found = download(&self.device, &self.queue, &count, 4)?;
        let found = u32::from_le_bytes([found[0], found[1], found[2], found[3]]);
        if found > capacity {
            return Err(format!("{found} surface points, more than {capacity}"));
        }
        if found == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let n = u64::from(found);
        let coords = download(&self.device, &self.queue, &points, n * 12)?;
        let keys = download(&self.device, &self.queue, &keys, n * 4)?;
        let floats: Vec<f32> = coords
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok((
            floats.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect(),
            keys.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ))
    }

    /// The volume's distances and weights.
    pub fn read(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        let floats = |buffer: &wgpu::Buffer| -> Result<Vec<f32>, String> {
            let bytes = download(&self.device, &self.queue, buffer, buffer.size())?;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        Ok((floats(&self.tsdf)?, floats(&self.weight)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> LeftCamera {
        LeftCamera {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            projection: [
                4000.0, 0.0, 160.0, 0.0, 0.0, 4000.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        }
    }

    #[test]
    fn depth_grid_places_points_at_their_nodes() {
        let point =
            |x: f64, y: f64, z: f64| [(x - 160.0) * z / 4000.0, (y - 120.0) * z / 4000.0, z];
        let grid = depth_grid(
            &[point(8.0, 8.0, 0.5), point(12.0, 16.0, 0.6)],
            &camera(),
            4,
            8,
            (320, 240),
        );
        assert_eq!((grid.width, grid.height), (76, 56));
        assert!((grid.depth[0] - 0.5).abs() < 1e-7);
        assert!((grid.depth[2 * 76 + 1] - 0.6).abs() < 1e-7);
        assert_eq!(grid.depth.iter().filter(|&&d| d > 0.0).count(), 2);
    }
}
