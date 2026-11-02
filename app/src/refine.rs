//! Second-order refinement of stereo matches on the GPU: one WGSL workgroup per sample,
//! following the NumPy reference `second_order_matches` in `slscan/stereo.py`.

use crate::gpu::{download, upload, Gpu};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

/// Floats per sample: left point, right point, then the affine part of its window,
/// row-major.
pub const SAMPLE_FLOATS: usize = 8;

/// Shape terms, move limit and correlation gate of one refinement stage.
pub type Stage = (u32, f32, Option<f32>);

#[derive(Debug, Clone, PartialEq)]
pub struct RefineParams {
    pub width: u32,
    pub height: u32,
    pub window: u32,
    pub iterations: u32,
    /// Shape terms per axis: 3 for affine windows, 6 for second-order windows.
    pub terms: u32,
    /// A sample is not kept when its column moves this far or farther from its start.
    pub max_move: f32,
    /// A sample is not kept when its final window correlates below this.
    pub min_correlation: Option<f32>,
}

pub struct Refiner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gradients: wgpu::ComputePipeline,
    refine: wgpu::ComputePipeline,
    pub adapter: String,
}

fn groups(n: usize) -> Result<(u32, u32), String> {
    let groups = u32::try_from(n.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
    Ok((
        groups.clamp(1, MAX_GROUPS),
        groups.div_ceil(MAX_GROUPS).max(1),
    ))
}

impl Refiner {
    /// Open the first adapter wgpu offers, or a software one when `fallback` is set.
    pub fn new(fallback: bool) -> Result<Self, String> {
        Ok(Self::with(&Gpu::open(fallback)?))
    }

    /// Build the kernels on a shared device.
    pub fn with(gpu: &Gpu) -> Self {
        let device = gpu.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("refine"),
            source: wgpu::ShaderSource::Wgsl(include_str!("refine.wgsl").into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let gradients = pipeline("gradients");
        let refine = pipeline("refine");
        Self {
            device,
            queue: gpu.queue.clone(),
            gradients,
            refine,
            adapter: gpu.adapter.clone(),
        }
    }

    fn check(
        params: &RefineParams,
        left: &[u8],
        right: &[u8],
        samples: &[f32],
    ) -> Result<(), String> {
        let pixels = params.width as usize * params.height as usize;
        if left.len() != pixels
            || right.len() != pixels
            || !samples.len().is_multiple_of(SAMPLE_FLOATS)
        {
            return Err("image or sample sizes do not match the parameters".into());
        }
        Ok(())
    }

    /// Run refinement stages in order, each starting from the right points of the one before
    /// with an identity shape. A sample is kept when every stage keeps it.
    pub fn stages(
        &self,
        base: &RefineParams,
        left: &[u8],
        right: &[u8],
        samples: &[f32],
        stages: &[Stage],
    ) -> Result<(Vec<[f32; 2]>, Vec<bool>), String> {
        Self::check(base, left, right, samples)?;
        let left_buffer = upload(&self.device, "left", left);
        let right_buffer = upload(&self.device, "right", right);
        self.stages_buffers(base, &left_buffer, &right_buffer, samples, stages)
    }

    /// Run refinement stages on an uploaded rectified pair, as `stages` does.
    pub fn stages_buffers(
        &self,
        base: &RefineParams,
        left: &wgpu::Buffer,
        right: &wgpu::Buffer,
        samples: &[f32],
        stages: &[Stage],
    ) -> Result<(Vec<[f32; 2]>, Vec<bool>), String> {
        let gradients = self.gradients(base, right)?;
        let mut current = samples.to_vec();
        let mut kept = vec![true; samples.len() / SAMPLE_FLOATS];
        let mut points = Vec::new();
        for &(terms, max_move, min_correlation) in stages {
            let params = RefineParams {
                terms,
                max_move,
                min_correlation,
                ..base.clone()
            };
            let (stage_points, stage_kept) =
                self.run(&params, left, right, &gradients, &current)?;
            for (all, stage) in kept.iter_mut().zip(&stage_kept) {
                *all &= *stage;
            }
            for (sample, point) in current.chunks_exact_mut(SAMPLE_FLOATS).zip(&stage_points) {
                sample[2..8].copy_from_slice(&[point[0], point[1], 1.0, 0.0, 0.0, 1.0]);
            }
            points = stage_points;
        }
        Ok((points, kept))
    }

    /// Refine affine matches on a rectified pair of row-major images. Returns each sample's
    /// refined right point and whether it is kept.
    pub fn refine(
        &self,
        params: &RefineParams,
        left: &[u8],
        right: &[u8],
        samples: &[f32],
    ) -> Result<(Vec<[f32; 2]>, Vec<bool>), String> {
        Self::check(params, left, right, samples)?;
        let left_buffer = upload(&self.device, "left", left);
        let right_buffer = upload(&self.device, "right", right);
        let gradients = self.gradients(params, &right_buffer)?;
        self.run(params, &left_buffer, &right_buffer, &gradients, samples)
    }

    fn uniform(
        params: &RefineParams,
        count: u32,
        pixel_stride: u32,
        sample_stride: u32,
    ) -> Vec<u8> {
        let mut uniform = Vec::with_capacity(48);
        for v in [
            params.width,
            params.height,
            params.window,
            params.iterations,
            count,
            pixel_stride,
            sample_stride,
            params.terms,
        ] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        uniform.extend_from_slice(&params.max_move.to_le_bytes());
        uniform.extend_from_slice(&params.min_correlation.unwrap_or(0.0).to_le_bytes());
        for v in [u32::from(params.min_correlation.is_some()), 0] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        uniform
    }

    fn bind(
        &self,
        pipeline: &wgpu::ComputePipeline,
        entries: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refine"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Scharr gradients of an uploaded right image, as storage buffers on the device.
    fn gradients(
        &self,
        params: &RefineParams,
        right: &wgpu::Buffer,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
        let pixels = params.width as usize * params.height as usize;
        if right.size() < (pixels.div_ceil(4) * 4) as u64 {
            return Err("image buffer is smaller than the parameters".into());
        }
        let (px, py) = groups(pixels)?;
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &Self::uniform(params, 0, px * WORKGROUP, 1),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffer = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (pixels * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let (gx, gy) = (buffer("gx"), buffer("gy"));
        let group = self.bind(
            &self.gradients,
            &[(0, &params_buffer), (2, right), (3, &gx), (4, &gy)],
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gradients"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gradients"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gradients);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(px, py, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok((gx, gy))
    }

    fn run(
        &self,
        params: &RefineParams,
        left: &wgpu::Buffer,
        right: &wgpu::Buffer,
        gradients: &(wgpu::Buffer, wgpu::Buffer),
        samples: &[f32],
    ) -> Result<(Vec<[f32; 2]>, Vec<bool>), String> {
        let count = samples.len() / SAMPLE_FLOATS;
        if count == 0 || !samples.len().is_multiple_of(SAMPLE_FLOATS) {
            return Ok((Vec::new(), Vec::new()));
        }
        let pixels = params.width as usize * params.height as usize;
        if left.size() < (pixels.div_ceil(4) * 4) as u64 {
            return Err("image buffer is smaller than the parameters".into());
        }
        let (px, _) = groups(pixels)?;
        let sample_groups = u32::try_from(count).map_err(|e| e.to_string())?;
        let (sx, sy) = (
            sample_groups.min(MAX_GROUPS),
            sample_groups.div_ceil(MAX_GROUPS),
        );
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let init = |label, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let params_buffer = init(
            "params",
            &Self::uniform(params, sample_groups, px * WORKGROUP, sx),
            wgpu::BufferUsages::UNIFORM,
        );
        let samples_buffer = init("samples", &sample_bytes, wgpu::BufferUsages::STORAGE);
        let buffer = |label, size: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let refined = buffer("refined", count * 8);
        let kept = buffer("kept", count * 4);
        let group = self.bind(
            &self.refine,
            &[
                (0, &params_buffer),
                (1, left),
                (2, right),
                (3, &gradients.0),
                (4, &gradients.1),
                (5, &samples_buffer),
                (6, &refined),
                (7, &kept),
            ],
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("refine"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refine"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.refine);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(sx, sy, 1);
        }
        self.queue.submit([encoder.finish()]);
        let refined_bytes = download(&self.device, &self.queue, &refined, (count * 8) as u64)?;
        let kept_bytes = download(&self.device, &self.queue, &kept, (count * 4) as u64)?;
        let points = refined_bytes
            .chunks_exact(8)
            .map(|c| {
                [
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                ]
            })
            .collect();
        let kept = kept_bytes.chunks_exact(4).map(|c| c[0] != 0).collect();
        Ok((points, kept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refiner() -> Option<Refiner> {
        match Refiner::new(false) {
            Ok(r) => Some(r),
            Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
                eprintln!("no GPU adapter, test skipped: {e}");
                None
            }
            Err(e) => panic!("no GPU adapter: {e}"),
        }
    }

    fn texture(width: u32, height: u32, shift: f32) -> Vec<u8> {
        let mut image = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let (x, y) = (x as f32 + shift, y as f32);
                let v =
                    128.0 + 60.0 * (0.31 * x + 0.17 * y).sin() + 40.0 * (0.13 * x - 0.29 * y).sin();
                image.push(v.round() as u8);
            }
        }
        image
    }

    #[test]
    fn shifted_texture_refines_to_its_shift() {
        let Some(refiner) = refiner() else { return };
        let (width, height, shift) = (96u32, 64u32, 2.3f32);
        let left = texture(width, height, 0.0);
        let right = texture(width, height, shift);
        let params = RefineParams {
            width,
            height,
            window: 25,
            iterations: 20,
            terms: 6,
            max_move: 1.0,
            min_correlation: None,
        };
        let samples = [48.0, 32.0, 46.0, 32.0, 1.0, 0.0, 0.0, 1.0];
        let (points, kept) = match refiner.refine(&params, &left, &right, &samples) {
            Ok(r) => r,
            Err(e) => panic!("refine failed: {e}"),
        };
        assert_eq!(kept, vec![true]);
        assert!(
            (points[0][0] - (48.0 - shift)).abs() < 0.1,
            "refined {:?}",
            points[0]
        );
        assert!(
            (points[0][1] - 32.0).abs() < 0.02,
            "refined {:?}",
            points[0]
        );
    }

    #[test]
    fn flat_images_are_not_kept() {
        let Some(refiner) = refiner() else { return };
        let params = RefineParams {
            width: 64,
            height: 64,
            window: 25,
            iterations: 20,
            terms: 6,
            max_move: 1.0,
            min_correlation: None,
        };
        let flat = vec![100u8; 64 * 64];
        let samples = [32.0, 32.0, 30.0, 32.0, 1.0, 0.0, 0.0, 1.0];
        let (_, kept) = match refiner.refine(&params, &flat, &flat, &samples) {
            Ok(r) => r,
            Err(e) => panic!("refine failed: {e}"),
        };
        assert_eq!(kept, vec![false]);
    }
}
