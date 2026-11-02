//! Second-order refinement of stereo matches on the GPU: one WGSL compute invocation per
//! sample, following the NumPy reference `second_order_matches` in `slscan/stereo.py`.

use crate::gpu::{open, read, words};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

/// Floats per sample: left point, right point, then the affine part of its window,
/// row-major.
pub const SAMPLE_FLOATS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct RefineParams {
    pub width: u32,
    pub height: u32,
    pub window: u32,
    pub iterations: u32,
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
        let (device, queue, adapter) = open(fallback, "refine")?;
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
        Ok(Self {
            device,
            queue,
            gradients,
            refine,
            adapter,
        })
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
        let pixels = params.width as usize * params.height as usize;
        if left.len() != pixels
            || right.len() != pixels
            || !samples.len().is_multiple_of(SAMPLE_FLOATS)
        {
            return Err("image or sample sizes do not match the parameters".into());
        }
        let count = samples.len() / SAMPLE_FLOATS;
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (px, py) = groups(pixels)?;
        let (sx, sy) = groups(count)?;
        let mut uniform = Vec::with_capacity(32);
        for v in [
            params.width,
            params.height,
            params.window,
            params.iterations,
            u32::try_from(count).map_err(|e| e.to_string())?,
            px * WORKGROUP,
            sx * WORKGROUP,
            0,
        ] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let init = |label, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let params_buffer = init("params", &uniform, wgpu::BufferUsages::UNIFORM);
        let left_buffer = init("left", &words(left), wgpu::BufferUsages::STORAGE);
        let right_buffer = init("right", &words(right), wgpu::BufferUsages::STORAGE);
        let samples_buffer = init("samples", &sample_bytes, wgpu::BufferUsages::STORAGE);
        let buffer = |label, size: usize, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let readback = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let gx = buffer("gx", pixels * 4, wgpu::BufferUsages::STORAGE);
        let gy = buffer("gy", pixels * 4, wgpu::BufferUsages::STORAGE);
        let refined = buffer("refined", count * 8, storage);
        let kept = buffer("kept", count * 4, storage);
        let refined_read = buffer("refined read", count * 8, readback);
        let kept_read = buffer("kept read", count * 4, readback);
        let bind = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
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
        };
        let gradient_group = bind(
            &self.gradients,
            &[(0, &params_buffer), (2, &right_buffer), (3, &gx), (4, &gy)],
        );
        let refine_group = bind(
            &self.refine,
            &[
                (0, &params_buffer),
                (1, &left_buffer),
                (2, &right_buffer),
                (3, &gx),
                (4, &gy),
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
        for (pipeline, group, (x, y)) in [
            (&self.gradients, &gradient_group, (px, py)),
            (&self.refine, &refine_group, (sx, sy)),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refine"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&refined, 0, &refined_read, 0, (count * 8) as u64);
        encoder.copy_buffer_to_buffer(&kept, 0, &kept_read, 0, (count * 4) as u64);
        self.queue.submit([encoder.finish()]);
        let refined_bytes = read(&self.device, &refined_read)?;
        let kept_bytes = read(&self.device, &kept_read)?;
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
