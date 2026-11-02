//! Stereo rectification on the GPU: one WGSL compute invocation per four output pixels,
//! following OpenCV's undistort-rectify map and bilinear remap as `slscan/stereo.py` uses them.

use crate::gpu::{read, words, Gpu};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

/// One camera's rectification, as `slscan.stereo.rectification_params` writes it.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RectifyParams {
    /// Inverse of the rectified projection times the rectifying rotation, row-major.
    pub inverse: [f32; 9],
    pub focal: [f32; 2],
    pub centre: [f32; 2],
    /// k1, k2, p1, p2, k3.
    pub distortion: [f32; 5],
    /// Rectified projection, 3 x 4 row-major, for triangulation.
    #[serde(default)]
    pub projection: [f64; 12],
    /// Rectifying rotation, 3 x 3 row-major, for triangulation.
    #[serde(default)]
    pub rotation: [f64; 9],
}

pub struct Rectifier {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter: String,
}

impl Rectifier {
    /// Open the first adapter wgpu offers, or a software one when `fallback` is set.
    pub fn new(fallback: bool) -> Result<Self, String> {
        Ok(Self::with(&Gpu::open(fallback)?))
    }

    /// Build the kernel on a shared device.
    pub fn with(gpu: &Gpu) -> Self {
        let device = gpu.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rectify"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rectify.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rectify"),
            layout: None,
            module: &module,
            entry_point: Some("rectify"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue: gpu.queue.clone(),
            pipeline,
            adapter: gpu.adapter.clone(),
        }
    }

    /// Rectify one row-major 8-bit image of `width` by `height` pixels.
    pub fn rectify(
        &self,
        params: &RectifyParams,
        width: u32,
        height: u32,
        raw: &[u8],
    ) -> Result<Vec<u8>, String> {
        let pixels = width as usize * height as usize;
        if raw.len() != pixels {
            return Err("image size does not match the parameters".into());
        }
        let words_out = pixels.div_ceil(4);
        let groups =
            u32::try_from(words_out.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
        let (x, y) = (
            groups.clamp(1, MAX_GROUPS),
            groups.div_ceil(MAX_GROUPS).max(1),
        );
        let mut uniform = Vec::with_capacity(96);
        for v in [width, height, x * WORKGROUP, 0] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        let floats = params
            .inverse
            .iter()
            .chain(&params.focal)
            .chain(&params.centre)
            .chain(&params.distortion)
            .chain(&[0.0, 0.0]);
        for v in floats {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        let init = |label, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let params_buffer = init("params", &uniform, wgpu::BufferUsages::UNIFORM);
        let raw_buffer = init("raw", &words(raw), wgpu::BufferUsages::STORAGE);
        let size = (words_out * 4) as u64;
        let rectified = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rectified"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rectified read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rectify"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: raw_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rectified.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rectify"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rectify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&rectified, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let mut bytes = read(&self.device, &readback)?;
        bytes.truncate(pixels);
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectifier() -> Option<Rectifier> {
        match Rectifier::new(false) {
            Ok(r) => Some(r),
            Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
                eprintln!("no GPU adapter, test skipped: {e}");
                None
            }
            Err(e) => panic!("no GPU adapter: {e}"),
        }
    }

    fn camera(f: f32, cx: f32, cy: f32, distortion: [f32; 5]) -> RectifyParams {
        RectifyParams {
            inverse: [1.0 / f, 0.0, -cx / f, 0.0, 1.0 / f, -cy / f, 0.0, 0.0, 1.0],
            focal: [f, f],
            centre: [cx, cy],
            distortion,
            projection: [0.0; 12],
            rotation: [0.0; 9],
        }
    }

    #[test]
    fn identity_rectification_returns_the_image() {
        let Some(rectifier) = rectifier() else { return };
        let (width, height) = (37u32, 23u32);
        let raw: Vec<u8> = (0..width * height).map(|i| (i * 7 % 251) as u8).collect();
        let params = camera(500.0, 18.0, 11.0, [0.0; 5]);
        match rectifier.rectify(&params, width, height, &raw) {
            Ok(out) => assert_eq!(out, raw),
            Err(e) => panic!("rectify failed: {e}"),
        }
    }

    #[test]
    fn pixels_mapped_outside_the_image_are_zero() {
        let Some(rectifier) = rectifier() else { return };
        let (width, height) = (16u32, 8u32);
        let raw = vec![200u8; (width * height) as usize];
        let mut params = camera(100.0, 8.0, 4.0, [0.0; 5]);
        params.inverse[2] += 1.0;
        match rectifier.rectify(&params, width, height, &raw) {
            Ok(out) => assert!(out.iter().all(|&v| v == 0)),
            Err(e) => panic!("rectify failed: {e}"),
        }
    }
}
