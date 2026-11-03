//! Decode on the GPU: one WGSL compute invocation per camera pixel, following the NumPy
//! reference in `slscan/decode.py`.

use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use std::sync::mpsc;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeParams {
    pub width: u32,
    pub height: u32,
    pub planes: u32,
    pub steps: u32,
    pub projector_width: u32,
    pub pitch: f32,
    pub min_contrast: f32,
    pub min_modulation: f32,
    pub max_disagreement: f32,
}

pub struct Decoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter: String,
}

/// Pad bytes to whole 32-bit words, as WGSL storage arrays of `u32` need.
pub fn words(bytes: &[u8]) -> Vec<u8> {
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().div_ceil(4) * 4, 0);
    padded
}

/// A device on the first adapter wgpu offers, or on a software one when `fallback` is set,
/// shared by every kernel built from it.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,
    /// The adapter runs on the CPU, as lavapipe does.
    pub software: bool,
}

impl Gpu {
    pub fn open(fallback: bool) -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: fallback,
            compatible_surface: None,
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sls-scan"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        Ok(Self {
            device,
            queue,
            adapter: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
            software: info.device_type == wgpu::DeviceType::Cpu,
        })
    }

    /// Upload an 8-bit image as a storage buffer of packed words.
    pub fn upload(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        upload(&self.device, label, bytes)
    }

    /// Copy the first `size` bytes of a buffer the kernels wrote to host memory.
    pub fn download(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
        download(&self.device, &self.queue, buffer, size)
    }
}

/// Upload an 8-bit image as a storage buffer of packed words.
pub fn upload(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &words(bytes),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Copy the first `size` bytes of a buffer the kernels wrote to host memory.
pub fn download(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, String> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    read(device, &readback)
}

/// Map a readback buffer after the queue's work and copy its contents to host memory.
pub fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
    let (tx, rx) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let bytes = buffer.get_mapped_range(..).to_vec();
    buffer.unmap();
    Ok(bytes)
}

impl Decoder {
    /// Open the first adapter wgpu offers, or a software one when `fallback` is set.
    pub fn new(fallback: bool) -> Result<Self, String> {
        Ok(Self::with(&Gpu::open(fallback)?))
    }

    /// Build the kernel on a shared device.
    pub fn with(gpu: &Gpu) -> Self {
        let device = gpu.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decode"),
            source: wgpu::ShaderSource::Wgsl(include_str!("decode.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("decode"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
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

    /// Decode stacks laid out as in `slscan`: plane-major, then rows, then columns.
    pub fn decode(
        &self,
        params: &DecodeParams,
        gray: &[u8],
        phase: &[u8],
    ) -> Result<(Vec<f32>, Vec<bool>), String> {
        let n = params.width as usize * params.height as usize;
        if gray.len() != 2 * params.planes as usize * n || phase.len() != params.steps as usize * n
        {
            return Err("stack sizes do not match the parameters".into());
        }
        let groups = u32::try_from(n.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
        let (x, y) = (groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS));
        let mut uniform = Vec::with_capacity(48);
        for v in [
            params.width,
            params.height,
            params.planes,
            params.steps,
            params.projector_width,
        ] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        for v in [
            params.pitch,
            params.min_contrast,
            params.min_modulation,
            params.max_disagreement,
        ] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        for v in [x * WORKGROUP, 0, 0] {
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
        let gray_buffer = init("gray", &words(gray), wgpu::BufferUsages::STORAGE);
        let phase_buffer = init("phase", &words(phase), wgpu::BufferUsages::STORAGE);
        let size = (n * 4) as u64;
        let output = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let readback = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (columns, mask) = (output("columns"), output("mask"));
        let (columns_read, mask_read) = (readback("columns read"), readback("mask read"));
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decode"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: phase_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: columns.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mask.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&columns, 0, &columns_read, 0, size);
        encoder.copy_buffer_to_buffer(&mask, 0, &mask_read, 0, size);
        self.queue.submit([encoder.finish()]);
        let column_bytes = read(&self.device, &columns_read)?;
        let mask_bytes = read(&self.device, &mask_read)?;
        let columns = column_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mask = mask_bytes.chunks_exact(4).map(|c| c[0] != 0).collect();
        Ok((columns, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder() -> Option<Decoder> {
        match Decoder::new(false) {
            Ok(d) => Some(d),
            Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
                eprintln!("no GPU adapter, test skipped: {e}");
                None
            }
            Err(e) => panic!("no GPU adapter: {e}"),
        }
    }

    #[test]
    fn exact_patterns_decode_to_their_columns() {
        let Some(decoder) = decoder() else { return };
        let (projector_width, pitch, steps) = (64u32, 8.0f32, 4u32);
        let planes = 6u32;
        let columns: Vec<f32> = (0..40).map(|i| 3.0 + 1.37 * i as f32).collect();
        let n = columns.len();
        let mut gray = vec![0u8; 2 * planes as usize * n];
        let mut phase = vec![0u8; steps as usize * n];
        for (p, &u) in columns.iter().enumerate() {
            let c = u.round() as u32;
            let code = c ^ (c >> 1);
            for k in 0..planes {
                let bit = (code >> (planes - 1 - k)) & 1;
                gray[(2 * k as usize) * n + p] = if bit == 1 { 230 } else { 25 };
                gray[(2 * k as usize + 1) * n + p] = if bit == 1 { 25 } else { 230 };
            }
            for i in 0..steps {
                let angle = std::f32::consts::TAU * (u / pitch + i as f32 / steps as f32);
                phase[i as usize * n + p] = (127.5 + 100.0 * angle.cos()).round() as u8;
            }
        }
        let params = DecodeParams {
            width: n as u32,
            height: 1,
            planes,
            steps,
            projector_width,
            pitch,
            min_contrast: 10.0,
            min_modulation: 10.0,
            max_disagreement: 1.0,
        };
        let (decoded, mask) = match decoder.decode(&params, &gray, &phase) {
            Ok(r) => r,
            Err(e) => panic!("decode failed: {e}"),
        };
        assert!(mask.iter().all(|&m| m));
        for (d, u) in decoded.iter().zip(&columns) {
            assert!((d - u).abs() < 0.05, "decoded {d} for column {u}");
        }
    }

    #[test]
    fn low_contrast_pixels_are_masked() {
        let Some(decoder) = decoder() else { return };
        let params = DecodeParams {
            width: 2,
            height: 1,
            planes: 1,
            steps: 3,
            projector_width: 2,
            pitch: 2.0,
            min_contrast: 10.0,
            min_modulation: 10.0,
            max_disagreement: 1.0,
        };
        let gray = [100, 100, 104, 100];
        let phase = [100, 100, 100, 100, 100, 100];
        let (_, mask) = match decoder.decode(&params, &gray, &phase) {
            Ok(r) => r,
            Err(e) => panic!("decode failed: {e}"),
        };
        assert_eq!(mask, vec![false, false]);
    }
}
