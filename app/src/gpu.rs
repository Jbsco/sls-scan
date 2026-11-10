//! Decode on the GPU: one WGSL compute invocation per camera pixel, following the NumPy
//! reference in `slscan/decode.py`.

use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use std::sync::{mpsc, Mutex};
use std::time::Instant;

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
    /// Buffers of the last decode, kept while the stack sizes stay the same.
    kept: Mutex<Option<Bound>>,
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

/// Map a readback buffer after the queue's work, run `f` on its bytes and unmap it.
pub fn read_with<T>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    f: impl FnOnce(&[u8]) -> T,
) -> Result<T, String> {
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
    let result = f(&buffer.get_mapped_range(..));
    buffer.unmap();
    Ok(result)
}

/// Map a readback buffer after the queue's work and copy its contents to host memory.
pub fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
    read_with(device, buffer, <[u8]>::to_vec)
}

/// The 32-bit floats a kernel wrote, from their bytes.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

/// One flag per 32-bit word a kernel wrote, set where the word is not zero.
pub fn flags(bytes: &[u8]) -> Vec<bool> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| u32::from_le_bytes(*w) != 0).collect()
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
            kept: Mutex::new(None),
        }
    }

    /// Allocate the buffers and bind group of a decode with stacks of these byte lengths
    /// over n pixels.
    fn allocate(&self, lengths: (usize, usize, usize)) -> Result<Bound, String> {
        let (gray, phase, n) = lengths;
        let groups = u32::try_from(n.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
        let size = (n * 4) as u64;
        let buffer = |label, size, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stack = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let output = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let readback = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let params = buffer(
            "params",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let gray_buffer = buffer("gray", gray.div_ceil(4) as u64 * 4, stack);
        let phase_buffer = buffer("phase", phase.div_ceil(4) as u64 * 4, stack);
        let (columns, mask) = (
            buffer("columns", size, output),
            buffer("mask", size, output),
        );
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decode"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
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
        Ok(Bound {
            lengths,
            params,
            gray: gray_buffer,
            phase: phase_buffer,
            columns_read: buffer("columns read", size, readback),
            mask_read: buffer("mask read", size, readback),
            columns,
            mask,
            bind_group,
            groups: (groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS)),
            size,
        })
    }

    /// Write the parameters and stacks into the kept buffers, allocating them when the
    /// sizes change, and run `f` on them.
    fn with_bound<T>(
        &self,
        params: &DecodeParams,
        gray: &[u8],
        phase: &[u8],
        f: impl FnOnce(&Bound) -> Result<T, String>,
    ) -> Result<T, String> {
        let n = params.width as usize * params.height as usize;
        if gray.len() != 2 * params.planes as usize * n || phase.len() != params.steps as usize * n
        {
            return Err("stack sizes do not match the parameters".into());
        }
        let lengths = (gray.len(), phase.len(), n);
        let mut kept = self.kept.lock().map_err(|e| e.to_string())?;
        let bound = match kept.take() {
            Some(b) if b.lengths == lengths => b,
            _ => self.allocate(lengths)?,
        };
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
        for v in [bound.groups.0 * WORKGROUP, 0, 0] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        self.queue.write_buffer(&bound.params, 0, &uniform);
        write_words(&self.queue, &bound.gray, gray);
        write_words(&self.queue, &bound.phase, phase);
        let result = f(&bound);
        *kept = Some(bound);
        result
    }

    fn encode(&self, bound: &Bound) -> wgpu::CommandEncoder {
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
            pass.set_bind_group(0, &bound.bind_group, &[]);
            pass.dispatch_workgroups(bound.groups.0, bound.groups.1, 1);
        }
        encoder
    }

    /// Decode stacks laid out as in `slscan`: plane-major, then rows, then columns.
    pub fn decode(
        &self,
        params: &DecodeParams,
        gray: &[u8],
        phase: &[u8],
    ) -> Result<(Vec<f32>, Vec<bool>), String> {
        self.with_bound(params, gray, phase, |bound| {
            let mut encoder = self.encode(bound);
            encoder.copy_buffer_to_buffer(&bound.columns, 0, &bound.columns_read, 0, bound.size);
            encoder.copy_buffer_to_buffer(&bound.mask, 0, &bound.mask_read, 0, bound.size);
            self.queue.submit([encoder.finish()]);
            let columns = read_with(&self.device, &bound.columns_read, floats)?;
            let mask = read_with(&self.device, &bound.mask_read, flags)?;
            Ok((columns, mask))
        })
    }

    /// Time the kernel alone, in milliseconds: the fastest of `runs` dispatches on stacks
    /// already on the device, with no readback.
    pub fn time_kernel(
        &self,
        params: &DecodeParams,
        gray: &[u8],
        phase: &[u8],
        runs: usize,
    ) -> Result<f64, String> {
        self.with_bound(params, gray, phase, |bound| {
            let mut fastest = f64::INFINITY;
            for _ in 0..runs {
                let start = Instant::now();
                self.queue.submit([self.encode(bound).finish()]);
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| e.to_string())?;
                fastest = fastest.min(start.elapsed().as_secs_f64() * 1000.0);
            }
            Ok(fastest)
        })
    }
}

/// Buffers and bind group of a decode, with the byte lengths of its stacks, its pixel
/// count and its dispatch size.
struct Bound {
    lengths: (usize, usize, usize),
    params: wgpu::Buffer,
    gray: wgpu::Buffer,
    phase: wgpu::Buffer,
    columns: wgpu::Buffer,
    mask: wgpu::Buffer,
    columns_read: wgpu::Buffer,
    mask_read: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    groups: (u32, u32),
    size: u64,
}

/// Write bytes to a buffer of whole 32-bit words, the last word padded with zeros.
fn write_words(queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: &[u8]) {
    let whole = bytes.len() / 4 * 4;
    if whole > 0 {
        queue.write_buffer(buffer, 0, &bytes[..whole]);
    }
    if whole < bytes.len() {
        let mut last = [0u8; 4];
        last[..bytes.len() - whole].copy_from_slice(&bytes[whole..]);
        queue.write_buffer(buffer, whole as u64, &last);
    }
}

/// One device for every GPU test, opened once, so tests on parallel threads never open
/// devices at the same time. None without an adapter, unless `SLS_REQUIRE_GPU` is set.
#[cfg(test)]
pub(crate) fn test_device() -> Option<&'static Gpu> {
    static DEVICE: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| match Gpu::open(false) {
            Ok(gpu) => Some(gpu),
            Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
                eprintln!("no GPU adapter, GPU tests skipped: {e}");
                None
            }
            Err(e) => panic!("no GPU adapter: {e}"),
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder() -> Option<Decoder> {
        crate::gpu::test_device().map(Decoder::with)
    }

    /// Stacks that encode the given columns exactly, with their parameters.
    fn exact_stacks(columns: &[f32], steps: u32) -> (DecodeParams, Vec<u8>, Vec<u8>) {
        let (projector_width, pitch, planes) = (64u32, 8.0f32, 6u32);
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
        (params, gray, phase)
    }

    fn decode_exact(decoder: &Decoder, columns: &[f32], steps: u32) -> Vec<f32> {
        let (params, gray, phase) = exact_stacks(columns, steps);
        let (decoded, mask) = match decoder.decode(&params, &gray, &phase) {
            Ok(r) => r,
            Err(e) => panic!("decode failed: {e}"),
        };
        assert!(mask.iter().all(|&m| m));
        for (d, u) in decoded.iter().zip(columns) {
            assert!((d - u).abs() < 0.05, "decoded {d} for column {u}");
        }
        decoded
    }

    #[test]
    fn exact_patterns_decode_to_their_columns() {
        let Some(decoder) = decoder() else { return };
        let columns: Vec<f32> = (0..40).map(|i| 3.0 + 1.37 * i as f32).collect();
        decode_exact(&decoder, &columns, 4);
    }

    #[test]
    fn kept_buffers_follow_changing_stack_sizes() {
        let Some(decoder) = decoder() else { return };
        let wide: Vec<f32> = (0..40).map(|i| 3.0 + 1.37 * i as f32).collect();
        let narrow: Vec<f32> = (0..7).map(|i| 50.0 - 4.1 * i as f32).collect();
        let first = decode_exact(&decoder, &wide, 4);
        decode_exact(&decoder, &narrow, 3);
        assert_eq!(decode_exact(&decoder, &wide, 4), first);
    }

    #[test]
    fn readback_bytes_convert_to_floats_and_flags() {
        let bytes: Vec<u8> = [1.5f32, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(floats(&bytes), vec![1.5, -2.25]);
        let words: Vec<u8> = [0u32, 1, 256]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(flags(&words), vec![false, true, true]);
        assert_eq!(floats(&bytes[..7]), vec![1.5]);
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

    #[test]
    fn kernel_time_is_measured() {
        let Some(decoder) = decoder() else { return };
        let params = DecodeParams {
            width: 64,
            height: 8,
            planes: 6,
            steps: 4,
            projector_width: 64,
            pitch: 8.0,
            min_contrast: 10.0,
            min_modulation: 10.0,
            max_disagreement: 1.0,
        };
        let n = 64 * 8;
        let (gray, phase) = (vec![0u8; 12 * n], vec![0u8; 4 * n]);
        let ms = match decoder.time_kernel(&params, &gray, &phase, 3) {
            Ok(ms) => ms,
            Err(e) => panic!("kernel timing failed: {e}"),
        };
        assert!(ms.is_finite() && ms >= 0.0, "kernel time {ms}");
    }
}
