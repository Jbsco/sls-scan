//! Coarse disparity on the GPU: block matching by normalized cross-correlation at reduced
//! resolution, following the NumPy reference `block_disparity_small` in `slscan/stereo.py`.

use crate::gpu::{open, read, words};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

#[derive(Debug, Clone, PartialEq)]
pub struct CoarseParams {
    /// Full-resolution image size; both must be multiples of `scale`.
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub block: u32,
    /// Lowest disparity searched, in reduced pixels.
    pub lo: i32,
    /// Number of disparities searched.
    pub count: u32,
    pub min_correlation: f32,
    pub margin: f32,
    pub fill: u32,
}

pub struct BlockMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    downsample: wgpu::ComputePipeline,
    correlate: wgpu::ComputePipeline,
    choose: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    pub adapter: String,
}

fn dispatch(n: usize) -> Result<(u32, u32), String> {
    let groups = u32::try_from(n.div_ceil(WORKGROUP as usize)).map_err(|e| e.to_string())?;
    Ok((
        groups.clamp(1, MAX_GROUPS),
        groups.div_ceil(MAX_GROUPS).max(1),
    ))
}

impl BlockMatcher {
    /// Open the first adapter wgpu offers, or a software one when `fallback` is set.
    pub fn new(fallback: bool) -> Result<Self, String> {
        let (device, queue, adapter) = open(fallback, "coarse")?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coarse"),
            source: wgpu::ShaderSource::Wgsl(include_str!("coarse.wgsl").into()),
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
        let (downsample, correlate, choose, fill) = (
            pipeline("downsample"),
            pipeline("correlate"),
            pipeline("choose"),
            pipeline("fill"),
        );
        Ok(Self {
            device,
            queue,
            downsample,
            correlate,
            choose,
            fill,
            adapter,
        })
    }

    /// Match a rectified pair of row-major 8-bit images. Returns the disparity of each
    /// reduced pixel in reduced pixels, row-major, and whether it is valid.
    pub fn disparity(
        &self,
        params: &CoarseParams,
        left: &[u8],
        right: &[u8],
    ) -> Result<(Vec<f32>, Vec<bool>), String> {
        let pixels = params.width as usize * params.height as usize;
        if left.len() != pixels || right.len() != pixels {
            return Err("image sizes do not match the parameters".into());
        }
        if params.scale == 0
            || !params.width.is_multiple_of(params.scale)
            || !params.height.is_multiple_of(params.scale)
            || params.count == 0
        {
            return Err("image size must be a multiple of the scale".into());
        }
        let (sw, sh) = (params.width / params.scale, params.height / params.scale);
        let small = sw as usize * sh as usize;
        let (px, py) = dispatch(small)?;
        let (vx, vy) = dispatch(small * params.count as usize)?;
        let mut uniform = Vec::with_capacity(48);
        for v in [
            params.width,
            params.height,
            sw,
            sh,
            params.scale,
            params.block,
        ] {
            uniform.extend_from_slice(&v.to_le_bytes());
        }
        uniform.extend_from_slice(&params.lo.to_le_bytes());
        uniform.extend_from_slice(&params.count.to_le_bytes());
        uniform.extend_from_slice(&params.min_correlation.to_le_bytes());
        uniform.extend_from_slice(&params.margin.to_le_bytes());
        for v in [px * WORKGROUP, vx * WORKGROUP] {
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
        let left_buffer = init("left", &words(left), wgpu::BufferUsages::STORAGE);
        let right_buffer = init("right", &words(right), wgpu::BufferUsages::STORAGE);
        let buffer = |label, size: usize, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let small_left = buffer("small left", small * 4, wgpu::BufferUsages::STORAGE);
        let small_right = buffer("small right", small * 4, wgpu::BufferUsages::STORAGE);
        let volume = buffer(
            "volume",
            small * params.count as usize * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let (disparity_a, valid_a) = (
            buffer("disparity a", small * 4, storage),
            buffer("valid a", small * 4, storage),
        );
        let (disparity_b, valid_b) = (
            buffer("disparity b", small * 4, storage),
            buffer("valid b", small * 4, storage),
        );
        let readback = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let (disparity_read, valid_read) = (
            buffer("disparity read", small * 4, readback),
            buffer("valid read", small * 4, readback),
        );
        let bind = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: b.as_entire_binding(),
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("coarse"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let downsample_group = bind(
            &self.downsample,
            &[
                (0, &params_buffer),
                (1, &left_buffer),
                (2, &right_buffer),
                (3, &small_left),
                (4, &small_right),
            ],
        );
        let correlate_group = bind(
            &self.correlate,
            &[
                (0, &params_buffer),
                (3, &small_left),
                (4, &small_right),
                (5, &volume),
            ],
        );
        let choose_group = bind(
            &self.choose,
            &[
                (0, &params_buffer),
                (5, &volume),
                (8, &disparity_a),
                (9, &valid_a),
            ],
        );
        let fill_groups = [
            bind(
                &self.fill,
                &[
                    (0, &params_buffer),
                    (6, &disparity_a),
                    (7, &valid_a),
                    (8, &disparity_b),
                    (9, &valid_b),
                ],
            ),
            bind(
                &self.fill,
                &[
                    (0, &params_buffer),
                    (6, &disparity_b),
                    (7, &valid_b),
                    (8, &disparity_a),
                    (9, &valid_a),
                ],
            ),
        ];
        let mut passes = vec![
            (&self.downsample, &downsample_group, (px, py)),
            (&self.correlate, &correlate_group, (vx, vy)),
            (&self.choose, &choose_group, (px, py)),
        ];
        for pass in 0..params.fill as usize {
            passes.push((&self.fill, &fill_groups[pass % 2], (px, py)));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("coarse"),
            });
        for (pipeline, group, (x, y)) in passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("coarse"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        let (disparity, valid) = if params.fill.is_multiple_of(2) {
            (&disparity_a, &valid_a)
        } else {
            (&disparity_b, &valid_b)
        };
        encoder.copy_buffer_to_buffer(disparity, 0, &disparity_read, 0, (small * 4) as u64);
        encoder.copy_buffer_to_buffer(valid, 0, &valid_read, 0, (small * 4) as u64);
        self.queue.submit([encoder.finish()]);
        let disparity_bytes = read(&self.device, &disparity_read)?;
        let valid_bytes = read(&self.device, &valid_read)?;
        let disparity = disparity_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let valid = valid_bytes.chunks_exact(4).map(|c| c[0] != 0).collect();
        Ok((disparity, valid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher() -> Option<BlockMatcher> {
        match BlockMatcher::new(false) {
            Ok(m) => Some(m),
            Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
                eprintln!("no GPU adapter, test skipped: {e}");
                None
            }
            Err(e) => panic!("no GPU adapter: {e}"),
        }
    }

    #[test]
    fn shifted_texture_matches_at_its_shift() {
        let Some(matcher) = matcher() else { return };
        let (width, height, shift) = (256u32, 128u32, 8usize);
        let mut state = 12345u32;
        let texture: Vec<u8> = (0..(width as usize + shift) * height as usize)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (state >> 24) as u8
            })
            .collect();
        let row = width as usize + shift;
        let mut left = Vec::with_capacity((width * height) as usize);
        let mut right = Vec::with_capacity((width * height) as usize);
        for y in 0..height as usize {
            left.extend_from_slice(&texture[y * row + shift..y * row + shift + width as usize]);
            right.extend_from_slice(&texture[y * row..y * row + width as usize]);
        }
        let params = CoarseParams {
            width,
            height,
            scale: 4,
            block: 7,
            lo: -6,
            count: 9,
            min_correlation: 0.6,
            margin: 0.05,
            fill: 2,
        };
        let (disparity, valid) = match matcher.disparity(&params, &left, &right) {
            Ok(r) => r,
            Err(e) => panic!("coarse failed: {e}"),
        };
        let kept: Vec<f32> = disparity
            .iter()
            .zip(&valid)
            .filter(|(_, v)| **v)
            .map(|(d, _)| *d)
            .collect();
        assert!(kept.len() > disparity.len() / 2, "{} valid", kept.len());
        assert!(
            kept.iter().all(|d| (d + 2.0).abs() < 0.5),
            "disparities {kept:?}"
        );
    }
}
