//! One depth frame from a raw stereo pair: rectification, block disparity and direct
//! refinement on the GPU, sample selection and triangulation on the CPU, following
//! `coarse_samples`, `direct_refine` and `triangulate` in `slscan/stereo.py`. A preview
//! skips refinement and triangulates block disparity, following `preview_matches`.

use crate::coarse::{BlockMatcher, CoarseParams};
use crate::gpu::Gpu;
use crate::rectify::{Rectifier, RectifyParams};
use crate::refine::{RefineParams, Refiner, SAMPLE_FLOATS};
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct DepthParams {
    pub coarse: CoarseParams,
    /// Sample spacing in pixels.
    pub step: u32,
    /// Matching window in pixels.
    pub window: u32,
    pub iterations: u32,
    /// Refinement stops once a step moves a match less than this, in pixels.
    pub exit_px: f32,
    /// A sample's window must lie where the 5 x 5 mean of the left image rounds to this or
    /// more.
    pub min_level: f32,
    pub max_move: f32,
    pub min_correlation: f32,
    pub near_m: f64,
    pub far_m: f64,
}

/// Block matching search and settings, as `slscan depth-params` writes them.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct CoarseSetup {
    pub scale: u32,
    pub block: u32,
    pub lo: i32,
    pub count: u32,
    pub min_correlation: f32,
    pub margin: f32,
    pub fill: u32,
}

/// Depth bounds and sampling, as `slscan depth-params` writes them.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DepthSection {
    pub near: f64,
    pub far: f64,
    pub step: u32,
    pub min_level: f32,
}

/// Block matching scale and search of the preview, as `slscan depth-params` writes them.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct PreviewSection {
    pub scale: u32,
    pub lo: i32,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Cameras {
    pub left: RectifyParams,
    pub right: RectifyParams,
}

/// Refinement exit of parameter files written before the exit was a setting.
pub fn earlier_exit() -> f32 {
    1e-5
}

/// Everything a depth frame takes for one calibration, as `slscan depth-params` writes it.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DepthSetup {
    pub width: u32,
    pub height: u32,
    pub window: u32,
    pub iterations: u32,
    #[serde(default = "earlier_exit")]
    pub exit_px: f32,
    pub max_move: f32,
    pub min_correlation: f32,
    pub coarse: CoarseSetup,
    pub depth: DepthSection,
    pub cameras: Cameras,
    #[serde(default)]
    pub preview: Option<PreviewSection>,
}

impl DepthSetup {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn params(&self) -> DepthParams {
        DepthParams {
            coarse: CoarseParams {
                width: self.width,
                height: self.height,
                scale: self.coarse.scale,
                block: self.coarse.block,
                lo: self.coarse.lo,
                count: self.coarse.count,
                min_correlation: self.coarse.min_correlation,
                margin: self.coarse.margin,
                fill: self.coarse.fill,
            },
            step: self.depth.step,
            window: self.window,
            iterations: self.iterations,
            exit_px: self.exit_px,
            min_level: self.depth.min_level,
            max_move: self.max_move,
            min_correlation: self.min_correlation,
            near_m: self.depth.near,
            far_m: self.depth.far,
        }
    }

    /// Block matching parameters of the preview: the depth frame's, at the preview's scale
    /// and search.
    pub fn preview_params(&self) -> Option<CoarseParams> {
        let preview = self.preview.as_ref()?;
        Some(CoarseParams {
            scale: preview.scale,
            lo: preview.lo,
            count: preview.count,
            ..self.params().coarse
        })
    }
}

/// Write points as a binary little-endian PLY file of float vertices, as `slscan` does.
pub fn write_ply(path: &Path, points: &[[f64; 3]]) -> io::Result<()> {
    let mut file = io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        file,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
        points.len()
    )?;
    for p in points {
        for v in p {
            file.write_all(&(*v as f32).to_le_bytes())?;
        }
    }
    file.flush()
}

pub struct DepthResult {
    /// Rectified left pixel of each point.
    pub samples: Vec<[f32; 2]>,
    /// Points in metres in the left camera's frame.
    pub points: Vec<[f64; 3]>,
    /// Milliseconds per stage, in order.
    pub timings: Vec<(&'static str, f64)>,
}

pub struct DepthFrame {
    gpu: Gpu,
    rectifier: Rectifier,
    matcher: BlockMatcher,
    refiner: Refiner,
    pub adapter: String,
}

/// Summed-area table with a zero first row and column.
fn integral(width: usize, height: usize, value: impl Fn(usize, usize) -> u32) -> Vec<u64> {
    let mut table = vec![0u64; (width + 1) * (height + 1)];
    for y in 0..height {
        let mut row = 0u64;
        for x in 0..width {
            row += u64::from(value(x, y));
            table[(y + 1) * (width + 1) + x + 1] = table[y * (width + 1) + x + 1] + row;
        }
    }
    table
}

fn box_sum(table: &[u64], width: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
    let w = width + 1;
    table[y1 * w + x1] + table[y0 * w + x0] - table[y0 * w + x1] - table[y1 * w + x0]
}

fn reflect101(i: isize, n: isize) -> usize {
    let r = if i < 0 {
        -i
    } else if i >= n {
        2 * n - 2 - i
    } else {
        i
    };
    r as usize
}

/// Pick grid samples whose whole window lies on lit, matched surface, with right guesses
/// from the coarse disparity.
pub fn select(
    params: &DepthParams,
    left: &[u8],
    disparity: &[f32],
    valid: &[bool],
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let (w, h) = (params.coarse.width as usize, params.coarse.height as usize);
    let scale = params.coarse.scale as usize;
    let sw = w / scale;
    let padded = integral(w + 4, h + 4, |x, y| {
        let sx = reflect101(x as isize - 2, w as isize);
        let sy = reflect101(y as isize - 2, h as isize);
        u32::from(left[sy * w + sx])
    });
    let threshold = 25.0 * (f64::from(params.min_level) - 0.5);
    let lit = |x: usize, y: usize| box_sum(&padded, w + 4, x, y, x + 5, y + 5) as f64 >= threshold;
    let matched = integral(w, h, |x, y| {
        u32::from(lit(x, y) && valid[(y / scale) * sw + x / scale])
    });
    let (window, step) = (params.window as usize, params.step as usize);
    let kernel = window + 8;
    let mut samples = Vec::new();
    let mut guesses = Vec::new();
    for y in (window..h.saturating_sub(window)).step_by(step) {
        for x in (window..w.saturating_sub(window)).step_by(step) {
            let (x0, y0) = (x - kernel / 2, y - kernel / 2);
            let (x1, y1) = ((x0 + kernel).min(w), (y0 + kernel).min(h));
            let full = ((x1 - x0) * (y1 - y0)) as u64;
            if box_sum(&matched, w, x0, y0, x1, y1) != full {
                continue;
            }
            let d = disparity[(y / scale) * sw + x / scale] * scale as f32;
            samples.push([x as f32, y as f32]);
            guesses.push([x as f32 - d, y as f32]);
        }
    }
    (samples, guesses)
}

/// Match every reduced pixel that block matching matched and whose block of the rectified
/// left image averages `min_level` or more, at the centre of its block, as
/// `preview_matches` in `slscan/stereo.py` does.
pub fn preview_matches(
    coarse: &CoarseParams,
    min_level: f32,
    left: &[u8],
    disparity: &[f32],
    valid: &[bool],
) -> Vec<([f32; 2], [f32; 2])> {
    let scale = coarse.scale as usize;
    let width = coarse.width as usize;
    let small_width = width / scale;
    let centre = (coarse.scale as f32 - 1.0) / 2.0;
    let threshold = f64::from(min_level) * (scale * scale) as f64;
    let lit = |i: usize| {
        let (x0, y0) = ((i % small_width) * scale, (i / small_width) * scale);
        let sum: u32 = (y0..y0 + scale)
            .map(|y| {
                left[y * width + x0..y * width + x0 + scale]
                    .iter()
                    .map(|&v| u32::from(v))
                    .sum::<u32>()
            })
            .sum();
        f64::from(sum) >= threshold
    };
    valid
        .iter()
        .zip(disparity)
        .enumerate()
        .filter(|(i, (v, _))| **v && lit(*i))
        .map(|(i, (_, d))| {
            let x = ((i % small_width) * scale) as f32 + centre;
            let y = ((i / small_width) * scale) as f32 + centre;
            ([x, y], [x - d * coarse.scale as f32, y])
        })
        .collect()
}

/// Triangulate rectified matches, as OpenCV's linear method does for a consistent pair,
/// and keep points between 0.8 near and 1.2 far, rotated into the left camera's frame.
pub fn triangulate(
    left: &RectifyParams,
    right: &RectifyParams,
    matches: &[([f32; 2], [f32; 2])],
    near_m: f64,
    far_m: f64,
) -> Vec<Option<[f64; 3]>> {
    let row = |p: &[f64; 12], r: usize, c: usize| p[r * 4 + c];
    matches
        .iter()
        .map(|(l, r)| {
            let (xl, y, xr) = (f64::from(l[0]), f64::from(l[1]), f64::from(r[0]));
            let mut a = [[0.0f64; 4]; 4];
            for (k, (p, u, v)) in [(&left.projection, xl, y), (&right.projection, xr, y)]
                .into_iter()
                .enumerate()
            {
                a[2 * k] = [0, 1, 2, 3].map(|c| u * row(p, 2, c) - row(p, 0, c));
                a[2 * k + 1] = [0, 1, 2, 3].map(|c| v * row(p, 2, c) - row(p, 1, c));
            }
            let mut n = [[0.0f64; 3]; 3];
            let mut b = [0.0f64; 3];
            for i in 0..3 {
                for j in 0..3 {
                    n[i][j] = (0..4).map(|k| a[k][i] * a[k][j]).sum();
                }
                b[i] = -(0..4).map(|k| a[k][i] * a[k][3]).sum::<f64>();
            }
            let det = |m: &[[f64; 3]; 3]| {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            };
            let d = det(&n);
            if d.abs() < 1e-300 {
                return None;
            }
            let mut x = [0.0f64; 3];
            for (i, xi) in x.iter_mut().enumerate() {
                let mut m = n;
                for (r, row) in m.iter_mut().enumerate() {
                    row[i] = b[r];
                }
                *xi = det(&m) / d;
            }
            if !(x[2] > near_m * 0.8 && x[2] < far_m * 1.2) {
                return None;
            }
            let rot = &left.rotation;
            Some([0, 1, 2].map(|i| (0..3).map(|j| x[j] * rot[j * 3 + i]).sum()))
        })
        .collect()
}

/// Triangulate matches into a result with the left pixel of each point in range.
fn depth_result(
    cameras: (&RectifyParams, &RectifyParams),
    matches: &[([f32; 2], [f32; 2])],
    near_m: f64,
    far_m: f64,
) -> DepthResult {
    let points = triangulate(cameras.0, cameras.1, matches, near_m, far_m);
    let mut out = DepthResult {
        samples: Vec::new(),
        points: Vec::new(),
        timings: Vec::new(),
    };
    for ((s, _), p) in matches.iter().zip(points) {
        if let Some(p) = p {
            out.samples.push(*s);
            out.points.push(p);
        }
    }
    out
}

impl DepthFrame {
    /// Open one device on the first adapter wgpu offers, or on a software one when
    /// `fallback` is set, and build every kernel on it.
    pub fn new(fallback: bool) -> Result<Self, String> {
        let gpu = Gpu::open(fallback)?;
        Ok(Self {
            rectifier: Rectifier::with(&gpu),
            matcher: BlockMatcher::with(&gpu),
            refiner: Refiner::with(&gpu),
            adapter: gpu.adapter.clone(),
            gpu,
        })
    }

    /// The device the kernels share, for more kernels on it.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The adapter runs on the CPU, where a full depth frame takes seconds.
    pub fn software(&self) -> bool {
        self.gpu.software
    }

    /// Reference dots in a raw pair taken with the pattern light off, placed in the left
    /// camera's frame against `surface`, the points of a pattern-lit pair taken next to it.
    pub fn dots(
        &self,
        params: &DepthParams,
        cameras: (&RectifyParams, &RectifyParams),
        left_raw: &[u8],
        right_raw: &[u8],
        surface: &[[f64; 3]],
    ) -> Result<Vec<[f64; 3]>, String> {
        let (w, h) = (params.coarse.width, params.coarse.height);
        let left = self.rectifier.rectify(cameras.0, w, h, left_raw)?;
        let right = self.rectifier.rectify(cameras.1, w, h, right_raw)?;
        Ok(crate::dots::find_dots(
            cameras.0,
            cameras.1,
            (&left, &right),
            w as usize,
            h as usize,
            surface,
            params.near_m,
            params.far_m,
        ))
    }

    /// Measure one raw stereo pair: a preview at `preview` settings when given, else a full
    /// depth frame.
    pub fn measure(
        &self,
        params: &DepthParams,
        preview: Option<&CoarseParams>,
        cameras: (&RectifyParams, &RectifyParams),
        left_raw: &[u8],
        right_raw: &[u8],
    ) -> Result<DepthResult, String> {
        match preview {
            Some(coarse) => self.preview(params, coarse, cameras, left_raw, right_raw),
            None => self.depth(params, cameras, left_raw, right_raw),
        }
    }

    /// Measure one raw stereo pair.
    pub fn depth(
        &self,
        params: &DepthParams,
        cameras: (&RectifyParams, &RectifyParams),
        left_raw: &[u8],
        right_raw: &[u8],
    ) -> Result<DepthResult, String> {
        let (w, h) = (params.coarse.width, params.coarse.height);
        let pixels = w as usize * h as usize;
        if left_raw.len() != pixels || right_raw.len() != pixels {
            return Err("image sizes do not match the parameters".into());
        }
        let mut timings = Vec::new();
        let mut clock = Instant::now();
        let mut lap = |name: &'static str, timings: &mut Vec<(&'static str, f64)>| {
            timings.push((name, clock.elapsed().as_secs_f64() * 1000.0));
            clock = Instant::now();
        };
        let left_raw = self.gpu.upload("left raw", left_raw);
        let right_raw = self.gpu.upload("right raw", right_raw);
        let left_buffer = self.rectifier.rectify_buffer(cameras.0, w, h, &left_raw)?;
        let right_buffer = self.rectifier.rectify_buffer(cameras.1, w, h, &right_raw)?;
        let mut left = self.gpu.download(&left_buffer, left_buffer.size())?;
        left.truncate(pixels);
        lap("rectify", &mut timings);
        let (disparity, valid) =
            self.matcher
                .disparity_buffers(&params.coarse, &left_buffer, &right_buffer)?;
        let (disparity, valid) = self.matcher.download(&disparity, &valid)?;
        lap("coarse", &mut timings);
        let (samples, guesses) = select(params, &left, &disparity, &valid);
        let mut start = Vec::with_capacity(samples.len() * SAMPLE_FLOATS);
        for (s, g) in samples.iter().zip(&guesses) {
            start.extend_from_slice(&[s[0], s[1], g[0], g[1], 1.0, 0.0, 0.0, 1.0]);
        }
        lap("select", &mut timings);
        let base = RefineParams {
            width: w,
            height: h,
            window: params.window,
            iterations: params.iterations,
            terms: 6,
            max_move: 1.0,
            min_correlation: None,
            exit: params.exit_px,
        };
        let stages = [
            (3, params.max_move, Some(params.min_correlation)),
            (6, 1.0, None),
        ];
        let (refined, kept) = if samples.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.refiner
                .stages_buffers(&base, &left_buffer, &right_buffer, &start, &stages)?
        };
        lap("refine", &mut timings);
        let matches: Vec<([f32; 2], [f32; 2])> = samples
            .iter()
            .zip(&refined)
            .zip(&kept)
            .filter(|(_, k)| **k)
            .map(|((s, r), _)| (*s, *r))
            .collect();
        let mut out = depth_result(cameras, &matches, params.near_m, params.far_m);
        lap("triangulate", &mut timings);
        out.timings = timings;
        Ok(out)
    }

    /// Measure one raw stereo pair from block disparity alone at the preview's `coarse`
    /// settings, with no refinement.
    pub fn preview(
        &self,
        params: &DepthParams,
        coarse: &CoarseParams,
        cameras: (&RectifyParams, &RectifyParams),
        left_raw: &[u8],
        right_raw: &[u8],
    ) -> Result<DepthResult, String> {
        let (w, h) = (coarse.width, coarse.height);
        let pixels = w as usize * h as usize;
        if left_raw.len() != pixels || right_raw.len() != pixels {
            return Err("image sizes do not match the parameters".into());
        }
        let mut timings = Vec::new();
        let mut clock = Instant::now();
        let mut lap = |name: &'static str, timings: &mut Vec<(&'static str, f64)>| {
            timings.push((name, clock.elapsed().as_secs_f64() * 1000.0));
            clock = Instant::now();
        };
        let left_raw = self.gpu.upload("left raw", left_raw);
        let right_raw = self.gpu.upload("right raw", right_raw);
        let left_buffer = self.rectifier.rectify_buffer(cameras.0, w, h, &left_raw)?;
        let right_buffer = self.rectifier.rectify_buffer(cameras.1, w, h, &right_raw)?;
        let mut left = self.gpu.download(&left_buffer, left_buffer.size())?;
        left.truncate(pixels);
        lap("rectify", &mut timings);
        let (disparity, valid) =
            self.matcher
                .disparity_buffers(coarse, &left_buffer, &right_buffer)?;
        let (disparity, valid) = self.matcher.download(&disparity, &valid)?;
        lap("coarse", &mut timings);
        let matches = preview_matches(coarse, params.min_level, &left, &disparity, &valid);
        let mut out = depth_result(cameras, &matches, params.near_m, params.far_m);
        lap("triangulate", &mut timings);
        out.timings = timings;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_matches_stand_at_block_centres() {
        let coarse = CoarseParams {
            width: 16,
            height: 8,
            scale: 8,
            block: 7,
            lo: 0,
            count: 4,
            min_correlation: 0.6,
            margin: 0.05,
            fill: 2,
        };
        let mut left = vec![6u8; 16 * 8];
        left[15] = 5;
        let matches = preview_matches(&coarse, 6.0, &left, &[1.5, 2.0], &[true, true]);
        assert_eq!(matches, vec![([3.5, 3.5], [-8.5, 3.5])]);
    }
}
