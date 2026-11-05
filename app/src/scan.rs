//! Phase-shift scans: each camera ray meets the plane of its decoded projector column,
//! following `slscan/triangulate.py`.

use crate::fuse::chunked;
use crate::gpu::{DecodeParams, Decoder};
use serde::Deserialize;
use std::collections::HashMap;

/// A pinhole device as `slscan scan-params` writes it: size, row-major matrix and the
/// distortion coefficients k1, k2, p1, p2, k3.
#[derive(Debug, Clone, Deserialize)]
pub struct Device {
    pub width: u32,
    pub height: u32,
    pub matrix: [f64; 9],
    pub distortion: [f64; 5],
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecodeSettings {
    pub planes: u32,
    pub steps: u32,
    pub projector_width: u32,
    pub pitch: f32,
    pub min_contrast: f32,
    pub min_modulation: f32,
    pub max_disagreement: f32,
    /// Exposures of a scan set relative to the node's nominal one, the nominal alone when
    /// empty.
    #[serde(default)]
    pub exposures: Vec<f64>,
    /// Whether a scan set ends with the projector's black, red, green and blue fields.
    #[serde(default)]
    pub color: bool,
    /// A white card's response per channel, for point colors against white, or empty for
    /// colors in grey levels.
    #[serde(default)]
    pub white: Vec<f64>,
}

/// Camera and projector, the camera-to-projector rotation (row-major) and translation,
/// and the decode settings of one scan.
#[derive(Debug, Clone, Deserialize)]
pub struct ScanParams {
    pub camera: Device,
    pub projector: Device,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
    pub decode: DecodeSettings,
}

impl ScanParams {
    pub fn decode_params(&self) -> DecodeParams {
        DecodeParams {
            width: self.camera.width,
            height: self.camera.height,
            planes: self.decode.planes,
            steps: self.decode.steps,
            projector_width: self.decode.projector_width,
            pitch: self.decode.pitch,
            min_contrast: self.decode.min_contrast,
            min_modulation: self.decode.min_modulation,
            max_disagreement: self.decode.max_disagreement,
        }
    }
}

impl Device {
    /// Undistorted normalized coordinates of a pixel by the five fixed iterations of
    /// `cv2.undistortPoints`.
    pub fn undistort(&self, u: f64, v: f64) -> (f64, f64) {
        let m = &self.matrix;
        let [k1, k2, p1, p2, k3] = self.distortion;
        let (x0, y0) = ((u - m[2]) / m[0], (v - m[5]) / m[4]);
        let (mut x, mut y) = (x0, y0);
        for _ in 0..5 {
            let r2 = x * x + y * y;
            let icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
            if icdist < 0.0 {
                return (x0, y0);
            }
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (x0 - dx) * icdist;
            y = (y0 - dy) * icdist;
        }
        (x, y)
    }

    /// Pixel of a point in the device's frame, with distortion, as `cv2.projectPoints`.
    pub fn project(&self, p: [f64; 3]) -> (f64, f64) {
        let m = &self.matrix;
        let [k1, k2, p1, p2, k3] = self.distortion;
        let (x, y) = (p[0] / p[2], p[1] / p[2]);
        let r2 = x * x + y * y;
        let radial = 1.0 + ((k3 * r2 + k2) * r2 + k1) * r2;
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        (m[0] * xd + m[2], m[4] * yd + m[5])
    }
}

/// The Gray code and phase stacks of a scan set's frames, keyed by capture name, in the
/// plane-major layout the decode kernel takes, one pair per exposure, or none when a
/// frame is missing.
pub fn stacks(
    frames: &HashMap<String, Vec<u8>>,
    decode: &DecodeSettings,
) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
    let levels = decode.exposures.len().max(1);
    (0..levels)
        .map(|e| {
            let prefix = if levels > 1 {
                format!("exposure_{e:03}/")
            } else {
                String::new()
            };
            let stack = |name: &str, n: u32| -> Option<Vec<u8>> {
                let mut bytes = Vec::new();
                for i in 0..n {
                    bytes.extend_from_slice(frames.get(&format!("{prefix}{name}_{i:03}"))?);
                }
                Some(bytes)
            };
            Some((
                stack("gray", 2 * decode.planes)?,
                stack("phase", decode.steps)?,
            ))
        })
        .collect()
}

/// The black, red, green and blue field images of a scan set's frames, or none when one
/// is missing.
pub fn color_fields(frames: &HashMap<String, Vec<u8>>) -> Option<[Vec<u8>; 4]> {
    let field = |name: &str| frames.get(&format!("color/{name}")).cloned();
    Some([
        field("black")?,
        field("red")?,
        field("green")?,
        field("blue")?,
    ])
}

/// Red, green and blue bytes for each pixel, as `colors.point_colors`: each field less
/// the black one, over the white response per channel times 255 when `white` lists all
/// three, rounded half to even and clipped.
pub fn point_colors(
    fields: &[Vec<u8>; 4],
    pixels: &[[u32; 2]],
    width: u32,
    white: &[f64],
) -> Vec<[u8; 3]> {
    let white: Option<&[f64; 3]> = white.try_into().ok();
    pixels
        .iter()
        .map(|p| {
            let i = p[1] as usize * width as usize + p[0] as usize;
            let black = f64::from(fields[0][i]);
            let mut out = [0u8; 3];
            for (k, byte) in out.iter_mut().enumerate() {
                let mut v = f64::from(fields[k + 1][i]) - black;
                if let Some(w) = white {
                    v = 255.0 * v / w[k];
                }
                *byte = v.round_ties_even().clamp(0.0, 255.0) as u8;
            }
            out
        })
        .collect()
}

/// Amplitude of the sinusoid in pixel `p`'s phase samples, as `patterns.phase_modulation`.
fn modulation(phase: &[u8], n: usize, p: usize, steps: usize) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..steps {
        let v = f64::from(phase[i * n + p]);
        let angle = std::f64::consts::TAU * i as f64 / steps as f64;
        num += v * angle.sin();
        den += v * angle.cos();
    }
    2.0 / steps as f64 * num.hypot(den)
}

/// Decode a scan set's stacks on the GPU, one pair per exposure. At more than one
/// exposure each pixel takes the exposure with the largest phase modulation among those
/// where it decoded, as `decode.decode_exposures`.
pub fn decode_scan(
    decoder: &Decoder,
    params: &ScanParams,
    stacks: &[(Vec<u8>, Vec<u8>)],
) -> Result<(Vec<f32>, Vec<bool>), String> {
    let n = params.camera.width as usize * params.camera.height as usize;
    let steps = params.decode.steps as usize;
    if let [(gray, phase)] = stacks {
        return decoder.decode(&params.decode_params(), gray, phase);
    }
    let mut columns = vec![0.0f32; n];
    let mut mask = vec![false; n];
    let mut best = vec![f64::NEG_INFINITY; n];
    for (gray, phase) in stacks {
        let (decoded, kept) = decoder.decode(&params.decode_params(), gray, phase)?;
        for p in 0..n {
            if kept[p] {
                let m = modulation(phase, n, p, steps);
                if m > best[p] {
                    best[p] = m;
                    columns[p] = decoded[p];
                    mask[p] = true;
                }
            }
        }
    }
    Ok((columns, mask))
}

fn point_of(params: &ScanParams, x: usize, y: usize, column: f64) -> Option<[f64; 3]> {
    let r = &params.rotation;
    let t = &params.translation;
    let (xn, yn) = params.camera.undistort(x as f64, y as f64);
    let ray = [xn, yn, 1.0];
    let rotated = [
        r[0] * ray[0] + r[1] * ray[1] + r[2],
        r[3] * ray[0] + r[4] * ray[1] + r[5],
        r[6] * ray[0] + r[7] * ray[1] + r[8],
    ];
    let mut v = params.projector.matrix[5];
    let mut point = [0.0; 3];
    for _ in 0..3 {
        let (px, _) = params.projector.undistort(column, v);
        let scale = -(t[0] - px * t[2]) / (rotated[0] - px * rotated[2]);
        point = [ray[0] * scale, ray[1] * scale, scale];
        let q = [
            r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0],
            r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1],
            r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2],
        ];
        v = params.projector.project(q).1;
    }
    (point[2] > 0.0).then_some(point)
}

/// Points in metres in the camera frame and their pixels as x, y, in row-major pixel
/// order, for every masked pixel whose ray meets its column's plane in front of the camera.
/// Each plane comes from the undistorted projector point at the row where the previous
/// estimate lands, over three iterations.
pub fn triangulate(
    columns: &[f32],
    mask: &[bool],
    params: &ScanParams,
) -> (Vec<[f64; 3]>, Vec<[u32; 2]>) {
    let width = params.camera.width as usize;
    let parts = chunked(mask.len(), |range| {
        let mut points = Vec::new();
        let mut pixels = Vec::new();
        for i in range {
            if !mask[i] {
                continue;
            }
            let (x, y) = (i % width, i / width);
            if let Some(p) = point_of(params, x, y, f64::from(columns[i])) {
                points.push(p);
                pixels.push([x as u32, y as u32]);
            }
        }
        (points, pixels)
    });
    let mut points = Vec::new();
    let mut pixels = Vec::new();
    for (p, q) in parts {
        points.extend(p);
        pixels.extend(q);
    }
    (points, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(f: f64, distortion: [f64; 5]) -> Device {
        Device {
            width: 640,
            height: 480,
            matrix: [f, 0.0, 320.0, 0.0, f, 240.0, 0.0, 0.0, 1.0],
            distortion,
        }
    }

    #[test]
    fn undistort_inverts_projection() {
        let d = device(800.0, [-0.08, 0.05, 0.0004, -0.0003, 0.0]);
        for &(x, y) in &[(0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)] {
            let (u, v) = d.project([x, y, 1.0]);
            let (xu, yu) = d.undistort(u, v);
            assert!((xu - x).abs() < 1e-6 && (yu - y).abs() < 1e-6, "{xu} {yu}");
        }
    }

    #[test]
    fn point_colors_follow_the_fields_and_white() {
        let fields = [
            vec![25u8, 25],
            vec![125u8, 225],
            vec![75u8, 25],
            vec![225u8, 255],
        ];
        let pixels = [[0, 0], [1, 0]];
        assert_eq!(
            point_colors(&fields, &pixels, 2, &[]),
            [[100, 50, 200], [200, 0, 230]]
        );
        assert_eq!(
            point_colors(&fields, &pixels, 2, &[200.0, 200.0, 200.0]),
            [[128, 64, 255], [255, 0, 255]]
        );
    }

    #[test]
    fn rays_meet_their_column_planes_on_a_plate() {
        let params = ScanParams {
            camera: device(800.0, [0.0; 5]),
            projector: device(700.0, [0.0; 5]),
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [-0.3, 0.0, 0.0],
            decode: DecodeSettings {
                planes: 10,
                steps: 4,
                projector_width: 640,
                pitch: 16.0,
                min_contrast: 10.0,
                min_modulation: 10.0,
                max_disagreement: 1.0,
                exposures: Vec::new(),
                color: false,
                white: Vec::new(),
            },
        };
        let (w, h) = (640usize, 480usize);
        let mut columns = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = [(x as f64 - 320.0) / 800.0, (y as f64 - 240.0) / 800.0, 1.0];
                columns[y * w + x] = (700.0 * (p[0] - 0.3) + 320.0) as f32;
            }
        }
        let mask = vec![true; w * h];
        let (points, pixels) = triangulate(&columns, &mask, &params);
        assert_eq!(points.len(), w * h);
        assert_eq!(pixels[w + 2], [2, 1]);
        for p in &points {
            assert!((p[2] - 1.0).abs() < 1e-5, "{p:?}");
        }
    }
}
