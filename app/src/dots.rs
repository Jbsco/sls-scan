//! Reference dots seen with the pattern light off, following `dot_centres`,
//! `rectified_valid`, `surface_disparity` and `find_dots` in `slscan/stereo.py`.

use crate::depth;
use crate::rectify::RectifyParams;

/// Window of the local background mean, in pixels.
const BACKGROUND: usize = 61;
/// A pixel is dark when it lies below its background by this share of it.
const CONTRAST: f64 = 0.4;
const MIN_AREA: usize = 6;
const MAX_AREA: usize = 20_000;
/// Pixels by which the rectified image's valid area shrinks.
const MARGIN: usize = 3;
const SURFACE_RADIUS_PX: f64 = 60.0;
const SURFACE_NEIGHBOURS: usize = 12;
const ROW_TOLERANCE_PX: f64 = 1.5;
const DEPTH_TOLERANCE_M: f64 = 0.01;

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

/// Mean over a `size` × `size` window with OpenCV's reflect-101 border, rounded to 8 bits,
/// as `cv2.blur` gives it.
fn box_mean(image: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    let half = (size / 2) as isize;
    let mut rows = vec![0u32; width * height];
    for y in 0..height {
        let line = &image[y * width..(y + 1) * width];
        let at = |x: isize| u32::from(line[reflect101(x, width as isize)]);
        let mut sum: u32 = (-half..=half).map(at).sum();
        for x in 0..width {
            rows[y * width + x] = sum;
            let x = x as isize;
            sum = sum + at(x + half + 1) - at(x - half);
        }
    }
    let count = (size * size) as u32;
    let mut out = vec![0u8; width * height];
    for x in 0..width {
        let at = |y: isize| rows[reflect101(y, height as isize) * width + x];
        let mut sum: u32 = (-half..=half).map(at).sum();
        for y in 0..height {
            out[y * width + x] = ((sum + count / 2) / count) as u8;
            let y = y as isize;
            sum = sum + at(y + half + 1) - at(y - half);
        }
    }
    out
}

/// Rectified pixels whose map lands inside the raw image at its nearest pixel, shrunk by
/// `margin` pixels, as `cv2.remap` with nearest sampling and `cv2.erode` give them.
pub fn rectified_valid(
    params: &RectifyParams,
    width: usize,
    height: usize,
    margin: usize,
) -> Vec<bool> {
    let ir = &params.inverse;
    let [k1, k2, p1, p2, k3] = params.distortion;
    let mut inside = vec![false; width * height];
    for i in 0..height {
        for j in 0..width {
            let (jf, i_f) = (j as f32, i as f32);
            let hx = jf * ir[0] + i_f * ir[1] + ir[2];
            let hy = jf * ir[3] + i_f * ir[4] + ir[5];
            let hw = jf * ir[6] + i_f * ir[7] + ir[8];
            let (x, y) = (hx / hw, hy / hw);
            let r2 = x * x + y * y;
            let radial = 1.0 + ((k3 * r2 + k2) * r2 + k1) * r2;
            let u = params.focal[0] * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
                + params.centre[0];
            let v = params.focal[1] * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
                + params.centre[1];
            let (u, v) = (u.round(), v.round());
            inside[i * width + j] =
                u >= 0.0 && v >= 0.0 && u <= (width - 1) as f32 && v <= (height - 1) as f32;
        }
    }
    let shrink = |mask: &[bool], horizontal: bool| -> Vec<bool> {
        let mut out = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let (lo, hi, along) = if horizontal {
                    (x.saturating_sub(margin), (x + margin).min(width - 1), y)
                } else {
                    (y.saturating_sub(margin), (y + margin).min(height - 1), x)
                };
                out[y * width + x] = (lo..=hi).all(|k| {
                    if horizontal {
                        mask[along * width + k]
                    } else {
                        mask[k * width + along]
                    }
                });
            }
        }
        out
    };
    shrink(&shrink(&inside, true), false)
}

/// Centres (x, y) of dark dots in an 8-bit image under room light: 8-connected regions
/// darker than their local background by `CONTRAST` of it, centred by their darkness. A
/// region touching the image border, or a pixel outside `valid` when given, is dropped,
/// since part of its dot may be missing.
pub fn dot_centres(
    image: &[u8],
    width: usize,
    height: usize,
    valid: Option<&[bool]>,
) -> Vec<[f64; 2]> {
    let background = box_mean(image, width, height, BACKGROUND);
    let depth: Vec<f64> = image
        .iter()
        .zip(&background)
        .map(|(&p, &b)| f64::from(b) - f64::from(p))
        .collect();
    let dark: Vec<bool> = depth
        .iter()
        .zip(&background)
        .map(|(&d, &b)| d > CONTRAST * f64::from(b))
        .collect();
    let mut seen = vec![false; width * height];
    let mut centres = Vec::new();
    let mut stack = Vec::new();
    for start in 0..width * height {
        if !dark[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut area, mut cut) = (0usize, false);
        let (mut sx, mut sy, mut total) = (0.0f64, 0.0f64, 0.0f64);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % width, i / width);
            area += 1;
            cut |= x == 0 || y == 0 || x == width - 1 || y == height - 1;
            cut |= valid.is_some_and(|v| !v[i]);
            sx += depth[i] * x as f64;
            sy += depth[i] * y as f64;
            total += depth[i];
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let n = ny as usize * width + nx as usize;
                    if dark[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        if !cut && (MIN_AREA..=MAX_AREA).contains(&area) {
            centres.push([sx / total, sy / total]);
        }
    }
    centres
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Disparity and depth that a surface (points in the left camera's frame) gives at
/// rectified left pixels: medians over its nearest `SURFACE_NEIGHBOURS` points that
/// project within `SURFACE_RADIUS_PX`, or none.
fn surface_disparity(
    left: &RectifyParams,
    right: &RectifyParams,
    surface: &[[f64; 3]],
    pixels: &[[f64; 2]],
) -> Vec<Option<(f64, f64)>> {
    let (pl, pr, r) = (&left.projection, &right.projection, &left.rotation);
    let projected: Vec<([f64; 2], f64, f64)> = surface
        .iter()
        .map(|p| {
            let x = [0, 1, 2].map(|i| (0..3).map(|j| r[i * 3 + j] * p[j]).sum::<f64>());
            let z = x[2];
            let uv = [pl[0] * x[0] / z + pl[2], pl[5] * x[1] / z + pl[6]];
            (uv, pl[2] - pr[2] - pr[3] / z, z)
        })
        .collect();
    pixels
        .iter()
        .map(|q| {
            let mut near: Vec<(f64, f64, f64)> = projected
                .iter()
                .filter_map(|(uv, d, z)| {
                    let distance = ((uv[0] - q[0]).powi(2) + (uv[1] - q[1]).powi(2)).sqrt();
                    (distance <= SURFACE_RADIUS_PX).then_some((distance, *d, *z))
                })
                .collect();
            if near.is_empty() {
                return None;
            }
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            near.truncate(SURFACE_NEIGHBOURS);
            let mut d: Vec<f64> = near.iter().map(|n| n.1).collect();
            let mut z: Vec<f64> = near.iter().map(|n| n.2).collect();
            Some((median(&mut d), median(&mut z)))
        })
        .collect()
}

/// Positions in the left camera's frame of dark dots seen in a rectified pair taken under
/// room light, next to a pattern-lit pair whose points are `surface`. A left dot pairs
/// with its one right dot on the row whose disparity lies within `DEPTH_TOLERANCE_M` of
/// the surface there. Dots with no partner or several are dropped.
#[allow(clippy::too_many_arguments)]
pub fn find_dots(
    left: &RectifyParams,
    right: &RectifyParams,
    rectified: (&[u8], &[u8]),
    width: usize,
    height: usize,
    surface: &[[f64; 3]],
    near_m: f64,
    far_m: f64,
) -> Vec<[f64; 3]> {
    let valid_left = rectified_valid(left, width, height, MARGIN);
    let valid_right = rectified_valid(right, width, height, MARGIN);
    let cl = dot_centres(rectified.0, width, height, Some(&valid_left));
    let cr = dot_centres(rectified.1, width, height, Some(&valid_right));
    let expected = surface_disparity(left, right, surface, &cl);
    let mut matches = Vec::new();
    for (p, e) in cl.iter().zip(&expected) {
        let Some((disparity, depth)) = *e else {
            continue;
        };
        let tolerance = right.projection[3].abs() / (depth * depth) * DEPTH_TOLERANCE_M;
        let partners: Vec<&[f64; 2]> = cr
            .iter()
            .filter(|q| {
                (q[1] - p[1]).abs() <= ROW_TOLERANCE_PX
                    && (p[0] - q[0] - disparity).abs() <= tolerance
            })
            .collect();
        if let [q] = partners[..] {
            matches.push(([p[0] as f32, p[1] as f32], [q[0] as f32, q[1] as f32]));
        }
    }
    depth::triangulate(left, right, &matches, near_m, far_m)
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mean_matches_a_direct_sum() {
        let (w, h) = (9, 7);
        let image: Vec<u8> = (0..w * h).map(|i| ((i * 37) % 251) as u8).collect();
        let out = box_mean(&image, w, h, 5);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0u32;
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let xx = reflect101(x as isize + dx, w as isize);
                        let yy = reflect101(y as isize + dy, h as isize);
                        sum += u32::from(image[yy * w + xx]);
                    }
                }
                assert_eq!(u32::from(out[y * w + x]), (sum + 12) / 25);
            }
        }
    }

    #[test]
    fn dots_cut_by_an_edge_are_dropped() {
        let (w, h) = (300usize, 200usize);
        let mut image = vec![120u8; w * h];
        for (cx, cy) in [(150.0f64, 100.0f64), (4.0, 60.0), (250.0, 150.0)] {
            for y in 0..h {
                for x in 0..w {
                    if (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) <= 100.0 {
                        image[y * w + x] = 10;
                    }
                }
            }
        }
        let valid: Vec<bool> = (0..w * h).map(|i| i % w < 240).collect();
        let centres = dot_centres(&image, w, h, Some(&valid));
        assert_eq!(centres.len(), 1);
        assert!((centres[0][0] - 150.0).abs() < 0.01 && (centres[0][1] - 100.0).abs() < 0.01);
    }
}
