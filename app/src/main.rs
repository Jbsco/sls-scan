use slscan_app::calibration;
use slscan_app::client::{self, Event};
use slscan_app::coarse;
use slscan_app::depth;
use slscan_app::gpu;
use slscan_app::protocol::SequenceRequest;
use slscan_app::rectify;
use slscan_app::refine;
use slscan_app::ui::ScanApp;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn headless_capture(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut wait = Duration::ZERO;
    let mut request = SequenceRequest {
        set: "scan".to_owned(),
        pitch: 32,
        steps: 4,
        frames: 1,
    };
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs: --node, --set, --pitch, --steps, --frames, --wait");
            return ExitCode::FAILURE;
        };
        let parsed = match key.as_str() {
            "--node" => {
                node.clone_from(value);
                Ok(())
            }
            "--set" => {
                request.set.clone_from(value);
                Ok(())
            }
            "--pitch" => value.parse().map(|v| request.pitch = v),
            "--steps" => value.parse().map(|v| request.steps = v),
            "--frames" => value.parse().map(|v| request.frames = v),
            "--wait" => match value.parse::<f64>() {
                Ok(seconds) if seconds >= 0.0 => {
                    wait = Duration::from_secs_f64(seconds);
                    Ok(())
                }
                _ => {
                    eprintln!("--wait takes seconds");
                    return ExitCode::FAILURE;
                }
            },
            _ => {
                eprintln!("unknown argument {key}");
                return ExitCode::FAILURE;
            }
        };
        if parsed.is_err() {
            eprintln!("{key} takes a number");
            return ExitCode::FAILURE;
        }
    }
    let result = client::capture(&node, &request, wait, |event| {
        if let Event::Connected(hello) = event {
            println!("connected to {} at {node}", hello.node);
        }
    });
    match result {
        Ok(stats) => {
            let interval = stats
                .mean_interval_ns
                .map_or_else(|| "none".to_owned(), |ns| format!("{:.3} ms", ns / 1.0e6));
            let missing = if stats.missing.is_empty() {
                "none".to_owned()
            } else {
                format!("{:?}", stats.missing)
            };
            println!(
                "{} frames, missing {missing}, in order {}, mean interval {interval}",
                stats.frames, stats.in_order
            );
            if stats.missing.is_empty() && stats.in_order {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("capture failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn headless_depth(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut wait = Duration::ZERO;
    let mut frames = 1u32;
    let mut params = PathBuf::from("data/depth.json");
    let mut out: Option<PathBuf> = None;
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs: --node, --params, --frames, --wait, --out");
            return ExitCode::FAILURE;
        };
        match key.as_str() {
            "--node" => node.clone_from(value),
            "--params" => params = PathBuf::from(value),
            "--out" => out = Some(PathBuf::from(value)),
            "--frames" => match value.parse() {
                Ok(v) => frames = v,
                Err(_) => {
                    eprintln!("--frames takes a number");
                    return ExitCode::FAILURE;
                }
            },
            "--wait" => match value.parse::<f64>() {
                Ok(seconds) if seconds >= 0.0 => wait = Duration::from_secs_f64(seconds),
                _ => {
                    eprintln!("--wait takes seconds");
                    return ExitCode::FAILURE;
                }
            },
            _ => {
                eprintln!("unknown argument {key}");
                return ExitCode::FAILURE;
            }
        }
    }
    let setup = match depth::DepthSetup::load(&params) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("depth parameters: {e}");
            return ExitCode::FAILURE;
        }
    };
    let frame = match depth::DepthFrame::new(false) {
        Ok(f) => f,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, depth frames skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("adapter {}", frame.adapter);
    let depth_params = setup.params();
    let request = SequenceRequest {
        set: "stereo".to_owned(),
        pitch: 32,
        steps: 4,
        frames,
    };
    let mut left: Option<Vec<u8>> = None;
    let mut results: Vec<(String, Result<depth::DepthResult, String>)> = Vec::new();
    let captured = client::capture(&node, &request, wait, |event| match event {
        Event::Connected(hello) => println!("connected to {} at {node}", hello.node),
        Event::Frame { name, pixels, .. } => {
            if name.ends_with("/left") {
                left = Some(pixels);
            } else if let Some(l) = left.take() {
                let pair = name.trim_end_matches("/right").to_owned();
                let cameras = (&setup.cameras.left, &setup.cameras.right);
                results.push((pair, frame.depth(&depth_params, cameras, &l, &pixels)));
            }
        }
        _ => {}
    });
    if let Err(e) = captured {
        eprintln!("capture failed: {e}");
        return ExitCode::FAILURE;
    }
    let mut ok = !results.is_empty();
    let mut last = None;
    for (pair, result) in &results {
        match result {
            Ok(d) => {
                let total: f64 = d.timings.iter().map(|(_, ms)| ms).sum();
                let stages: Vec<String> = d
                    .timings
                    .iter()
                    .map(|(name, ms)| format!("{name} {ms:.1}"))
                    .collect();
                println!(
                    "{pair}: {} points in {total:.1} ms ({} ms)",
                    d.points.len(),
                    stages.join(", ")
                );
                ok &= !d.points.is_empty();
                last = Some(d);
            }
            Err(e) => {
                println!("{pair}: depth failed: {e}");
                ok = false;
            }
        }
    }
    if let (Some(path), Some(d)) = (&out, last) {
        match depth::write_ply(path, &d.points) {
            Ok(()) => println!("{} points written to {}", d.points.len(), path.display()),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!("depth: {} pairs", results.len());
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn headless_calibrate(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut poses = 12u32;
    let mut out = PathBuf::from("data/calibration");
    let mut tool = "python3 -m slscan.cli".to_owned();
    let mut wait = Duration::ZERO;
    let mut board = calibration::Board::default();
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs");
            return ExitCode::FAILURE;
        };
        let parsed = match key.as_str() {
            "--node" => {
                node.clone_from(value);
                true
            }
            "--out" => {
                out = PathBuf::from(value);
                true
            }
            "--tool" => {
                tool.clone_from(value);
                true
            }
            "--poses" => value.parse().map(|v| poses = v).is_ok(),
            "--wait" => match value.parse::<f64>() {
                Ok(s) if s >= 0.0 => {
                    wait = Duration::from_secs_f64(s);
                    true
                }
                _ => false,
            },
            "--squares-x" => value.parse().map(|v| board.squares_x = v).is_ok(),
            "--squares-y" => value.parse().map(|v| board.squares_y = v).is_ok(),
            "--square-mm" => value.parse().map(|v| board.square_mm = v).is_ok(),
            "--marker-mm" => value.parse().map(|v| board.marker_mm = v).is_ok(),
            _ => false,
        };
        if !parsed {
            eprintln!("bad argument {key} {value}");
            return ExitCode::FAILURE;
        }
    }
    let Some(tool) = calibration::Tool::parse(&tool) else {
        eprintln!("--tool is empty");
        return ExitCode::FAILURE;
    };
    let mut projector = None;
    for _ in 0..poses {
        match calibration::capture_pose(&node, wait, &out) {
            Ok((dir, p)) => {
                let name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match calibration::detect(&tool, &dir, &board) {
                    Ok(d) => println!("{name}: {} of {} corners", d.corners, d.of),
                    Err(e) => println!("{name}: detection failed: {e}"),
                }
                projector = Some(p);
            }
            Err(e) => {
                eprintln!("pose capture failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(projector) = projector else {
        eprintln!("no poses captured");
        return ExitCode::FAILURE;
    };
    match calibration::solve(&tool, &out, &board, &projector) {
        Ok(s) => {
            println!(
                "{} views, camera {:.3} px (limit {}), projector {:.3} px (limit {}), baseline {:.2} mm, written to {}",
                s.views, s.camera_rms_px, s.camera_limit_px, s.projector_rms_px, s.projector_limit_px,
                1000.0 * s.baseline_m, s.out
            );
            if s.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("solve failed: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(serde::Deserialize)]
struct VectorMeta {
    width: u32,
    height: u32,
    planes: u32,
    steps: u32,
    projector_width: u32,
    pitch: f32,
    min_contrast: f32,
    min_modulation: f32,
    max_disagreement: f32,
}

fn decode_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan decode-check <vector directory> [--fallback]");
        return ExitCode::FAILURE;
    };
    let fallback = args.iter().any(|a| a == "--fallback");
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, gray, phase, columns, mask) = match (
        read("meta.json"),
        read("gray.bin"),
        read("phase.bin"),
        read("columns.bin"),
        read("mask.bin"),
    ) {
        (Ok(m), Ok(g), Ok(p), Ok(c), Ok(k)) => (m, g, p, c, k),
        _ => {
            eprintln!("decode vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: VectorMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let decoder = match gpu::Decoder::new(fallback) {
        Ok(d) => d,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, decode check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let params = gpu::DecodeParams {
        width: meta.width,
        height: meta.height,
        planes: meta.planes,
        steps: meta.steps,
        projector_width: meta.projector_width,
        pitch: meta.pitch,
        min_contrast: meta.min_contrast,
        min_modulation: meta.min_modulation,
        max_disagreement: meta.max_disagreement,
    };
    let mut timings = Vec::new();
    let mut result = None;
    for _ in 0..3 {
        let start = Instant::now();
        match decoder.decode(&params, &gray, &phase) {
            Ok(r) => result = Some(r),
            Err(e) => {
                eprintln!("decode failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let Some((decoded, decoded_mask)) = result else {
        return ExitCode::FAILURE;
    };
    let reference: Vec<f32> = columns
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference_mask: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
    let mismatched = decoded_mask
        .iter()
        .zip(&reference_mask)
        .filter(|(a, b)| a != b)
        .count();
    let max_difference = decoded
        .iter()
        .zip(&reference)
        .zip(decoded_mask.iter().zip(&reference_mask))
        .filter(|(_, (a, b))| **a && **b)
        .map(|((d, r), _)| (d - r).abs())
        .fold(0.0f32, f32::max);
    let pixels = reference_mask.len();
    let fraction = mismatched as f64 / pixels as f64;
    let fastest = timings.iter().copied().fold(f64::INFINITY, f64::min);
    println!("adapter {}", decoder.adapter);
    println!(
        "{} × {} pixels decoded in {fastest:.1} ms (fastest of three, with upload and readback)",
        meta.width, meta.height
    );
    println!(
        "mask disagreements {mismatched} of {pixels} ({:.4} %), largest column difference {max_difference:.5} px",
        100.0 * fraction
    );
    if fraction <= 1e-4 && max_difference <= 0.01 {
        println!("decode check: GPU matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("decode check: GPU differs from the NumPy reference");
        ExitCode::FAILURE
    }
}

#[derive(serde::Deserialize)]
struct StereoMeta {
    width: u32,
    height: u32,
    window: u32,
    iterations: u32,
    samples: usize,
    #[serde(default)]
    direct_samples: usize,
    #[serde(default)]
    max_move: f32,
    #[serde(default)]
    min_correlation: f32,
    #[serde(default)]
    coarse: Option<CoarseMeta>,
    #[serde(default)]
    depth: Option<DepthMeta>,
}

#[derive(serde::Deserialize)]
struct DepthMeta {
    near: f64,
    far: f64,
    step: u32,
    min_level: f32,
}

#[derive(serde::Deserialize)]
struct CoarseMeta {
    scale: u32,
    block: u32,
    lo: i32,
    count: u32,
    min_correlation: f32,
    margin: f32,
    fill: u32,
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Refine one vector set three times and compare the last result with its reference.
fn check_set(
    label: &str,
    refiner: &refine::Refiner,
    meta: &StereoMeta,
    images: (&[u8], &[u8]),
    samples: &[f32],
    reference: (&[f32], &[u8]),
    stages: &[refine::Stage],
) -> Result<bool, String> {
    let count = reference.1.len();
    if samples.len() != count * refine::SAMPLE_FLOATS || reference.0.len() != 2 * count {
        return Err(format!("{label} vectors do not match meta.json"));
    }
    let mut fastest = f64::INFINITY;
    let mut result = None;
    for _ in 0..3 {
        let start = Instant::now();
        let base = refine::RefineParams {
            width: meta.width,
            height: meta.height,
            window: meta.window,
            iterations: meta.iterations,
            terms: 6,
            max_move: 1.0,
            min_correlation: None,
        };
        result = Some(refiner.stages(&base, images.0, images.1, samples, stages)?);
        fastest = fastest.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    let Some((points, kept)) = result else {
        return Err("no refinement ran".into());
    };
    let reference_kept: Vec<bool> = reference.1.iter().map(|&k| k != 0).collect();
    let mismatched = kept
        .iter()
        .zip(&reference_kept)
        .filter(|(a, b)| a != b)
        .count();
    let max_difference = points
        .iter()
        .zip(reference.0.chunks_exact(2))
        .zip(kept.iter().zip(&reference_kept))
        .filter(|(_, (a, b))| **a && **b)
        .map(|((p, r), _)| (p[0] - r[0]).abs().max((p[1] - r[1]).abs()))
        .fold(0.0f32, f32::max);
    let fraction = mismatched as f64 / count.max(1) as f64;
    println!(
        "{label}: {count} samples refined with {} px windows in {fastest:.1} ms (fastest of three, with upload and readback)",
        meta.window
    );
    println!(
        "{label}: kept disagreements {mismatched} of {count} ({:.2} %), largest difference {max_difference:.5} px",
        100.0 * fraction
    );
    Ok(fraction <= 0.005 && max_difference <= 0.01)
}

fn refine_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan refine-check <vector directory> [--fallback]");
        return ExitCode::FAILURE;
    };
    let fallback = args.iter().any(|a| a == "--fallback");
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, left, right, samples, refined, kept) = match (
        read("meta.json"),
        read("left.bin"),
        read("right.bin"),
        read("samples.bin"),
        read("refined.bin"),
        read("kept.bin"),
    ) {
        (Ok(m), Ok(l), Ok(r), Ok(s), Ok(f), Ok(k)) => (m, l, r, s, f, k),
        _ => {
            eprintln!("stereo vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: StereoMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let refiner = match refine::Refiner::new(fallback) {
        Ok(r) => r,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, refine check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    if kept.len() != meta.samples {
        eprintln!("stereo vectors do not match meta.json");
        return ExitCode::FAILURE;
    }
    println!("adapter {}", refiner.adapter);
    let images = (left.as_slice(), right.as_slice());
    let affine = check_set(
        "affine start",
        &refiner,
        &meta,
        images,
        &floats(&samples),
        (&floats(&refined), &kept),
        &[(6, 1.0, None)],
    );
    let direct = if meta.direct_samples == 0 {
        Ok(true)
    } else {
        match (
            read("direct-samples.bin"),
            read("direct-refined.bin"),
            read("direct-kept.bin"),
        ) {
            (Ok(s), Ok(f), Ok(k)) => check_set(
                "direct",
                &refiner,
                &meta,
                images,
                &floats(&s),
                (&floats(&f), &k),
                &[
                    (3, meta.max_move, Some(meta.min_correlation)),
                    (6, 1.0, None),
                ],
            ),
            _ => Err(format!("direct vectors missing in {}", dir.display())),
        }
    };
    match (affine, direct) {
        (Ok(true), Ok(true)) => {
            println!("refine check: GPU matches the NumPy reference");
            ExitCode::SUCCESS
        }
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("refine failed: {e}");
            ExitCode::FAILURE
        }
        _ => {
            println!("refine check: GPU differs from the NumPy reference");
            ExitCode::FAILURE
        }
    }
}

#[derive(serde::Deserialize)]
struct Rectifications {
    left: rectify::RectifyParams,
    right: rectify::RectifyParams,
}

fn rectify_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan rectify-check <vector directory> [--fallback]");
        return ExitCode::FAILURE;
    };
    let fallback = args.iter().any(|a| a == "--fallback");
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, params, files) = match (
        read("meta.json"),
        read("rectify.json"),
        [
            read("left-raw.bin"),
            read("right-raw.bin"),
            read("left.bin"),
            read("right.bin"),
        ],
    ) {
        (Ok(m), Ok(p), [Ok(a), Ok(b), Ok(c), Ok(d)]) => (m, p, [a, b, c, d]),
        _ => {
            eprintln!("stereo vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let (meta, params): (StereoMeta, Rectifications) = match (
        serde_json::from_slice(&meta),
        serde_json::from_slice(&params),
    ) {
        (Ok(m), Ok(p)) => (m, p),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("vector metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let rectifier = match rectify::Rectifier::new(fallback) {
        Ok(r) => r,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, rectify check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let [left_raw, right_raw, left_ref, right_ref] = files;
    let mut fastest = f64::INFINITY;
    let mut outputs = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let pair = [(&params.left, &left_raw), (&params.right, &right_raw)]
            .map(|(p, raw)| rectifier.rectify(p, meta.width, meta.height, raw));
        fastest = fastest.min(start.elapsed().as_secs_f64() * 1000.0);
        outputs = Vec::new();
        for r in pair {
            match r {
                Ok(o) => outputs.push(o),
                Err(e) => {
                    eprintln!("rectify failed: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    let mut largest = 0u8;
    let mut differing = 0usize;
    for (out, reference) in outputs.iter().zip([&left_ref, &right_ref]) {
        for (a, b) in out.iter().zip(reference.iter()) {
            let d = a.abs_diff(*b);
            largest = largest.max(d);
            differing += usize::from(d > 0);
        }
    }
    let pixels = 2 * meta.width as usize * meta.height as usize;
    let fraction = differing as f64 / pixels as f64;
    println!("adapter {}", rectifier.adapter);
    println!(
        "two {} × {} images rectified in {fastest:.1} ms (fastest of three, with upload and readback)",
        meta.width, meta.height
    );
    println!(
        "pixels differing {differing} of {pixels} ({:.4} %), largest difference {largest}",
        100.0 * fraction
    );
    if largest <= 1 && fraction <= 1e-3 {
        println!("rectify check: GPU matches the OpenCV reference");
        ExitCode::SUCCESS
    } else {
        println!("rectify check: GPU differs from the OpenCV reference");
        ExitCode::FAILURE
    }
}

fn coarse_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan coarse-check <vector directory> [--fallback]");
        return ExitCode::FAILURE;
    };
    let fallback = args.iter().any(|a| a == "--fallback");
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, left, right, reference) = match (
        read("meta.json"),
        read("left.bin"),
        read("right.bin"),
        read("coarse.bin"),
    ) {
        (Ok(m), Ok(l), Ok(r), Ok(c)) => (m, l, r, floats(&c)),
        _ => {
            eprintln!("coarse vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: StereoMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(c) = meta.coarse else {
        eprintln!("meta.json has no coarse section");
        return ExitCode::FAILURE;
    };
    let matcher = match coarse::BlockMatcher::new(fallback) {
        Ok(m) => m,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, coarse check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let params = coarse::CoarseParams {
        width: meta.width,
        height: meta.height,
        scale: c.scale,
        block: c.block,
        lo: c.lo,
        count: c.count,
        min_correlation: c.min_correlation,
        margin: c.margin,
        fill: c.fill,
    };
    let mut fastest = f64::INFINITY;
    let mut result = None;
    for _ in 0..3 {
        let start = Instant::now();
        match matcher.disparity(&params, &left, &right) {
            Ok(r) => result = Some(r),
            Err(e) => {
                eprintln!("coarse failed: {e}");
                return ExitCode::FAILURE;
            }
        }
        fastest = fastest.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    let Some((disparity, valid)) = result else {
        return ExitCode::FAILURE;
    };
    if disparity.len() != reference.len() {
        eprintln!("coarse vectors do not match meta.json");
        return ExitCode::FAILURE;
    }
    let mut mismatched = 0usize;
    let mut both = 0usize;
    let mut differing = 0usize;
    let mut largest = 0.0f32;
    for ((d, v), r) in disparity.iter().zip(&valid).zip(&reference) {
        if *v != r.is_finite() {
            mismatched += 1;
        } else if *v {
            both += 1;
            let difference = (d - r).abs();
            largest = largest.max(difference);
            differing += usize::from(difference > 0.01);
        }
    }
    let pixels = reference.len();
    let fraction = mismatched as f64 / pixels as f64;
    let differing_fraction = differing as f64 / both.max(1) as f64;
    println!("adapter {}", matcher.adapter);
    println!(
        "{} × {} reduced pixels over {} disparities matched in {fastest:.1} ms (fastest of three, with upload and readback)",
        meta.width / c.scale,
        meta.height / c.scale,
        c.count
    );
    println!(
        "validity disagreements {mismatched} of {pixels} ({:.3} %), over 0.01 apart {differing} of {both} ({:.3} %), largest difference {largest:.4}",
        100.0 * fraction,
        100.0 * differing_fraction
    );
    if fraction <= 0.005 && differing_fraction <= 0.005 {
        println!("coarse check: GPU matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("coarse check: GPU differs from the NumPy reference");
        ExitCode::FAILURE
    }
}

fn depth_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan depth-check <vector directory> [--fallback]");
        return ExitCode::FAILURE;
    };
    let fallback = args.iter().any(|a| a == "--fallback");
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, cameras, left, right, reference) = match (
        read("meta.json"),
        read("rectify.json"),
        read("left-raw.bin"),
        read("right-raw.bin"),
        read("depth.bin"),
    ) {
        (Ok(m), Ok(c), Ok(l), Ok(r), Ok(d)) => (m, c, l, r, d),
        _ => {
            eprintln!("depth vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let (meta, cameras): (StereoMeta, Rectifications) = match (
        serde_json::from_slice(&meta),
        serde_json::from_slice(&cameras),
    ) {
        (Ok(m), Ok(c)) => (m, c),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("vector metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (Some(c), Some(d)) = (&meta.coarse, &meta.depth) else {
        eprintln!("meta.json has no coarse or depth section");
        return ExitCode::FAILURE;
    };
    let params = depth::DepthParams {
        coarse: coarse::CoarseParams {
            width: meta.width,
            height: meta.height,
            scale: c.scale,
            block: c.block,
            lo: c.lo,
            count: c.count,
            min_correlation: c.min_correlation,
            margin: c.margin,
            fill: c.fill,
        },
        step: d.step,
        window: meta.window,
        iterations: meta.iterations,
        min_level: d.min_level,
        max_move: meta.max_move,
        min_correlation: meta.min_correlation,
        near_m: d.near,
        far_m: d.far,
    };
    let frame = match depth::DepthFrame::new(fallback) {
        Ok(f) => f,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, depth check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut best: Option<(f64, depth::DepthResult)> = None;
    for _ in 0..3 {
        let start = Instant::now();
        let result = match frame.depth(&params, (&cameras.left, &cameras.right), &left, &right) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("depth failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let total = start.elapsed().as_secs_f64() * 1000.0;
        if best.as_ref().is_none_or(|(t, _)| total < *t) {
            best = Some((total, result));
        }
    }
    let Some((total, result)) = best else {
        return ExitCode::FAILURE;
    };
    let mut expected = std::collections::HashMap::new();
    for row in reference.chunks_exact(40) {
        let v: Vec<f64> = row
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        expected.insert((v[0] as i64, v[1] as i64), [v[2], v[3], v[4]]);
    }
    let mut extra = 0usize;
    let mut distances = Vec::new();
    for (s, p) in result.samples.iter().zip(&result.points) {
        match expected.get(&(s[0] as i64, s[1] as i64)) {
            Some(r) => {
                let squares = (0..3).map(|i| (p[i] - r[i]).powi(2)).sum::<f64>();
                distances.push(squares.sqrt());
            }
            None => extra += 1,
        }
    }
    let missing = expected.len() - distances.len();
    distances.sort_by(f64::total_cmp);
    let median = distances.get(distances.len() / 2).copied().unwrap_or(0.0);
    let p99 = distances
        .get(distances.len() * 99 / 100)
        .copied()
        .unwrap_or(0.0);
    let largest = distances.last().copied().unwrap_or(0.0);
    let stages: Vec<String> = result
        .timings
        .iter()
        .map(|(name, ms)| format!("{name} {ms:.1}"))
        .collect();
    println!("adapter {}", frame.adapter);
    println!(
        "{} points from a {} × {} pair in {total:.1} ms (fastest of three, with upload and readback): {} ms",
        result.points.len(),
        meta.width,
        meta.height,
        stages.join(", ")
    );
    println!(
        "against {} reference points: {missing} missing, {extra} extra, distance median {:.3} µm, 99th percentile {:.3} µm, largest {:.3} µm",
        expected.len(),
        median * 1e6,
        p99 * 1e6,
        largest * 1e6
    );
    let differing = (missing + extra) as f64 / expected.len().max(1) as f64;
    if differing <= 0.005 && p99 <= 1e-6 && largest <= 5e-6 {
        println!("depth check: GPU matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("depth check: GPU differs from the NumPy reference");
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("capture") => return headless_capture(&args[2..]),
        Some("calibrate") => return headless_calibrate(&args[2..]),
        Some("depth") => return headless_depth(&args[2..]),
        Some("decode-check") => return decode_check(&args[2..]),
        Some("refine-check") => return refine_check(&args[2..]),
        Some("rectify-check") => return rectify_check(&args[2..]),
        Some("coarse-check") => return coarse_check(&args[2..]),
        Some("depth-check") => return depth_check(&args[2..]),
        _ => {}
    }
    let options = eframe::NativeOptions::default();
    let result = eframe::run_native(
        "sls-scan",
        options,
        Box::new(|_cc| Ok(Box::new(ScanApp::default()))),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("application failed: {e}");
            ExitCode::FAILURE
        }
    }
}
