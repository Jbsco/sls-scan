use slscan_app::calibration;
use slscan_app::client::{self, Event};
use slscan_app::coarse;
use slscan_app::depth;
use slscan_app::dots;
use slscan_app::fuse;
use slscan_app::gpu;
use slscan_app::protocol::SequenceRequest;
use slscan_app::rectify;
use slscan_app::refine;
use slscan_app::register;
use slscan_app::tsdf;
use slscan_app::ui::ScanApp;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn headless_capture(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut wait = Duration::ZERO;
    let mut out: Option<PathBuf> = None;
    let mut request = SequenceRequest {
        set: "scan".to_owned(),
        pitch: 32,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
    };
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!(
                "arguments come in pairs: --node, --set, --pitch, --steps, --frames, --rate, --dark-every, --wait, --out"
            );
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
            "--rate" => value.parse().map(|v| request.rate_hz = v),
            "--dark-every" => value.parse().map(|v| request.dark_every = v),
            "--out" => {
                out = Some(PathBuf::from(value));
                Ok(())
            }
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
    let mut written = Ok(());
    let result = client::capture(&node, &request, wait, |event| match event {
        Event::Connected(hello) => println!("connected to {} at {node}", hello.node),
        Event::Frame {
            header,
            name,
            pixels,
        } => {
            if let (Some(dir), Ok(())) = (&out, &written) {
                let path = dir.join(format!("{name}.png"));
                written = path
                    .parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|()| {
                        calibration::write_png(
                            &path,
                            header.width.into(),
                            header.height.into(),
                            &pixels,
                        )
                    });
            }
        }
        _ => {}
    });
    if let Err(e) = written {
        eprintln!("recording failed: {e}");
        return ExitCode::FAILURE;
    }
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
            let gap = stats
                .largest_gap_ns
                .map_or_else(|| "none".to_owned(), |ns| format!("{:.3} ms", ns / 1.0e6));
            println!(
                "{} frames, missing {missing}, in order {}, pairs in step {}, mean interval {interval}, largest gap {gap}",
                stats.frames, stats.in_order, stats.in_step
            );
            if stats.missing.is_empty() && stats.in_order && stats.in_step {
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

/// Voxel size of the fused model, as `track.fuse` uses by default.
const FUSION_VOXEL_M: f64 = 0.0005;

fn headless_depth(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut wait = Duration::ZERO;
    let mut frames = 1u32;
    let mut params = PathBuf::from("data/depth.json");
    let mut out: Option<PathBuf> = None;
    let mut mode = "auto".to_owned();
    let mut surface_out: Option<PathBuf> = None;
    let mut volume_out: Option<PathBuf> = None;
    let mut mesh_out: Option<PathBuf> = None;
    let mut faces = 0usize;
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!(
                "arguments come in pairs: --node, --params, --frames, --wait, --out, --mode, --surface, --volume, --mesh, --faces"
            );
            return ExitCode::FAILURE;
        };
        match key.as_str() {
            "--node" => node.clone_from(value),
            "--params" => params = PathBuf::from(value),
            "--out" => out = Some(PathBuf::from(value)),
            "--mode" => mode.clone_from(value),
            "--surface" => surface_out = Some(PathBuf::from(value)),
            "--volume" => volume_out = Some(PathBuf::from(value)),
            "--mesh" => mesh_out = Some(PathBuf::from(value)),
            "--faces" => match value.parse() {
                Ok(v) => faces = v,
                Err(_) => {
                    eprintln!("--faces takes a number");
                    return ExitCode::FAILURE;
                }
            },
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
    let preview = match mode.as_str() {
        "depth" => None,
        "auto" => setup.preview_params().filter(|_| frame.software()),
        "preview" => match setup.preview_params() {
            Some(p) => Some(p),
            None => {
                eprintln!("{} has no preview section", params.display());
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("--mode takes auto, depth or preview");
            return ExitCode::FAILURE;
        }
    };
    let kind = if preview.is_some() {
        "previews"
    } else {
        "depth frames"
    };
    println!("{kind} on adapter {}", frame.adapter);
    let depth_params = setup.params();
    let request = SequenceRequest {
        set: "stereo".to_owned(),
        pitch: 32,
        steps: 4,
        frames,
        rate_hz: 0,
        dark_every: 0,
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
                let result = frame.measure(&depth_params, preview.as_ref(), cameras, &l, &pixels);
                results.push((pair, result));
            }
        }
        _ => {}
    });
    if let Err(e) = captured {
        eprintln!("capture failed: {e}");
        return ExitCode::FAILURE;
    }
    let mut ok = !results.is_empty();
    let mut fusion = fuse::Fusion::new(FUSION_VOXEL_M);
    let mut last: &[[f64; 3]] = &[];
    let mut volume = tsdf::Live::coarse();
    let mut surface = Vec::new();
    let camera = tsdf::LeftCamera {
        rotation: setup.cameras.left.rotation,
        projection: setup.cameras.left.projection,
    };
    for (pair, result) in &results {
        match result {
            Ok(d) => {
                let total: f64 = d.timings.iter().map(|(_, ms)| ms).sum();
                let stages: Vec<String> = d
                    .timings
                    .iter()
                    .map(|(name, ms)| format!("{name} {ms:.1}"))
                    .collect();
                let fused = if preview.is_some() {
                    String::new()
                } else {
                    let points = fuse::drop_isolated(
                        &d.points,
                        f64::from(depth_params.step),
                        camera.projection[0],
                        fuse::ISOLATED_SPACINGS,
                    );
                    let dropped = d.points.len() - points.len();
                    let start = Instant::now();
                    fusion.add(&points);
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let rms = fusion
                        .last_rms
                        .map_or_else(String::new, |r| format!(", ICP residual {:.1} µm", r * 1e6));
                    let start = Instant::now();
                    let grid = tsdf::depth_grid(
                        &points,
                        &camera,
                        depth_params.step,
                        depth_params.window,
                        (setup.width, setup.height),
                    );
                    let tsdf = match fusion.poses.last() {
                        Some(pose) => {
                            let rotation: [f64; 9] =
                                std::array::from_fn(|k| pose.rotation[k / 3][k % 3]);
                            volume.add(
                                frame.gpu(),
                                &grid,
                                &points,
                                &camera,
                                (&rotation, &pose.translation),
                            )
                        }
                        None => Ok(Vec::new()),
                    };
                    let tsdf_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let tsdf = match tsdf {
                        Ok(s) => {
                            surface = s;
                            format!(", TSDF surface {} points in {tsdf_ms:.1} ms", surface.len())
                        }
                        Err(e) => {
                            ok = false;
                            format!(", TSDF failed: {e}")
                        }
                    };
                    format!(
                        ", {dropped} isolated dropped, fused in {ms:.1} ms into {} points{rms}{tsdf}",
                        fusion.model().len()
                    )
                };
                println!(
                    "{pair}: {} points in {total:.1} ms ({} ms){fused}",
                    d.points.len(),
                    stages.join(", ")
                );
                last = &d.points;
                ok &= !d.points.is_empty();
            }
            Err(e) => {
                println!("{pair}: depth failed: {e}");
                ok = false;
            }
        }
    }
    if let Some(path) = &out {
        let (points, what) = if preview.is_some() {
            (last, "preview")
        } else {
            (fusion.model(), "fused")
        };
        match depth::write_ply(path, points) {
            Ok(()) => println!(
                "{} {what} points written to {}",
                points.len(),
                path.display()
            ),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(path) = &surface_out {
        match depth::write_ply(path, &surface) {
            Ok(()) => println!(
                "{} TSDF surface points written to {}",
                surface.len(),
                path.display()
            ),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(dir) = &volume_out {
        match volume.save(dir) {
            Ok(()) => println!("TSDF volume written to {}", dir.display()),
            Err(e) => {
                eprintln!("volume: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(path) = &mesh_out {
        let start = Instant::now();
        let meshed = volume.mesh(0.01, faces).and_then(|m| {
            slscan_app::mesh::write_mesh(path, &m)
                .map(|()| m)
                .map_err(|e| format!("{}: {e}", path.display()))
        });
        match meshed {
            Ok(m) => println!(
                "mesh of {} triangles and {} vertices written to {} in {:.1} ms",
                m.faces.len(),
                m.vertices.len(),
                path.display(),
                start.elapsed().as_secs_f64() * 1000.0
            ),
            Err(e) => {
                eprintln!("mesh: {e}");
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

fn headless_calibrate_stereo(
    node: &str,
    wait: Duration,
    poses: u32,
    out: &Path,
    tool: &calibration::Tool,
    board: &calibration::Board,
) -> ExitCode {
    for _ in 0..poses {
        let dir = match calibration::capture_stereo_pose(node, wait, out) {
            Ok(dir) => dir,
            Err(e) => {
                eprintln!("pose capture failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let sides = ["left.png", "right.png"]
            .map(|image| calibration::detect_image(tool, &dir, board, image));
        match sides {
            [Ok(l), Ok(r)] => println!(
                "{name}: left {} of {}, right {} of {} corners",
                l.corners, l.of, r.corners, r.of
            ),
            [Err(e), _] | [_, Err(e)] => println!("{name}: detection failed: {e}"),
        }
    }
    let solve = match calibration::solve_stereo(tool, out, board) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("solve failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "{} views, left {:.3} px, right {:.3} px (limit {}), pair {:.3} px, baseline {:.2} mm, written to {}",
        solve.views,
        solve.left_rms_px,
        solve.right_rms_px,
        solve.camera_limit_px,
        solve.stereo_rms_px,
        1000.0 * solve.baseline_m,
        solve.out
    );
    if !solve.ok {
        return ExitCode::FAILURE;
    }
    match calibration::depth_params(tool, Path::new(&solve.out), &out.join("depth.json")) {
        Ok(p) => {
            println!(
                "depth parameters written to {}: {} × {}, reduced disparities {} to {}",
                p.out,
                p.width,
                p.height,
                p.lo,
                p.lo + p.count as i32 - 1
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("depth parameters failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn headless_calibrate(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut set = "calibration".to_owned();
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
            "--set" => {
                set.clone_from(value);
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
    if set == "stereo" {
        return headless_calibrate_stereo(&node, wait, poses, &out, &tool, &board);
    }
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
    #[serde(default = "depth::earlier_exit")]
    exit_px: f32,
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
            exit: meta.exit_px,
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

#[derive(serde::Deserialize)]
struct DotMeta {
    width: usize,
    height: usize,
    near: f64,
    far: f64,
}

fn points_of(bytes: &[u8]) -> Vec<[f64; 3]> {
    bytes
        .chunks_exact(24)
        .map(|c| {
            [0, 1, 2].map(|k| {
                let mut word = [0u8; 8];
                word.copy_from_slice(&c[8 * k..8 * k + 8]);
                f64::from_le_bytes(word)
            })
        })
        .collect()
}

/// Largest distance `dot-check` accepts between a Rust dot and its NumPy dot, whose
/// rectified images may differ by a grey level at some pixels.
const DOT_METRES: f64 = 2e-6;

/// Rectify the light-off pair `slscan dot-vectors` wrote on the GPU, find its dots and
/// compare them with the NumPy dots.
fn dot_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan dot-check <dot vector directory> [--fallback]");
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
            read("surface.bin"),
            read("dots.bin"),
        ],
    ) {
        (Ok(m), Ok(p), [Ok(a), Ok(b), Ok(c), Ok(d)]) => (m, p, [a, b, c, d]),
        _ => {
            eprintln!("dot vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let (meta, params): (DotMeta, Rectifications) = match (
        serde_json::from_slice(&meta),
        serde_json::from_slice(&params),
    ) {
        (Ok(m), Ok(p)) => (m, p),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("dot vector metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let rectifier = match rectify::Rectifier::new(fallback) {
        Ok(r) => r,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, dot check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let [left_raw, right_raw, surface, reference] = files;
    let (w, h) = (meta.width as u32, meta.height as u32);
    let rectified = rectifier
        .rectify(&params.left, w, h, &left_raw)
        .and_then(|l| {
            rectifier
                .rectify(&params.right, w, h, &right_raw)
                .map(|r| (l, r))
        });
    let (left, right) = match rectified {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("rectify failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let surface = points_of(&surface);
    let reference = points_of(&reference);
    let clock = Instant::now();
    let found = dots::find_dots(
        &params.left,
        &params.right,
        (&left, &right),
        meta.width,
        meta.height,
        &surface,
        meta.near,
        meta.far,
    );
    let ms = clock.elapsed().as_secs_f64() * 1000.0;
    let largest = found
        .iter()
        .map(|p| {
            reference
                .iter()
                .map(|q| (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    println!(
        "adapter {}: {} dots found in {ms:.1} ms against {} NumPy dots, largest distance {:.3} µm",
        rectifier.adapter,
        found.len(),
        reference.len(),
        largest * 1e6
    );
    if found.len() == reference.len() && largest < DOT_METRES {
        println!("dot check: Rust dot finding matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!(
            "dot check: Rust and NumPy dots differ in count or by more than {} µm",
            DOT_METRES * 1e6
        );
        ExitCode::FAILURE
    }
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

/// How depth frame points compare with reference points.
struct Comparison {
    missing: usize,
    extra: usize,
    count: usize,
    median: f64,
    p99: f64,
    largest: f64,
    /// Distance limit of the 99th percentile in metres.
    unit: f64,
    /// Points more than five units from their reference.
    beyond: usize,
}

impl Comparison {
    /// At most 0.5 % missing or extra, 99th percentile within one unit, 0.1 % beyond five.
    fn matches(&self) -> bool {
        let count = self.count.max(1) as f64;
        (self.missing + self.extra) as f64 / count <= 0.005
            && self.p99 <= self.unit
            && self.beyond as f64 / count <= 0.001
    }

    fn summary(&self) -> String {
        format!(
            "against {} reference points: {} missing, {} extra, distance median {:.3} µm, 99th percentile {:.3} µm, largest {:.3} µm, {} beyond {:.0} µm",
            self.count,
            self.missing,
            self.extra,
            self.median * 1e6,
            self.p99 * 1e6,
            self.largest * 1e6,
            self.beyond,
            self.unit * 5e6
        )
    }
}

/// Match depth frame points with reference rows (left x, left y, then the point) by their
/// left pixel, with distance limits in multiples of `unit` metres.
fn compare_points(
    samples: &[[f32; 2]],
    points: &[[f64; 3]],
    reference: &[u8],
    unit: f64,
) -> Comparison {
    let mut expected = std::collections::HashMap::new();
    for row in reference.chunks_exact(40) {
        let v = doubles(row);
        expected.insert((v[0] as i64, v[1] as i64), [v[2], v[3], v[4]]);
    }
    let mut extra = 0usize;
    let mut distances = Vec::new();
    for (s, p) in samples.iter().zip(points) {
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
    let at = |q: usize| distances.get(q).copied().unwrap_or(0.0);
    Comparison {
        missing,
        extra,
        count: expected.len(),
        median: at(distances.len() / 2),
        p99: at(distances.len() * 99 / 100),
        largest: distances.last().copied().unwrap_or(0.0),
        unit,
        beyond: distances.iter().filter(|&&d| d > 5.0 * unit).count(),
    }
}

/// Print one replayed frame and whether it matches its reference file, when there is one,
/// with distance limits in multiples of `unit` metres.
fn replay_frame(
    label: &str,
    result: Result<depth::DepthResult, String>,
    reference: &Path,
    unit: f64,
) -> bool {
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            println!("{label}: failed: {e}");
            return false;
        }
    };
    let total: f64 = result.timings.iter().map(|(_, ms)| ms).sum();
    let Ok(reference) = std::fs::read(reference) else {
        println!("{label}: {} points in {total:.1} ms", result.points.len());
        return true;
    };
    let comparison = compare_points(&result.samples, &result.points, &reference, unit);
    let matched = comparison.matches();
    println!(
        "{label}: {} points in {total:.1} ms; {}, {}",
        result.points.len(),
        comparison.summary(),
        if matched { "matches" } else { "differs" }
    );
    matched
}

fn replay(args: &[String]) -> ExitCode {
    let mut frames_dir: Option<PathBuf> = None;
    let mut params = PathBuf::from("data/depth.json");
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs: --frames, --params");
            return ExitCode::FAILURE;
        };
        match key.as_str() {
            "--frames" => frames_dir = Some(PathBuf::from(value)),
            "--params" => params = PathBuf::from(value),
            _ => {
                eprintln!("unknown argument {key}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(frames_dir) = frames_dir else {
        eprintln!("usage: sls-scan replay --frames <directory> [--params depth.json]");
        return ExitCode::FAILURE;
    };
    let setup = match depth::DepthSetup::load(&params) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("depth parameters: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut pairs: Vec<PathBuf> = match std::fs::read_dir(&frames_dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("frame_"))
            })
            .collect(),
        Err(e) => {
            eprintln!("{}: {e}", frames_dir.display());
            return ExitCode::FAILURE;
        }
    };
    pairs.sort();
    if pairs.is_empty() {
        eprintln!("no frame_* directories in {}", frames_dir.display());
        return ExitCode::FAILURE;
    }
    let frame = match depth::DepthFrame::new(false) {
        Ok(f) => f,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, replay skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("adapter {}", frame.adapter);
    let depth_params = setup.params();
    let preview = setup.preview_params();
    let mut ok = true;
    for pair in &pairs {
        let name = pair
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let images = (
            calibration::read_png(&pair.join("left.png")),
            calibration::read_png(&pair.join("right.png")),
        );
        let ((lw, lh, left), (rw, rh, right)) = match images {
            (Ok(l), Ok(r)) => (l, r),
            (Err(e), _) | (_, Err(e)) => {
                println!("{name}: {e}");
                ok = false;
                continue;
            }
        };
        if (lw, lh) != (setup.width, setup.height) || (rw, rh) != (setup.width, setup.height) {
            println!("{name}: images are not {} × {}", setup.width, setup.height);
            ok = false;
            continue;
        }
        let cameras = (&setup.cameras.left, &setup.cameras.right);
        ok &= replay_frame(
            &name,
            frame.depth(&depth_params, cameras, &left, &right),
            &pair.join("reference.bin"),
            1e-6,
        );
        if let Some(coarse) = &preview {
            ok &= replay_frame(
                &format!("{name} preview"),
                frame.preview(&depth_params, coarse, cameras, &left, &right),
                &pair.join("preview.bin"),
                1e-6 * f64::from(coarse.scale),
            );
        }
    }
    println!(
        "replay: {} pairs, {}",
        pairs.len(),
        if ok { "all match" } else { "some differ" }
    );
    if ok {
        ExitCode::SUCCESS
    } else {
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
        exit_px: meta.exit_px,
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
    let comparison = compare_points(&result.samples, &result.points, &reference, 1e-6);
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
    println!("{}", comparison.summary());
    if comparison.matches() {
        println!("depth check: GPU matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("depth check: GPU differs from the NumPy reference");
        ExitCode::FAILURE
    }
}

#[derive(serde::Deserialize)]
struct VolumeMeta {
    lower: [f64; 3],
    shape: [u32; 3],
    voxel: f64,
    truncation: f64,
}

/// Mesh a saved TSDF volume, as `slscan mesh-volume` does, without Python.
#[derive(serde::Deserialize)]
struct PoseRecord {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

#[derive(serde::Deserialize)]
struct RegisterVectors {
    voxel: f64,
    rounds: usize,
    views: Vec<String>,
    numpy: Vec<PoseRecord>,
    truth: Vec<PoseRecord>,
}

/// Angle in degrees and translation distance in metres between two poses.
fn pose_difference(pose: &fuse::Pose, other: &PoseRecord) -> (f64, f64) {
    let r = &pose.rotation;
    let o = &other.rotation;
    let trace: f64 = (0..3)
        .map(|i| (0..3).map(|k| r[i][k] * o[i][k]).sum::<f64>())
        .sum();
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    let d: f64 = (0..3)
        .map(|i| (pose.translation[i] - other.translation[i]).powi(2))
        .sum();
    (angle, d.sqrt())
}

/// Read view clouds, failing with a message naming the file.
fn read_views(paths: &[PathBuf]) -> Option<Vec<Vec<[f64; 3]>>> {
    let mut views = Vec::new();
    for path in paths {
        match register::read_ply(path) {
            Ok(v) => views.push(v),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return None;
            }
        }
    }
    Some(views)
}

/// Register the views `slscan register-vectors` wrote and compare the poses with the NumPy
/// registration and the truth.
fn register_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan register-check <register vectors directory>");
        return ExitCode::FAILURE;
    };
    let meta: RegisterVectors = match std::fs::read(dir.join("register.json"))
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("register.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let paths: Vec<PathBuf> = meta.views.iter().map(|v| dir.join(v)).collect();
    let Some(views) = read_views(&paths) else {
        return ExitCode::FAILURE;
    };
    let clock = Instant::now();
    let (poses, records) = register::register_views(&views, meta.voxel, meta.rounds);
    println!(
        "{} views registered in {:.0} ms at {} mm voxels, overlaps {}",
        views.len(),
        clock.elapsed().as_secs_f64() * 1000.0,
        meta.voxel * 1000.0,
        records
            .iter()
            .map(|r| format!("{:.2}", r.overlap))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let (mut worst_numpy, mut worst_truth) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64));
    for (pose, (numpy, truth)) in poses.iter().zip(meta.numpy.iter().zip(&meta.truth)) {
        let (a, t) = pose_difference(pose, numpy);
        worst_numpy = (worst_numpy.0.max(a), worst_numpy.1.max(t));
        let (a, t) = pose_difference(pose, truth);
        worst_truth = (worst_truth.0.max(a), worst_truth.1.max(t));
    }
    println!(
        "against NumPy poses: rotation up to {:.6}°, translation up to {:.3} µm; against truth: {:.5}°, {:.2} µm",
        worst_numpy.0,
        worst_numpy.1 * 1e6,
        worst_truth.0,
        worst_truth.1 * 1e6
    );
    if worst_numpy.0 < REGISTER_DEGREES && worst_numpy.1 < REGISTER_METRES {
        println!("register check: Rust registration matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!(
            "register check: Rust and NumPy poses differ by more than {REGISTER_DEGREES}° or {} µm",
            REGISTER_METRES * 1e6
        );
        ExitCode::FAILURE
    }
}

/// Largest pose difference `register-check` accepts between the Rust and NumPy
/// registrations, whose RANSAC draws differ.
const REGISTER_DEGREES: f64 = 0.001;
const REGISTER_METRES: f64 = 5e-6;

/// Register view clouds in order around an object, merge them and write the poses, as
/// `slscan register` does.
fn register_views_command(args: &[String]) -> ExitCode {
    let split = args
        .iter()
        .position(|a| a.starts_with("--"))
        .unwrap_or(args.len());
    let paths: Vec<PathBuf> = args[..split].iter().map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!(
            "usage: sls-scan register <view.ply>... [--voxel-mm N] [--rounds N] [--merge-voxel-mm N] [--out merged.ply] [--poses poses.json]"
        );
        return ExitCode::FAILURE;
    }
    let (mut voxel_mm, mut rounds, mut merge_mm) = (10.0f64, 1usize, 0.5f64);
    let (mut out, mut pose_file) = (
        PathBuf::from("data/merged.ply"),
        PathBuf::from("data/poses.json"),
    );
    for pair in args[split..].chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!(
                "arguments come in pairs: --voxel-mm, --rounds, --merge-voxel-mm, --out, --poses"
            );
            return ExitCode::FAILURE;
        };
        let parsed = match key.as_str() {
            "--voxel-mm" => value.parse().map(|v| voxel_mm = v).map_err(|_| ()),
            "--rounds" => value.parse().map(|v| rounds = v).map_err(|_| ()),
            "--merge-voxel-mm" => value.parse().map(|v| merge_mm = v).map_err(|_| ()),
            "--out" => {
                out = PathBuf::from(value);
                Ok(())
            }
            "--poses" => {
                pose_file = PathBuf::from(value);
                Ok(())
            }
            _ => Err(()),
        };
        if parsed.is_err() {
            eprintln!("bad argument {key} {value}");
            return ExitCode::FAILURE;
        }
    }
    let Some(views) = read_views(&paths) else {
        return ExitCode::FAILURE;
    };
    let (poses, records) = register::register_views(&views, voxel_mm / 1000.0, rounds);
    let mut merged: Vec<[f64; 3]> = views
        .iter()
        .zip(&poses)
        .flat_map(|(v, pose)| v.iter().map(|p| pose.apply(p)))
        .collect();
    if merge_mm > 0.0 {
        merged = register::voxel_average(&merged, merge_mm / 1000.0);
    }
    if let Err(e) = depth::write_ply(&out, &merged) {
        eprintln!("{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    let overlaps: Vec<Option<f64>> = std::iter::once(None)
        .chain(records.iter().map(|r| Some(r.overlap)))
        .collect();
    let listed: Vec<serde_json::Value> = paths
        .iter()
        .zip(&poses)
        .zip(&overlaps)
        .map(|((path, pose), overlap)| {
            serde_json::json!({
                "file": path.display().to_string(),
                "rotation": pose.rotation,
                "translation": pose.translation,
                "overlap_with_previous": overlap,
            })
        })
        .collect();
    let document = serde_json::json!({"frame": paths[0].display().to_string(), "views": listed});
    let written = serde_json::to_string_pretty(&document)
        .map_err(std::io::Error::other)
        .and_then(|text| std::fs::write(&pose_file, text));
    if let Err(e) = written {
        eprintln!("{}: {e}", pose_file.display());
        return ExitCode::FAILURE;
    }
    for (path, record) in paths[1..].iter().zip(&records) {
        println!(
            "{}: overlap with the view before {:.2}",
            path.display(),
            record.overlap
        );
    }
    println!(
        "{} points written to {}, poses in the frame of {} written to {}",
        merged.len(),
        out.display(),
        paths[0].display(),
        pose_file.display()
    );
    ExitCode::SUCCESS
}

fn mesh_volume(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!(
            "usage: sls-scan mesh-volume <volume directory> --out <mesh> [--fill-mm N] [--faces N]"
        );
        return ExitCode::FAILURE;
    };
    let (mut out, mut fill_mm, mut faces) = (None, 0.0f64, 0usize);
    for pair in args[1..].chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs: --out, --fill-mm, --faces");
            return ExitCode::FAILURE;
        };
        let parsed = match key.as_str() {
            "--out" => {
                out = Some(PathBuf::from(value));
                Ok(())
            }
            "--fill-mm" => value.parse().map(|v| fill_mm = v).map_err(|_| ()),
            "--faces" => value.parse().map(|v| faces = v).map_err(|_| ()),
            _ => Err(()),
        };
        if parsed.is_err() {
            eprintln!("bad argument {key} {value}");
            return ExitCode::FAILURE;
        }
    }
    let Some(out) = out else {
        eprintln!("--out is required");
        return ExitCode::FAILURE;
    };
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, distances, weights) = match (read("meta.json"), read("tsdf.bin"), read("weight.bin"))
    {
        (Ok(m), Ok(t), Ok(w)) => (m, floats(&t), floats(&w)),
        _ => {
            eprintln!("volume files missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: VolumeMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let spec = tsdf::VolumeSpec {
        lower: meta.lower,
        voxel: meta.voxel,
        truncation: meta.truncation,
        shape: meta.shape,
    };
    if distances.len() != spec.voxels() || weights.len() != spec.voxels() {
        eprintln!("volume files do not match meta.json");
        return ExitCode::FAILURE;
    }
    let mut clock = Instant::now();
    let mut lap = || {
        let ms = clock.elapsed().as_secs_f64() * 1000.0;
        clock = Instant::now();
        ms
    };
    let mut mesh = slscan_app::mesh::surface_nets(&spec, &distances, &weights);
    let nets = (mesh.faces.len(), lap());
    let holes = slscan_app::mesh::fill_holes(&mut mesh, fill_mm / 1000.0);
    let filled = lap();
    if faces > 0 {
        mesh = slscan_app::mesh::decimate(&mesh, faces, 1000.0);
    }
    let decimated = lap();
    if mesh.faces.is_empty() {
        eprintln!("{}: the volume has no surface", dir.display());
        return ExitCode::FAILURE;
    }
    if let Err(e) = slscan_app::mesh::write_mesh(&out, &mesh) {
        eprintln!("{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    println!(
        "{} triangles in {:.1} ms, {holes} holes filled in {filled:.1} ms, decimated in {decimated:.1} ms; {} vertices and {} triangles written to {}",
        nets.0,
        nets.1,
        mesh.vertices.len(),
        mesh.faces.len(),
        out.display()
    );
    ExitCode::SUCCESS
}

#[derive(serde::Deserialize)]
struct TsdfMeta {
    counts: Vec<usize>,
    step: u32,
    window: u32,
    width: u32,
    height: u32,
    rotation: [f64; 9],
    projection: [f64; 12],
    lower: [f64; 3],
    shape: [u32; 3],
    voxel: f64,
    truncation: f64,
}

fn tsdf_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan tsdf-check <vector directory>");
        return ExitCode::FAILURE;
    };
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, frames, poses, tsdf, weight, surface, keys, mesh_files) = match (
        read("meta.json"),
        read("frames.bin"),
        read("poses.bin"),
        read("tsdf.bin"),
        read("weight.bin"),
        read("surface.bin"),
        read("surface-keys.bin"),
        read("mesh-vertices.bin").and_then(|v| read("mesh-faces.bin").map(|f| (v, f))),
    ) {
        (Ok(m), Ok(f), Ok(p), Ok(t), Ok(w), Ok(s), Ok(k), Ok(mf)) => (
            m,
            doubles(&f),
            doubles(&p),
            floats(&t),
            floats(&w),
            doubles(&s),
            k.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect::<Vec<u32>>(),
            mf,
        ),
        _ => {
            eprintln!("TSDF vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: TsdfMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let spec = tsdf::VolumeSpec {
        lower: meta.lower,
        voxel: meta.voxel,
        truncation: meta.truncation,
        shape: meta.shape,
    };
    if frames.len() != 3 * meta.counts.iter().sum::<usize>()
        || poses.len() != 12 * meta.counts.len()
        || tsdf.len() != spec.voxels()
        || weight.len() != spec.voxels()
        || surface.len() != 3 * keys.len()
    {
        eprintln!("TSDF vectors do not match meta.json");
        return ExitCode::FAILURE;
    }
    let integrator = match tsdf::Integrator::new(false, spec.clone()) {
        Ok(i) => i,
        Err(e) if std::env::var_os("SLS_REQUIRE_GPU").is_none() => {
            println!("no GPU adapter, TSDF check skipped: {e}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("adapter {}", integrator.adapter);
    let camera = tsdf::LeftCamera {
        rotation: meta.rotation,
        projection: meta.projection,
    };
    let start = Instant::now();
    let mut offset = 0;
    for (&count, pose) in meta.counts.iter().zip(poses.chunks_exact(12)) {
        let points: Vec<[f64; 3]> = frames[offset..offset + 3 * count]
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        offset += 3 * count;
        let grid = tsdf::depth_grid(
            &points,
            &camera,
            meta.step,
            meta.window,
            (meta.width, meta.height),
        );
        let rotation: [f64; 9] = std::array::from_fn(|i| pose[i]);
        let translation = [pose[9], pose[10], pose[11]];
        if let Err(e) = integrator.integrate(&grid, &camera, &rotation, &translation) {
            eprintln!("integration failed: {e}");
            return ExitCode::FAILURE;
        }
    }
    let (gpu_tsdf, gpu_weight) = match integrator.read() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("readback failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let observed = weight.iter().filter(|&&w| w > 0.0).count();
    let differing = weight
        .iter()
        .zip(&gpu_weight)
        .filter(|(a, b)| a != b)
        .count();
    let mut distances: Vec<f64> = tsdf
        .iter()
        .zip(&gpu_tsdf)
        .zip(weight.iter().zip(&gpu_weight))
        .filter(|(_, (a, b))| **a > 0.0 && a == b)
        .map(|((a, b), _)| f64::from((a - b).abs()))
        .collect();
    distances.sort_by(f64::total_cmp);
    let at = |q: usize| distances.get(q).copied().unwrap_or(0.0);
    let (median, p99) = (at(distances.len() / 2), at(distances.len() * 99 / 100));
    let largest = distances.last().copied().unwrap_or(0.0);
    let scale = meta.truncation * 1e6;
    println!(
        "{} frames into {} voxels in {elapsed:.1} ms with upload and readback; {observed} observed, {differing} with other weights, distance difference median {:.4} µm, 99th percentile {:.4} µm, largest {:.4} µm",
        meta.counts.len(),
        spec.voxels(),
        median * scale,
        p99 * scale,
        largest * scale
    );
    let volume_ok = differing as f64 <= 0.001 * observed.max(1) as f64 && p99 * scale <= 0.1;
    let expected: std::collections::HashMap<u32, [f64; 3]> = keys
        .iter()
        .zip(surface.chunks_exact(3))
        .map(|(&k, p)| (k, [p[0], p[1], p[2]]))
        .collect();
    let capacity = u32::try_from(2 * expected.len() + 1024).unwrap_or(u32::MAX);
    let start = Instant::now();
    let (points, found) = match integrator.surface(capacity) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("surface failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let surface_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut extra = 0usize;
    let mut gaps = Vec::new();
    for (p, k) in points.iter().zip(&found) {
        match expected.get(k) {
            Some(r) => {
                let squares: f64 = (0..3).map(|d| (f64::from(p[d]) - r[d]).powi(2)).sum();
                gaps.push(squares.sqrt());
            }
            None => extra += 1,
        }
    }
    let missing = expected.len() - gaps.len();
    gaps.sort_by(f64::total_cmp);
    let gap = |q: usize| gaps.get(q).copied().unwrap_or(0.0) * 1e6;
    let surface_p99 = gap(gaps.len() * 99 / 100);
    let beyond = gaps.iter().filter(|&&g| g > 5e-6).count();
    println!(
        "{} surface points in {:.1} ms against {}: {missing} missing, {extra} extra, distance median {:.4} µm, 99th percentile {surface_p99:.4} µm, largest {:.4} µm, {beyond} beyond 5 µm",
        points.len(),
        surface_ms,
        expected.len(),
        gap(gaps.len() / 2),
        gaps.last().copied().unwrap_or(0.0) * 1e6
    );
    let reference = expected.len().max(1) as f64;
    let surface_ok = (missing + extra) as f64 <= 0.005 * reference
        && surface_p99 <= 1.0
        && beyond as f64 <= 0.005 * reference;
    let reference_vertices = doubles(&mesh_files.0);
    let reference_faces: Vec<u32> = mesh_files
        .1
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let start = Instant::now();
    let mesh = slscan_app::mesh::surface_nets(&spec, &tsdf, &weight);
    let mesh_ms = start.elapsed().as_secs_f64() * 1000.0;
    let same_faces = mesh
        .faces
        .iter()
        .flatten()
        .copied()
        .eq(reference_faces.iter().copied());
    let largest = if mesh.vertices.len() * 3 == reference_vertices.len() {
        mesh.vertices
            .iter()
            .flatten()
            .zip(&reference_vertices)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    } else {
        f64::INFINITY
    };
    println!(
        "mesh of the NumPy volume in {mesh_ms:.1} ms: {} vertices against {}, {} triangles against {}, triangles {}, vertex difference largest {:.6} µm",
        mesh.vertices.len(),
        reference_vertices.len() / 3,
        mesh.faces.len(),
        reference_faces.len() / 3,
        if same_faces { "equal" } else { "differ" },
        largest * 1e6
    );
    let mesh_ok = same_faces && largest <= 1e-9;
    if volume_ok && surface_ok && mesh_ok {
        println!("TSDF check: GPU matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("TSDF check: GPU differs from the NumPy reference");
        ExitCode::FAILURE
    }
}

#[derive(serde::Deserialize)]
struct FusionMeta {
    counts: Vec<usize>,
    voxel: f64,
    #[serde(default)]
    dot_counts: Vec<usize>,
}

fn doubles(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

fn fusion_check(args: &[String]) -> ExitCode {
    let Some(dir) = args.first().map(PathBuf::from) else {
        eprintln!("usage: sls-scan fusion-check <vector directory>");
        return ExitCode::FAILURE;
    };
    let read = |name: &str| std::fs::read(dir.join(name));
    let (meta, frames, poses, model) = match (
        read("meta.json"),
        read("frames.bin"),
        read("poses.bin"),
        read("model.bin"),
    ) {
        (Ok(m), Ok(f), Ok(p), Ok(d)) => (m, doubles(&f), doubles(&p), doubles(&d)),
        _ => {
            eprintln!("fusion vectors missing in {}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let meta: FusionMeta = match serde_json::from_slice(&meta) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta.json: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dots = if meta.dot_counts.is_empty() {
        Vec::new()
    } else {
        match read("dots.bin") {
            Ok(d) => doubles(&d),
            Err(e) => {
                eprintln!("dots.bin: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    if frames.len() != 3 * meta.counts.iter().sum::<usize>()
        || poses.len() != 12 * meta.counts.len()
        || dots.len() != 3 * meta.dot_counts.iter().sum::<usize>()
        || !(meta.dot_counts.is_empty() || meta.dot_counts.len() == meta.counts.len())
    {
        eprintln!("fusion vectors do not match meta.json");
        return ExitCode::FAILURE;
    }
    let mut fusion = fuse::Fusion::new(meta.voxel);
    let (mut offset, mut dot_offset) = (0, 0);
    let mut times = Vec::new();
    for (k, &count) in meta.counts.iter().enumerate() {
        let points: Vec<fuse::Vec3> = frames[offset..offset + 3 * count]
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        offset += 3 * count;
        let dot_count = meta.dot_counts.get(k).copied().unwrap_or(0);
        let seen: Vec<fuse::Vec3> = dots[dot_offset..dot_offset + 3 * dot_count]
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        dot_offset += 3 * dot_count;
        let start = Instant::now();
        fusion.add_with_dots(&points, &seen);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let (mut translation, mut rotation) = (0.0f64, 0.0f64);
    for (pose, reference) in fusion.poses.iter().zip(poses.chunks_exact(12)) {
        let r = &pose.rotation;
        let trace: f64 = (0..3)
            .map(|i| (0..3).map(|k| r[i][k] * reference[i * 3 + k]).sum::<f64>())
            .sum();
        rotation = rotation.max(((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees());
        let d: f64 = (0..3)
            .map(|i| (pose.translation[i] - reference[9 + i]).powi(2))
            .sum();
        translation = translation.max(d.sqrt());
    }
    let key = |p: &[f64]| [0, 1, 2].map(|d| (p[d] / meta.voxel).floor() as i64);
    let expected: std::collections::HashMap<[i64; 3], [f64; 3]> = model
        .chunks_exact(3)
        .map(|p| (key(p), [p[0], p[1], p[2]]))
        .collect();
    let (mut found, mut largest) = (0usize, 0.0f64);
    for p in fusion.model() {
        if let Some(r) = expected.get(&key(p)) {
            found += 1;
            let d: f64 = (0..3).map(|i| (p[i] - r[i]).powi(2)).sum();
            largest = largest.max(d.sqrt());
        }
    }
    let differing = fusion.model().len() + expected.len() - 2 * found;
    let fraction = differing as f64 / expected.len().max(1) as f64;
    let total: f64 = times.iter().sum();
    println!(
        "{} frames tracked and fused in {total:.1} ms, {:.1} ms per frame at most, {} model points",
        meta.counts.len(),
        times.iter().copied().fold(0.0, f64::max),
        fusion.model().len()
    );
    println!(
        "pose difference up to {:.3} µm and {:.6}°, voxels in one model only {differing} of {} ({:.3} %), point difference up to {:.3} µm",
        translation * 1e6,
        rotation,
        expected.len(),
        100.0 * fraction,
        largest * 1e6
    );
    if translation <= 1e-6 && rotation <= 1e-4 && fraction <= 0.005 && largest <= 1e-6 {
        println!("fusion check: Rust matches the NumPy reference");
        ExitCode::SUCCESS
    } else {
        println!("fusion check: Rust differs from the NumPy reference");
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
        Some("dot-check") => return dot_check(&args[2..]),
        Some("coarse-check") => return coarse_check(&args[2..]),
        Some("depth-check") => return depth_check(&args[2..]),
        Some("fusion-check") => return fusion_check(&args[2..]),
        Some("tsdf-check") => return tsdf_check(&args[2..]),
        Some("mesh-volume") => return mesh_volume(&args[2..]),
        Some("register") => return register_views_command(&args[2..]),
        Some("register-check") => return register_check(&args[2..]),
        Some("replay") => return replay(&args[2..]),
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
