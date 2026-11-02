use slscan_app::calibration;
use slscan_app::client::{self, Event};
use slscan_app::protocol::SequenceRequest;
use slscan_app::ui::ScanApp;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

fn headless_capture(args: &[String]) -> ExitCode {
    let mut node = "127.0.0.1:5600".to_owned();
    let mut wait = Duration::ZERO;
    let mut request = SequenceRequest {
        set: "scan".to_owned(),
        pitch: 32,
        steps: 4,
    };
    for pair in args.chunks(2) {
        let (Some(key), Some(value)) = (pair.first(), pair.get(1)) else {
            eprintln!("arguments come in pairs: --node, --set, --pitch, --steps, --wait");
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("capture") => return headless_capture(&args[2..]),
        Some("calibrate") => return headless_calibrate(&args[2..]),
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
