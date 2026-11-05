//! The three panels: configuration, preview and processing, and the scene.

use crate::calibration::{self, Board, Detection, Solve, StereoSolve, Tool};
use crate::client::{self, Event, Stats};
use crate::depth::{DepthFrame, DepthResult, DepthSetup};
use crate::fuse::{self, Fusion, Vec3};
use crate::protocol::{FrameHeader, Hello, Projector, SequenceRequest};
use crate::scan::{self, ScanParams};
use crate::{depth, gpu, mesh, tsdf};
use eframe::egui;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

type StereoDetection = Result<(Detection, Detection), String>;
type ScanResult = Result<(Vec<[f64; 3]>, String), String>;

enum CalibrationEvent {
    Pose(String, Result<Detection, String>, Projector),
    StereoPose(String, StereoDetection),
    Failed(String),
    Solved(Result<Solve, String>),
    StereoSolved(Result<(StereoSolve, String), String>),
}

fn limit_label(value: f64, limit: f64, what: &str) -> egui::RichText {
    let colour = if value < limit {
        egui::Color32::LIGHT_GREEN
    } else {
        egui::Color32::LIGHT_RED
    };
    egui::RichText::new(format!("{what} {value:.3} px (limit {limit})")).color(colour)
}

/// Voxel size of the fused model, as `track.fuse` uses by default.
const FUSION_VOXEL_M: f64 = 0.0005;
/// With reference dots on, every this many-th stereo pair is taken with the pattern
/// light off (`DESIGN.md` D77).
const DARK_EVERY: u32 = 4;
/// Points drawn in the scene at most, every so many of a larger cloud.
const MAX_DRAWN: usize = 20_000;

enum DepthJob {
    Pair(Vec<u8>, Vec<u8>),
    /// A pair taken with the pattern light off, kept until the next lit pair's points
    /// place its dots.
    Dark(Vec<u8>, Vec<u8>),
    Reset,
    /// Mesh the live volume, closing holes up to a size in metres and decimating to a
    /// triangle count when it is not 0, and reply with the mesh.
    Mesh(f64, usize, mpsc::Sender<Result<mesh::Mesh, String>>),
}

struct LiveFrame {
    depth: Result<DepthResult, String>,
    /// The TSDF surface, or the fused points when the TSDF failed.
    model: Result<Vec<Vec3>, (String, Vec<Vec3>)>,
    fuse_ms: f64,
    rms: Option<f64>,
    /// Dots found in the light-off pair before this frame, and dots in the map.
    dots: Option<(Result<usize, String>, usize)>,
}

struct DepthWorker {
    params_path: String,
    jobs: mpsc::Sender<DepthJob>,
    results: Receiver<LiveFrame>,
    in_flight: usize,
}

pub struct ScanApp {
    node: String,
    session: String,
    tool: String,
    board: Board,
    calibration_events: Option<Receiver<CalibrationEvent>>,
    poses: Vec<(String, Result<Detection, String>)>,
    projector: Option<Projector>,
    solve: Option<Result<Solve, String>>,
    stereo_poses: Vec<(String, StereoDetection)>,
    stereo_solve: Option<Result<(StereoSolve, String), String>>,
    set: String,
    pitch: u32,
    steps: u32,
    hello: Option<Hello>,
    status: String,
    events: Option<Receiver<Event>>,
    preview: Option<egui::TextureHandle>,
    last: Option<(String, FrameHeader)>,
    frames: usize,
    stats: Option<Stats>,
    params_path: String,
    stereo_frames: u32,
    /// Take every `DARK_EVERY`-th stereo pair with the pattern light off for reference dots.
    dots: bool,
    depth: Option<DepthWorker>,
    pending_left: Option<Vec<u8>>,
    cloud: Vec<[f64; 3]>,
    fusion_status: String,
    depth_status: String,
    depth_frames: usize,
    depth_skipped: usize,
    mesh_path: String,
    mesh_fill_mm: f64,
    mesh_faces: u32,
    export: Option<Receiver<Result<(usize, usize), String>>>,
    export_status: String,
    scan_params_path: String,
    scan_params: Option<ScanParams>,
    scan_frames: HashMap<String, Vec<u8>>,
    scan_job: Option<Receiver<ScanResult>>,
    scan_status: String,
    points_path: String,
}

impl Default for ScanApp {
    fn default() -> Self {
        Self {
            node: "127.0.0.1:5600".to_owned(),
            session: "data/calibration".to_owned(),
            tool: "python3 -m slscan.cli".to_owned(),
            board: Board::default(),
            calibration_events: None,
            poses: Vec::new(),
            projector: None,
            solve: None,
            stereo_poses: Vec::new(),
            stereo_solve: None,
            set: "scan".to_owned(),
            pitch: 32,
            steps: 4,
            hello: None,
            status: "idle".to_owned(),
            events: None,
            preview: None,
            last: None,
            frames: 0,
            stats: None,
            params_path: "data/depth.json".to_owned(),
            stereo_frames: 30,
            dots: false,
            depth: None,
            pending_left: None,
            cloud: Vec::new(),
            fusion_status: String::new(),
            depth_status: String::new(),
            depth_frames: 0,
            depth_skipped: 0,
            mesh_path: "data/scan.stl".to_owned(),
            mesh_fill_mm: 10.0,
            mesh_faces: 0,
            export: None,
            export_status: String::new(),
            scan_params_path: "data/scan.json".to_owned(),
            scan_params: None,
            scan_frames: HashMap::new(),
            scan_job: None,
            scan_status: String::new(),
            points_path: "data/scan.ply".to_owned(),
        }
    }
}

impl ScanApp {
    /// Load the depth parameters and start a worker that owns the device and runs one
    /// depth frame per pair it is sent.
    fn start_depth(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let setup = DepthSetup::load(Path::new(&self.params_path))?;
        let frame = DepthFrame::new(false)?;
        let adapter = frame.adapter.clone();
        let params = setup.params();
        let preview = setup.preview_params().filter(|_| frame.software());
        let kind = if preview.is_some() {
            "previews"
        } else {
            "depth frames"
        };
        let (jobs_tx, jobs_rx) = mpsc::channel::<DepthJob>();
        let (results_tx, results_rx) = mpsc::channel();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let mut fusion = Fusion::new(FUSION_VOXEL_M);
            let mut volume = tsdf::Live::coarse();
            let camera = tsdf::LeftCamera {
                rotation: setup.cameras.left.rotation,
                projection: setup.cameras.left.projection,
            };
            let size = (params.coarse.width, params.coarse.height);
            let mut pending_dark: Option<(Vec<u8>, Vec<u8>)> = None;
            for job in jobs_rx {
                let (left, right) = match job {
                    DepthJob::Pair(left, right) => (left, right),
                    DepthJob::Dark(left, right) => {
                        pending_dark = Some((left, right));
                        continue;
                    }
                    DepthJob::Reset => {
                        fusion = Fusion::new(FUSION_VOXEL_M);
                        volume.reset();
                        pending_dark = None;
                        continue;
                    }
                    DepthJob::Mesh(fill_m, faces, reply) => {
                        let _ = reply.send(volume.mesh(fill_m, faces));
                        continue;
                    }
                };
                let cameras = (&setup.cameras.left, &setup.cameras.right);
                let depth = frame.measure(&params, preview.as_ref(), cameras, &left, &right);
                let start = std::time::Instant::now();
                let mut surface = Ok(Vec::new());
                let mut dots = None;
                if let (Ok(d), None) = (&depth, &preview) {
                    let points = fuse::drop_isolated(
                        &d.points,
                        f64::from(params.step),
                        camera.projection[0],
                        fuse::ISOLATED_SPACINGS,
                    );
                    if let Some((l, r)) = pending_dark.take() {
                        let found = frame.dots(&params, cameras, &l, &r, &points);
                        if let Ok(seen) = &found {
                            fusion.add_dark(seen);
                        }
                        dots = Some((found.map(|seen| seen.len()), fusion.dots().len()));
                    }
                    fusion.add(&points);
                    if let Some(pose) = fusion.poses.last() {
                        let grid =
                            tsdf::depth_grid(&points, &camera, params.step, params.window, size);
                        let rotation: [f64; 9] =
                            std::array::from_fn(|k| pose.rotation[k / 3][k % 3]);
                        surface = volume.add(
                            frame.gpu(),
                            &grid,
                            &points,
                            &camera,
                            (&rotation, &pose.translation),
                        );
                    }
                }
                let live = LiveFrame {
                    depth,
                    model: surface.map_err(|e| (e, fusion.model().to_vec())),
                    fuse_ms: start.elapsed().as_secs_f64() * 1000.0,
                    rms: fusion.last_rms,
                    dots,
                };
                let _ = results_tx.send(live);
                repaint.request_repaint();
            }
        });
        self.depth = Some(DepthWorker {
            params_path: self.params_path.clone(),
            jobs: jobs_tx,
            results: results_rx,
            in_flight: 0,
        });
        self.depth_status = format!("{kind} on {adapter}");
        Ok(())
    }

    fn start(&mut self, ctx: &egui::Context) {
        let stereo = self.set == "stereo";
        if stereo
            && self
                .depth
                .as_ref()
                .is_none_or(|w| w.params_path != self.params_path)
        {
            if let Err(e) = self.start_depth(ctx) {
                self.depth = None;
                self.depth_status = format!("no depth frames: {e}");
            }
        }
        self.pending_left = None;
        self.depth_frames = 0;
        self.depth_skipped = 0;
        if let Some(worker) = &self.depth {
            let _ = worker.jobs.send(DepthJob::Reset);
        }
        self.cloud.clear();
        self.fusion_status.clear();
        self.scan_frames.clear();
        self.scan_params = None;
        self.scan_status.clear();
        if self.set == "scan" {
            match std::fs::read(&self.scan_params_path)
                .map_err(|e| e.to_string())
                .and_then(|b| serde_json::from_slice::<ScanParams>(&b).map_err(|e| e.to_string()))
            {
                Ok(params) => {
                    self.pitch = params.decode.pitch.round() as u32;
                    self.steps = params.decode.steps;
                    self.scan_params = Some(params);
                }
                Err(e) => {
                    self.scan_status =
                        format!("frames only, no points: {}: {e}", self.scan_params_path);
                }
            }
        }
        let (tx, rx) = mpsc::channel();
        let node = self.node.clone();
        let request = SequenceRequest {
            set: self.set.clone(),
            pitch: self.pitch,
            steps: self.steps,
            frames: if stereo { self.stereo_frames } else { 1 },
            rate_hz: 0,
            dark_every: if stereo && self.dots { DARK_EVERY } else { 0 },
            exposures: self
                .scan_params
                .as_ref()
                .filter(|_| self.set == "scan")
                .map(|p| p.decode.exposures.clone())
                .unwrap_or_default(),
        };
        let repaint = ctx.clone();
        thread::spawn(move || {
            let send = |event| {
                let _ = tx.send(event);
                repaint.request_repaint();
            };
            if let Err(e) = client::capture(&node, &request, Duration::ZERO, send) {
                let _ = tx.send(Event::Failed(e.to_string()));
                repaint.request_repaint();
            }
        });
        self.events = Some(rx);
        self.frames = 0;
        self.stats = None;
        self.status = format!("running {} sequence", self.set);
    }

    fn busy(&self) -> bool {
        self.events.is_some() || self.calibration_events.is_some() || self.scan_job.is_some()
    }

    /// Decode and triangulate the scan set just received on a thread of its own, with a
    /// device made for it.
    fn start_scan(&mut self, ctx: &egui::Context) {
        let Some(params) = self.scan_params.clone() else {
            return;
        };
        let Some(stacks) = scan::stacks(&self.scan_frames, &params.decode) else {
            self.scan_status = "the node sent fewer frames than the scan parameters need".into();
            return;
        };
        self.scan_frames.clear();
        let (tx, rx) = mpsc::channel();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = gpu::Decoder::new(false).and_then(|decoder| {
                let start = std::time::Instant::now();
                let (columns, mask) = scan::decode_scan(&decoder, &params, &stacks)?;
                let decoded = start.elapsed().as_secs_f64() * 1000.0;
                let start = std::time::Instant::now();
                let (points, _) = scan::triangulate(&columns, &mask, &params);
                let note = format!(
                    "{} points, decoded in {decoded:.0} ms and triangulated in {:.0} ms",
                    points.len(),
                    start.elapsed().as_secs_f64() * 1000.0
                );
                Ok((points, note))
            });
            let _ = tx.send(result);
            repaint.request_repaint();
        });
        self.scan_job = Some(rx);
        self.scan_status = "decoding the scan".to_owned();
    }

    fn poll_scan(&mut self) {
        let Some(rx) = &self.scan_job else {
            return;
        };
        if let Ok(result) = rx.try_recv() {
            match result {
                Ok((points, note)) => {
                    self.cloud = points;
                    self.scan_status = note;
                }
                Err(e) => self.scan_status = format!("scan failed: {e}"),
            }
            self.scan_job = None;
        }
    }

    fn save_points(&mut self) {
        let path = PathBuf::from(&self.points_path);
        self.scan_status = match depth::write_ply(&path, &self.cloud) {
            Ok(()) => format!("{} points written to {}", self.cloud.len(), path.display()),
            Err(e) => format!("{}: {e}", path.display()),
        };
    }

    fn capture_stereo_pose(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        let (node, session, tool, board) = (
            self.node.clone(),
            PathBuf::from(&self.session),
            Tool::parse(&self.tool),
            self.board.clone(),
        );
        let repaint = ctx.clone();
        thread::spawn(move || {
            let event = match calibration::capture_stereo_pose(&node, Duration::ZERO, &session) {
                Ok(dir) => {
                    let name = dir
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let detections =
                        tool.ok_or_else(|| "no tool command".to_owned())
                            .and_then(|t| {
                                let side = |image| {
                                    calibration::detect_image(&t, &dir, &board, image)
                                        .map_err(|e| e.to_string())
                                };
                                Ok((side("left.png")?, side("right.png")?))
                            });
                    CalibrationEvent::StereoPose(name, detections)
                }
                Err(e) => CalibrationEvent::Failed(e.to_string()),
            };
            let _ = tx.send(event);
            repaint.request_repaint();
        });
        self.calibration_events = Some(rx);
        self.status = "capturing stereo pose".to_owned();
    }

    fn start_stereo_solve(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        let (session, tool, board) = (
            PathBuf::from(&self.session),
            Tool::parse(&self.tool),
            self.board.clone(),
        );
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = tool
                .ok_or_else(|| "no tool command".to_owned())
                .and_then(|t| {
                    let solve = calibration::solve_stereo(&t, &session, &board)
                        .map_err(|e| e.to_string())?;
                    let params = calibration::depth_params(
                        &t,
                        Path::new(&solve.out),
                        &session.join("depth.json"),
                    )
                    .map_err(|e| e.to_string())?;
                    Ok((solve, params.out))
                });
            let _ = tx.send(CalibrationEvent::StereoSolved(result));
            repaint.request_repaint();
        });
        self.calibration_events = Some(rx);
        self.status = "solving stereo calibration".to_owned();
    }

    fn capture_pose(&mut self, ctx: &egui::Context) {
        if self.set == "stereo" {
            return self.capture_stereo_pose(ctx);
        }
        let (tx, rx) = mpsc::channel();
        let (node, session, tool, board) = (
            self.node.clone(),
            PathBuf::from(&self.session),
            Tool::parse(&self.tool),
            self.board.clone(),
        );
        let repaint = ctx.clone();
        thread::spawn(move || {
            let event = match calibration::capture_pose(&node, Duration::ZERO, &session) {
                Ok((dir, projector)) => {
                    let name = dir
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let detection =
                        tool.ok_or_else(|| "no tool command".to_owned())
                            .and_then(|t| {
                                calibration::detect(&t, &dir, &board).map_err(|e| e.to_string())
                            });
                    CalibrationEvent::Pose(name, detection, projector)
                }
                Err(e) => CalibrationEvent::Failed(e.to_string()),
            };
            let _ = tx.send(event);
            repaint.request_repaint();
        });
        self.calibration_events = Some(rx);
        self.status = "capturing calibration pose".to_owned();
    }

    fn start_solve(&mut self, ctx: &egui::Context) {
        if self.set == "stereo" {
            return self.start_stereo_solve(ctx);
        }
        let Some(projector) = self.projector.clone() else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        let (session, tool, board) = (
            PathBuf::from(&self.session),
            Tool::parse(&self.tool),
            self.board.clone(),
        );
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = tool
                .ok_or_else(|| "no tool command".to_owned())
                .and_then(|t| {
                    calibration::solve(&t, &session, &board, &projector).map_err(|e| e.to_string())
                });
            let _ = tx.send(CalibrationEvent::Solved(result));
            repaint.request_repaint();
        });
        self.calibration_events = Some(rx);
        self.status = "solving calibration".to_owned();
    }

    fn poll_calibration(&mut self) {
        let Some(events) = &self.calibration_events else {
            return;
        };
        if let Ok(event) = events.try_recv() {
            match event {
                CalibrationEvent::Pose(name, detection, projector) => {
                    self.status = format!("captured {name}");
                    self.poses.push((name, detection));
                    self.projector = Some(projector);
                }
                CalibrationEvent::Failed(message) => {
                    self.status = format!("pose capture failed: {message}");
                }
                CalibrationEvent::Solved(result) => {
                    self.status = "calibration solved".to_owned();
                    self.solve = Some(result);
                }
                CalibrationEvent::StereoPose(name, detections) => {
                    self.status = format!("captured {name}");
                    self.stereo_poses.push((name, detections));
                }
                CalibrationEvent::StereoSolved(result) => {
                    self.status = "stereo calibration solved".to_owned();
                    if let Ok((_, params)) = &result {
                        self.params_path.clone_from(params);
                    }
                    self.stereo_solve = Some(result);
                }
            }
            self.calibration_events = None;
        }
    }

    fn calibration_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.heading("Calibration");
        ui.label("Session directory");
        ui.text_edit_singleline(&mut self.session);
        ui.label("Calibration tool");
        ui.text_edit_singleline(&mut self.tool);
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.board.squares_x)
                    .range(3..=30)
                    .prefix("across "),
            );
            ui.add(
                egui::DragValue::new(&mut self.board.squares_y)
                    .range(3..=30)
                    .prefix("down "),
            );
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.board.square_mm)
                    .range(5.0..=200.0)
                    .prefix("square mm "),
            );
            ui.add(
                egui::DragValue::new(&mut self.board.marker_mm)
                    .range(3.0..=150.0)
                    .prefix("marker mm "),
            );
        });
        let idle = !self.busy();
        let stereo = self.set == "stereo";
        ui.horizontal(|ui| {
            if ui
                .add_enabled(idle, egui::Button::new("Capture pose"))
                .clicked()
            {
                self.capture_pose(ctx);
            }
            let enough = if stereo {
                self.stereo_poses.len() >= 3
            } else {
                self.poses.len() >= 3 && self.projector.is_some()
            };
            if ui
                .add_enabled(idle && enough, egui::Button::new("Solve"))
                .clicked()
            {
                self.start_solve(ctx);
            }
        });
        if stereo {
            for (name, detections) in &self.stereo_poses {
                match detections {
                    Ok((l, r)) => ui.label(format!(
                        "{name}: left {} of {}, right {} of {} corners",
                        l.corners, l.of, r.corners, r.of
                    )),
                    Err(e) => ui.label(format!("{name}: detection failed: {e}")),
                };
            }
            match &self.stereo_solve {
                Some(Ok((s, params))) => {
                    ui.label(limit_label(s.left_rms_px, s.camera_limit_px, "left"));
                    ui.label(limit_label(s.right_rms_px, s.camera_limit_px, "right"));
                    ui.label(format!(
                        "pair {:.3} px, baseline {:.2} mm, {} views",
                        s.stereo_rms_px,
                        1000.0 * s.baseline_m,
                        s.views
                    ));
                    ui.label(format!("written to {}", s.out));
                    ui.label(format!("depth parameters {params}"));
                }
                Some(Err(e)) => {
                    ui.label(
                        egui::RichText::new(format!("solve failed: {e}"))
                            .color(egui::Color32::LIGHT_RED),
                    );
                }
                None => {}
            }
            return;
        }
        for (name, detection) in &self.poses {
            match detection {
                Ok(d) => ui.label(format!("{name}: {} of {} corners", d.corners, d.of)),
                Err(e) => ui.label(format!("{name}: detection failed: {e}")),
            };
        }
        match &self.solve {
            Some(Ok(s)) => {
                ui.label(limit_label(s.camera_rms_px, s.camera_limit_px, "camera"));
                ui.label(limit_label(
                    s.projector_rms_px,
                    s.projector_limit_px,
                    "projector",
                ));
                ui.label(format!(
                    "baseline {:.2} mm, {} views",
                    1000.0 * s.baseline_m,
                    s.views
                ));
                ui.label(format!("written to {}", s.out));
            }
            Some(Err(e)) => {
                ui.label(
                    egui::RichText::new(format!("solve failed: {e}"))
                        .color(egui::Color32::LIGHT_RED),
                );
            }
            None => {}
        }
    }

    /// Keep a left frame until its right arrives, then send the pair to the worker when it
    /// is idle, or count the pair as skipped. A light-off pair is always sent, since the
    /// lit pair after it needs it.
    fn pair_frame(&mut self, name: &str, pixels: Vec<u8>) {
        if name.ends_with("/left") {
            self.pending_left = Some(pixels);
            return;
        }
        let (Some(left), Some(worker)) = (self.pending_left.take(), self.depth.as_mut()) else {
            return;
        };
        if name.starts_with("dark_") {
            let _ = worker.jobs.send(DepthJob::Dark(left, pixels));
            return;
        }
        if worker.in_flight == 0 && worker.jobs.send(DepthJob::Pair(left, pixels)).is_ok() {
            worker.in_flight += 1;
        } else {
            self.depth_skipped += 1;
        }
    }

    fn poll_depth(&mut self) {
        let Some(worker) = self.depth.as_mut() else {
            return;
        };
        while let Ok(live) = worker.results.try_recv() {
            worker.in_flight = worker.in_flight.saturating_sub(1);
            let rms = live
                .rms
                .map_or_else(String::new, |r| format!(", ICP residual {:.1} µm", r * 1e6));
            let rms = match &live.dots {
                Some((Ok(found), map)) => format!("{rms}, {found} dots seen, {map} in map"),
                Some((Err(e), _)) => format!("{rms}, dot finding failed: {e}"),
                None => rms,
            };
            let (model, kind) = match live.model {
                Ok(surface) => (surface, "TSDF surface".to_owned()),
                Err((e, fused)) => (fused, format!("fused points (TSDF failed: {e})")),
            };
            self.fusion_status = format!(
                "model: {} {kind}, fused in {:.0} ms{rms}",
                model.len(),
                live.fuse_ms
            );
            self.cloud = model;
            match live.depth {
                Ok(d) => {
                    let total: f64 = d.timings.iter().map(|(_, ms)| ms).sum();
                    let stages: Vec<String> = d
                        .timings
                        .iter()
                        .map(|(name, ms)| format!("{name} {ms:.0}"))
                        .collect();
                    self.depth_status = format!(
                        "{} points in {total:.0} ms ({} ms)",
                        d.points.len(),
                        stages.join(", ")
                    );
                    self.depth_frames += 1;
                }
                Err(e) => self.depth_status = format!("depth frame failed: {e}"),
            }
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let Some(events) = self.events.take() else {
            return;
        };
        let mut done = false;
        while let Ok(event) = events.try_recv() {
            match event {
                Event::Connected(hello) => self.hello = Some(hello),
                Event::Frame {
                    header,
                    name,
                    pixels,
                } => {
                    let size = [usize::from(header.width), usize::from(header.height)];
                    let image = egui::ColorImage::from_gray(size, &pixels);
                    match &mut self.preview {
                        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                        None => {
                            self.preview = Some(ctx.load_texture(
                                "preview",
                                image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    self.frames += 1;
                    if self.set == "stereo" {
                        self.pair_frame(&name, pixels);
                    } else if self.set == "scan" && self.scan_params.is_some() {
                        self.scan_frames.insert(name.clone(), pixels);
                    }
                    self.last = Some((name, header));
                }
                Event::Finished(stats) => {
                    self.status = format!("finished, {} frames", stats.frames);
                    self.stats = Some(stats);
                    done = true;
                    if self.set == "scan" {
                        self.start_scan(ctx);
                    }
                }
                Event::Failed(message) => {
                    self.status = format!("failed: {message}");
                    done = true;
                }
            }
        }
        if !done {
            self.events = Some(events);
        }
    }

    fn configuration(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Configuration");
        ui.label("Capture node");
        ui.text_edit_singleline(&mut self.node);
        ui.label("Pattern set");
        ui.radio_value(&mut self.set, "scan".to_owned(), "Scan");
        ui.radio_value(&mut self.set, "calibration".to_owned(), "Calibration");
        ui.radio_value(&mut self.set, "stereo".to_owned(), "Stereo");
        if self.set == "stereo" {
            ui.add(
                egui::DragValue::new(&mut self.stereo_frames)
                    .range(1..=1000)
                    .prefix("frames "),
            );
            ui.label("Depth parameters");
            ui.text_edit_singleline(&mut self.params_path);
            ui.checkbox(&mut self.dots, "Reference dots (smooth panels)")
                .on_hover_text("Take every 4th pair with the pattern light off and track from the dots seen in it");
        }
        if self.set == "scan" {
            ui.label("Scan parameters")
                .on_hover_text("Written by slscan scan-params from a camera and projector calibration; their pitch and steps replace the ones below");
            ui.text_edit_singleline(&mut self.scan_params_path);
        }
        ui.add(
            egui::DragValue::new(&mut self.pitch)
                .range(4..=128)
                .prefix("pitch "),
        );
        ui.add(
            egui::DragValue::new(&mut self.steps)
                .range(3..=8)
                .prefix("steps "),
        );
        let idle = !self.busy();
        if ui
            .add_enabled(idle, egui::Button::new("Run sequence"))
            .clicked()
        {
            self.start(ctx);
        }
        ui.label(&self.status);
        if let Some(hello) = &self.hello {
            ui.separator();
            ui.label(format!("Node: {}", hello.node));
            let p = &hello.projector;
            ui.label(format!(
                "Projector: {} × {} at {} Hz",
                p.width, p.height, p.refresh_hz
            ));
            for camera in &hello.cameras {
                ui.label(format!(
                    "Camera {}: {} × {} {}",
                    camera.name, camera.width, camera.height, camera.format
                ));
            }
        }
        self.calibration_panel(ui, ctx);
    }

    fn preview(&self, ui: &mut egui::Ui) {
        ui.heading("Preview and processing");
        if let Some((name, header)) = &self.last {
            ui.label(format!(
                "{name}: pattern {}, exposure start {:.3} ms, {} frames received",
                header.index,
                header.timestamp_ns as f64 / 1.0e6,
                self.frames
            ));
        }
        if let Some(stats) = &self.stats {
            let timing = stats.mean_interval_ns.map_or_else(
                || "no interval".to_owned(),
                |ns| format!("mean interval {:.3} ms ({:.1} Hz)", ns / 1.0e6, 1.0e9 / ns),
            );
            ui.label(format!(
                "missing {:?}, in order {}, pairs in step {}, {timing}",
                stats.missing, stats.in_order, stats.in_step
            ));
        }
        if !self.depth_status.is_empty() {
            ui.label(&self.depth_status);
        }
        if !self.fusion_status.is_empty() {
            ui.label(&self.fusion_status);
        }
        if !self.scan_status.is_empty() {
            ui.label(&self.scan_status);
        }
        if let Some(texture) = &self.preview {
            ui.add(egui::Image::new(texture).shrink_to_fit());
        }
    }

    /// Mesh the live volume in the depth worker and write it from a thread, so the
    /// window does not wait on the file.
    fn export_mesh(&mut self, ctx: &egui::Context) {
        let Some(worker) = &self.depth else {
            self.export_status = "no live volume: run a stereo sequence first".to_owned();
            return;
        };
        let out = PathBuf::from(&self.mesh_path);
        let (mesh_tx, mesh_rx) = mpsc::channel();
        let job = DepthJob::Mesh(
            self.mesh_fill_mm / 1000.0,
            self.mesh_faces as usize,
            mesh_tx,
        );
        if worker.jobs.send(job).is_err() {
            self.export_status = "depth worker stopped".to_owned();
            return;
        }
        let (tx, rx) = mpsc::channel();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = mesh_rx
                .recv()
                .map_err(|e| e.to_string())
                .and_then(|m| m)
                .and_then(|m| {
                    mesh::write_mesh(&out, &m)
                        .map(|()| (m.faces.len(), m.vertices.len()))
                        .map_err(|e| format!("{}: {e}", out.display()))
                });
            let _ = tx.send(result);
            repaint.request_repaint();
        });
        self.export = Some(rx);
        self.export_status = "exporting".to_owned();
    }

    fn poll_export(&mut self) {
        let Some(rx) = &self.export else {
            return;
        };
        if let Ok(result) = rx.try_recv() {
            self.export_status = match result {
                Ok((triangles, vertices)) => format!(
                    "{triangles} triangles and {vertices} vertices written to {}",
                    self.mesh_path
                ),
                Err(e) => format!("export failed: {e}"),
            };
            self.export = None;
        }
    }

    /// Draw the fused model or the last scan from the front, shaded by depth, export the
    /// model as a mesh and save the points.
    fn scene(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Scene");
        ui.horizontal(|ui| {
            ui.label("Mesh");
            ui.text_edit_singleline(&mut self.mesh_path);
        });
        ui.add(
            egui::DragValue::new(&mut self.mesh_fill_mm)
                .range(0.0..=100.0)
                .prefix("fill holes up to ")
                .suffix(" mm"),
        );
        ui.add(
            egui::DragValue::new(&mut self.mesh_faces)
                .range(0..=10_000_000)
                .prefix("triangles at most ")
                .suffix(" (0 keeps all)"),
        );
        let ready = self.depth.is_some() && self.export.is_none();
        if ui
            .add_enabled(ready, egui::Button::new("Export mesh"))
            .clicked()
        {
            self.export_mesh(ctx);
        }
        if !self.export_status.is_empty() {
            ui.label(&self.export_status);
        }
        ui.horizontal(|ui| {
            ui.label("Points");
            ui.text_edit_singleline(&mut self.points_path);
        });
        if ui
            .add_enabled(!self.cloud.is_empty(), egui::Button::new("Save points"))
            .clicked()
        {
            self.save_points();
        }
        if self.cloud.is_empty() {
            ui.label("No model yet.");
            return;
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in &self.cloud {
            for (i, v) in p.iter().enumerate() {
                lo[i] = lo[i].min(*v);
                hi[i] = hi[i].max(*v);
            }
        }
        ui.label(format!(
            "{} points, depth {:.3} to {:.3} m, {} frames, {} skipped",
            self.cloud.len(),
            lo[2],
            hi[2],
            self.depth_frames,
            self.depth_skipped
        ));
        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
        let rect = response.rect.shrink(4.0);
        let span = [0, 1, 2].map(|i| (hi[i] - lo[i]).max(1e-9));
        let scale = (f64::from(rect.width()) / span[0]).min(f64::from(rect.height()) / span[1]);
        let stride = self.cloud.len().div_ceil(MAX_DRAWN).max(1);
        for p in self.cloud.iter().step_by(stride) {
            let x = rect.left() + ((p[0] - lo[0]) * scale) as f32;
            let y = rect.top() + ((p[1] - lo[1]) * scale) as f32;
            let shade = (255.0 - 160.0 * (p[2] - lo[2]) / span[2]) as u8;
            painter.circle_filled(egui::pos2(x, y), 1.5, egui::Color32::from_gray(shade));
        }
    }
}

impl eframe::App for ScanApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.poll_calibration();
        self.poll_depth();
        self.poll_export();
        self.poll_scan();
        egui::Panel::left("configuration")
            .default_size(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.configuration(ui, &ctx));
            });
        egui::Panel::right("scene")
            .default_size(240.0)
            .show_inside(ui, |ui| self.scene(ui, &ctx));
        egui::CentralPanel::default().show_inside(ui, |ui| self.preview(ui));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::kittest::Queryable;
    use egui_kittest::Harness;
    use std::time::Instant;

    /// Run a scan set from the stand-in at `SLS_WINDOW_NODE` through the window with the
    /// scan parameters at `SLS_WINDOW_SCAN_PARAMS`, save its points to `SLS_WINDOW_OUT`
    /// and compare them with `sls-scan scan`'s points at `SLS_WINDOW_REFERENCE`. Skipped
    /// unless `SLS_WINDOW_NODE` is set, as the application gate sets it.
    #[test]
    fn window_scan_matches_the_headless_scan() -> Result<(), String> {
        let Ok(node) = std::env::var("SLS_WINDOW_NODE") else {
            return Ok(());
        };
        let var = |name: &str| std::env::var(name).map_err(|_| format!("{name} not set"));
        let (params, out, reference) = (
            var("SLS_WINDOW_SCAN_PARAMS")?,
            var("SLS_WINDOW_OUT")?,
            var("SLS_WINDOW_REFERENCE")?,
        );
        let mut harness = Harness::builder()
            .with_size(egui::vec2(1600.0, 1200.0))
            .build_eframe(|_| ScanApp::default());
        {
            let app = harness.state_mut();
            app.node = node;
            app.scan_params_path = params;
            app.points_path = out.clone();
        }
        harness.run_steps(2);
        harness.get_by_label("Run sequence").click();
        let start = Instant::now();
        loop {
            harness.run_steps(1);
            let app = harness.state();
            if !app.busy() && !app.cloud.is_empty() {
                break;
            }
            if start.elapsed() > Duration::from_secs(180) {
                return Err(format!(
                    "no scan after 180 s: {}, {}",
                    app.status, app.scan_status
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
        harness.get_by_label("Save points").click();
        harness.run_steps(2);
        let saved = crate::register::read_ply(Path::new(&out)).map_err(|e| e.to_string())?;
        let expected =
            crate::register::read_ply(Path::new(&reference)).map_err(|e| e.to_string())?;
        println!(
            "window scan: {} points, {}",
            saved.len(),
            harness.state().scan_status
        );
        if saved != expected {
            return Err(format!(
                "window saved {} points, headless scan {}",
                saved.len(),
                expected.len()
            ));
        }
        Ok(())
    }
}
