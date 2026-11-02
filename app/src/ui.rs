//! The three panels: configuration, preview and processing, and the scene.

use crate::calibration::{self, Board, Detection, Solve, Tool};
use crate::client::{self, Event, Stats};
use crate::protocol::{FrameHeader, Hello, Projector, SequenceRequest};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

enum CalibrationEvent {
    Pose(String, Result<Detection, String>, Projector),
    Failed(String),
    Solved(Result<Solve, String>),
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
        }
    }
}

impl ScanApp {
    fn start(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        let node = self.node.clone();
        let request = SequenceRequest {
            set: self.set.clone(),
            pitch: self.pitch,
            steps: self.steps,
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
        self.events.is_some() || self.calibration_events.is_some()
    }

    fn capture_pose(&mut self, ctx: &egui::Context) {
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
        ui.horizontal(|ui| {
            if ui
                .add_enabled(idle, egui::Button::new("Capture pose"))
                .clicked()
            {
                self.capture_pose(ctx);
            }
            let enough = self.poses.len() >= 3 && self.projector.is_some();
            if ui
                .add_enabled(idle && enough, egui::Button::new("Solve"))
                .clicked()
            {
                self.start_solve(ctx);
            }
        });
        for (name, detection) in &self.poses {
            match detection {
                Ok(d) => ui.label(format!("{name}: {} of {} corners", d.corners, d.of)),
                Err(e) => ui.label(format!("{name}: detection failed: {e}")),
            };
        }
        match &self.solve {
            Some(Ok(s)) => {
                let limit = |value: f64, limit: f64, what: &str| {
                    let colour = if value < limit {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    };
                    egui::RichText::new(format!("{what} {value:.3} px (limit {limit})"))
                        .color(colour)
                };
                ui.label(limit(s.camera_rms_px, s.camera_limit_px, "camera"));
                ui.label(limit(s.projector_rms_px, s.projector_limit_px, "projector"));
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

    fn poll(&mut self, ctx: &egui::Context) {
        let Some(events) = &self.events else { return };
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
                    self.last = Some((name, header));
                }
                Event::Finished(stats) => {
                    self.status = format!("finished, {} frames", stats.frames);
                    self.stats = Some(stats);
                    done = true;
                }
                Event::Failed(message) => {
                    self.status = format!("failed: {message}");
                    done = true;
                }
            }
        }
        if done {
            self.events = None;
        }
    }

    fn configuration(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Configuration");
        ui.label("Capture node");
        ui.text_edit_singleline(&mut self.node);
        ui.label("Pattern set");
        ui.radio_value(&mut self.set, "scan".to_owned(), "Scan");
        ui.radio_value(&mut self.set, "calibration".to_owned(), "Calibration");
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
                "missing {:?}, in order {}, {timing}",
                stats.missing, stats.in_order
            ));
        }
        if let Some(texture) = &self.preview {
            ui.add(egui::Image::new(texture).shrink_to_fit());
        }
    }

    fn scene(ui: &mut egui::Ui) {
        ui.heading("Scene");
        ui.label("No point cloud loaded.");
    }
}

impl eframe::App for ScanApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.poll_calibration();
        egui::Panel::left("configuration")
            .default_size(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.configuration(ui, &ctx));
            });
        egui::Panel::right("scene")
            .default_size(240.0)
            .show_inside(ui, Self::scene);
        egui::CentralPanel::default().show_inside(ui, |ui| self.preview(ui));
    }
}
