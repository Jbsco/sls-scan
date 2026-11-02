//! The three panels: configuration, preview and processing, and the scene.

use crate::client::{self, Event, Stats};
use crate::protocol::{FrameHeader, Hello, SequenceRequest};
use eframe::egui;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

pub struct ScanApp {
    node: String,
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
        let idle = self.events.is_none();
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
        egui::Panel::left("configuration")
            .default_size(260.0)
            .show_inside(ui, |ui| self.configuration(ui, &ctx));
        egui::Panel::right("scene")
            .default_size(240.0)
            .show_inside(ui, Self::scene);
        egui::CentralPanel::default().show_inside(ui, |ui| self.preview(ui));
    }
}
