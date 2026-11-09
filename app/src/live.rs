//! Live depth frames in two stages on two threads: one measures a stereo pair on the GPU
//! while the other fuses the pair before it into the model and the TSDF volume.

use crate::coarse::CoarseParams;
use crate::depth::{DepthFrame, DepthResult, DepthSetup};
use crate::fuse::{self, Fusion, Pose, Vec3};
use crate::mesh::Mesh;
use crate::tsdf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Voxel size of the fused model, as `track.fuse` uses by default.
pub const FUSION_VOXEL_M: f64 = 0.0005;

/// Lit pairs to keep in a pipeline at once: one in each stage and one waiting.
pub const IN_FLIGHT: usize = 3;

pub enum Job {
    /// A lit stereo pair, by name: left and right raw images.
    Pair(String, Vec<u8>, Vec<u8>),
    /// A pair taken with the pattern light off, kept until the next lit pair's points
    /// place its dots.
    Dark(String, Vec<u8>, Vec<u8>),
    /// Start again with no model and no volume.
    Reset,
    /// Mesh the live volume, closing holes up to a size in metres and decimating to a
    /// triangle count when it is not 0, and reply with the mesh.
    Mesh(f64, usize, Sender<Result<Mesh, String>>),
}

/// A light-off pair's dots against the lit pair after it.
pub struct Dark {
    pub name: String,
    /// Dots found and the pose they gave.
    pub found: Result<(usize, Pose), String>,
    /// Dots in the map before and after this pair.
    pub mapped: (usize, usize),
}

/// One lit pair, measured and fused.
pub struct LiveFrame {
    pub name: String,
    pub depth: Result<DepthResult, String>,
    /// Isolated points left out of fusion.
    pub dropped: usize,
    pub dark: Option<Dark>,
    /// Milliseconds in fusion and in the TSDF volume.
    pub fuse_ms: f64,
    pub tsdf_ms: f64,
    pub rms: Option<f64>,
    /// Points in the fused model.
    pub fused: usize,
    /// The TSDF surface, or the fused points when the TSDF failed.
    pub model: Result<Vec<Vec3>, (String, Vec<Vec3>)>,
}

struct Measured {
    name: String,
    depth: Result<DepthResult, String>,
    /// Points to fuse, when the pair was measured in full.
    points: Option<Vec<Vec3>>,
    dark: Option<(String, Result<Vec<Vec3>, String>)>,
}

enum Fuse {
    Frame(Measured),
    Reset,
    Mesh(f64, usize, Sender<Result<Mesh, String>>),
}

/// What a pipeline leaves when it closes.
pub struct Finished {
    pub fusion: Fusion,
    pub volume: tsdf::Live,
    /// Light-off pairs that no measured lit pair followed.
    pub unused_dark: usize,
}

pub struct Pipeline {
    jobs: Sender<Job>,
    /// One `LiveFrame` per lit pair, in the order sent.
    pub results: Receiver<LiveFrame>,
    measuring: JoinHandle<usize>,
    fusing: JoinHandle<(Fusion, tsdf::Live)>,
}

impl Pipeline {
    /// Start both stages. With `preview`, pairs are measured by block disparity alone and
    /// not fused. `notify` runs after each result is sent.
    pub fn start(
        setup: &DepthSetup,
        frame: DepthFrame,
        preview: Option<CoarseParams>,
        notify: impl Fn() + Send + 'static,
    ) -> Self {
        let (jobs, jobs_rx) = mpsc::channel::<Job>();
        let (measured_tx, measured_rx) = mpsc::channel::<Fuse>();
        let (results_tx, results) = mpsc::channel();
        let params = setup.params();
        let camera = tsdf::LeftCamera {
            rotation: setup.cameras.left.rotation,
            projection: setup.cameras.left.projection,
        };
        let gpu = frame.gpu().clone();
        let focal = camera.projection[0];
        let (measure_params, cameras) = (params.clone(), setup.cameras.clone());
        let measuring = thread::spawn(move || {
            let params = measure_params;
            let cameras = (&cameras.left, &cameras.right);
            let mut pending: Option<(String, Vec<u8>, Vec<u8>)> = None;
            let mut unused = 0;
            for job in jobs_rx {
                let message = match job {
                    Job::Dark(name, left, right) => {
                        unused += usize::from(pending.is_some());
                        pending = Some((name, left, right));
                        continue;
                    }
                    Job::Reset => {
                        pending = None;
                        Fuse::Reset
                    }
                    Job::Mesh(fill_m, faces, reply) => Fuse::Mesh(fill_m, faces, reply),
                    Job::Pair(name, left, right) => {
                        let depth =
                            frame.measure(&params, preview.as_ref(), cameras, &left, &right);
                        let points = match (&depth, &preview) {
                            (Ok(d), None) => Some(fuse::drop_isolated(
                                &d.points,
                                f64::from(params.step),
                                focal,
                                fuse::ISOLATED_SPACINGS,
                            )),
                            _ => None,
                        };
                        let dark = points.as_ref().and_then(|points| {
                            pending.take().map(|(dark, l, r)| {
                                (dark, frame.dots(&params, cameras, &l, &r, points))
                            })
                        });
                        Fuse::Frame(Measured {
                            name,
                            depth,
                            points,
                            dark,
                        })
                    }
                };
                if measured_tx.send(message).is_err() {
                    break;
                }
            }
            unused + usize::from(pending.is_some())
        });
        let fusing = thread::spawn(move || {
            let mut fusion = Fusion::new(FUSION_VOXEL_M);
            let mut volume = tsdf::Live::coarse();
            let size = (params.coarse.width, params.coarse.height);
            for message in measured_rx {
                let measured = match message {
                    Fuse::Reset => {
                        fusion = Fusion::new(FUSION_VOXEL_M);
                        volume.reset();
                        continue;
                    }
                    Fuse::Mesh(fill_m, faces, reply) => {
                        let _ = reply.send(volume.mesh(fill_m, faces));
                        continue;
                    }
                    Fuse::Frame(measured) => measured,
                };
                let mut live = LiveFrame {
                    name: measured.name,
                    dropped: 0,
                    dark: None,
                    fuse_ms: 0.0,
                    tsdf_ms: 0.0,
                    rms: None,
                    fused: 0,
                    model: Ok(Vec::new()),
                    depth: measured.depth,
                };
                if let (Ok(d), Some(points)) = (&live.depth, &measured.points) {
                    live.dropped = d.points.len() - points.len();
                    live.dark = measured.dark.map(|(name, found)| {
                        let before = fusion.dots().len();
                        let found = found.map(|dots| (dots.len(), fusion.add_dark(&dots)));
                        Dark {
                            name,
                            found,
                            mapped: (before, fusion.dots().len()),
                        }
                    });
                    let start = Instant::now();
                    fusion.add(points);
                    live.fuse_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let start = Instant::now();
                    let grid = tsdf::depth_grid(points, &camera, params.step, params.window, size);
                    let surface = match fusion.poses.last() {
                        Some(pose) => {
                            let rotation: [f64; 9] =
                                std::array::from_fn(|k| pose.rotation[k / 3][k % 3]);
                            volume.add(&gpu, &grid, points, &camera, (&rotation, &pose.translation))
                        }
                        None => Ok(Vec::new()),
                    };
                    live.tsdf_ms = start.elapsed().as_secs_f64() * 1000.0;
                    live.rms = fusion.last_rms;
                    live.fused = fusion.model().len();
                    live.model = surface.map_err(|e| (e, fusion.model().to_vec()));
                }
                if results_tx.send(live).is_err() {
                    break;
                }
                notify();
            }
            (fusion, volume)
        });
        Self {
            jobs,
            results,
            measuring,
            fusing,
        }
    }

    /// Queue a job. False when the pipeline has stopped.
    pub fn send(&self, job: Job) -> bool {
        self.jobs.send(job).is_ok()
    }

    /// Close the pipeline once every queued job is done, and return its model and volume.
    pub fn finish(self) -> Result<Finished, String> {
        drop(self.jobs);
        let unused_dark = self
            .measuring
            .join()
            .map_err(|_| "the measuring stage panicked".to_owned())?;
        let (fusion, volume) = self
            .fusing
            .join()
            .map_err(|_| "the fusing stage panicked".to_owned())?;
        Ok(Finished {
            fusion,
            volume,
            unused_dark,
        })
    }
}
