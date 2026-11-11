//! Live depth frames in two stages on two threads: one measures a stereo pair on the GPU
//! while the other fuses the pair before it into the model and the TSDF volume.

use crate::coarse::CoarseParams;
use crate::depth::{DepthFrame, DepthResult, DepthSetup};
use crate::fuse::{self, Fusion, Pose, Vec3};
use crate::mesh::Mesh;
use crate::tsdf;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Voxel size of the fused model, as `track.fuse` uses by default.
pub const FUSION_VOXEL_M: f64 = 0.0005;

/// Lit pairs a caller that sends every pair keeps in a pipeline at once: one in each
/// stage and one waiting. With light-off pairs it adds `WAITING_AT_MOST`.
pub const IN_FLIGHT: usize = 3;

/// Measured lit pairs the fusing stage keeps for a later light-off pair at most. Beyond
/// that it fuses the oldest from the light-off poses there are.
pub const WAITING_AT_MOST: usize = 8;

pub enum Job {
    /// A lit stereo pair: its name, its capture time in seconds, and its left and right
    /// raw images.
    Pair(String, f64, Vec<u8>, Vec<u8>),
    /// A pair taken with the pattern light off, kept until the next lit pair's points
    /// place its dots.
    Dark(String, f64, Vec<u8>, Vec<u8>),
    /// Start again with no model and no volume.
    Reset,
    /// Fuse the lit pairs still waiting for a light-off pair, as at the end of a stream.
    Flush,
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
    /// The pair was offered while the measuring stage was busy, and waited for it.
    pub held: bool,
    /// Offered pairs passed over for this newer one.
    pub skipped: usize,
    /// Milliseconds from the pair's arrival at the pipeline to this result.
    pub latency_ms: f64,
    pub depth: Result<DepthResult, String>,
    /// Isolated points left out of fusion.
    pub dropped: usize,
    pub dark: Option<Dark>,
    /// Milliseconds in fusion and in the TSDF volume.
    pub fuse_ms: f64,
    pub tsdf_ms: f64,
    pub rms: Option<f64>,
    /// Where fusion placed the pair in the model's frame.
    pub pose: Option<Pose>,
    /// Points in the fused model.
    pub fused: usize,
    /// The TSDF surface, or the fused points when the TSDF failed.
    pub model: Result<Vec<Vec3>, (String, Vec<Vec3>)>,
}

/// A lit pair on its way to the measuring stage.
struct Arrival {
    name: String,
    time: f64,
    left: Vec<u8>,
    right: Vec<u8>,
    arrived: Instant,
    /// Resets sent before this pair.
    epoch: usize,
    held: bool,
}

/// A light-off pair kept for the lit pair after it.
struct DarkPair {
    name: String,
    time: f64,
    epoch: usize,
    left: Vec<u8>,
    right: Vec<u8>,
}

/// Offered pairs that the measuring stage has not taken.
#[derive(Default)]
struct Waiting {
    /// The newest offered lit pair.
    lit: Option<Arrival>,
    /// The newest offered light-off pair.
    dark: Option<DarkPair>,
    /// `dark` was offered after `lit`, so it goes with a later lit pair.
    dark_after_lit: bool,
    /// Offered lit pairs replaced by a newer one since a pair was last taken.
    replaced: usize,
    /// Offered light-off pairs replaced by a newer one.
    unused_dark: usize,
}

impl Waiting {
    /// Keep a lit pair in place of one still waiting. True when none was waiting.
    fn offer(&mut self, pair: Arrival) -> bool {
        let first = self.lit.replace(pair).is_none();
        self.replaced += usize::from(!first);
        self.dark_after_lit = false;
        first
    }

    /// Keep a light-off pair in place of one still waiting.
    fn offer_dark(&mut self, pair: DarkPair) {
        self.unused_dark += usize::from(self.dark.replace(pair).is_some());
        self.dark_after_lit = self.lit.is_some();
    }

    /// The waiting lit pair, the light-off pair offered before it, and the count of lit
    /// pairs it replaced.
    fn take(&mut self) -> Option<(Arrival, Option<DarkPair>, usize)> {
        let lit = self.lit.take()?;
        let dark = if self.dark_after_lit {
            None
        } else {
            self.dark.take()
        };
        self.dark_after_lit = false;
        Some((lit, dark, std::mem::take(&mut self.replaced)))
    }

    /// Drop both waiting pairs, as a reset does.
    fn clear(&mut self) {
        self.lit = None;
        self.dark = None;
        self.dark_after_lit = false;
        self.replaced = 0;
    }
}

/// What callers and the measuring stage share.
#[derive(Default)]
struct Intake {
    waiting: Mutex<Waiting>,
    busy: AtomicBool,
    resets: AtomicUsize,
}

enum Message {
    /// A job with its arrival time and the resets sent before it.
    Job(Job, Instant, usize),
    /// A pair was offered.
    Offered,
}

struct Measured {
    name: String,
    time: f64,
    held: bool,
    skipped: usize,
    arrived: Instant,
    epoch: usize,
    depth: Result<DepthResult, String>,
    /// Points to fuse, when the pair was measured in full.
    points: Option<Vec<Vec3>>,
    /// Name, capture time and dots of a light-off pair before this one.
    dark: Option<(String, f64, Result<Vec<Vec3>, String>)>,
}

/// A measured lit pair in the fusing stage, until it is fused and its result sent.
struct Unfused {
    live: LiveFrame,
    time: f64,
    arrived: Instant,
    /// Points to fuse, when the pair was measured in full.
    points: Option<Vec<Vec3>>,
}

enum Fuse {
    Frame(Measured),
    /// Start again; the count is of resets up to this one.
    Reset(usize),
    Flush,
    Mesh(f64, usize, Sender<Result<Mesh, String>>),
}

/// What a pipeline leaves when it closes.
pub struct Finished {
    pub fusion: Fusion,
    pub volume: tsdf::Live,
    /// Light-off pairs that no measured lit pair followed.
    pub unused_dark: usize,
    /// Results not yet taken from `results`.
    pub rest: Vec<LiveFrame>,
}

pub struct Pipeline {
    /// None once the pipeline is closed.
    jobs: Option<Sender<Message>>,
    intake: Arc<Intake>,
    /// One `LiveFrame` per lit pair measured, in order. A pair measured before a reset
    /// and fused after it has none.
    pub results: Receiver<LiveFrame>,
    /// Until they are joined.
    stages: Option<Stages>,
}

/// The measuring thread, which returns its unused light-off pairs, and the fusing thread.
struct Stages {
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
        let (jobs, jobs_rx) = mpsc::channel::<Message>();
        let (measured_tx, measured_rx) = mpsc::sync_channel::<Fuse>(0);
        let (results_tx, results) = mpsc::channel();
        let intake = Arc::new(Intake::default());
        let shared = Arc::clone(&intake);
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
            let mut pending: Option<DarkPair> = None;
            let mut unused = 0;
            for message in jobs_rx {
                let (pair, skipped) = match message {
                    Message::Offered => {
                        let taken = shared.waiting.lock().ok().and_then(|mut w| w.take());
                        let Some((pair, dark, skipped)) = taken else {
                            continue;
                        };
                        if let Some(dark) = dark {
                            unused += usize::from(pending.is_some());
                            pending = Some(dark);
                        }
                        (pair, skipped)
                    }
                    Message::Job(Job::Pair(name, time, left, right), arrived, epoch) => {
                        let pair = Arrival {
                            name,
                            time,
                            left,
                            right,
                            arrived,
                            epoch,
                            held: false,
                        };
                        (pair, 0)
                    }
                    Message::Job(Job::Dark(name, time, left, right), _, epoch) => {
                        unused += usize::from(pending.is_some());
                        pending = Some(DarkPair {
                            name,
                            time,
                            epoch,
                            left,
                            right,
                        });
                        continue;
                    }
                    Message::Job(Job::Reset, _, epoch) => {
                        pending = None;
                        if measured_tx.send(Fuse::Reset(epoch)).is_err() {
                            break;
                        }
                        continue;
                    }
                    Message::Job(Job::Flush, ..) => {
                        if measured_tx.send(Fuse::Flush).is_err() {
                            break;
                        }
                        continue;
                    }
                    Message::Job(Job::Mesh(fill_m, faces, reply), ..) => {
                        if measured_tx.send(Fuse::Mesh(fill_m, faces, reply)).is_err() {
                            break;
                        }
                        continue;
                    }
                };
                shared.busy.store(true, Ordering::Release);
                let depth =
                    frame.measure(&params, preview.as_ref(), cameras, &pair.left, &pair.right);
                let points = match (&depth, &preview) {
                    (Ok(d), None) => Some(fuse::drop_isolated(
                        &d.points,
                        f64::from(params.step),
                        focal,
                        fuse::ISOLATED_SPACINGS,
                    )),
                    _ => None,
                };
                if pending.as_ref().is_some_and(|d| d.epoch != pair.epoch) {
                    pending = None;
                }
                let dark = points.as_ref().and_then(|points| {
                    pending.take().map(|d| {
                        let dots = frame.dots(&params, cameras, &d.left, &d.right, points);
                        (d.name, d.time, dots)
                    })
                });
                let sent = measured_tx.send(Fuse::Frame(Measured {
                    name: pair.name,
                    time: pair.time,
                    held: pair.held,
                    skipped,
                    arrived: pair.arrived,
                    epoch: pair.epoch,
                    depth,
                    points,
                    dark,
                }));
                shared.busy.store(false, Ordering::Release);
                if sent.is_err() {
                    break;
                }
            }
            let waiting = shared
                .waiting
                .lock()
                .map_or(0, |w| w.unused_dark + usize::from(w.dark.is_some()));
            unused + usize::from(pending.is_some()) + waiting
        });
        let fusing = thread::spawn(move || {
            let mut fusion = Fusion::new(FUSION_VOXEL_M);
            let mut volume = tsdf::Live::coarse();
            let size = (params.coarse.width, params.coarse.height);
            let mut epoch = 0;
            let mut waiting: VecDeque<Unfused> = VecDeque::new();
            // Fuse a pair that has points, and send its result. False once nobody takes
            // results.
            let complete = |fusion: &mut Fusion, volume: &mut tsdf::Live, pair: Unfused| {
                let mut live = pair.live;
                if let Some(points) = &pair.points {
                    let start = Instant::now();
                    let pose = fusion.add_at(pair.time, points, &[]);
                    live.fuse_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let start = Instant::now();
                    let grid = tsdf::depth_grid(points, &camera, params.step, params.window, size);
                    let rotation: [f64; 9] = std::array::from_fn(|k| pose.rotation[k / 3][k % 3]);
                    let surface =
                        volume.add(&gpu, &grid, points, &camera, (&rotation, &pose.translation));
                    live.tsdf_ms = start.elapsed().as_secs_f64() * 1000.0;
                    live.rms = fusion.last_rms;
                    live.pose = Some(pose);
                    live.fused = fusion.model().len();
                    live.model = surface.map_err(|e| (e, fusion.model().to_vec()));
                }
                live.latency_ms = pair.arrived.elapsed().as_secs_f64() * 1000.0;
                let sent = results_tx.send(live).is_ok();
                if sent {
                    notify();
                }
                sent
            };
            // Complete waiting pairs from the oldest on: down to `keep` of them, then those
            // that no later light-off pair would start differently, and those captured
            // before `passed`, which two light-off pairs have followed. After each one,
            // take what the measuring stage has ready into `inbox`, so that stage goes on.
            let release = |fusion: &mut Fusion,
                           volume: &mut tsdf::Live,
                           waiting: &mut VecDeque<Unfused>,
                           inbox: &mut VecDeque<Fuse>,
                           (keep, passed): (usize, f64)| {
                while let Some(front) = waiting.front() {
                    let ready =
                        front.points.is_none() || front.time < passed || !fusion.waits(front.time);
                    if waiting.len() <= keep && !ready {
                        break;
                    }
                    let done = waiting
                        .pop_front()
                        .is_none_or(|pair| complete(fusion, volume, pair));
                    if !done {
                        return false;
                    }
                    while inbox.len() < WAITING_AT_MOST {
                        match measured_rx.try_recv() {
                            Ok(message) => inbox.push_back(message),
                            Err(_) => break,
                        }
                    }
                }
                true
            };
            let mut inbox: VecDeque<Fuse> = VecDeque::new();
            // Capture times of the last two light-off pairs to come through, posed or not.
            let mut dark_times = [f64::NEG_INFINITY; 2];
            while let Some(message) = inbox.pop_front().or_else(|| measured_rx.recv().ok()) {
                let measured = match message {
                    Fuse::Reset(count) => {
                        if count > epoch {
                            epoch = count;
                            fusion = Fusion::new(FUSION_VOXEL_M);
                            volume.reset();
                            waiting.clear();
                            dark_times = [f64::NEG_INFINITY; 2];
                        }
                        continue;
                    }
                    Fuse::Flush => {
                        let all = (0, f64::INFINITY);
                        if !release(&mut fusion, &mut volume, &mut waiting, &mut inbox, all) {
                            break;
                        }
                        continue;
                    }
                    Fuse::Mesh(fill_m, faces, reply) => {
                        let _ = reply.send(volume.mesh(fill_m, faces));
                        continue;
                    }
                    Fuse::Frame(measured) => measured,
                };
                if measured.epoch < epoch {
                    continue;
                }
                if measured.epoch > epoch {
                    epoch = measured.epoch;
                    fusion = Fusion::new(FUSION_VOXEL_M);
                    volume.reset();
                    waiting.clear();
                    dark_times = [f64::NEG_INFINITY; 2];
                }
                let mut pair = Unfused {
                    live: LiveFrame {
                        name: measured.name,
                        held: measured.held,
                        skipped: measured.skipped,
                        latency_ms: 0.0,
                        dropped: 0,
                        dark: None,
                        fuse_ms: 0.0,
                        tsdf_ms: 0.0,
                        rms: None,
                        pose: None,
                        fused: 0,
                        model: Ok(Vec::new()),
                        depth: measured.depth,
                    },
                    time: measured.time,
                    arrived: measured.arrived,
                    points: None,
                };
                if let (Ok(d), Some(points)) = (&pair.live.depth, measured.points) {
                    pair.live.dropped = d.points.len() - points.len();
                    pair.live.dark = measured.dark.map(|(name, at, found)| {
                        dark_times = [dark_times[1], at];
                        let before = fusion.dots().len();
                        let found = found.map(|dots| (dots.len(), fusion.add_dark_at(at, &dots)));
                        Dark {
                            name,
                            found,
                            mapped: (before, fusion.dots().len()),
                        }
                    });
                    pair.points = Some(points);
                }
                waiting.push_back(pair);
                let some = (WAITING_AT_MOST, dark_times[0]);
                if !release(&mut fusion, &mut volume, &mut waiting, &mut inbox, some) {
                    break;
                }
            }
            let all = (0, f64::INFINITY);
            release(&mut fusion, &mut volume, &mut waiting, &mut inbox, all);
            (fusion, volume)
        });
        Self {
            jobs: Some(jobs),
            intake,
            results,
            stages: Some(Stages { measuring, fusing }),
        }
    }

    /// Queue a job, to be done in order. False when the pipeline has stopped.
    pub fn send(&self, job: Job) -> bool {
        let mut epoch = self.intake.resets.load(Ordering::Acquire);
        if matches!(job, Job::Reset) {
            epoch = self.intake.resets.fetch_add(1, Ordering::AcqRel) + 1;
            if let Ok(mut waiting) = self.intake.waiting.lock() {
                waiting.clear();
            }
        }
        self.jobs
            .as_ref()
            .is_some_and(|jobs| jobs.send(Message::Job(job, Instant::now(), epoch)).is_ok())
    }

    /// Offer a lit pair, as `Job::Pair` names it, to be measured once the measuring stage
    /// is free. It takes the place of an offered pair still waiting, so that stage always
    /// finds the newest one. False when the pipeline has stopped.
    pub fn offer(&self, name: String, time: f64, left: Vec<u8>, right: Vec<u8>) -> bool {
        let pair = Arrival {
            name,
            time,
            left,
            right,
            arrived: Instant::now(),
            epoch: self.intake.resets.load(Ordering::Acquire),
            held: self.intake.busy.load(Ordering::Acquire),
        };
        let Ok(mut waiting) = self.intake.waiting.lock() else {
            return false;
        };
        let first = waiting.offer(pair);
        drop(waiting);
        !first
            || self
                .jobs
                .as_ref()
                .is_some_and(|jobs| jobs.send(Message::Offered).is_ok())
    }

    /// Offer a light-off pair, as `Job::Dark` names it, among offered lit pairs. It goes
    /// with the next lit pair offered after it that the measuring stage takes, and takes
    /// the place of an offered light-off pair still waiting.
    pub fn offer_dark(&self, name: String, time: f64, left: Vec<u8>, right: Vec<u8>) {
        let pair = DarkPair {
            name,
            time,
            epoch: self.intake.resets.load(Ordering::Acquire),
            left,
            right,
        };
        if let Ok(mut waiting) = self.intake.waiting.lock() {
            waiting.offer_dark(pair);
        }
    }

    /// Take no more jobs, wait for both stages to finish those queued, and return the
    /// light-off pairs left unused, the model and the volume. None when already closed.
    fn close(&mut self) -> Option<Result<(usize, Fusion, tsdf::Live), String>> {
        self.jobs = None;
        let stages = self.stages.take()?;
        let unused = stages
            .measuring
            .join()
            .map_err(|_| "the measuring stage panicked".to_owned());
        let fused = stages
            .fusing
            .join()
            .map_err(|_| "the fusing stage panicked".to_owned());
        Some(unused.and_then(|unused| fused.map(|(fusion, volume)| (unused, fusion, volume))))
    }

    /// Close the pipeline once every queued job is done, and return its model and volume.
    pub fn finish(mut self) -> Result<Finished, String> {
        let (unused_dark, fusion, volume) = self.close().ok_or("the pipeline is closed")??;
        Ok(Finished {
            fusion,
            volume,
            unused_dark,
            rest: self.results.try_iter().collect(),
        })
    }
}

impl Drop for Pipeline {
    /// Wait for both stages, so no thread still uses the device once its owner goes on.
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(name: &str) -> Arrival {
        Arrival {
            name: name.to_owned(),
            time: 0.0,
            left: Vec::new(),
            right: Vec::new(),
            arrived: Instant::now(),
            epoch: 0,
            held: false,
        }
    }

    fn dark(name: &str) -> DarkPair {
        DarkPair {
            name: name.to_owned(),
            time: 0.0,
            epoch: 0,
            left: Vec::new(),
            right: Vec::new(),
        }
    }

    fn names(taken: Option<(Arrival, Option<DarkPair>, usize)>) -> Option<(String, String, usize)> {
        taken.map(|(l, d, n)| (l.name, d.map_or_else(String::new, |d| d.name), n))
    }

    fn took(lit: &str, dark: &str, replaced: usize) -> Option<(String, String, usize)> {
        Some((lit.to_owned(), dark.to_owned(), replaced))
    }

    #[test]
    fn newest_lit_pair_replaces_one_still_waiting() {
        let mut waiting = Waiting::default();
        assert!(waiting.offer(lit("1")));
        assert!(!waiting.offer(lit("2")));
        assert!(!waiting.offer(lit("3")));
        assert_eq!(names(waiting.take()), took("3", "", 2));
        assert!(waiting.take().is_none());
        assert!(waiting.offer(lit("4")));
        assert_eq!(names(waiting.take()), took("4", "", 0));
    }

    #[test]
    fn light_off_pair_goes_with_the_lit_pair_after_it() {
        let mut waiting = Waiting::default();
        waiting.offer_dark(dark("0"));
        assert!(waiting.take().is_none());
        waiting.offer(lit("1"));
        waiting.offer(lit("2"));
        assert_eq!(names(waiting.take()), took("2", "0", 1));
        waiting.offer(lit("3"));
        waiting.offer_dark(dark("4"));
        assert_eq!(names(waiting.take()), took("3", "", 0));
        assert!(waiting.take().is_none());
        waiting.offer(lit("5"));
        assert_eq!(names(waiting.take()), took("5", "4", 0));
    }

    #[test]
    fn light_off_pair_waits_for_a_lit_pair_newer_than_one_it_followed() {
        let mut waiting = Waiting::default();
        waiting.offer(lit("3"));
        waiting.offer_dark(dark("4"));
        waiting.offer(lit("5"));
        assert_eq!(names(waiting.take()), took("5", "4", 1));
    }

    #[test]
    fn newer_light_off_pair_replaces_one_still_waiting() {
        let mut waiting = Waiting::default();
        waiting.offer_dark(dark("0"));
        waiting.offer(lit("1"));
        waiting.offer_dark(dark("4"));
        assert_eq!(waiting.unused_dark, 1);
        assert_eq!(names(waiting.take()), took("1", "", 0));
        waiting.offer(lit("5"));
        assert_eq!(names(waiting.take()), took("5", "4", 0));
    }

    #[test]
    fn reset_drops_waiting_pairs() {
        let mut waiting = Waiting::default();
        waiting.offer_dark(dark("0"));
        waiting.offer(lit("1"));
        waiting.offer(lit("2"));
        waiting.clear();
        assert!(waiting.take().is_none());
        assert!(waiting.offer(lit("3")));
        assert_eq!(names(waiting.take()), took("3", "", 0));
    }
}
