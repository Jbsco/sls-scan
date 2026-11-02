//! Capture client: runs one pattern sequence on a node and reports each frame.

use crate::protocol::{self, FrameHeader, Hello, SequenceRequest};
use std::io;
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum Event {
    Connected(Hello),
    Frame {
        header: FrameHeader,
        name: String,
        pixels: Vec<u8>,
    },
    Finished(Stats),
    Failed(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub frames: usize,
    pub missing: Vec<usize>,
    pub in_order: bool,
    pub mean_interval_ns: Option<f64>,
    /// Largest step between consecutive distinct timestamps.
    pub largest_gap_ns: Option<f64>,
    /// Whether both frames of every received pair carry one timestamp; true for sets
    /// without pairs.
    pub in_step: bool,
}

impl Stats {
    fn from_frames(expected: usize, indices: &[u32], stamps: &[i64], paired: bool) -> Self {
        let mut seen = vec![false; expected];
        for &i in indices {
            if let Some(slot) = seen.get_mut(i as usize) {
                *slot = true;
            }
        }
        let missing = seen
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (!s).then_some(i))
            .collect();
        let mut distinct = stamps.to_vec();
        distinct.dedup();
        let mean_interval_ns = match (distinct.first(), distinct.last()) {
            (Some(first), Some(last)) if distinct.len() > 1 => {
                Some((last - first) as f64 / (distinct.len() - 1) as f64)
            }
            _ => None,
        };
        let largest_gap_ns = distinct
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .reduce(f64::max);
        let stamp_of: std::collections::HashMap<u32, i64> = indices
            .iter()
            .copied()
            .zip(stamps.iter().copied())
            .collect();
        let in_step = !paired
            || stamp_of
                .iter()
                .filter(|(i, _)| *i % 2 == 0)
                .all(|(i, s)| stamp_of.get(&(i + 1)).is_none_or(|r| r == s));
        Self {
            frames: indices.len(),
            missing,
            in_order: indices.windows(2).all(|w| w[0] < w[1]),
            mean_interval_ns,
            largest_gap_ns,
            in_step,
        }
    }
}

fn try_connect(address: &str) -> io::Result<(TcpStream, Hello)> {
    let mut stream = TcpStream::connect(address)?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let (kind, payload) = protocol::read_message(&mut stream)?;
    if kind != protocol::HELLO {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "node did not open with HELLO",
        ));
    }
    Ok((stream, serde_json::from_slice(&payload)?))
}

/// Connect and read HELLO, retrying for up to `wait` while the node starts.
pub fn connect(address: &str, wait: Duration) -> io::Result<(TcpStream, Hello)> {
    let deadline = Instant::now() + wait;
    loop {
        match try_connect(address) {
            Ok(connected) => return Ok(connected),
            Err(e) if e.kind() != io::ErrorKind::InvalidData && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Connect to a node, run one sequence and pass every event to `on_event`.
pub fn capture(
    address: &str,
    request: &SequenceRequest,
    wait: Duration,
    mut on_event: impl FnMut(Event),
) -> io::Result<Stats> {
    let (mut stream, hello) = connect(address, wait)?;
    let names = protocol::pattern_names(
        &request.set,
        hello.projector.width,
        hello.projector.height,
        request.steps,
        request.frames,
    )
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unknown pattern set"))?;
    on_event(Event::Connected(hello));
    protocol::write_message(
        &mut stream,
        protocol::SEQUENCE,
        &serde_json::to_vec(request)?,
    )?;
    let (mut indices, mut stamps) = (Vec::new(), Vec::new());
    loop {
        let (kind, payload) = protocol::read_message(&mut stream)?;
        match kind {
            protocol::FRAME => {
                let (header, pixels) = protocol::parse_frame(&payload)?;
                indices.push(header.index);
                stamps.push(header.timestamp_ns);
                let name = names
                    .get(header.index as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("frame_{:03}", header.index));
                on_event(Event::Frame {
                    header,
                    name,
                    pixels: pixels.to_vec(),
                });
            }
            protocol::END => break,
            protocol::ERROR => {
                let message = String::from_utf8_lossy(&payload).into_owned();
                return Err(io::Error::other(message));
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected message type {other}"),
                ));
            }
        }
    }
    let stats = Stats::from_frames(names.len(), &indices, &stamps, request.set == "stereo");
    on_event(Event::Finished(stats.clone()));
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_report_missing_and_order() {
        let stats = Stats::from_frames(4, &[0, 1, 3], &[0, 10, 30], false);
        assert_eq!(stats.missing, vec![2]);
        assert!(stats.in_order);
        assert_eq!(stats.mean_interval_ns, Some(15.0));
        assert_eq!(stats.largest_gap_ns, Some(20.0));
        assert!(!Stats::from_frames(2, &[1, 0], &[0, 1], false).in_order);
        let pairs = Stats::from_frames(4, &[0, 1, 2, 3], &[0, 0, 30, 30], true);
        assert_eq!(pairs.mean_interval_ns, Some(30.0));
        assert!(pairs.in_step);
        assert!(!Stats::from_frames(4, &[0, 1, 2, 3], &[0, 0, 30, 31], true).in_step);
    }
}
