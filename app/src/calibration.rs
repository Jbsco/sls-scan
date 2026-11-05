//! Calibration from the application: board poses captured over the protocol into a
//! session directory, and the Python tool run as a subprocess on them.

use crate::client::{self, Event};
use crate::protocol::{Projector, SequenceRequest};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct Board {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_mm: f64,
    pub marker_mm: f64,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            squares_x: 11,
            squares_y: 8,
            square_mm: 50.0,
            marker_mm: 37.0,
        }
    }
}

impl Board {
    fn args(&self) -> Vec<String> {
        vec![
            "--squares-x".into(),
            self.squares_x.to_string(),
            "--squares-y".into(),
            self.squares_y.to_string(),
            "--square-mm".into(),
            self.square_mm.to_string(),
            "--marker-mm".into(),
            self.marker_mm.to_string(),
        ]
    }
}

/// The calibration tool as a program and its leading arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub program: String,
    pub args: Vec<String>,
}

impl Tool {
    pub fn parse(command: &str) -> Option<Self> {
        let mut words = command.split_whitespace().map(str::to_owned);
        let program = words.next()?;
        Some(Self {
            program,
            args: words.collect(),
        })
    }

    fn json<T: for<'de> Deserialize<'de>>(&self, extra: &[String]) -> io::Result<T> {
        let output = Command::new(&self.program)
            .args(&self.args)
            .args(extra)
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().last().unwrap_or_default();
        if let Ok(failure) = serde_json::from_str::<ToolError>(line) {
            return Err(io::Error::other(failure.error));
        }
        serde_json::from_str(line).map_err(|e| {
            let stderr = String::from_utf8_lossy(&output.stderr);
            io::Error::other(format!(
                "{e}: {}",
                stderr.lines().last().unwrap_or_default()
            ))
        })
    }
}

#[derive(Deserialize)]
struct ToolError {
    error: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Detection {
    pub corners: u32,
    pub of: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct StereoSolve {
    pub views: u32,
    pub left_rms_px: f64,
    pub right_rms_px: f64,
    pub stereo_rms_px: f64,
    pub camera_limit_px: f64,
    pub baseline_m: f64,
    pub out: String,
    pub ok: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DepthParamsResult {
    pub out: String,
    pub width: u32,
    pub height: u32,
    pub lo: i32,
    pub count: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Solve {
    pub views: u32,
    pub camera_rms_px: f64,
    pub camera_limit_px: f64,
    pub projector_rms_px: f64,
    pub projector_limit_px: f64,
    pub baseline_m: f64,
    pub out: String,
    pub ok: bool,
}

pub fn write_png(path: &Path, width: u32, height: u32, pixels: &[u8]) -> io::Result<()> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().map_err(io::Error::other)?;
    writer.write_image_data(pixels).map_err(io::Error::other)
}

/// Read an 8-bit greyscale PNG as its width, height and row-major pixels.
pub fn read_png(path: &Path) -> io::Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(io::BufReader::new(File::open(path)?));
    let mut reader = decoder.read_info().map_err(io::Error::other)?;
    let mut pixels = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut pixels).map_err(io::Error::other)?;
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err(io::Error::other(format!(
            "{} is not 8-bit greyscale",
            path.display()
        )));
    }
    pixels.truncate(info.buffer_size());
    Ok((info.width, info.height, pixels))
}

/// The next unused `pose_NNN` directory of a session.
pub fn next_pose_dir(session: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(session)?;
    let taken = fs::read_dir(session)?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("pose_"))
        .count();
    Ok(session.join(format!("pose_{taken:03}")))
}

/// Capture one sequence of a set into a directory, one PNG per frame, and return the
/// node's projector.
fn capture_set(
    address: &str,
    wait: Duration,
    dir: &Path,
    set: &str,
) -> io::Result<Option<Projector>> {
    fs::create_dir_all(dir)?;
    let request = SequenceRequest {
        set: set.into(),
        pitch: 32,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
        exposures: Vec::new(),
        color: false,
    };
    let mut projector = None;
    let mut failure = None;
    client::capture(address, &request, wait, |event| match event {
        Event::Connected(hello) => projector = Some(hello.projector),
        Event::Frame {
            header,
            name,
            pixels,
        } => {
            let path = dir.join(format!("{name}.png"));
            let written = write_png(&path, header.width.into(), header.height.into(), &pixels);
            if let Err(e) = written {
                failure.get_or_insert(e);
            }
        }
        Event::Finished(_) | Event::Failed(_) => {}
    })?;
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(projector)
}

/// Capture one board pose into the next pose directory and return it with the projector.
pub fn capture_pose(
    address: &str,
    wait: Duration,
    session: &Path,
) -> io::Result<(PathBuf, Projector)> {
    let dir = next_pose_dir(session)?;
    let projector = capture_set(address, wait, &dir, "calibration")?
        .ok_or_else(|| io::Error::other("no HELLO from the node"))?;
    Ok((dir, projector))
}

/// Capture one board pair into the next pose directory and return it.
pub fn capture_stereo_pose(address: &str, wait: Duration, session: &Path) -> io::Result<PathBuf> {
    let dir = next_pose_dir(session)?;
    capture_set(address, wait, &dir, "stereo-calibration")?;
    Ok(dir)
}

pub fn detect(tool: &Tool, pose: &Path, board: &Board) -> io::Result<Detection> {
    detect_image(tool, pose, board, "white.png")
}

/// Detect board corners in one image of a pose directory.
pub fn detect_image(tool: &Tool, pose: &Path, board: &Board, image: &str) -> io::Result<Detection> {
    let mut extra = vec!["detect".into(), pose.to_string_lossy().into_owned()];
    extra.extend(board.args());
    extra.extend(["--image".into(), image.into(), "--json".into()]);
    tool.json(&extra)
}

pub fn solve_stereo(tool: &Tool, session: &Path, board: &Board) -> io::Result<StereoSolve> {
    let mut extra = vec![
        "calibrate-stereo".into(),
        session.to_string_lossy().into_owned(),
    ];
    extra.extend(board.args());
    extra.extend([
        "--out".into(),
        session.join("stereo.yaml").to_string_lossy().into_owned(),
        "--json".into(),
    ]);
    tool.json(&extra)
}

/// Write the depth frame parameters of a stereo calibration.
pub fn depth_params(tool: &Tool, calibration: &Path, out: &Path) -> io::Result<DepthParamsResult> {
    let extra = vec![
        "depth-params".into(),
        "--calibration".into(),
        calibration.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--json".into(),
    ];
    tool.json(&extra)
}

pub fn solve(
    tool: &Tool,
    session: &Path,
    board: &Board,
    projector: &Projector,
) -> io::Result<Solve> {
    let mut extra = vec!["calibrate".into(), session.to_string_lossy().into_owned()];
    extra.extend(board.args());
    extra.extend([
        "--projector-width".into(),
        projector.width.to_string(),
        "--projector-height".into(),
        projector.height.to_string(),
        "--out".into(),
        session
            .join("calibration.yaml")
            .to_string_lossy()
            .into_owned(),
        "--json".into(),
    ]);
    tool.json(&extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!("sls-scan-{name}-{}-{stamp}", std::process::id()))
    }

    #[test]
    fn png_round_trip() -> io::Result<()> {
        let dir = scratch("png");
        fs::create_dir_all(&dir)?;
        let path = dir.join("frame.png");
        let pixels: Vec<u8> = (0..12).collect();
        write_png(&path, 4, 3, &pixels)?;
        let (width, height, back) = read_png(&path)?;
        assert_eq!((width, height), (4, 3));
        assert_eq!(back, pixels);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn pose_directories_are_numbered() -> io::Result<()> {
        let session = scratch("session");
        assert_eq!(next_pose_dir(&session)?, session.join("pose_000"));
        fs::create_dir_all(session.join("pose_000"))?;
        fs::write(session.join("calibration.yaml"), "")?;
        assert_eq!(next_pose_dir(&session)?, session.join("pose_001"));
        fs::remove_dir_all(session)
    }

    #[test]
    fn tool_command_splits_into_program_and_arguments() {
        let tool = Tool::parse("python3 -m slscan.cli");
        assert_eq!(
            tool,
            Some(Tool {
                program: "python3".into(),
                args: vec!["-m".into(), "slscan.cli".into()]
            })
        );
        assert_eq!(Tool::parse("  "), None);
    }

    #[test]
    fn solve_result_parses() -> serde_json::Result<()> {
        let line = r#"{"views": 12, "camera_rms_px": 0.096, "camera_limit_px": 0.3, "projector_rms_px": 0.092, "projector_limit_px": 0.5, "baseline_m": 0.34985, "out": "c.yaml", "ok": true}"#;
        let solve: Solve = serde_json::from_str(line)?;
        assert!(solve.ok && solve.views == 12);
        Ok(())
    }
}
