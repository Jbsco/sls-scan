//! Messages between the application and a capture node, as specified in
//! `docs/capture-protocol.md`.

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"SLS1";
pub const HELLO: u8 = 1;
pub const SEQUENCE: u8 = 2;
pub const FRAME: u8 = 3;
pub const END: u8 = 4;
pub const ERROR: u8 = 5;
pub const FRAME_HEADER_LEN: usize = 23;
pub const MONO8: u8 = 0;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Projector {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Camera {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Hello {
    pub node: String,
    pub projector: Projector,
    pub cameras: Vec<Camera>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SequenceRequest {
    pub set: String,
    pub pitch: u32,
    pub steps: u32,
    #[serde(default = "one")]
    pub frames: u32,
    /// Trigger rate of a stereo set in pairs per second, or 0 for as fast as the node can.
    #[serde(default)]
    pub rate_hz: u32,
    /// Every this many-th stereo pair is taken with the pattern light off, none when 0.
    #[serde(default)]
    pub dark_every: u32,
}

const fn one() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub sequence: u32,
    pub index: u32,
    pub exposure: u16,
    pub timestamp_ns: i64,
    pub width: u16,
    pub height: u16,
    pub format: u8,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
}

pub fn write_message(w: &mut impl Write, kind: u8, payload: &[u8]) -> io::Result<()> {
    let length = u32::try_from(payload.len()).map_err(|_| invalid("payload over 4 GiB"))?;
    let mut header = Vec::with_capacity(9);
    header.extend_from_slice(MAGIC);
    header.push(kind);
    header.extend_from_slice(&length.to_le_bytes());
    w.write_all(&header)?;
    w.write_all(payload)
}

pub fn read_message(r: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut header = [0u8; 9];
    r.read_exact(&mut header)?;
    if &header[0..4] != MAGIC {
        return Err(invalid("message without the SLS1 magic"));
    }
    let length = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;
    let mut payload = vec![0u8; length];
    r.read_exact(&mut payload)?;
    Ok((header[4], payload))
}

pub fn encode_frame(header: &FrameHeader, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + pixels.len());
    out.extend_from_slice(&header.sequence.to_le_bytes());
    out.extend_from_slice(&header.index.to_le_bytes());
    out.extend_from_slice(&header.exposure.to_le_bytes());
    out.extend_from_slice(&header.timestamp_ns.to_le_bytes());
    out.extend_from_slice(&header.width.to_le_bytes());
    out.extend_from_slice(&header.height.to_le_bytes());
    out.push(header.format);
    out.extend_from_slice(pixels);
    out
}

pub fn parse_frame(payload: &[u8]) -> io::Result<(FrameHeader, &[u8])> {
    if payload.len() < FRAME_HEADER_LEN {
        return Err(invalid("frame shorter than its header"));
    }
    let u32_at =
        |i: usize| u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
    let u16_at = |i: usize| u16::from_le_bytes([payload[i], payload[i + 1]]);
    let mut stamp = [0u8; 8];
    stamp.copy_from_slice(&payload[10..18]);
    let header = FrameHeader {
        sequence: u32_at(0),
        index: u32_at(4),
        exposure: u16_at(8),
        timestamp_ns: i64::from_le_bytes(stamp),
        width: u16_at(18),
        height: u16_at(20),
        format: payload[22],
    };
    if header.format != MONO8 {
        return Err(invalid("unsupported pixel format"));
    }
    let size = usize::from(header.width) * usize::from(header.height);
    let pixels = payload
        .get(FRAME_HEADER_LEN..FRAME_HEADER_LEN + size)
        .ok_or_else(|| invalid("frame shorter than its image"))?;
    Ok((header, pixels))
}

/// Bit planes that index every position along an axis of the given length.
pub fn bit_planes(length: u32) -> u32 {
    let mut planes = 0;
    while (1u64 << planes) < u64::from(length) {
        planes += 1;
    }
    planes.max(1)
}

/// Whether stereo pair `pair` is taken with the pattern light off: every `dark_every`-th
/// pair from the first, none when `dark_every` is 0.
pub fn is_dark(pair: u32, dark_every: u32) -> bool {
    dark_every > 0 && pair.is_multiple_of(dark_every)
}

/// Capture file names of a set, in sequence order. A stereo set is `frames` pairs, left
/// then right, and a pair taken with the pattern light off is named `dark_` in place of
/// `frame_`. A stereo-calibration set is one board pair.
pub fn pattern_names(
    set: &str,
    width: u32,
    height: u32,
    steps: u32,
    frames: u32,
    dark_every: u32,
) -> Option<Vec<String>> {
    fn numbered(prefix: &'static str, n: u32) -> impl Iterator<Item = String> {
        (0..n).map(move |i| format!("{prefix}_{i:03}"))
    }
    match set {
        "stereo" => Some(
            (0..frames)
                .flat_map(|i| {
                    let prefix = if is_dark(i, dark_every) {
                        "dark"
                    } else {
                        "frame"
                    };
                    ["left", "right"].map(|side| format!("{prefix}_{i:03}/{side}"))
                })
                .collect(),
        ),
        "stereo-calibration" => Some(vec!["left".to_owned(), "right".to_owned()]),
        "calibration" => Some(
            std::iter::once("white".to_owned())
                .chain(numbered("columns", 2 * bit_planes(width)))
                .chain(numbered("rows", 2 * bit_planes(height)))
                .collect(),
        ),
        "scan" => Some(
            numbered("gray", 2 * bit_planes(width))
                .chain(numbered("phase", steps))
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_request_defaults_to_one_frame() -> serde_json::Result<()> {
        let request: SequenceRequest =
            serde_json::from_str(r#"{"set":"scan","pitch":32,"steps":4}"#)?;
        assert_eq!(request.frames, 1);
        assert_eq!(request.rate_hz, 0);
        Ok(())
    }

    #[test]
    fn message_round_trip() -> io::Result<()> {
        let mut buffer = Vec::new();
        write_message(&mut buffer, SEQUENCE, b"{}")?;
        let (kind, payload) = read_message(&mut buffer.as_slice())?;
        assert_eq!((kind, payload.as_slice()), (SEQUENCE, &b"{}"[..]));
        Ok(())
    }

    #[test]
    fn bad_magic_is_rejected() {
        let bytes = b"XXXX\x01\x00\x00\x00\x00";
        assert!(read_message(&mut &bytes[..]).is_err());
    }

    #[test]
    fn frame_round_trip() -> io::Result<()> {
        let header = FrameHeader {
            sequence: 7,
            index: 2,
            exposure: 1,
            timestamp_ns: 123_456_789,
            width: 4,
            height: 3,
            format: MONO8,
        };
        let pixels: Vec<u8> = (0..12).collect();
        let payload = encode_frame(&header, &pixels);
        assert_eq!(payload.len(), FRAME_HEADER_LEN + 12);
        let (back, image) = parse_frame(&payload)?;
        assert_eq!(back, header);
        assert_eq!(image, pixels.as_slice());
        Ok(())
    }

    #[test]
    fn short_frame_is_rejected() {
        let header = FrameHeader {
            sequence: 0,
            index: 0,
            exposure: 0,
            timestamp_ns: 0,
            width: 4,
            height: 3,
            format: MONO8,
        };
        let payload = encode_frame(&header, &[0; 5]);
        assert!(parse_frame(&payload).is_err());
    }

    #[test]
    fn pattern_names_follow_capture_layout() {
        let scan = pattern_names("scan", 1024, 768, 4, 1, 0).unwrap_or_default();
        assert_eq!(scan.len(), 24);
        assert_eq!(
            (scan[0].as_str(), scan[23].as_str()),
            ("gray_000", "phase_003")
        );
        let calibration = pattern_names("calibration", 1024, 768, 4, 1, 0).unwrap_or_default();
        assert_eq!(calibration.len(), 41);
        assert_eq!(calibration[1], "columns_000");
        assert_eq!(calibration[40], "rows_019");
        assert!(pattern_names("other", 1024, 768, 4, 1, 0).is_none());
        let stereo = pattern_names("stereo", 1024, 768, 4, 2, 0).unwrap_or_default();
        assert_eq!(
            stereo,
            [
                "frame_000/left",
                "frame_000/right",
                "frame_001/left",
                "frame_001/right"
            ]
        );
        let dark = pattern_names("stereo", 1024, 768, 4, 8, 4).unwrap_or_default();
        assert_eq!(dark[0..2], ["dark_000/left", "dark_000/right"]);
        assert_eq!(dark[8..10], ["dark_004/left", "dark_004/right"]);
        assert_eq!(
            pattern_names("stereo-calibration", 1024, 768, 4, 1, 0).unwrap_or_default(),
            ["left", "right"]
        );
    }

    #[test]
    fn bit_planes_match_the_pattern_generator() {
        assert_eq!(bit_planes(1), 1);
        assert_eq!(bit_planes(2), 1);
        assert_eq!(bit_planes(1024), 10);
        assert_eq!(bit_planes(1920), 11);
    }
}
