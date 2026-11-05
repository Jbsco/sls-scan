use slscan_app::client::{capture, Event};
use slscan_app::protocol::{self, FrameHeader, SequenceRequest, MONO8};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

fn serve_one(listener: TcpListener, frames: u32) -> thread::JoinHandle<std::io::Result<()>> {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept()?;
        let hello =
            br#"{"node":"fake","projector":{"width":16,"height":4,"refresh_hz":60},"cameras":[]}"#;
        protocol::write_message(&mut stream, protocol::HELLO, hello)?;
        let (kind, payload) = protocol::read_message(&mut stream)?;
        assert_eq!(kind, protocol::SEQUENCE);
        let request: SequenceRequest = serde_json::from_slice(&payload)?;
        assert_eq!(request.set, "scan");
        for index in 0..frames {
            let header = FrameHeader {
                sequence: 0,
                index,
                exposure: 0,
                timestamp_ns: i64::from(index) * 16_666_667,
                width: 3,
                height: 2,
                format: MONO8,
            };
            let pixels = [u8::try_from(index).unwrap_or(0); 6];
            protocol::write_message(
                &mut stream,
                protocol::FRAME,
                &protocol::encode_frame(&header, &pixels),
            )?;
        }
        protocol::write_message(&mut stream, protocol::END, b"{}")
    })
}

#[test]
fn client_receives_a_full_scan_sequence() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?.to_string();
    let names = protocol::pattern_names("scan", 16, 4, 4, 1, 0, 1).unwrap_or_default();
    let server = serve_one(listener, u32::try_from(names.len()).unwrap_or(0));
    let request = SequenceRequest {
        set: "scan".into(),
        pitch: 16,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
        exposures: Vec::new(),
    };
    let mut received = Vec::new();
    let stats = capture(&address, &request, Duration::ZERO, |event| {
        if let Event::Frame { name, pixels, .. } = event {
            received.push((name, pixels[0]));
        }
    })?;
    assert!(server.join().is_ok());
    assert_eq!(stats.frames, names.len());
    assert!(stats.missing.is_empty() && stats.in_order);
    assert_eq!(stats.mean_interval_ns, Some(16_666_667.0));
    assert_eq!(received[0].0, "gray_000");
    assert_eq!(received.last().map(|r| r.0.as_str()), Some("phase_003"));
    Ok(())
}

#[test]
fn client_reports_missing_frames() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?.to_string();
    let server = serve_one(listener, 5);
    let request = SequenceRequest {
        set: "scan".into(),
        pitch: 16,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
        exposures: Vec::new(),
    };
    let stats = capture(&address, &request, Duration::ZERO, |_| {})?;
    assert!(server.join().is_ok());
    assert_eq!(stats.frames, 5);
    assert_eq!(
        stats.missing.len(),
        protocol::pattern_names("scan", 16, 4, 4, 1, 0, 1)
            .unwrap_or_default()
            .len()
            - 5
    );
    Ok(())
}

#[test]
fn client_waits_for_a_late_node() -> std::io::Result<()> {
    let address = TcpListener::bind("127.0.0.1:0")?.local_addr()?;
    let late = thread::spawn(move || -> std::io::Result<()> {
        thread::sleep(Duration::from_millis(500));
        let listener = TcpListener::bind(address)?;
        serve_one(listener, 1)
            .join()
            .map_err(|_| std::io::Error::other("node thread panicked"))?
    });
    let request = SequenceRequest {
        set: "scan".into(),
        pitch: 16,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
        exposures: Vec::new(),
    };
    let stats = capture(
        &address.to_string(),
        &request,
        Duration::from_secs(10),
        |_| {},
    )?;
    assert!(late.join().is_ok());
    assert_eq!(stats.frames, 1);
    Ok(())
}

#[test]
fn client_without_wait_fails_fast() -> std::io::Result<()> {
    let address = TcpListener::bind("127.0.0.1:0")?.local_addr()?;
    let request = SequenceRequest {
        set: "scan".into(),
        pitch: 16,
        steps: 4,
        frames: 1,
        rate_hz: 0,
        dark_every: 0,
        exposures: Vec::new(),
    };
    assert!(capture(&address.to_string(), &request, Duration::ZERO, |_| {}).is_err());
    Ok(())
}
