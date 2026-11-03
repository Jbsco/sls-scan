#!/bin/bash
# Application gate: formatting, lints and tests of the Rust application; its capture
# client, stereo calibration, fused depth frames and recorded-pair replay against the
# stand-in node over the capture protocol; and its GPU decode, rectification, coarse
# disparity, refinement, depth frames, fusion and TSDF against the NumPy and OpenCV
# references.
exec >&2
if ! command -v cargo >/dev/null 2>&1; then
    echo "app: cargo not found - skipped"
    [ -n "${3:-}" ] && touch "$3"
    exit 0
fi
set -e
(cd app && cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings \
    && cargo test --locked && cargo build --locked)
PORT=${SLS_APP_PORT:-5604}
SESSION=$(mktemp -d)
python3 -m slscan.cli node-sim --port "$PORT" >/dev/null 2>&1 &
NODE=$!
trap 'kill "$NODE" 2>/dev/null || true; rm -rf "$SESSION"' EXIT
RUN="cargo run --locked --quiet --manifest-path app/Cargo.toml --"
$RUN capture --node "127.0.0.1:$PORT" --set scan --pitch 16 --wait 30
$RUN capture --node "127.0.0.1:$PORT" --set calibration --wait 30
$RUN capture --node "127.0.0.1:$PORT" --set stereo --frames 3 --wait 30
$RUN capture --node "127.0.0.1:$PORT" --set stereo --frames 150 --rate 30 --wait 30
$RUN calibrate --node "127.0.0.1:$PORT" --set stereo --wait 30 --poses 12 --out "$SESSION/stereo" \
    --squares-x 9 --squares-y 6 --square-mm 10 --marker-mm 7.5
$RUN depth --node "127.0.0.1:$PORT" --params "$SESSION/stereo/depth.json" --frames 2 --wait 30 \
    --mode depth
$RUN depth --node "127.0.0.1:$PORT" --params "$SESSION/stereo/depth.json" --frames 4 --wait 30 \
    --mode preview
$RUN capture --node "127.0.0.1:$PORT" --set stereo --frames 2 --wait 30 --out "$SESSION/recorded"
python3 -m slscan.cli depth-params --calibration "$SESSION/stereo/stereo.yaml" --step 16 \
    --out "$SESSION/replay.json" >/dev/null
python3 -m slscan.cli replay-reference "$SESSION/recorded" --params "$SESSION/replay.json" \
    --calibration "$SESSION/stereo/stereo.yaml" >/dev/null
$RUN replay --frames "$SESSION/recorded" --params "$SESSION/replay.json"
$RUN calibrate --node "127.0.0.1:$PORT" --wait 30 --poses 6 --out "$SESSION" \
    --squares-x 9 --squares-y 6 --square-mm 40 --marker-mm 30
python3 -m slscan.cli decode-vectors --out "$SESSION/vectors" >/dev/null
$RUN decode-check "$SESSION/vectors"
python3 -m slscan.cli stereo-vectors --out "$SESSION/stereo" >/dev/null
$RUN rectify-check "$SESSION/stereo"
$RUN coarse-check "$SESSION/stereo"
$RUN refine-check "$SESSION/stereo"
$RUN depth-check "$SESSION/stereo"
python3 -m slscan.cli fusion-vectors --out "$SESSION/fusion" >/dev/null
$RUN fusion-check "$SESSION/fusion"
python3 -m slscan.cli tsdf-vectors --out "$SESSION/tsdf" >/dev/null
$RUN tsdf-check "$SESSION/tsdf"
echo "app: all checks clean"
[ -n "${3:-}" ] && touch "$3"
exit 0
