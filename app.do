#!/bin/bash
# Application gate: formatting, lints and tests of the Rust application, its capture
# client against the stand-in node over the capture protocol, and its GPU decode and
# stereo refinement against the NumPy reference.
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
$RUN calibrate --node "127.0.0.1:$PORT" --wait 30 --poses 6 --out "$SESSION" \
    --squares-x 9 --squares-y 6 --square-mm 40 --marker-mm 30
python3 -m slscan.cli decode-vectors --out "$SESSION/vectors" >/dev/null
$RUN decode-check "$SESSION/vectors"
python3 -m slscan.cli stereo-vectors --out "$SESSION/stereo" >/dev/null
$RUN refine-check "$SESSION/stereo"
echo "app: all checks clean"
[ -n "${3:-}" ] && touch "$3"
exit 0
