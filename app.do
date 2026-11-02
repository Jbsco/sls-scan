#!/bin/bash
# Application gate: formatting, lints and tests of the Rust application, then its capture
# client against the stand-in node over the capture protocol.
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
python3 -m slscan.cli node-sim --port "$PORT" >/dev/null 2>&1 &
NODE=$!
trap 'kill "$NODE" 2>/dev/null || true' EXIT
RUN="cargo run --locked --quiet --manifest-path app/Cargo.toml --"
$RUN capture --node "127.0.0.1:$PORT" --set scan --pitch 16 --wait 30
$RUN capture --node "127.0.0.1:$PORT" --set calibration --wait 30
echo "app: all checks clean"
[ -n "${3:-}" ] && touch "$3"
exit 0
