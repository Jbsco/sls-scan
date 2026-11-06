#!/bin/bash
# Futhark gate: the Futhark decode against NumPy on synthetic scans, built for the C and
# multicore backends.
exec >&2
if ! command -v futhark >/dev/null 2>&1; then
    echo "fut: futhark not found - skipped"
    [ -n "${3:-}" ] && touch "$3"
    exit 0
fi
set -e
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
futhark fmt --check fut/decode.fut
for scene in plate sphere strips; do
    python3 -m slscan.cli decode-vectors --scene "$scene" --out "$WORK/$scene" >/dev/null
done
python3 -m slscan.cli futhark-check --exact "$WORK/plate" "$WORK/sphere" "$WORK/strips"
python3 -m slscan.cli futhark-check --precision f32 "$WORK/plate"
python3 -m slscan.cli futhark-check --backend multicore --exact "$WORK/plate"
echo "fut: all checks clean"
