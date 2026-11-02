#!/bin/bash
# Virtual bench over the capture protocol: the stand-in node in a container serves a
# calibration capture and a plate scan, and the pipeline calibrates, scans and measures.
exec >&2
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    echo "node: docker not available - skipped"
    [ -n "${3:-}" ] && touch "$3"
    exit 0
fi
set -e
OUT=$(mktemp -d)
NAME=sls-scan-node-bench
PORT=${SLS_NODE_PORT:-5601}
cleanup() { docker stop "$NAME" >/dev/null 2>&1 || true; rm -rf "$OUT"; }
trap cleanup EXIT
docker build -q -t sls-scan-node . >/dev/null
docker run -d --rm -p "127.0.0.1:$PORT:5600" --name "$NAME" sls-scan-node --scene plate >/dev/null
for i in $(seq -f %03g 0 11); do
    python3 -m slscan.cli capture --node "127.0.0.1:$PORT" --set calibration --wait 60 \
        --out "$OUT/calibration/pose_$i" >/dev/null
done
python3 -m slscan.cli calibrate "$OUT/calibration" --squares-x 9 --squares-y 6 --square-mm 40 \
    --marker-mm 30 --projector-width 1024 --projector-height 768 --out "$OUT/calibration.yaml"
python3 -m slscan.cli capture --node "127.0.0.1:$PORT" --set scan --pitch 16 --out "$OUT/plate"
python3 -m slscan.cli scan "$OUT/plate" --calibration "$OUT/calibration.yaml" --pitch 16 \
    --out "$OUT/plate.ply"
python3 -m slscan.cli measure "$OUT/plate.ply" --shape plane
echo "node: virtual bench over the capture protocol clean"
[ -n "${3:-}" ] && touch "$3"
exit 0
