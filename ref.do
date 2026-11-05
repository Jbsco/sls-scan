#!/bin/bash
# Reference gate: the Haskell reference of the decode and triangulation against NumPy on
# synthetic scans, built from GHC's own libraries alone.
exec >&2
if ! command -v ghc >/dev/null 2>&1; then
    echo "ref: ghc not found - skipped"
    [ -n "${3:-}" ] && touch "$3"
    exit 0
fi
set -e
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
build () {
    ghc -O2 -Wall -Werror -iref -outputdir "$WORK/$1" -o "$WORK/sls-ref" ref/Main.hs \
        -hide-all-packages -package base -package bytestring -package array \
        -package filepath "${@:2}" >/dev/null
}
build dynamic -dynamic 2>/dev/null || build static
for scene in plate sphere strips; do
    python3 -m slscan.cli decode-vectors --scene "$scene" --out "$WORK/$scene" >/dev/null
    "$WORK/sls-ref" decode-check "$WORK/$scene"
done
python3 -m slscan.cli scan-vectors --out "$WORK/scan" >/dev/null
"$WORK/sls-ref" scan-check "$WORK/scan"
echo "ref: all checks clean"
