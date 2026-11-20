#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 8 ]]; then
  echo "Usage: $0 SMOKE SIGHT_MODULE SIGHT_DATA HYPER_MODULE HYPER_DATA DEPTH_MODEL U2NET PORTRAIT" >&2
  exit 2
fi
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
SIGHT_SMOKE_DEPTH_BRIDGE=1 SIGHT_SMOKE_WIDTH=512 SIGHT_SMOKE_HEIGHT=512 SIGHT_SMOKE_EFFECT=sight_depth_filter \
SIGHT_SMOKE_IMAGE_PLUGIN="${OBS_IMAGE_SOURCE_MODULE:-/usr/local/lib/obs-plugins/image-source.so}" SIGHT_SMOKE_IMAGE_FILE="$8" \
SIGHT_SMOKE_DEPTH_COMPOSE="$7" SIGHT_SMOKE_HYPERSHADER_MODULE="$4" SIGHT_SMOKE_HYPERSHADER_DATA="$5" \
SIGHT_SMOKE_DEPTH_PROBE="$5/shaders/tests/sight_depth_composition_probe.effect" \
SIGHT_SMOKE_DEPTH_SPARKS="$5/shaders/root/filter.sight_flow_sparks.effect" \
SIGHT_SMOKE_DEPTH_GLASS="$5/shaders/root/filter.sight_depth_glass.effect" \
timeout 90 xvfb-run -a "$1" "$2" "$6" "$3" > "$log" 2>&1 || { cat "$log"; exit 1; }
cat "$log"
grep -Fq 'SIGHT_DEPTH_SNAPSHOT ready=1 range=1 metadata=1 known=1 abi=1 disabled=1 expired=1 recovered=1' "$log"
grep -Fq 'SIGHT_DEPTH_COMPOSED all=1 values=1 reordered=1 missing=1 back=1 barrier=1 restored=1' "$log"
grep -Fq 'SIGHT_DEPTH_SPARKS quiet=1 emitted=1' "$log"
grep -Fq 'SIGHT_DEPTH_GLASS_BRIDGE warped=1' "$log"
grep -Fq 'SIGHT_DEPTH_BRIDGE_20261118' "$log"
if grep -Eq 'set to invalid size|no active effect|Error compiling shader|Actual model inference failed' "$log"; then exit 1; fi
