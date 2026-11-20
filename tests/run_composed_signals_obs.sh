#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 7 ]]; then
    echo "Usage: $0 SMOKE SIGHT_MODULE SIGHT_DATA HYPER_MODULE HYPER_DATA U2NET PORTRAIT" >&2
    exit 2
fi
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
SIGHT_SMOKE_WIDTH=512 SIGHT_SMOKE_HEIGHT=512 SIGHT_SMOKE_EFFECT=sight_u2net_filter \
SIGHT_SMOKE_IMAGE_PLUGIN="${OBS_IMAGE_SOURCE_MODULE:-/usr/local/lib/obs-plugins/image-source.so}" SIGHT_SMOKE_IMAGE_FILE="$7" \
SIGHT_SMOKE_HYPERSHADER_MODULE="$4" SIGHT_SMOKE_HYPERSHADER_DATA="$5" \
SIGHT_SMOKE_COMPOSED_EFFECT="$5/shaders/tests/sight_composed_flow_matte_probe.effect" \
SIGHT_SMOKE_COMPOSED_SPARKS="$5/shaders/root/filter.sight_flow_sparks.effect" \
timeout 80 "$repo/tests/run_obs_sight_render.sh" "$1" "$2" "$6" "$3" > "$log" 2>&1 || { cat "$log"; exit 1; }
cat "$log"
grep -Fq 'SIGHT_COMPOSED_SIGNALS first=1 reverse=1 disabled=1 recovered=1 blocked=1 restored=1' "$log"
grep -Fq 'SIGHT_MASKED_SPARKS quiet=1 subject=1' "$log"
if grep -Eq 'set to invalid size|no active effect|Error compiling shader' "$log"; then exit 1; fi
