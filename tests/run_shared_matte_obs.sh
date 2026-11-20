#!/usr/bin/env bash
# Actual Sight->HyperShaderFX runtime integration; no normal OBS/profile changes.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 5 ]]; then
    echo "Usage: $0 SIGHT_BUILD HYPERSHADER_MODULE HYPERSHADER_DATA U2NET_MODEL PORTRAIT_IMAGE" >&2
    exit 2
fi
build="$1"; hs_module="$2"; hs_data="$3"; model="$4"; image="$5"
image_module="${OBS_IMAGE_SOURCE_MODULE:-/usr/local/lib/obs-plugins/image-source.so}"
for path in "$build/sight_obs_effect_render_smoke" "$build/live-latent-lab-sight.so" "$hs_module" "$hs_data/shaders/tests/sight_matte_pipeline_probe.effect" "$hs_data/shaders/root/filter.sight_subject_frame.effect" "$hs_data/shaders/root/filter.sight_motion_pulse.effect" "$model" "$image" "$image_module"; do
    [[ -f "$path" ]] || { echo "Required fixture missing: $path" >&2; exit 2; }
done
SIGHT_SMOKE_WIDTH=512 SIGHT_SMOKE_HEIGHT=512 SIGHT_SMOKE_EFFECT=sight_u2net_filter \
SIGHT_SMOKE_IMAGE_PLUGIN="$image_module" SIGHT_SMOKE_IMAGE_FILE="$image" \
SIGHT_SMOKE_EXPECT_FOREGROUND=1 SIGHT_SMOKE_SHARED_MATTE=1 \
SIGHT_SMOKE_HYPERSHADER_MODULE="$hs_module" SIGHT_SMOKE_HYPERSHADER_DATA="$hs_data" \
SIGHT_SMOKE_MOTION_EFFECT="$hs_data/shaders/root/filter.sight_motion_pulse.effect" \
SIGHT_SMOKE_SUBJECT_FRAME_EFFECT="$hs_data/shaders/root/filter.sight_subject_frame.effect" \
SIGHT_SMOKE_HYPERSHADER_EFFECT="$hs_data/shaders/tests/sight_matte_pipeline_probe.effect" \
"$repo/tests/run_obs_sight_render.sh" "$build/sight_obs_effect_render_smoke" "$build/live-latent-lab-sight.so" "$model" "$repo/data"
