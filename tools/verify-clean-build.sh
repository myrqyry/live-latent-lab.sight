#!/usr/bin/env bash
# Clean model-free build gate. No runtime installation or package publication.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 1 ]]; then echo "Usage: $0 new-build-directory" >&2; exit 2; fi
build="$(realpath -m "$1")"
if [[ -e "$build" ]]; then echo "Refusing existing build directory: $build" >&2; exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 SIGHT_ONNX_DEVICE=cpu
if [[ -n "${SIGHT_OBS_SDK_ROOT:-}" ]]; then
  test -s "$SIGHT_OBS_SDK_ROOT/include/obs/obs-config.h"
  export LD_LIBRARY_PATH="$SIGHT_OBS_SDK_ROOT/lib:$SIGHT_OBS_SDK_ROOT/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python3 "$root/tools/verify-dependency-provenance.py"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_EXPECTED_LIBOBS_MINOR=2 -DSIGHT_ENABLE_UNVERIFIED_EFFECTS=OFF \
  -DSIGHT_ENABLE_NATIVE_SD_GENERATOR=OFF -DSIGHT_ENABLE_NATIVE_SD_LIVE_FILTER=OFF \
  -DSIGHT_ENABLE_NEURAL_RENDERING=OFF -DSIGHT_ENABLE_RTMW=OFF \
  -DSIGHT_STAGE_LOCAL_MEDIAPIPE=OFF
cmake --build "$build" --parallel 4
# No per-user models, test media or SDKs supplied. Explicitly isolated OBS configuration.
mkdir -p "$build/isolated-home" "$build/isolated-config" "$build/isolated-cache" "$build/isolated-data" "$build/isolated-tmp"
export HOME="$build/isolated-home" XDG_CONFIG_HOME="$build/isolated-config" XDG_CACHE_HOME="$build/isolated-cache" XDG_DATA_HOME="$build/isolated-data" TMPDIR="$build/isolated-tmp"
ctest --test-dir "$build" --output-on-failure --output-junit "$build/model-free.xml"
python3 - "$build/model-free.xml" <<'PY'
import sys, xml.etree.ElementTree as E
cases=E.parse(sys.argv[1]).findall('.//testcase')
names={c.get('name') for c in cases}
required={'sight_release_safety','sight_acquire_mediapipe_unit','sight_native_vrm_obs_render','sight_ocr_output_unit','sight_dense_motion_unit'}
if not required <= names or any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):
    raise SystemExit(f'Missing or failed required model-free tests: {sorted(required-names)}')
print(f'{len(cases)} model-free tests verified; model-backed inference is a separate qualification')
PY
"$root/tools/verify-release-stage.sh" "$build"
