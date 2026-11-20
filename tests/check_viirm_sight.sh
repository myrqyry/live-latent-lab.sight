#!/usr/bin/env bash
# Interop boundary: ViiRM-authored VRM 1.0 -> Sight native VRM.
# No network, no OBS profile changes, no private models in the repository.
set -euo pipefail
if test "$#" -ne 1 || test ! -f "$1";then
  echo "Usage: SIGHT_BUILD_DIR=/path/to/configured/build $0 /path/to/avatar.vrm" >&2
  exit 2
fi
avatar="$(realpath "$1")"
repo="$(cd "$(dirname "$0")/.." && pwd)"
build="${SIGHT_BUILD_DIR:-$repo/build}"
module="$build/live-latent-lab-sight.so"
if test ! -f "$build/CMakeCache.txt";then
  echo "Configure Sight's build before running this check: $build" >&2
  exit 3
fi
command -v xvfb-run >/dev/null || { echo "xvfb-run is needed for the headless native OBS check" >&2;exit 4; }
cmake --build "$build" --target sight_vrm_transform_interop_test sight_obs_native_vrm_smoke -j4
"$build/sight_vrm_transform_interop_test" "$avatar"
for mode in default body hands;do
  echo "Checking real native libobs render: $mode"
  if test "$mode" = default;then
    xvfb-run -a -s "-screen 0 1280x1024x24"       "$build/sight_obs_native_vrm_smoke" "$module" "$avatar" "$repo/data"
  else
    xvfb-run -a -s "-screen 0 1280x1024x24"       "$build/sight_obs_native_vrm_smoke" "$module" "$avatar" "$repo/data" "--$mode"
  fi
done
echo "ViiRM -> Sight checks PASSED for: $avatar"
