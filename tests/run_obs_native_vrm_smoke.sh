#!/usr/bin/env bash
set -euo pipefail
test_bin="$1"
module="$2"
fixture_bin="$3"
xvfb_run="$4"
module_data="$5"
work="$(mktemp -d /tmp/sight-native-vrm-XXXXXX)"
trap 'rm -rf -- "$work"' EXIT
"$fixture_bin" --make-fixture "$work/test.vrm" > "$work/fixture.log"
"$xvfb_run" -a -s "-screen 0 1280x1024x24" \
    "$test_bin" "$module" "$work/test.vrm" "$module_data" > "$work/obs.log" 2>&1 || {
        tail -65 "$work/obs.log"
        exit 1
    }
grep -F "NATIVE_VRM_SOURCE_READY_20261104" "$work/obs.log" >/dev/null
grep -F "NATIVE_VRM_OBS_PIXEL_PASS" "$work/obs.log"
grep -F "NATIVE_VRM_SIGNAL_TO_RENDER PASS" "$work/obs.log"
