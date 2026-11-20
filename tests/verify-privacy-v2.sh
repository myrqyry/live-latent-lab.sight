#!/usr/bin/env bash
# Real native face CNN and OBS video regression; no user's webcam or scenes.
set -eo pipefail
if [[ "$#" != 5 ]]; then
  echo "usage: $0 smoke-binary sight.so data-root 512rgba reference-model" >&2; exit 2
fi
smoke="$1";module="$2";data="$3";rgba="$4";reference_model="$5"
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d /tmp/sight-privacy-v2-XXXXXX)"
trap 'rm -rf "$work"' EXIT
image_plugin="${SIGHT_SMOKE_IMAGE_PLUGIN:-}"
if [[ -z "$image_plugin" || ! -s "$image_plugin" ]]; then
  echo "Missing OBS image-source module; set SIGHT_SMOKE_IMAGE_PLUGIN to image-source.so" >&2
  exit 2
fi
test -s "$rgba" && test -s "$smoke" && test -s "$module"
python3 - "$rgba" "$work" <<'PY'
from pathlib import Path
from PIL import Image
import sys
source=Image.frombytes('RGBA',(512,512),Path(sys.argv[1]).read_bytes())
out=Path(sys.argv[2]);source.save(out/'single.png')
small=source.resize((256,256),Image.Resampling.LANCZOS)
two=Image.new('RGBA',(512,512),(90,90,90,255))
two.paste(small,(0,128));two.paste(small,(256,128))
two.save(out/'two.png')
PY
export SIGHT_SMOKE_EFFECT=sight_privacy_v2_filter
export SIGHT_SMOKE_WIDTH=512 SIGHT_SMOKE_HEIGHT=512
# Use the tested module directory, never an older user-installed runtime.
LD_LIBRARY_PATH="$(dirname "$module")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
for mode in single two empty; do
  echo "REAL_PRIVACY_OBS_MODE=$mode"
  unset SIGHT_SMOKE_PRIVACY_EXPECT_MASKED SIGHT_SMOKE_PRIVACY_EXPECT_TWO_MASKS
  unset SIGHT_SMOKE_PRIVACY_REQUIRE_BLACK SIGHT_SMOKE_HOLD_MS SIGHT_SMOKE_IMAGE_FILE
  unset SIGHT_SMOKE_PRIVACY_CHECK_FREEZE SIGHT_SMOKE_EXPECT_PASSTHROUGH
  if [[ "$mode" == single ]]; then export SIGHT_SMOKE_PRIVACY_CHECK_FREEZE=1; fi
  if [[ "$mode" == empty ]]; then
    export SIGHT_SMOKE_EXPECT_PASSTHROUGH=1
    export SIGHT_SMOKE_HOLD_MS=1750
    unset SIGHT_SMOKE_IMAGE_PLUGIN
  else
    export SIGHT_SMOKE_IMAGE_PLUGIN="$image_plugin"
    export SIGHT_SMOKE_PRIVACY_EXPECT_MASKED=1
    export SIGHT_SMOKE_IMAGE_FILE="$work/$mode.png"
    export SIGHT_SMOKE_HOLD_MS=1750
    if [[ "$mode" == two ]]; then
      export SIGHT_SMOKE_PRIVACY_EXPECT_TWO_MASKS=1
    fi
  fi
  "$repo/tests/run_obs_sight_render.sh" "$smoke" "$module" "$reference_model" "$data" \
    >"$work/$mode.log" 2>&1 || { tail -65 "$work/$mode.log";exit 1; }
  if [[ "$mode" == single ]]; then
    grep -q 'privacy_live_source_updates=1 privacy_never_black=1' "$work/$mode.log" ||
      { echo "FAIL: native privacy render did not stay live without black";exit 1; }
  fi
  if [[ "$mode" == two ]]; then
    grep -q 'PRIVACY_V2_REAL_FACES_20261011 faces=2' "$work/$mode.log" ||
      { echo "FAIL: native detector did not find both faces";exit 1; }
  elif [[ "$mode" == empty ]]; then
    grep -q 'PRIVACY_V2_VALID_ZERO_FACES_20261011' "$work/$mode.log" ||
      { echo "FAIL: empty scene did not complete real native detection";exit 1; }
    grep -q 'privacy_zero_face_passthrough=1' "$work/$mode.log" ||
      { echo "FAIL: successful zero-face detection did not display camera";exit 1; }
  fi
  grep -E 'PRIVACY_V2|privacy_.*pixels|privacy_left_changed|privacy_live_source_updates|privacy_no_face_passthrough|privacy_zero_face_passthrough|privacy_stale_frame|PASS:' "$work/$mode.log" | tail -12
done
echo "PASS: one/two real faces masked and zero-face camera remains visible in OBS"
