#!/usr/bin/env bash
# Run a *real* OBS graphics cycle and reject errors even if the frame changes.
set -euo pipefail
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
status=0
xvfb-run -a -s '-screen 0 1280x720x24' "$@" > "$log" 2>&1 || status=$?
cat "$log"
if grep -Eq 'No vertex shader specified|missing vertex shader|Error compiling shader|effect_setval_inline: invalid param|device_draw [(]GL[)] failed|obs_source_draw: no active effect|Actual ONNX frame inference failed|Actual subject mask inference failed|Actual model inference failed|[[]Sight/Tasks[]] Real tracking failed|[[]Sight/Tasks[]] Tracker load failed|[[]Sight/Depth[]] Inference failed|[[]SegProcessor[]] Infer failed|BroadcastIterator::Append|[[]Enhance[]] Process failed' "$log"; then
    echo 'FAIL: OBS graphics or ONNX reported a real inference/draw error' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_DEPTH_REJECT_STALE:-}" ]] &&
    ! grep -Fq '[Sight/Depth] Change-aware history blending active' "$log"; then
    echo 'FAIL: Depth change-aware blending branch never executed' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_DEPTH_VERIFY_EXPIRY:-}" ]] &&
    ! grep -Fq 'Last depth result expired; passing through source until inference recovers' "$log"; then
    echo 'FAIL: native Depth Camera never expired its old output' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_DEPTH_VERIFY_RESIZE:-}" ]] &&
    ! grep -Fq 'Source geometry changed to' "$log"; then
    echo 'FAIL: native Depth Camera never invalidated source geometry' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_EXPECT_RTMW_BACKEND:-}" ]] &&
   ! grep -Fq "RTMW_DETECTOR_BACKEND_20261118_Y927 detector=${SIGHT_SMOKE_EXPECT_RTMW_BACKEND} compiled=2" "$log"; then
    echo "FAIL: expected real persistent YOLOX+RTMW GPU backend" >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_EXPECT_TASK:-}" ]] && ! grep -Fq     "First real ${SIGHT_SMOKE_EXPECT_TASK} tracking output:" "$log"; then
    echo "FAIL: no real MediaPipe Tasks tracking output for ${SIGHT_SMOKE_EXPECT_TASK}" >&2
    exit 1
fi
if [[ "${SIGHT_SMOKE_EFFECT:-}" == "sight_value_router_filter" ]] &&
    ! grep -Fq 'First numeric effect update:' "$log"; then
    echo "FAIL: no real effect property updated by value router" >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_ROUTER_GATE:-}" ]] &&
    ! grep -Fq 'SIGHT_AR_THRESHOLD_GATE_20261118_G419' "$log"; then
    echo 'FAIL: threshold gate identity marker absent' >&2
    exit 1
fi
if [[ "${SIGHT_SMOKE_EFFECT:-}" == "sight_touch_field_filter" ]]; then
    if [[ -n "${SIGHT_SMOKE_EXPECT_PASSTHROUGH:-}" ]]; then
        if grep -Fq 'First live hand-driven GPU refraction' "$log"; then
            echo 'FAIL: Touch Field rendered without a valid hand' >&2
            exit 1
        fi
    elif ! grep -Fq 'First live hand-driven GPU refraction:' "$log"; then
        echo 'FAIL: Touch Field never consumed a valid hand tracking signal' >&2
        exit 1
    fi
fi
if [[ "${SIGHT_SMOKE_EFFECT:-}" == "sight_tracking_receiver_filter" ]] &&
    ! grep -Fq 'First shared tracking signal' "$log"; then
    echo "FAIL: no cross-instance tracking signal consumed" >&2
    exit 1
fi
if { [[ -n "${SIGHT_SMOKE_HOLISTIC_OVERLAY:-}" ]] ||
       [[ -n "${SIGHT_SMOKE_HOLISTIC_LIVE_TOGGLE:-}" ]]; } &&
    ! grep -Fq 'SIGHT_HOLISTIC_GPU_OVERLAY_20261112_J728' "$log"; then
    echo 'FAIL: verified Holistic GPU overlay renderer never ran' >&2
    exit 1
fi
if { [[ -n "${SIGHT_SMOKE_HOLISTIC_PERSON_MASK:-}" ]] ||
       [[ -n "${SIGHT_SMOKE_HOLISTIC_MASK_TOGGLE:-}" ]]; } &&
    ! grep -Fq 'SIGHT_HOLISTIC_PERSON_MASK_20261113_R862' "$log"; then
    echo 'FAIL: native Holistic person mask never reached overlay stability' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_HOLISTIC_TRIPLE:-}" ]] &&
    ! grep -Fq 'SIGHT_HOLISTIC_NATIVE_V1_20261112_H153' "$log"; then
    echo 'FAIL: correct Holistic producer module identity never appeared' >&2
    exit 1
fi
if ! grep -Fq 'ACTIVE_SKETCH_CUTOUT_DEPTH_TASKS_SIGNALS_HAND_TOUCH_EVENTS_20261010' "$log"; then
    echo 'FAIL: plugin identity marker absent (wrong binary loaded)' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_REFINE_EDGES:-}" ]] &&
    ! grep -Fq 'SIGHT_EDGE_MATTE_REFINEMENT_20261117' "$log"; then
    echo 'FAIL: real Subject Cutout refinement branch never executed' >&2
    exit 1
fi
if [[ -n "${SIGHT_SMOKE_MATTE_CONTROLS:-}" ]] &&
    { ! grep -Fq 'SIGHT_MATTE_CALIBRATION_20261117' "$log" ||
      ! grep -Fq 'SIGHT_MATTE_HISTORY_20261117' "$log" ||
      ! grep -Fq 'SIGHT_MATTE_LIVE_VIEWS=1' "$log"; }; then
    echo 'FAIL: real matte controls/history/views not exercised' >&2
    exit 1
fi
# A Sight-only instance skips native models. An explicit switch back to
# native tracking initializes the original runner on demand.
if [[ -v SIGHT_SMOKE_HYPERSHADER_MODULE ]]; then
    if [[ -v SIGHT_SMOKE_TEST_NATIVE_SWITCH ]]; then
        if ! grep -Fq '[HyperShader] MediaPipe Runner initialized' "$log"; then
            echo 'FAIL: switching back to native tracking did not initialize MediaPipe' >&2
            exit 1
        fi
    elif grep -Fq '[HyperShader] MediaPipe Runner initialized' "$log"; then
        echo 'FAIL: Sight-shared HyperShader instance redundantly loaded its own MediaPipe models' >&2
        exit 1
    fi
fi
exit "$status"
