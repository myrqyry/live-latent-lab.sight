#!/usr/bin/env bash
# Run the actual Sight Depth Camera ONNX inference; no mocked model execution.
# Arguments: model.onnx image.rgb width height repetitions output.gray
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "Usage: $0 model.onnx image.rgb width height repetitions output.gray" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
ort_inc="$repo/external/onnxruntime/include"
ort_lib="$repo/external/onnxruntime/lib/libonnxruntime.so.1.17.1"
test -f "$ort_lib" || { echo "ONNX Runtime library unavailable: $ort_lib" >&2; exit 2; }
test -f "$1" && test -f "$2"
work="$(mktemp -d /tmp/sight-depth-benchmark-XXXXXX)"
trap 'rm -rf "$work"' EXIT
g++ -std=c++20 -O3 -I"$repo/src" -I"$ort_inc" \
  "$repo/tests/benchmark_depth_onnx.cpp" \
  "$repo/src/core/onnx_depth_processor.cpp" \
  "$ort_lib" -Wl,-rpath,"$(dirname "$ort_lib")" -pthread \
  -o "$work/benchmark-onnx"
"$work/benchmark-onnx" "$@"
