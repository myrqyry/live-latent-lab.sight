#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || -z ${ORT_ROOT:-} ]]; then
    echo 'Usage: ORT_ROOT=/path/to/onnxruntime tests/run_animegan_inference.sh /path/to/animegan.onnx' >&2
    exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
"${CXX:-c++}" -std=c++20 -O2 -DUSE_ONNX_MEDIAPIPE \
    -I"$repo_root/src" -I"$ORT_ROOT/include" \
    "$repo_root/tests/animegan_inference_test.cpp" \
    "$repo_root/src/core/enhancement_processor.cpp" \
    -L"$ORT_ROOT/lib" -Wl,-rpath,"$ORT_ROOT/lib" \
    -lonnxruntime -o "$build_dir/animegan_inference_test"
"$build_dir/animegan_inference_test" "$1"
