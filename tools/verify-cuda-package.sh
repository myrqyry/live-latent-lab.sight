#!/usr/bin/env bash
# Real local qualification of a matched, externally acquired Linux CUDA SDK.
# No system installation, model downloads, or OBS launch.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: SIGHT_CUDA_RUNTIME_PATH=/vendor/lib:... $0 build-directory" >&2; exit 2; }
build="$(realpath "$1")"
: "${SIGHT_CUDA_RUNTIME_PATH:?Set to the matching CUDA runtime library directories}"
[[ -x "$build/sight_native_ort_smoke" ]] || { echo "Build sight_native_ort_smoke first" >&2; exit 2; }
root="$(mktemp -d /tmp/sight-cuda-stage.XXXXXX)"
trap 'rm -rf -- "$root"' EXIT
prefix="$root/package"
cmake --install "$build" --prefix "$prefix" > "$root/install.log"
lib="$prefix/bin/64bit"
provider="$lib/libonnxruntime_providers_cuda.so"
test -s "$provider" && test -s "$lib/libonnxruntime_providers_shared.so"
# Use only the staged ORT and explicitly supplied NVIDIA dependencies.
export LD_LIBRARY_PATH="$lib:$SIGHT_CUDA_RUNTIME_PATH"
ldd "$provider" > "$root/provider-ldd.txt"
ldd "$build/sight_native_ort_smoke" > "$root/smoke-ldd.txt"
if grep -q 'not found' "$root/provider-ldd.txt" "$root/smoke-ldd.txt"; then
    cat "$root/provider-ldd.txt" "$root/smoke-ldd.txt" >&2; exit 1
fi
# Do not allow the test to silently use an already installed ORT.
grep -F "=> $lib/libonnxruntime.so" "$root/smoke-ldd.txt"
env -u SIGHT_ONNX_DEVICE "$build/sight_native_ort_smoke" --cuda
printf 'SIGHT_MATCHED_CUDA_STAGE_PASS=1\n'
