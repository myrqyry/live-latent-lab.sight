#!/usr/bin/env bash
# Actual NVIDIA OpenGL display required; Xvfb software rendering is not proof.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: SIGHT_CUDA_RUNTIME_PATH=/vendor/lib:... $0 build-directory" >&2;exit 2; }
: "${SIGHT_CUDA_RUNTIME_PATH:?Set matching CUDA/cuDNN runtime directories}"
: "${DISPLAY:?Use a real NVIDIA OpenGL display}"
build=$(realpath "$1")
cache="$build/CMakeCache.txt"
grep -Fxq 'SIGHT_ENABLE_CUDA_TEXTURE_INTEROP:BOOL=ON' "$cache"
model=$(sed -n 's/^SIGHT_ANIMEGAN_MODEL:FILEPATH=//p' "$cache")
[[ -f "$model" ]] || { echo "Configure SIGHT_ANIMEGAN_MODEL to the qualified local model." >&2;exit 2; }
cmake --build "$build" --target live-latent-lab-sight sight_obs_effect_render_smoke --parallel 4
temp=$(mktemp -d /tmp/sight-cuda-texture-verify.XXXXXX)
trap 'rm -rf -- "$temp"' EXIT
cmake --install "$build" --prefix "$temp/package" >"$temp/install.log"
mkdir "$temp/home"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export HOME="$temp/home" XDG_CONFIG_HOME="$temp/home/config" XDG_CACHE_HOME="$temp/home/cache" XDG_DATA_HOME="$temp/home/data"
export LD_LIBRARY_PATH="$temp/package/bin/64bit:$SIGHT_CUDA_RUNTIME_PATH"
export SIGHT_SMOKE_CUDA_TEXTURE=1 SIGHT_SMOKE_CUDA_TEXTURE_SWITCH=1
export SIGHT_SMOKE_WIDTH=320 SIGHT_SMOKE_HEIGHT=180 SIGHT_SMOKE_SKETCH_SIZE=256
unset SIGHT_ONNX_DEVICE
timeout 90 "$build/sight_obs_effect_render_smoke" "$temp/package/bin/64bit/live-latent-lab-sight.so" "$model" "$temp/package/data" >"$temp/test.log" 2>&1 || { cat "$temp/test.log";exit 1; }
cat "$temp/test.log"
grep -Eq 'SIGHT_SKETCH_CUDA_RESIDENT .*pass=1' "$temp/test.log"
grep -Eq 'SIGHT_SKETCH_CUDA_RESIZE .*loads=1 pass=1' "$temp/test.log"
grep -Eq 'SIGHT_SKETCH_CUDA_SWITCH .*pass=1' "$temp/test.log"
if grep -Eq 'No vertex shader specified|missing vertex shader|Error compiling shader|device_draw [(]GL[)] failed|obs_source_draw: no active effect|Actual ONNX frame inference failed|error: \[Sight/CUDA texture\]' "$temp/test.log";then exit 1;fi
