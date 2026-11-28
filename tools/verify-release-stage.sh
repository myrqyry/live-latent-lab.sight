#!/usr/bin/env bash
# Check the install contents using a new private staging prefix.
# No changes to live OBS, system libraries, or any user's home plugin files.
set -euo pipefail
if [[ "$#" != 1 ]]; then
  echo "usage: $0 build-directory" >&2
  exit 2
fi
build="$(realpath "$1")"
if [[ ! -f "$build/CMakeCache.txt" ]]; then
  echo "Not a configured Sight build: $build" >&2
  exit 2
fi
stage="$(mktemp -d /tmp/sight-release-stage-XXXXXX)"
trap 'rm -rf -- "$stage"' EXIT
prefix="$stage/package"
cmake --install "$build" --prefix "$prefix" > "$stage/install.log"
require_file() {
  if [[ ! -s "$prefix/$1" ]]; then
    echo "RELEASE_STAGE_MISSING=$1" >&2
    exit 1
  fi
}
require_file "bin/64bit/live-latent-lab-sight.so"
module="$prefix/bin/64bit/live-latent-lab-sight.so"
dynamic="$(readelf -d "$module")"
# Match the actual linked SONAME, not the version of a former local build.
mapfile -t needed < <(printf '%s\n' "$dynamic" | sed -n 's/.*(NEEDED).*\[\([^]]*\)\].*/\1/p')
ort_found=0
for soname in "${needed[@]}"; do
  case "$soname" in
    libonnxruntime.so|libonnxruntime.so.*)
      require_file "bin/64bit/$soname"
      ort_found=1 ;;
    libstable-diffusion.so)
      require_file "bin/64bit/$soname"
      require_file "data/sight-generator-preview.effect"
      require_file "data/licenses/stable-diffusion.cpp/LICENSE"
      require_file "data/licenses/ggml/LICENSE"
      require_file "data/licenses/utf8proc/LICENSE.md"
      require_file "data/licenses/oniguruma/COPYING" ;;
    libLiteRt.so)
      require_file "bin/64bit/$soname"
      require_file "data/licenses/litert/LICENSE" ;;
  esac
done
if [[ "$ort_found" != 1 ]]; then
  echo "RELEASE_STAGE_MISSING_ONNX_NEEDED=YES" >&2
  exit 1
fi
require_file "data/sight-tracking-overlay.effect"
require_file "data/sight-touch-field.effect"
require_file "data/sight-depth-camera.effect"
require_file "data/locale/en-US.ini"
require_file "data/licenses/sight/LICENSE"
require_file "data/licenses/sight/LICENSING.md"
cmp -s "$(dirname "$0")/../LICENSE" "$prefix/data/licenses/sight/LICENSE"
grep -Fq "GPL-3.0-only" "$prefix/data/licenses/sight/LICENSING.md"
require_file "data/licenses/onnxruntime/LICENSE"
require_file "data/licenses/onnxruntime/ThirdPartyNotices.txt"
require_file "data/licenses/libfacedetection/LICENSE"
require_file "data/licenses/tinygltf/LICENSE"
require_file "data/licenses/stb/LICENSE"
if [[ -e "$prefix/data/sight-neural-rendering.effect" ]]; then
  for file in LICENSE-OpenDLSS-NR.txt UPSTREAM-NOTICE.txt fused.cu fused.py input_conditioning.py primitives.cuh quantized.py replay.py still.py worker.py; do
    require_file "data/neural-rendering/$file"
  done
  echo "SIGHT_STAGE_NEURAL_SOURCE_INCLUDED=YES (external graph and weights not included)"
fi
if [[ -e "$prefix/bin/64bit/libmediapipe.so" ]]; then
  require_file "data/licenses/mediapipe/LICENSE"
  echo "SIGHT_STAGE_MEDIAPIPE_INCLUDED=YES (redistribution rights still require review)"
else
  echo "SIGHT_STAGE_MEDIAPIPE_INCLUDED=NO (face/body/hand require a separate runtime)"
fi
if find "$prefix" -type f \( -iname '*.onnx' -o -iname '*.task' -o -iname '*.tflite' -o -iname '*.safetensors' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.gguf' -o -iname '*.ckpt' -o -iname '*.moc3' -o -iname '*.engine' -o -iname '*.plan' -o -iname '*.npz' \) -print -quit | grep -q .; then
  echo "RELEASE_STAGE_UNEXPECTED_MODEL_BUNDLE=YES" >&2
  exit 1
fi
if ! readelf -d "$prefix/bin/64bit/live-latent-lab-sight.so" |
    grep -E 'RPATH|RUNPATH' | grep -Fq "\$ORIGIN"; then
  echo "RELEASE_STAGE_NON_RELOCATABLE_RPATH=YES" >&2
  exit 1
fi
# CUDA is optional and its dependencies may be provisioned by the target.
# Keep host-loader diagnostics distinct from the contents check.
provider="$prefix/bin/64bit/libonnxruntime_providers_cuda.so"
if [[ -e "$provider" ]]; then
  require_file "bin/64bit/libonnxruntime_providers_shared.so"
  if provider_links="$(env -u LD_LIBRARY_PATH ldd "$provider" 2>&1)" &&
      ! grep -q 'not found' <<< "$provider_links"; then
    echo "SIGHT_STAGE_CUDA_PROVIDER_LINKS=RESOLVED (inference not tested)"
  else
    echo "SIGHT_STAGE_CUDA_PROVIDER_LINKS=UNRESOLVED (local GPU qualification required)"
    grep 'not found' <<< "$provider_links" || true
  fi
fi
echo "SIGHT_RELEASE_STAGE_FILES=$(find "$prefix" -type f | wc -l)"
echo "SIGHT_RELEASE_STAGE_BYTES=$(du -sb "$prefix" | awk '{print $1}')"
echo "SIGHT_RELEASE_STAGE_CONTENTS_PASS=1"
echo "This is technical staging only; NOT a public redistribution approval."
