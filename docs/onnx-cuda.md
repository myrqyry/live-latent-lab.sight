# Optional NVIDIA inference in Sight

Sight prefers CUDA for Portrait Sketch, Subject Cutout, Depth Camera and
the optional ONNX semantic detector when
its ONNX Runtime can create a CUDA session. Unsupported graph operators may
still run on CPU. A CPU-only runtime or CUDA session initialization failure
uses the existing CPU path. Set `SIGHT_ONNX_DEVICE=cpu` before launching OBS
to skip CUDA even with a GPU-enabled runtime. This is not a live toggle.

Subject Cutout additionally offers a per-instance **Inference device**
setting: Auto (default, same fallback policy), CPU, or CUDA (required).
Required CUDA rejects session initialization failure and the global CPU
override; the status snapshot explains the failure and the filter uses
its normal passthrough behavior. CUDA session selection does not imply
all operators execute on GPU. Changing the setting rebuilds the session
on the existing worker without changing the model file or other sources.
Use FP32 models for CPU; the locally qualified FP16 conversions have
only been validated on CUDA. See [cutout precision](cutout-precision.md).
Provider selection is separate from texture and tensor transfers. See the
[CUDA texture and tensor I/O guide](cuda-texture-interop.md) for opt-in
build flags, runtime selectors, and the concurrent-effects test.


Use a separate build directory with a Linux ONNX Runtime GPU distribution
and its matching CUDA libraries and cuDNN if required by that distribution.
The GPU and CPU ONNX Runtime binaries and provider libraries should not be
mixed. See the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

```sh
cmake -S . -B build-sight-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ORT_ROOT=/absolute/path/to/onnxruntime-linux-x64-gpu \
  -DCMAKE_INSTALL_PREFIX="$HOME/.config/obs-studio/plugins/live-latent-lab-sight"
cmake --build build-sight-cuda --target live-latent-lab-sight -j2
```

Stage into a separate user-owned directory with `cmake --install
build-sight-cuda --prefix /absolute/path/to/stage`. Sight stages the selected
ONNX Runtime, its CUDA provider and shared provider helper if present.
The separate NVIDIA dependencies must already be visible to the OBS
process. The default release stage check targets the bundled CPU runtime.

After loading a model, look in the OBS log for `[Sight/ONNX]` and
`CUDA session ready` or `CPU session ready`. Session creation and inference
remain on the existing workers. A later inference failure follows the
filter's existing error handling; it does not switch providers mid-frame.
MediaPipe Tasks tracking and the face-detection path of Privacy V2 are unaffected, as is the separate
opt-in LiteRT RTMW GPU mode.

## Local qualification — 2026-11-24

Historical development qualification (not this curated build): the development build with ONNX Runtime GPU 1.30.0 and its matching
installed CUDA/shared provider libraries passed actual libobs rendering
tests for Portrait Sketch (landscape), Subject Cutout, Depth Camera,
and the Depth Anything V2, TEED, PiDiNet and HED native preprocessors.
Each run required the CUDA-session marker via
`SIGHT_SMOKE_EXPECT_ONNX_DEVICE=cuda`. That confirms CUDA session
creation and transformed pixels, not exclusive GPU operator placement.
The CPU override also passed the combined neural/depth and shared
matte tests. No automatic model downloads, runtime installation or
public release qualification were performed. The installed provider
library path configuration is machine-specific; a distribution must
supply its matching CUDA/cuDNN dependencies.

## Curated CUDA validation and package boundary (2026-11-26)

Actual curated OBS inference passed with the local ONNX Runtime GPU 1.30.0
library, matching CUDA 13/cuDNN dependencies and the older 1.17.1 header
API. Both Longpipe XS matte and flow sessions explicitly reported CUDA.
The shared matte ABI, stationary stabilization, source movement, motion
compensation and restored position passed on the RTX 2070 SUPER.
This split SDK is for local qualification only. CMake now refuses to stage
such a configuration before writing any files: headers, shared library
and notices must come from one matching `SIGHT_ORT_ROOT` distribution.
The separately acquired CUDA and cuDNN runtime dependencies must be visible
on the target system; they are not bundled by Sight. Use a fresh build
folder when changing SDKs, and verify actual providers and model-backed
inference. This is not a portable GPU installer certification.

## Matched publisher CUDA SDK (2026-11-26)

For Linux x86_64, the official Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0
NuGet package contains matching native headers, library, CUDA provider and
third-party notices. The acquisition tool verifies the complete publisher
package SHA-256 and every selected output. TensorRT, .NET dependencies,
NVIDIA runtimes and model weights are not extracted. The ELF SONAME is
`libonnxruntime.so.1`; the SDK contains both the linker and loader names.

```sh
python3 tools/acquire-native-ort.py linux-x86_64 "$HOME/.local/share/sight-sdk/ort-cuda-1.30.0" --variant cuda-linux
cmake -S . -B build-sight-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ORT_ROOT="$HOME/.local/share/sight-sdk/ort-cuda-1.30.0"
cmake --build build-sight-cuda --parallel 4
bash tools/verify-release-stage.sh build-sight-cuda
# Supply actual matching NVIDIA CUDA 13 runtime library directories.
SIGHT_CUDA_RUNTIME_PATH=/absolute/path/to/cuda13/lib bash tools/verify-cuda-package.sh build-sight-cuda
```

The staged qualification helper checks all provider loader dependencies and
verifies that the test process uses the staged ONNX Runtime, then requires a
real CUDA-only MatMul with CPU execution fallback disabled. It launches no
OBS instance and downloads no model. The NVIDIA driver and matching CUDA
13 runtime libraries must be installed separately from their publishers.
Never mix CUDA 12 and CUDA 13 library directories or substitute a CPU
runtime. Consult the [ONNX Runtime provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
and [NVIDIA CUDA installation guidance](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

The matched SDK compiled Sight and passed actual staged CUDA-only inference
on the local RTX 2070 SUPER. The 19-file stage passed content and SONAME
checks. Without the explicit runtime directories, the provider reports
missing CUDA 13 libraries, as expected. This qualifies the local staged
runtime with external dependencies, not independent machine portability,
Windows CUDA, macOS acceleration, or an unrestricted public binary release.
The matched SDK was not tested with a live OBS filter in this pass because
OBS was already open; previous split-SDK CUDA Longpipe tests remain separate.
See [publisher package receipt](qualification/onnxruntime-cuda-sdk.json).
