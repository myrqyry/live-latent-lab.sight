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
Other ONNX filters retain their existing selection behavior.


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
