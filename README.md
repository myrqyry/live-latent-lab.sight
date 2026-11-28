# lll.sight

Native OBS Studio tools for computer vision, visual effects and signal-driven
interactions.

**Source:** [live-latent-lab.sight](https://github.com/myrqyry/live-latent-lab.sight)
(currently private). This is a curated, separately rooted release-candidate
repository, not the historical development repository. There are currently
**no GitHub releases or redistribution-cleared installers**.

**Start here:** Review [models and downloads](docs/model-downloads.md), then
[build and verify](#build-and-verify) locally on Linux with OBS 32. Optional
GPU tracking needs the extra LiteRT runtime and models listed below.
[Release readiness](docs/release-readiness.md) records what still blocks
public source/binary distribution; [release status](RELEASE_STATUS.md) records
the specific verification evidence.

## Linux x86_64 candidate

All entries use the `lll.sight` prefix in OBS.

| Filter or source | Function |
| --- | --- |
| Tracking | Face, body, hands and gestures, holistic, objects, and optional GPU WholeBody (RTMW) |
| Portrait Sketch (ControlNet Prep) | Model-based portrait line art |
| Preprocessors | Native image transforms, Depth Anything V2, TEED, PiDiNet, HED and scribble |
| Subject Cutout (U2Net, MODNet, Longpipe) | Foreground separation, optional motion-compensated Longpipe stabilization and shared matte |
| Depth Camera (Depth Anything) | Relative inverse depth and shared depth map |
| Spatial Motion (DIS) | Optional optical flow and motion signals |
| Text Obfuscation (OCR) | Optional local OCR, named regions, visual editor and confirmed text routing |
| Privacy V2 — Conceal All Faces (Experimental) | Experimental face concealment |
| Tracking Signal Receiver | Shared tracking snapshots without another model |
| Value Router (Signals + Waves) | Drive OBS properties from signals and waves |
| VRM Avatar (Native) | Native avatar source |

Choose a mode and model in **lll.sight Tracking**. Each instance runs
one selected model pipeline; add another instance for simultaneous
tracking. The former standalone trackers are registered as hidden compatibility
types for existing saved scenes (WholeBody requires the RTMW backend).
New tracking filters use the single unified entry.

### Use with hypershader

Sight provides reusable tracking, matte, depth, motion, optical flow and
OCR-region signals. hypershader consumes them for creative effects.
For hand ripples, place Tracking in Hands & gestures mode upstream of
hypershader on the same source, disable its overlay, enable shared Sight
snapshots in hypershader, select the hand channel and choose
**Sight: Hand Ripples**. The former standalone Touch Field filter is retained
as a hidden compatibility type so existing scenes retain their effect.

The expanded Linux build exposes the eleven entries above, plus hidden
legacy compatibility registrations. Preprocessors, Spatial Motion
and Text Obfuscation require optional system libraries; WholeBody needs
LiteRT GPU support. Native Live2D remains an [optional local development build](docs/live2d-native.md) using a separately acquired official SDK; it is excluded from the default preview binary, not removed from the source. Unvalidated generative effects remain gated.

Subject Cutout supports U²-Net/U²-NetP, MODNet and Longpipe, per-instance
Auto/CPU/required CUDA selection, adaptive optional flow and bounded shared
matte/motion. See [cutout models](docs/model-downloads.md#subject-cutout-models),
[precision](docs/cutout-precision.md) and [flow](docs/longpipe-flow.md).
Model/runtime preflight, actual backend status and recent inference health are
available in supported filters. Optional [semantic obfuscation and geometry](docs/semantic-obfuscation.md)
use user-supplied ONNX weights; not a guarantee of redaction.

Model-backed effects require separately acquired compatible models and
runtime libraries. See [optional CUDA](docs/onnx-cuda.md), [model setup](docs/model-downloads.md) and
[MediaPipe ABI notes](docs/mediapipe-1.2.3.md). Privacy V2 is experimental;
it is not a guarantee of anonymity or complete redaction.

## Native Preprocessors

On Linux, with OpenCV available, add **lll.sight Preprocessors**. The
single filter includes native Canny, binary, lineart, PyraCanny, scribble,
luminance, intensity, palette and tile shuffle modes, plus local Depth
Anything V2, TEED, PiDiNet and HED ONNX inference. Model-backed modes
require compatible local ONNX files; unavailable modes do not fabricate
outputs. No ComfyUI or Python service is required at runtime.

The optional dependency-free Rust normalization/postprocessing core is
enabled with `-DSIGHT_ENABLE_RUST_PREPROCESSOR_CORE=ON` and built with
Cargo offline. The default C++ path remains functional. Disable the
filter with `-DSIGHT_ENABLE_PREPROCESSORS=OFF`. See the
[native interface contract](docs/native-preprocessor-interface.md) for
per-instance counters, ABI and deterministic checks. Windows/macOS
native baseline does not yet include this OpenCV-dependent filter.

## Optional native diffusion (experimental)

The native generator and live image-to-image filter are available as
opt-in additions to the validated build. They support local checkpoints,
recursive feedback, camera/object-motion preview, temporal stabilization,
and an optional Q8_0 weight mode in Live Diffusion. Real CUDA acceptance
passed on Turing (RTX 2070 SUPER); other devices and sustained
high-resolution performance have not been qualified.

See [native diffusion setup](docs/native-diffusion.md) for source builds,
requirements, and verification. Neither checkpoints nor proprietary GPU
model assets are included. The optional [neural-rendering integration](docs/neural-rendering.md)
is also available as source; its external graph and compatible checkpoint
remain user-provisioned, not bundled.

## Plugin preferences

With Qt6 Widgets available, open **Settings → Plugins → lll.sight**
to choose model and runtime browsing folders and access model download,
setup and licensing resources. Sound and Sight share one Plugins category
when both are installed, but maintain independent preferences. Apply or OK
saves; Cancel discards unapplied edits. Existing selected models and
automatic tracking discovery are unchanged. Build with
`-DSIGHT_SETTINGS_PAGE=OFF` to omit this optional Qt integration.
The OBS 32.2 dialog integration uses a private widget layout and is not an
official frontend API; other versions are unqualified.

## Build and verify

Windows x64 and macOS (Apple Silicon/Intel) native baseline builds are under
qualification through the shared CI matrix. They use the same maintained
CPU/ONNX and VRM sources with matching native SDKs, not Linux runtime binaries.
Optional Linux-only backends and private Qt integration are not enabled in
that baseline. See [shared qualification](.github/scripts/SHARED-CI.md).

Requirements: Linux, OBS 32 development headers and libobs, C++20,
CMake, nlohmann JSON development headers (`nlohmann-json3-dev` on Ubuntu),
the matching ONNX Runtime 1.17.1, and xvfb-run for graphics tests.
The [pinned ONNX Runtime dependency](docs/onnx-runtime.md) is required for
vision inference. Its headers, shared library and legal notices are retained;
redistribution review remains open. A separately acquired MediaPipe runtime
is not copied into the package merely because it is supplied for testing.
For explicit local staging only, set `SIGHT_STAGE_LOCAL_MEDIAPIPE=ON` with
the ABI-verified `SIGHT_MEDIAPIPE_LIB` path.

```sh
cmake -S . -B build-public -DCMAKE_BUILD_TYPE=Release
cmake --build build-public --parallel 4
ctest --test-dir build-public --output-on-failure
./tools/verify-release-stage.sh build-public
cmake --install build-public --prefix "$PWD/stage"
```

Tests with actual inference require explicit model/runtime/fixture paths;
a reduced model-free test matrix does not qualify all tracking modes.
Do not install into an active OBS process. No separately acquired model files are shipped. The vendored BSD
libfacedetection implementation includes its embedded CNN arrays.

## Optional GPU tracking

RTMW whole-body tracking and the YOLOX GPU detector were verified in a
separate Linux build using the existing matched LiteRT 2.3 native runtime
and SDK. Supply those files explicitly; do not redistribute them until
their terms have been reviewed.

```sh
cmake -S . -B build-public \
  -DSIGHT_ENABLE_RTMW=ON \
  -DSIGHT_LITERT_SDK_DIR=/path/to/litert_cc_sdk \
  -DSIGHT_LITERT_RUNTIME_DIR=/path/to/ai_edge_litert \
  -DSIGHT_LITERT_LICENSE=/path/to/LICENSE
cmake --build build-public --parallel 4
```

Select the corresponding compatible model files in OBS. Real GPU acceptance
requires the RTMW and YOLOX model files plus the MediaPipe tracking fixtures.
The earlier expanded Linux configuration passed all 92 configured CTest
checks and real hypershader integration. The latest curated integration
completed 114 of 115 tests on the first run; the corrected VRM easing
assertion passed on a targeted rebuild. See the release status for scope.
See [release status](RELEASE_STATUS.md) for exact evidence and limitations.

## Release gate

Public distribution is not yet cleared. Sight’s original code is licensed
under [GPL-3.0-only](LICENSE). See [licensing and third-party boundaries](LICENSING.md).
Runtime provenance, model terms, clean-machine acceptance and package
compatibility must still be reviewed before publication. Windows and macOS
are not certified.
