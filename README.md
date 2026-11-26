# Live Latent Lab: Sight

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

All entries use the `LLL.sight` prefix in OBS.

| Filter or source | Function |
| --- | --- |
| Tracking | Face, body, hands and gestures, holistic, objects, and optional GPU WholeBody (RTMW) |
| Portrait Sketch (ControlNet Prep) | Model-based portrait line art |
| Subject Cutout (U2Net) | Foreground separation and shared subject matte |
| Depth Camera (Depth Anything) | Relative inverse depth and shared depth map |
| Spatial Motion (DIS) | Optional optical flow and motion signals |
| Text Obfuscation (OCR) | Optional local text-region detection and treatment |
| Privacy V2 — Conceal All Faces (Experimental) | Experimental face concealment |
| Tracking Signal Receiver | Shared tracking snapshots without another model |
| Value Router (Signals + Waves) | Drive OBS properties from signals and waves |
| VRM Avatar (Native) | Native avatar source |

Choose a mode and model in **LLL.sight Tracking**. Each instance runs
one selected model pipeline; add another instance for simultaneous
tracking. The six former standalone trackers are no longer registered.

### Use with HyperShaderFX

Sight provides reusable tracking, matte, depth, motion, optical flow and
OCR-region signals. HyperShaderFX consumes them for creative effects.
For hand ripples, place Tracking in Hands & gestures mode upstream of
HyperShaderFX on the same source, disable its overlay, enable shared Sight
snapshots in HyperShaderFX, select the hand channel and choose
**Sight: Hand Ripples**. The former standalone Touch Field filter is removed.

The expanded Linux build registers the ten entries above. Spatial Motion
and Text Obfuscation require optional system libraries; WholeBody needs
LiteRT GPU support. Native Live2D remains an [optional local development build](docs/live2d-native.md) using a separately acquired official SDK; it is excluded from the default preview binary, not removed from the source. Unvalidated generative effects remain gated.

Model-backed effects require separately acquired compatible models and
runtime libraries. See [model setup](docs/model-downloads.md) and
[MediaPipe ABI notes](docs/mediapipe-1.2.3.md). Privacy V2 is experimental;
it is not a guarantee of anonymity or complete redaction.

## Plugin preferences

With Qt6 Widgets available, open **Settings → Plugins → LLL.sight**
to choose model and runtime browsing folders and access model download,
setup and licensing resources. Sound and Sight share one Plugins category
when both are installed, but maintain independent preferences. Apply or OK
saves; Cancel discards unapplied edits. Existing selected models and
automatic tracking discovery are unchanged. Build with
`-DSIGHT_SETTINGS_PAGE=OFF` to omit this optional Qt integration.
The OBS 32.2 dialog integration uses a private widget layout and is not an
official frontend API; other versions are unqualified.

## Build and verify

Requirements: Linux, OBS 32 development headers and libobs, C++20,
CMake, the matching ONNX Runtime 1.17.1, and xvfb-run for graphics tests.
The [pinned ONNX Runtime dependency](docs/onnx-runtime.md) is required for
vision inference. Its headers, shared library and legal notices are retained;
redistribution review remains open.

```sh
cmake -S . -B build-public -DCMAKE_BUILD_TYPE=Release
cmake --build build-public --parallel 4
ctest --test-dir build-public --output-on-failure
./tools/verify-release-stage.sh build-public
cmake --install build-public --prefix "$PWD/stage"
```

Tests with actual inference require explicit model/runtime/fixture paths;
a reduced model-free test matrix does not qualify all tracking modes.
Do not install into an active OBS process. No user models are shipped.

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
The final expanded Linux configuration passed all 92 configured CTest
checks, and release staging and the real HyperShaderFX integration passed.
See [release status](RELEASE_STATUS.md) for exact evidence and limitations.

## Release gate

Public distribution is not cleared. The original README declared
GPL-2.0, but a top-level original-code license has not yet been confirmed.
Runtime notices, model terms, clean-machine acceptance and package
compatibility must be reviewed before publication. Windows and macOS
are not certified.
