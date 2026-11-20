# Live Latent Lab: Sight

Native OBS Studio tools for computer vision, visual effects and signal-driven
interactions. This is a local public-release candidate, not a published release.

## Linux x86_64 candidate

The validated build includes subject cutout, depth, portrait sketch,
face/body/hand/holistic/object tracking, unified tracking, a tracking
signal receiver, value routing, touch-field effects, privacy experiments,
and native VRM avatars. Spatial motion and text obfuscation depend on
optional system libraries. The optional GPU whole-body path needs a
separate LiteRT configuration. Native Live2D needs the private SDK and
is not part of this candidate. Legacy generative effects remain gated.

Model-backed effects require separately acquired compatible models and
runtime libraries. See [model setup](docs/model-downloads.md) and
[MediaPipe ABI notes](docs/mediapipe-1.2.3.md). Privacy V2 is experimental;
it is not a guarantee of anonymity or complete redaction.

## Build and verify

Requirements: Linux, OBS 32 development headers and libobs, C++20,
CMake, the matching ONNX Runtime 1.17.1, and xvfb-run for graphics tests.
The tracked ONNX Runtime copy is retained pending distribution review.

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

## Release gate

Public distribution is not cleared. The original README declared
GPL-2.0, but a top-level original-code license has not yet been confirmed.
Runtime notices, model terms, clean-machine acceptance and package
compatibility must be reviewed before publication. Windows and macOS
are not certified.
