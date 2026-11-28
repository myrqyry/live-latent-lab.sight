# Tests

The validated Linux build registers model-free checks and conditionally
registers real-model and OBS graphics tests. The experimental full build
contains the older LiteRT, Stable Diffusion and other processor tests.

## Default validated-effects tests

Follow the [Linux build prerequisites](../README.md#build-and-verify), then
supply the model paths below to enable the Portrait Sketch and U²-Net checks.
Other cases have their own model, fixture and runtime requirements; see
[`cmake/validated_effects.cmake`](../cmake/validated_effects.cmake) for the
registration conditions. Use `ctest -N` to see what a build will run.

The [OBS render smoke](obs_sight_render_smoke.cpp) loads the built module
under `xvfb-run`, checks expected filter IDs and display names, and verifies
rendered output according to the case (effect output or passthrough). The
expected filter set depends on the build; see
[`src/plugin-main.cpp`](../src/plugin-main.cpp) for registrations. Native VRM
is a source, with its own `sight_native_vrm_obs_render` test.

```bash
cmake -S . -B build-verified -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ANIMEGAN_MODEL=/absolute/path/to/animegan_v3_portrait.onnx \
  -DSIGHT_U2NET_MODEL=/absolute/path/to/u2net.onnx
cmake --build build-verified
ctest --test-dir build-verified -N
ctest --test-dir build-verified --output-on-failure
```

ONNX models are not bundled. Keep real-image fixtures outside the repository.
MediaPipe cases take the matching runtime, model paths, and case-specific
fixtures such as `SIGHT_MEDIAPIPE_TEST_IMAGE` (512×512 PNG/JPG) or
`SIGHT_MEDIAPIPE_TEST_RGBA` (raw 512×512 RGBA). The Privacy V2 integration
uses the raw RGBA face fixture and `SIGHT_MEDIAPIPE_FACE_MODEL`; it also
requires OBS's `image-source` module. CMake checks common Linux plugin
locations; override with `-DSIGHT_IMAGE_SOURCE_MODULE=/path/to/image-source.so`.
The original privacy reference fixture used a licensed public portrait
(Pixabay girl-2704869), cropped to square and converted on 2026-10-27.

The render runner loads the built Sight module and rejects known ONNX or
OpenGL errors even if pixels change. These checks do not measure live OBS
preview FPS or webcam performance.

## Experimental LiteRT tests

In a build configured with `SIGHT_ENABLE_UNVERIFIED_EFFECTS=ON` and its
legacy SDK dependencies, run the focused LiteRT suite with:

```bash
ctest --test-dir build -R 'litert_' --output-on-failure
```

The tests cover SDK linkage, shared runtime failure handling, model-manager
checksum and atomic-install behavior, and geometry processor validation.

## Existing tests

- `test_transform` checks image transformation logic without the full OBS
  runtime.
- `test_stable_diffusion_backend` checks backend input validation and callback
  setup without loading a model.
- `test_memory_leak` exercises plugin source lifecycles when the OBS-linked
  test target is configured.
- `test_leaks` and `verify_sd_api` are diagnostic sources for their respective
  checks; they are not substitutes for an OBS graphics smoke test.
- `test_integration.py` is an OBS Python scripting experiment. It requires an
  OBS environment with `obspython` and is not part of the standalone CTest
  suite.

Tests that create or destroy OBS graphics resources require a valid OBS
graphics context. Do not treat a headless process without that context as
evidence that graphics callbacks are safe.

## AnimeGANv3 real-model smoke test

With `SIGHT_ANIMEGAN_MODEL` configured and the validated build compiled,
run the processor test and the Portrait Sketch OBS cases with:

```bash
ctest --test-dir build-verified -R '^sight_animegan_(real_inference|obs_.*)$' --output-on-failure
```

The processor test runs a 128×128 RGB frame through a real float32 NHWC ONNX
model and checks that the output differs from the input. The `sight_animegan_obs_*`
cases exercise the registered Portrait Sketch filter and real OBS texture
rendering, including square, landscape, portrait, and misaligned sizes. They
do not establish live FPS on a user camera or GPU acceleration.

## U²-Net Subject Cutout

The U²-Net model test checks the actual 320×320 NCHW ONNX input and the
first 1×1×320×320 fused mask output. The OBS test registers the real
`sight_u2net_filter` source and verifies the filter does not emit shader
errors. An input with no recognizable foreground can validly yield an
all-transparent matte; do not require synthetic gradients to have a subject.

To enable `sight_u2net_obs_subject`, configure `SIGHT_U2NET_TEST_IMAGE` with
the absolute path to a local 750×1000 subject image. The test requires OBS's
`image-source` module; its separate override is `SIGHT_U2NET_IMAGE_MODULE`.
This fixture is for tests only; keep it outside the repository.

For optional CUDA-bound matte tests and their model requirements, see the
[CUDA texture and tensor I/O guide](../docs/cuda-texture-interop.md).

## U²-Net performance diagnostic

The U²-Net processor sets ONNX intra-op threads to half the reported logical
CPU count, rounded down and clamped to 1–6. The graphics thread captures when
the worker requests a frame; previous results can render while the worker
is busy.

When eight masks have finished, OBS logs the first-eight inference average
and throughput, for example
`[Sight/U2Net] Inference average: ... ms/frame (... inference fps, first 8 frames)`.

For a controlled comparison of 1 versus 2, 4, 6, or 8 ONNX CPU threads
without GPU readbacks, run the `sight_u2net_benchmark` binary with
`/path/to/u2net.onnx <thread_count>`. It measures five actual inference runs
on a 320×320 RGB-like tensor, including first and warmed-up latency. This
benchmark is not a replacement for measuring live OBS preview FPS.

## Native-resolution cutout quality regression

The U²-Net Image-source smoke verifies that foreground RGB remains near
the original **full-resolution** pixel colors where the segmentation alpha
is nearly opaque (mean absolute RGB difference at most 8/255 across at least
1000 pixels), rather than stretching the low-resolution inference RGB.
This checks original color preservation, **not** fine mask-edge accuracy.

The optional Portrait Sketch detail test verifies 768-pixel inference loads
and renders successfully. The default setting is 512 px and a 256 px
quick option remains available. Higher resolution increases inference time.
The full-size image and custom shader must not trigger OBS/GL errors.
