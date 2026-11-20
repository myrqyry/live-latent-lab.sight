# Tests

The default build contains real PortraitSketch, U²-Net, Depth Anything,
MediaPipe Tasks, and Privacy V2 inference plus OBS graphics
smoke tests. The experimental full build also contains the older LiteRT, Stable
Diffusion and other processor tests.

## Default validated-effects tests

Configure with the model paths below to enable each effect group's
regression checks; tests register only when their models, fixtures, and the
pinned MediaPipe runtime are present. The OBS smoke runs through the actual
plugin with an OpenGL context provided by `xvfb-run`, checks that **exactly
the verified Sight filter set is registered** (Portrait Sketch, Subject
Cutout, Depth Camera, Tracking, Privacy V2,
Tracking Signal Receiver, Value Router) and the output video differs from
the input. See the main `README.md` for the full flag list.

```bash
cmake -S . -B build-verified \
  -DSIGHT_ANIMEGAN_MODEL=/absolute/path/to/animegan_v3_portrait.onnx \
  -DSIGHT_U2NET_MODEL=/absolute/path/to/u2net.onnx
cmake --build build-verified
ctest --test-dir build-verified --output-on-failure
```

ONNX models are not bundled. The MediaPipe/Privacy OBS tests additionally need
real-image fixtures — a 512x512 face picture (`SIGHT_MEDIAPIPE_TEST_IMAGE`,
PNG/JPG) and the same picture as raw 512x512 RGBA
(`SIGHT_MEDIAPIPE_TEST_RGBA`) — kept outside the repository, for example under
`~/.cache/sight-test-fixtures/`. The privacy reference is a licensed public
portrait (pixabay girl-2704869, cropped to square) converted on 2026-10-27;
the pinned detector requires a *real* face, so synthetic or gradient images
are rejected by design. The Privacy V2 OBS integration test also
requires OBS's image-source module. CMake checks common Linux plugin
locations; use -DSIGHT_IMAGE_SOURCE_MODULE=/path/to/image-source.so to
override. The runner uses the built module's runtime instead of an installed
Sight copy. The tests cover square, landscape, portrait,
and misaligned sizes. A U²-Net mask test validates real segmentation and
confirms a nonconstant alpha channel on a local subject image. The tests fail
on ONNX broadcasting or OpenGL shader errors
rather than merely checking a changed pixel count. If the system has OBS's
`image-source` module, an additional test generates a temporary BMP and
runs the filter on that real OBS source type. The tests exercise CPU inference
and capture/render callbacks under software OpenGL; they do not measure
real-time FPS on the user's active GPU or camera.

## Experimental LiteRT tests

Run the focused LiteRT suite with:

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

This checks the existing Anime Style mode with a real ONNX model, rather than
mocking the inference path. The supplied model must have float32 NHWC inputs
and outputs. The script compiles `src/core/enhancement_processor.cpp` against
an existing ONNX Runtime C++ SDK and runs a 128 x 128 RGB frame through it.

```bash
ORT_ROOT=/path/to/onnxruntime \
  tests/run_animegan_inference.sh /path/to/animegan_v3_portrait.onnx
```

It requires the ONNX Runtime `include/` and `lib/` directories. No model is
bundled with the test. Passing means the model loaded, produced a correctly
sized tensor, and changed actual pixel values. The separate
`sight_animegan_obs_render` test verifies plugin loading, filter registration,
and real OBS texture rendering. Neither test establishes live FPS on a user
camera or GPU acceleration.

## U²-Net Subject Cutout

The U²-Net model test checks the actual 320×320 NCHW ONNX input and the
first 1×1×320×320 fused mask output. The OBS test registers the real
`sight_u2net_filter` source and verifies the filter does not emit shader
errors. An input with no recognizable foreground can validly yield an
all-transparent matte; do not require synthetic gradients to have a subject.

To additionally test alpha variation on a local subject image with
dimensions 750×1000, configure `SIGHT_U2NET_TEST_IMAGE` with its absolute
path. This is a **test-only** option; no user image is committed.

## U²-Net performance diagnostic

The U²-Net filter uses up to six physical-core-equivalent ONNX CPU threads
when available, leaving other cores for OBS. Its graphics thread only captures
another frame after an inference finishes; previous results continue to
render normally while the worker is busy.

When eight masks have finished, OBS logs an actual rolling inference average
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
