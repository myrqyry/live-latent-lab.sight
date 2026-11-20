# MediaPipe 1.2.3 in Sight (Linux x86_64)

**Status: verified in a development build, not installed into the normal OBS
session.** Sight can load either its known legacy Tasks C library or the
official MediaPipe **1.2.3** manylinux x86_64 C library. The correct ABI
is selected by the runtime's known file size; the CMake packager also
requires the full SHA-256. Unknown builds are rejected, not guessed.

## Getting the official runtime

Download the **Linux x86_64 wheel** for
[mediapipe 1.2.3 on PyPI](https://pypi.org/project/mediapipe/1.2.3/).
This is a distribution of the C runtime plus Python bindings; Sight does
not require you to install the Python package.

- Wheel filename: mediapipe-1.2.3-py3-none-manylinux_2_28_x86_64.whl
- Wheel size: 43,342,798 bytes
- Wheel SHA-256: 1fcda646cc7b96f73453acc416ce31d0fba287d0f1fc095865272a7ea6986497
- Archive member: mediapipe/tasks/c/libmediapipe.so
- Extracted ELF size: 127,908,512 bytes
- Extracted ELF SHA-256: 7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63

For a locally installed copy renamed libmediapipe-1.2.3.so next to
Sight's plugin binary, new Holistic filters select the matching 1.2.3
runtime automatically. A missing or unrecognized runtime leaves the
field empty, without silently falling back to the old MediaPipe ABI.
This local discovery checks filename and known ABI size, not a live
SHA-256: validate any manually supplied runtime before installing it.

A wheel is a ZIP archive. Extract just that shared-library member into
a user-controlled directory, **outside** the existing Sight installation,
then verify its checksum using:

    sha256sum /full/path/to/1.2.3/libmediapipe.so

In OBS, select that file in the **MediaPipe Tasks C runtime** field of a
Sight Face, Body, Hand or Holistic Tracking filter. Keep selecting your own local
Face/Pose/Gesture .task model in the separate model field. **Sight never
bundles or downloads model weights.** The original legacy runtime remains
a supported alternative for existing scenes.

Don't rename the wheel to .so, copy the entire wheel into the plugin
folder, or overwrite a loaded runtime while OBS is running.

## Default selection follow-up (2026-11-18)

Face, Body and Hand now prefer the recognized local 1.2.3 runtime
when present, using the same discovery boundary as Holistic. They
retain the tested legacy ABI alternative. Existing explicitly saved
paths are not migrated. Package/test configuration verifies SHA-256;
runtime discovery uses known ABI size and requires a regular file
rather than following a symlink. The 1.2.3 runtime remains CPU-delegate
in this implementation. See
[tracking modernization](verification/2026-11-18-tracking-modernization.md).

## What has been verified

- MediaPipe's official 1.2.3 wheel/library SHA-256 and exported C
  FaceLandmarker, PoseLandmarker and GestureRecognizer entry points.
- The upstream 1.2.3 header's MpBaseOptions has a different layout and
  delegate values. Sight maintains separate C ABI option structs with
  sizeof/offsetof assertions against the published Linux x86_64 headers.
- Actual local model calls: 478 face landmarks, 52 blendshape scores and
  head transform; 33 pose landmarks plus 33 world-space points; 21 hand
  landmarks, hand-world points, and Thumb_Up gesture.
- Both legacy and 1.2.3 paths passed all three focused model tests.
  New-runtime Face/Body/Hand OBS filters passed actual offscreen GPU
  rendering and process teardown under Xvfb.
- The full validated-effects suite passed **54/54 CTests** with the
  optional 1.2.3 regression tests enabled.
- CMake's isolated staged install with the new runtime passed 16 file
  checks and packaged no model weights. Approximate staged package size:
  **145 MB** with 1.2.3 versus **51 MB** with the legacy runtime.

To repeat additional 1.2.3 checks with local model test fixtures:

    cmake -S . -B build-verified \
      -DSIGHT_MEDIAPIPE_123_LIB=/full/path/to/1.2.3/libmediapipe.so \
      -DSIGHT_GESTURE_TEST_RGBA=/full/path/to/hand-rgba-512.raw
    cmake --build build-verified -j2
    ctest --test-dir build-verified --output-on-failure

The existing model/fixture options documented in README.md must also be
configured for those optional real-model tests to register. No runtime
downloads or installations occur through CTest.

## Holistic Landmarker: OBS producer and conversion proof

The official standalone float16 Holistic model is available at:

https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

- Model size: 13,683,609 bytes
- SHA-256 of the observed download:
  e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
- The model was downloaded to a **temporary test directory**, never bundled,
  installed into OBS, committed, or fetched by the Sight plugin.
- Isolated C API VIDEO inference returned 478 face points, 33 pose and
  33 pose-world points, both 21-point hands on a hand test image, 52
  face blendshapes, and a pose segmentation mask. On two 512x512 reference
  images, the third-frame inference was about 19–20 ms (not a webcam
  benchmark or guarantee of faster total OBS performance).
- Sight now has a *pure conversion adapter* at
  src/core/holistic_signal_adapter.hpp. It copies raw Holistic results
  into separate existing Face/Body/Hand TrackingData payloads, preserving
  per-hand world coordinates, and rejects invalid/partial modalities.
  A real model probe validated the adapter against frames and confirmed
  result-buffer lifetime independence after MediaPipe cleanup.
- The adapter deliberately **does not fabricate** gesture labels,
  face-transformation matrices, or a segmentation-mask snapshot. Holistic
  does not classify gestures; only Gesture Recognizer produces names such
  as Thumb_Up. Holistic mask data must be copied/owned before exposing it
  to other components.

The **LLL.sight Tracking → Holistic** filter runs one
VIDEO graph, publishing synchronized Face/Body/Hand snapshots on
independently named channels (defaults holistic/face, holistic/body,
holistic/hand). Model weights and native runtime remain user-supplied.
A local **Show tracking overlay here** checkbox defaults off. Enabling
it draws **All, Face, Body, or Hands (both)** directly on the camera
without starting another model or stopping signal publication.

**Tracking stability (overlay only)** defaults to Balanced with **55%
minimum landmark confidence** and fading temporarily lost joints.
Off (raw) bypasses it; Responsive, Balanced and Stable require 1, 2
or 3 consecutive plausible observations before displaying new joints.
The preview softens small jitter, rejects implausible large arm jumps,
and uses body-wrist proximity plus temporal consistency for left/right
hands. Low-confidence joints fade briefly then clear. **Published
face, body and hand tracking data remains unmodified.** Holistic hands
often lack independent per-point visibility confidence, so a percentage
threshold alone cannot reject all phantom hands. The
optional **Use person mask to suppress background limbs** gating now
uses the same Holistic graph. When enabled it copies the model's
Vec32F1 pose mask (including stride) into owned memory before closing
the native result and checks plausible elbows/wrists/hand roots against
that silhouette. It is **off by default**, requires reloading the graph
when toggled, leaves missing/uncredible masks as passthrough, and
does not publish filtered landmarks or masks to other consumers.
This is background-aware detection filtering, not background removal;
morphological mask cleanup and live low-light acceptance remain
separate work.

Isolated real OBS 32.2 OpenGL tests verified camera capture,
one-graph inference, three same-frame read-only snapshots, unchanged
video with overlay off, GPU overlays in all four modes, and a live
checkbox/mode update without restarting inference, plus teardown. OBS may delay actual object destruction
after a filter is detached, so a stricter test confirmed the last
snapshots become unreadable when their requested max-age expires
(5 seconds in the test). This does not prove immediate erasure.
Normal webcam performance, user GUI and downstream ViiRM acceptance
remain unverified.

## What this does *not* add automatically

- **Holistic production-camera acceptance:** isolated native and OBS
  tests pass. Continuous camera FPS/soak, mirroring, and actual ViiRM
  receiver experience remain separate, unverified work.
- GPU/NPU delegates: the new release adds NPU and LiteRT pathways, but
  Sight's verified Face/Body/Hand path remains CPU-only.
- Newer-model or arbitrary-runtime compatibility: a 1.2.4/1.3.0 .so is
  *not* accepted simply because it exports the same symbol names.
- Commercial distribution approval: shipping a library from a wheel
  separately still requires its dependency and redistribution review.
  This runtime is different from the model weights users download.

The [MediaPipe 1.2.2 release](https://github.com/google-ai-edge/mediapipe/releases/tag/v1.2.2)
also clarifies that HandLandmarker/GestureRecognizer Left/Right refer to
the **subject**, not the viewer. Passing horizontally mirrored frames
inverts the labels; use Sight's existing VRM mirror-label control only
where the producer's actual image orientation calls for it.

Upstream: [MediaPipe 1.2.3 release](https://github.com/google-ai-edge/mediapipe/releases/tag/v1.2.3).
