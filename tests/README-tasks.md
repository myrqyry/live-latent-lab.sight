# Sight MediaPipe Tasks tracking evidence

Three OBS filters use an independent native Tasks backend rather than copying
the user's HyperShaderFX or Exeldro Move fork implementation.

## Reproducible pinned artifacts

- Legacy runtime SHA256: 7ba887d5e09d87440b3713816ec02897d6a849eadf2325c9aabe3707cbdac86b
- Official MediaPipe 1.2.3 Linux x86_64 runtime SHA256:
  7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63
  (from the verified PyPI wheel). See [runtime setup](../docs/mediapipe-1.2.3.md).
- face_landmarker.task SHA256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
- gesture_recognizer.task SHA256: 97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482
- pose_landmarker.task SHA256: 64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b

The native library is bundled only on validated local installations, and
task models remain in the Sight persistent configuration models folder.
Neither binary nor model weights are in git. The C API declarations are
Apache-2.0 derived; legacy headers remain unchanged, and Sight has a
separate 1.2.3 C options ABI overlay. See the legacy header README for
provenance. CMake accepts only the two tested SHA-256 binaries at package
time; the runtime loader identifies their ABI by known size and checks
the required C symbols. It does not hash the library on each filter load.

## What has actually been tested

A standalone native probe and a compiled Sight backend both executed
VIDEO mode on a real face photograph. The face task returned 478 landmark
coordinates, 52 expression scores, and a 4x4 head transformation matrix.
The pose task returned 33 body landmarks. The gesture task returned
21 hand landmarks and correctly classified the official Thumb Up,
Victory and Pointing Up photographs; simultaneous two-hand detection
has not been validated in a live session. Succeeding model calls on a
static photograph do not establish live video tracking quality.

Face, Body and Hand OBS tests render their actual overlays on an OBS Image source,
verify genuine inference occurred, then remove each filter and tear OBS down.
Older Sight effects retain their own ONNX/OBS regressions. Both trackers
allow user-selectable inference cadence and landmark temporal smoothing;
the output source RGB remains native resolution.

A crash in OBS shutdown was reproduced and attributed via core backtrace
to pthread TLS destructors calling unmapped libmediapipe code. Loading the
verified library with RTLD_NODELETE fixes this in isolated OBS tests.
All graphs are still closed on their worker thread. Do not swap the library
for a different ABI without testing its complete load/inference/shutdown path.

## Holistic integration

The Holistic 1.2.3 OBS filter runs a single graph, publishing
Face/Body/Hand channels. Overlay rendering defaults off, but users can
enable same-filter overlays for All / Face / Body / Hands without running
another model. Real OBS/OpenGL checks exercise the invisible passthrough,
four visible GPU modes, and live toggle+mode update.
The HolisticOverlayStabilizer deterministic suite tests low confidence,
occluded or disappearing joints, sudden impossible jumps, sustained
motion recovery, small jitter, independent left/right hand identity,
phantom detached hands, and fading. Real OBS tests also cover the
default Balanced / 55% / fade-on configuration and alternative
Off / Stable / higher threshold / no-fade controls. Original
TrackingSignalBus snapshots remain raw model observations.
The separate person-mask unit test exercises plausible off-person limbs,
missing/invalid/empty/full masks and original-signal preservation.
Real native 1.2.3 inference verifies an owned 512x512 pose mask that
survives result teardown and is credible with the test person's torso.
Two real OBS tests exercise the optional default-off mask checkbox
and enabling it on a running source, including an actual graph reload.
A separate test
checks one-model inference, three atomic same-frame snapshots,
versioned procedure interoperability and max-age expiry after removal.
The native real-model runner and collision unit tests are independent.

Holistic does not emit gesture classification or a face/head transform.
Its segmentation image is not exported. Normal webcam performance and
ViiRM user experience remain unverified.

## Native object detection and short-lived IDs

A separate Object Tracking filter uses the official MediaPipe 1.2.3
Object Detector VIDEO API, with Google EfficientDet-Lite0 float32.
Official sample photo: cat 0.790057, dog 0.752158 on a real 512x333
RGBA frame. A model-backed CTest validates the same result over
monotonically increasing timestamps. The ObjectTracks unit tests
exercise identity retention across reordering, short disappearance,
short crossing motion, and malformed boxes. The independent overlay
unit verifies pixel output and visible identity badges.

Two isolated OBS/OpenGL tests on the exact built Sight module verify
real category labels and IDs through sight_object_read_v1, box overlay
pixels when enabled, and unchanged source pixels when disabled.
These checks require an existing compatible TFLite model and the pinned
MediaPipe 1.2.3 runtime. No weights are bundled or recorded in Git.
With all local model/fixture paths supplied, the full build passed
75 of 75 configured CTests. See docs/object-tracking-mediapipe.md.

## Deliberately not claimed

- No live-webcam, multi-subject, occlusion, or tracking-latency guarantee
- No automatic OBS scene item movement or Move actions yet
- In-process cross-filter snapshots are available through the typed signal
  bus and a versioned private OBS procedure; no WebSocket or cross-process
  interface is implemented
- No face landmarks using the old HyperShaderFX ONNX wrapper yet
- No LiteRT GPU backend or CUDA execution path for these tasks
- No simulated or placeholder inference
