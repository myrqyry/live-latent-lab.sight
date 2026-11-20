# Hand & Gesture Motion — verified Linux/OBS path

**LLL.sight Hand & Gesture Motion (MediaPipe Tasks)** is a standalone OBS filter
backed by the **Gesture Recognizer**, which returns hand landmarks *and*
built-in classifications from one model invocation.

## Artifacts and provenance

- MediaPipe native legacy Linux library: the exact ABI-pinned Tasks C library
  described in [README-tasks.md](README-tasks.md).
- Gesture Recognizer (float16/1):
  `https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`
- Expected task SHA256:
  `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`
- Task model filename:
  `~/.config/obs-studio/plugin_config/live-latent-lab-sight/models/gesture_recognizer.task`
- C API declarations for Gesture Recognizer are Apache-2.0 derived and
  checked against the pinned legacy binary, as confirmed by the real runtime
  VIDEO graph test. Models are deliberately not committed to git.

The official MediaPipe gesture documentation provides image fixtures:
`https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg`,
`victory.jpg` and `pointing_up.jpg` in the same directory.

## Actual functional boundary

- Uses background MediaPipe VIDEO-mode inference (CPU), 2 hands maximum.
- For each detected hand: 21 normalized XYZ landmarks, handedness label,
  top gesture class and its confidence, and a continuous 0..1 pinch value.
  Pinch is calculated from thumb-tip/index-tip distance relative to palm
  length; the normalized mapping is heuristic, not a calibrated physical
  distance or an independent ML gesture classifier.
- Hand snapshots are published to a channel (default `hand`) with an explicit
  empty result on no detection, single publisher ownership and expiry.
- The first detected hand is mirrored into the old `landmarks` field for
  compatible consumers. Two hands are available independently through the
  `hands` array. **Hand index is detection order**, not permanent identity;
  index order may change as hands overlap or leave the frame.
- The OBS filter draws a GPU-composited 21-point skeletal overlay for every
  detected hand, up to 2; its original-resolution source stays intact.
- The Tracking Signal Receiver can draw the same hands on another OBS source
  without creating a second model. Coordinates remain normalized to producer
  framing.
- Touch Field can consume the same published hand channel to create
  real OBS GPU refraction around detected index fingertips, without
  another model. See [Touch Field](README-touch-field.md).
- Value Router adds features: hand point X/Y, normalized pinch and confidence
  of a chosen built-in gesture category. Use hand number 0 or 1 and
  point index 0..20 for coordinates.
- Recognized built-in classes: Closed_Fist, Open_Palm, Pointing_Up,
  Thumb_Down, Thumb_Up, Victory, ILoveYou; `None` and unrecognized
  categories are not treated as positive gesture matches.
- Tracking cadence, input edge length, temporal smoothing, and optional
  local overlay match Face/Body Motion. Missing models produce passthrough,
  not invented landmarks.

## Validation

- Standalone VIDEO C API probe against official real hand photographs:
  Thumb_Up, Victory, Pointing_Up classified correctly; returned 21
  landmarks and exited cleanly.
- `sight_tasks_hand_real_inference`: Sight backend with *real* hand RGBA
  frame, three VIDEO timestamps, 21 landmarks, valid pinch and Thumb_Up.
- `sight_tasks_obs_hand`: actual OBS image source and GPU composited hand
  overlay with real inference, plus GPU/OBS shutdown.
- `sight_signals_obs_hand_receiver`: separate invisible hand producer and
  receiver with real landmark overlay pixels.
- `sight_value_router_obs_hand_pinch`: separate invisible producer and
  independent numeric OBS setting driven by actual hand pinch value.
- `sight_tracking_overlay_unit`, `sight_tracking_signal_bus_unit`,
  `sight_value_modulation_unit`: two-hand renderer, complete signal
  snapshot, feature extraction and arbitrary range mapping.
- Existing Face, Body, Sketch, Depth, Cutout, Signal Receiver, Router
  regressions remain in the same CTest suite.

**Not verified**: simultaneous live two-hand detections, live webcam tracking
and identity continuity as hands cross, end-user mobile GPU performance, or
a direct external-plugin/MIDI/scene-transform route. No mock fallback
is installed. Source pictures were converted to 512x512 RGBA for tests;
they are not committed or distributed as model data.


## Repeated render-frame inference (2026-11-04)

OBS may render 60 times per second while a webcam supplies only 30
camera frames. Hand Motion's worker compares **exact rendered RGBA
bytes** with the preceding successfully inferred image, suppressing
identical-image model calls. A 250ms heartbeat continues processing
unchanged images to keep the tracking bus valid. This comparison
happens outside the OBS graphics thread. It is NOT guaranteed to find
all duplicated camera frames when upstream effects re-render with
different pixel values.

Regression: `sight_hand_frame_dedup_unit` (identical/one-pixel changes,
resizing, reset, heartbeat, and 60 render calls / 30 distinct images).
After loading a new plugin instance verify the runtime marker
`HAND_RENDER_DEDUP_ACTIVE_20261104_Q247`, and inspect ten-second
`Hand delivery` lines for `skipped=N byte-identical rendered images`.
If this number is near zero, the rendered frames are not byte-identical,
and real capture-timestamp integration remains an alternative; don't
infer that the webcam is truly delivering 60 distinct frames/second.


Installed-file runtime regression (2026-11-04): used the existing
`obs_sight_render_smoke.cpp` image-source test with a real hand photograph
resized to 512px (temporary local fixture, not committed) and the
pinned local Gesture Recognizer task model. Test loaded **the installed**
user plugin, logged its dedup marker and produced 21 real landmarks,
then ~10 seconds of held, static input logged 29 hand-containing model
results and 53 byte-identical rendered images skipped. This validates
the actual OBS filter/worker/model path; live camera duplicate rate
and subjective motion smoothness are unverified until user restarts.
