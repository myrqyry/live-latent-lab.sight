# Sight Tracking Signal Bus

## Contract

A MediaPipe Tasks Face/Body/Hand filter is a producer, not the owner of every
effect using its output. The producer publishes an immutable snapshot after
each actual inference to a named channel, defaulting to `face`, `body` or `hand`.

Each snapshot contains:
- 478 face, 33 pose, or up to two sets of 21 hand landmarks;
- optional face expression blendshapes and facial transformation matrix;
- optional hand gesture class/score, handedness and pinch amount per hand;
- model kind, producer input dimensions, monotonic revision and timestamp;
- explicit empty subject landmarks if inference succeeds without a detection.

The `Sight: Tracking Signal Receiver` is a separate OBS filter. It can
draw the latest body/face/hand landmarks from a specified channel without
opening a model or running inference. The producer's visual overlay can
be disabled independently. GPU compositing still preserves native-resolution
source colors.

The **bus is internal to a single Sight plugin inside one OBS process**.
It is not an OBS WebSocket API or cross-process service. External plugins can
now request a **read-only versioned snapshot** through the receiver's native
libobs procedure handler (described below). The receiver itself still only
draws landmark overlays; other consumers, including HyperShaderFX, decide
which fields to visualize.

## Interoperability: read-only OBS snapshot procedure (v1)

Create a **private** source of type \`sight_tracking_receiver_filter\` after
Sight registers its source type. Obtain \`obs_source_get_proc_handler(reader)\`
and call \`sight_tracking_read_v1\` with:

- \`channel\` (string): publisher's named channel (\`face\`, \`body\`, \`hand\`
  by default).
- \`max_age_ms\` (int): 1–5000 ms; readers should usually use 750 ms.
- \`snapshot\` (pointer): caller-owned \`sight_snapshot_v1\`, initialized with
  \`abi_version=1\` and \`struct_size=sizeof(sight_snapshot_v1)\`.
- \`available\` (output bool): whether the requested immutable snapshot is
  fresh and copied successfully.

The fixed C struct is in \`src/core/sight_snapshot_v1.h\`. This has no C++
type/runtime dependency between modules: it includes the kind (1=face,
2=body, 3=hand), up to 478 normalized image-space points, up to two 21-point
hands, their pinch/gesture score/labels, up to 64 face blendshape scores,
optional 4×4 head transform, source dimensions and revision. Consumers must
copy the exact v1 header and check its version and size before reading.

Missing, expired and unsupported results are **not** silently reused.
A fresh no-detection result is available with a zero landmark count; a missing
or expired result has \`available=false\`. When a channel or publisher
disappears, consumers must clear their old textures or parameters rather
than keep the last result. The source is private to the consumer and should
be released when the consumer is destroyed. Sight's producing filter remains
the only task/model owner. An OBS filter is not a network protocol or an
unrestricted cross-plugin memory API.


## Lifetime and concurrency

- Multiple consumers may read snapshots independently.
- A channel has one owner; another producer must use another channel.
- Publisher IDs prevent an unrelated filter from withdrawing someone else's
  data. Deletion and changed model/resolution remove stale snapshots.
- The receiver defaults to 750 ms expiry (configurable 100-5000 ms), so a
  frozen producer cannot leave stale points permanently attached.
- A successful no-subject inference clears receiver overlays immediately.
- The producer uses its existing frame cadence and temporal smoothing; the
  receiver adds no inference work.
- Coordinates refer to the producer camera's normalized framing, not a
  global OBS canvas transform. Align source frames or add a coordinate
  transform if overlaying another camera.

## Tests

- `sight_tracking_signal_bus_unit` tests multiple readers, copy isolation,
  frame expiry, duplicate-writer rejection, publisher-only withdrawal and
  explicit no-subject frames. It also registers an exit handler before the bus
  is first constructed, exercising late exit-time reads and publishes. Run
  the test with AddressSanitizer to catch static-destruction use-after-free:
  `g++ -std=c++20 -O1 -g -fsanitize=address -fno-omit-frame-pointer -pthread -Isrc -o /tmp/sight-bus-asan tests/tracking_signal_bus_test.cpp src/core/tracking_signal_bus.cpp && ASAN_OPTIONS=detect_leaks=0 /tmp/sight-bus-asan`.
  The pre-fix version produced an ASan `heap-use-after-free` during process
  exit. Sight now keeps the bus alive for the process lifetime. Normal OBS
  window close was separately verified with real Face Motion and VRM sources
  in an isolated profile. Abruptly terminating the X server instead can
  still trigger a third-party OpenCV TLS abort; this is not a clean OBS exit.
- `sight_signals_obs_face_receiver`: creates **two separate filter instances**
  on a real OBS Image source, disables the publisher's own overlay and proves
  a separate receiver visualizes **478 real MediaPipe landmarks**.
- `sight_signals_obs_body_receiver`: repeats for **33 body landmarks**.
- `sight_signals_obs_hand_receiver`: repeats for **21 real hand landmarks**
  and verifies publisher/receiver sharing without second inference.
- \`tests/sight_snapshot_export_test.cpp\` checks the bounded, versioned
  snapshot conversion, face/head/hand fields, missing results and ABI mismatches.
- \`tests/sight_snapshot_proc_obs_test.cpp\` loads the real Sight module,
  creates its private receiver and calls the libobs procedure; it verifies
  unavailable snapshots remain clear.
- The face/body/hand receiver smoke tests additionally assert the actual
  MediaPipe Tasks producer's landmark counts can be retrieved through that
  same native procedure (478/33/21). The hand run needs an explicit public
  512×512 image and installed Gesture Recognizer model.
- An opt-in \`SIGHT_SMOKE_HYPERSHADER_MODULE\` integration path also loads the
  separately built HyperShader module with its \`SIGHT_SMOKE_HYPERSHADER_DATA\`
  and \`SIGHT_SMOKE_HYPERSHADER_EFFECT\`, verifies actual shader pixels from
  the producer's point texture, then withdraws the producer and checks the
  750 ms expiry clears the output. The isolated test provides a queued UI
  dispatcher (normal OBS already provides one); it never modifies the user's
  scenes or OBS installation.
- These tests exercise graphics initialization and teardown, rejecting OBS
  shader errors and crashes, and retain the rest of Sight's existing suite.

## Exeldro Move comparison

In the user's Move fork, `mediapipe-move-filter.cpp` may directly change
another source/filter's settings using `obs_source_update()`, or directly
apply transforms using `obs_sceneitem_set_pos/rot/scale()`. That's effective
for a selected target, but it isn't a general pub/sub bus. Sight's bus
decouples obtaining a signal from consuming it, permitting multiple effects
with one inference. A Move-style scene-item action consumer remains future
work. We have not copied GPL code from Move into Sight.
