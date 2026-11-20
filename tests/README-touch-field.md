# Touch Field — hand-controlled GPU image refraction

**LLL.sight Touch Field (Hand Ripples)** reads the existing `Hand & Gesture
Motion` channel. It does not load another MediaPipe model or run its own
inference. Its GPU shader refracts the *full native-resolution source image*
around each detected **index fingertip (landmark 8)**. Up to two hands are
supported. Higher pinch amount boosts refraction; the ripples use deterministic
radial wave motion, without temporal image feedback. New short-lived
expanding ripple events are emitted when fingertips enter the frame and after
each accumulated 2.2%-of-frame-distance movement (aspect corrected). Up to
eight events persist for 1.6 seconds; no duplicate events are created merely
because OBS renders the same tracking snapshot again.

## OBS setup

On the **same video source**, add:
1. `LLL.sight Hand & Gesture Motion (MediaPipe Tasks)`; set tracking channel to
   `hand` and optionally disable its visible landmark overlay
2. `LLL.sight Touch Field (Hand Ripples)`; receive channel `hand`

Moving an index fingertip moves a circular ripple field and leaves short
expanding ripple fronts behind it. Slow movements accumulate rather than
requiring a sudden >0.004 normalized jump in one tracking update.
Controls are radius, refraction strength, number of rings, wave speed, pinch
amplification and maximum signal age. An absent/expired/empty tracking signal,
zero refraction strength, missing shader, or genuinely mismatched source dimensions
causes passthrough without simulated coordinates.

**Fixed camera-resolution bug:** earlier builds incorrectly tagged the
tracking snapshot with the **model's downscaled input dimensions** instead of
the native OBS camera dimensions. A 1920×1080 webcam inferred at 512×288
therefore drove the tracking overlay, but Touch Field rejected every
snapshot. Producer snapshots now carry the native source dimensions while
MediaPipe still operates on the smaller frame. This preserves strict
protection against attaching effects to genuinely mismatched sources.

**Important:** coordinates are relative to the original tracking camera,
not transformed OBS scene coordinates. Place both effects on the same
untransformed source with matching dimensions. If the sender and receiver
use different sizes, the receiver deliberately refuses to warp.
Finger labels are **detection-order** only; identity is not persistent across
overlap/crossing. Pinch is a heuristic normalized distance.

No fake ripple texture is synthesized when a hand disappears. Wave phase
animates only while a fresh tracking position is available. No particle
engine, collision, fluid simulation, texture persistence, MIDI control or
cross-plugin network transport is claimed.

## Diagnosing one-ripple or intermittent webcam output

A single first-success log entry does **not** demonstrate continuing
tracking. With OBS `--verbose`, Touch Field emits one **debug** summary
every six seconds:

- `rendered / total`: frames where the shader used a valid hand signal
- `new_snapshots`: distinct MediaPipe signal publications observed
- `moving_snapshots`: unique tracking updates with measurable movement
- `finger_travel`: accumulated aspect-corrected normalized fingertip distance
- `new_ripple_batches`: snapshots that launched one or more new ripple fronts
- `no_fresh_signal`: stale/missing hand channel or zero effect strength
- `no_hands`: fresh snapshots with no valid hand landmarks
- `source_mismatch`: genuine capture-dimension mismatch
- `gpu_fail`: failed shader/texture render path

This is one aggregated line rather than per-webcam-frame log spam.
When debugging, compare the six-second summaries before and after hand
movement. On a long-lived OBS camera, zero `new_snapshots` alongside
`no_fresh_signal` means the receiver is not getting fresh tracking, even
if the one-time startup message remains in the log.

The continuous OBS regression keeps a real MediaPipe hand publisher and
GPU shader active across 56 further rendered frames and asserts both
ongoing distortion and changing wave output. Unlike previous
first-change-only tests, it fails if ripples freeze or stop.

## Verification boundaries

- `sight_touch_field_events_unit`: 80 tiny fingertip movements emit multiple
  independent ripples, expire on time, do not repeat for one signal sequence,
  and allow reentry without fabricating tracking
- `sight_touch_field_signal_unit`: checks two fingertip snapshots, pinch,
  invalid/non-finite coordinates, mismatched source dimensions and no hands
- `sight_touch_field_obs_hand`: real Google test image, real MediaPipe VIDEO
  task on a background worker, cross-instance signal publisher, headless
  OBS/OpenGL source rendering and a nonzero native output pixel difference
- `sight_touch_field_obs_native_wide`: the actual failure reproduced with
  640×360 camera input, downscaled to 512×288 for real MediaPipe inference.
  Verifies Touch Field consumes native-size metadata and distorts pixels.
  Requires ImageMagick's `convert`/`magick` to prepare a test-only fixture
  from the supplied, externally located original image.
- `sight_touch_field_obs_no_hand`: real OBS image and Touch Field filter
  with no hand publisher; every sampled source pixel stays unchanged
- All pre-existing Portrait Sketch, U2Net, Depth, Face/Body/Hand tracking,
  signal receiver and numeric router tests run in the same CTest suite

Tests do not establish live camera calibration, two simultaneous tracked
hands, achievable FPS on the real GPU, physical accuracy, or compatibility
with transformed/mismatched sources. The online-hand pose fixture is a
test-only image and not checked into git. No OBS install/restart is necessary
to run the isolated build and test suite.
