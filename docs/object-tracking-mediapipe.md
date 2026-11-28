# Native MediaPipe 1.2.3 Object Tracking for Sight

Checkpoint: 2026-11-13. Grounded in actual Google EfficientDet-Lite0
inference and isolated real OBS rendering. This is an additional
standalone object detector, not the Holistic pose graph.

## What is implemented

- Select Objects (MediaPipe) inside lll.sight Tracking.
- Filter ID: sight_unified_tracking_filter; mode: object.
- Dedicated native MediaPipe 1.2.3 VIDEO Object Detector with explicit
  user-selected TFLite model. Defaults to Google EfficientDet-Lite0
  in Sight's persistent configuration models folder; it is not bundled.
- The existing Sight GPU capture/render pipeline sends reduced-size RGBA
  frames to an owning model worker; graphics operations stay on OBS's
  graphics thread and native model calls stay off that thread.
- ObjectTracks confirms each new ID across two detections, matches by
  category and predicted box position, and retains missing detections
  for at most four model frames. IDs are filter-local and temporary.
- Labeled, premultiplied transparent box overlay, with optional checkbox
  to disable drawing while continuing inference.
- Read-only OBS procedure sight_object_read_v1 publishes recent JSON
  with category labels, scores, stable IDs and normalized x/y/w/h.
  Additional fields: count, source_width, source_height, available.
  No network server, file writes or automatic camera follow/zoom.
- Results expire after 2 seconds without new inference; model changes
  and confidence changes reload the graph and clear old output.
  Empty/missing/incompatible models do not invent detections.

## Model and runtime

Verified native runtime (already installed beside Sight):
  SHA256 7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63

Model: official Google EfficientDet Lite0 float32, 13,836,895 bytes.
  SHA256 40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58

Source:
https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite

The model remains user-owned, outside Git and outside the staged plugin.
Arbitrary YOLO World models or ONNX exports are NOT promised compatible
with the native MediaPipe Object Detector API.

## Verification

- Tests object_tracks_test.cpp: reordered detections, short disappearance,
  bounded expiry and renewed IDs, category isolation, malformed boxes,
  and same-category crossing. Test-first RED then GREEN.
- Tests object_overlay_test.cpp: RGBA box pixels, transparent background,
  varying ID label, invalid frame dimensions.
- Tests object_detector_real_test.cpp: real MediaPipe VIDEO inference on
  a Google sample cat-and-dog photograph (512x333 RGBA), 3 consecutive
  timestamps; model reports cat confidence 0.790057 and dog 0.752158.
  Invalid model, unopened detector and nonmonotonic timestamps rejected.
- Isolated libobs/OpenGL image source with exactly 13 registered filters:
  sight_object_123_obs_overlay and sight_object_123_obs_on_source_only
  both passed. Tests query the actual sight_object_read_v1 procedure
  and verify cat/dog, dimensions and object IDs. Invisible mode retains
  detections while preserving original source pixels.
- Full build and 75/75 CTests passed with all available private test
  models, including previous Holistic segmentation and tracking tests.
- Release-stage content verification passed, including no bundled model
  weights. Four release-safety tests passed. This does NOT certify
  distribution rights or sustained performance.

## Not implemented or established

Automatic camera crop, pan, zoom, persistent recognition across long
occlusion, arbitrary new detector models, OBS text file emission,
Streamer.bot integration, websocket cross-process transport, live
webcam performance, and second-machine installation are NOT verified
or implemented by this step. Normal OBS user-scene appearance requires
the actual installed plugin and model path to be verified separately.

The original GPLv2 OBS Detect project informed the feature direction,
but no GPLv2 tracking code was copied. This implementation uses its own
small independent category-aware motion association rather than claiming
the upstream Hungarian/SORT algorithm.
