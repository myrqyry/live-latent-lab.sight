# Semantic obfuscation (experimental)

Use the existing Privacy V2 filter and enable **Use custom semantic detector**.
Select an ONNX detector with a single float32 NCHW input [1,3,H,W], fixed
32–1280 pixel dimensions, RGB 0–1, gray-114 letterbox, and a single raw
channel-major float32 output [1,4+classes,N]. Coordinates are center x/y and
width/height in model pixels; remaining channels are probabilities. Class count
is derived from static output channels by default; dynamic class dimensions
require an explicit count. Class IDs remain model-specific. End-to-end
NMS exports, objectness-based YOLOv5, classifiers, quantized I/O and arbitrary
segmentation outputs are not supported. Raw overlapping boxes are unioned
rather than suppressed because this is coverage, not object counting.

## Candidate models (2026-11-25)

- [Nevermore 1.0](https://huggingface.co/kinkyrvn/nevermore): YOLO26n small,
  512x288 FP32/FP16 weights, 14 classes; choose class 1 for face-only testing.
  The model card reports 2.98 ms median on RTX 4090/DirectML; that is not a
  Sight measurement. The tiny model is faster but reports substantially worse
  detection coverage in thumbnails. Model is AGPL-3.0. Do not bundle or
  redistribute it without licensing review. Credit: Kinky Raven.
- [EraX Anti-NSFW V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1):
  YOLO11 nano, five classes, 640x640. Model card claims Apache-2.0;
  upstream training/export rights still require review. Raw ONNX export
  is a candidate, not tested.
- [NudeNet](https://github.com/notAI-tech/NudeNet): established 320n/640m
  ONNX baseline; AGPL-3.0, preprocessing and output contract must be
  verified separately.
- [RF-DETR](https://github.com/roboflow/rf-detr): Apache-2.0 detection Nano
  and segmentation variants for general object masking. Different output
  contract; not supported by this raw YOLO adapter.

## Local validation

Nevermore small 512x288 FP32, revision
0e05d6c3d9555b05e6af786a220d83d347b50d4e, SHA256
b923c5c0b578533234c3d5cc773e0edc69e892dd1b48b58608f9dc019c4e1992.
Stored separately in ~/.cache/sight-semantic-models, not in git.
Native ONNX Runtime 1.30 on Ryzen 7 7700: warm single portrait inference
plus preprocessing 9.95–13.25 ms (six samples, no OBS render time).
Actual OBS 512x512 portrait test with class 1 at confidence .20 produced
31,892 changed and 230,252 unchanged pixels. CPU and CUDA session both
passed; OBS graphics in Xvfb were software OpenGL, not a hardware-render
throughput benchmark. No explicit-content accuracy claim is made.

### Optimized exports, measured 2026-11-25

30 warm native C++ inference samples after 6 initial iterations, including
letterbox and decoding, on the Ryzen 7 7700 / RTX 2070 SUPER. GPU timings
use CUDA-preferred sessions; separate real OBS runs explicitly assert CUDA
session initialization for both FP16 variants.

| Export | Classes | CPU median / p95 (ms) | CUDA-preferred median / p95 (ms) |
| --- | ---: | ---: | ---: |
| Small FP32 512x288 | 14 | 9.48 / 10.40 | 4.03 / 7.56 |
| Small FP16 512x288 | 14 | 9.76 / 10.43 | 3.23 / 6.23 |
| Tiny FP32 320x192 | 10 | 4.10 / 4.77 | 2.67 / 6.35 |
| Tiny FP16 320x192 | 10 | 4.33 / 4.59 | 2.12 / 5.78 |

Small FP16 is the balanced starting choice; Tiny is a throughput option,
not an accuracy-equivalent replacement. Both retain float32 graph I/O.
Both produced real CUDA sessions and passed the OBS selective masking
test with automatically inferred class counts. Cold first inference in
this run ranged from 300 to 985 ms on CUDA-preferred sessions.

All files at the same pinned revision, local cache only:

- det_v11_small/det_v11_small_512x288_fp16.onnx:
  4a094d6995e0bc4e870df9e2ff52954e3c610dc1229e038322e6b7ce1a85992e
- det_v11_tiny/det_v11_tiny_320x192.onnx:
  2acfbce8411b3089aeb69ae48a810fba99d0527fc7272bdba2c6801d353c1d94
- det_v11_tiny/det_v11_tiny_320x192_fp16.onnx:
  91fa349a93806b6d1ab2b8631eb4f1d1443e669f72eb89d6894dbfd3c8648ace

The face class is 1 in both exports. Other classes differ between variants;
consult ONNX metadata before using a selection. Optional
SIGHT_SEMANTIC_TEST_SMALL_FP16 and SIGHT_SEMANTIC_TEST_TINY_FP16 enable
additional real-model CTests.

Optional tests: configure SIGHT_SEMANTIC_TEST_MODEL, SIGHT_SEMANTIC_TEST_RGBA
(512x512 RGBA portrait), SIGHT_SEMANTIC_TEST_IMAGE (matching PNG). Build
sight_semantic_detector_real_test, sight_semantic_regions_test,
sight_privacy_text_obs_smoke and live-latent-lab-sight, then
ctest -R 'sight_semantic|sight_privacy' --output-on-failure.

## Temporal coverage and buffering

Region persistence (default 180 ms; 0 disables persistence across frames) immediately
covers new boxes, unions overlapping same-class boxes, associates recent
boxes by class and overlap, and retains last coverage briefly through a
missed detection. Current detections are never shrunk by smoothing. This
is an overlap-based tracker with bounded constant-velocity prediction,
not optical flow. It extrapolates missed detections for at most 120 ms
and one observed box extent per axis, unions with previous coverage, and
clips to the frame. Held predictions never feed back into velocity.
Class confidence overrides accept ID:confidence pairs, e.g. 1:0.20,
3:0.55. Class selection still applies; invalid rules fail explicitly.

Approved frames only (buffered) is an opt-in alternative to the original
live-first behavior. It draws only the analyzed candidate frame and its
matching mask, or an opaque neutral slate while no fresh approval exists.
A valid empty result approves the original analyzed image. Output motion
can be less smooth; detector false negatives remain possible.

The filter exposes a read-only OBS proc sight_semantic_read_v1 returning
regions_json, count, source_width, source_height, available. Boxes are
normalized source coordinates, with numeric category, score, track id and
held flag. The snapshot rejects stale, failed or old-generation results.
A second caller-owned typed ABI, sight_semantic_geometry_v1, powers
HyperShader's semantic mask, bounds and category metadata. Enable Analysis
only (shader signals; no concealment) and disable buffered mode. The source
passes original video; HyperShader must be downstream on the same source.
Typed snapshots reject stale, oversized (>256 regions), incompatible and
non-analysis-only results. Existing JSON proc remains unchanged.

## Limits

The model selection must match the expected contract and class IDs. No
guarantee of obscuring every region; this filter preserves the existing
live-first Privacy V2 behavior (fresh mask, bounded hold, then passthrough).
Use opaque cover for stronger concealment of detected regions, never
rely on it alone for broadcast safety. The optional buffered mode reduces frame/result mismatch but is not proof of complete detection coverage.
