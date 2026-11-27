# Longpipe motion-compensated matte stabilization (2026-11-26)

Subject Cutout can optionally run the official Longpipe flow graph on its
existing worker. In **Model family**, select Longpipe and the refined
matting ONNX file. Select its matching tier and orientation under
**Longpipe optical flow (optional)**, then set **Matte stabilization**
above zero. The same per-filter device selection applies to both sessions;
required CUDA refuses CPU fallback. The normal cutout remains available
when the auxiliary flow graph cannot be loaded. Zero retention stops flow
inference and clears temporal history, but leaves the optional session
loaded for re-enabling without a model reload. Clearing the file unloads
the session. Older sources and other model families behave as before.

## Qualified model

- Matting: XS landscape, `longpipe-xs-landscape.onnx`.
- Flow: [official XS landscape flow ONNX](https://huggingface.co/Longpipe/longpipe/blob/main/onnx/xs/flow_landscape.onnx).
- Local flow file: `~/loam/.cache/sight-realtime-models/longpipe-xs-flow-landscape.onnx`.
- Size: 3,202,933 bytes. SHA-256:
  `21e498f6c2b0b47d27b7a1d17f96e524cbfcb589652bb175f737d652f8049ffe`
  (matches publisher manifest).
- Both files use original FP32 precision. Large and XL landscape FP32
  pairs have also been qualified; see the adaptive comparison below. Other
  matched tiers/orientations remain individually unqualified.

Upstream [ONNX contract](https://longpipe.dev/docs/onnx/):
previous RGB then current RGB as NCHW float32 [0,1]; output is backward
dx/dy in base-resolution pixels on a lower-resolution grid. The XS
landscape canvas is 384×240, base 128×80, flow grid 64×40.
Sight uses the corresponding canvas/base/grid dimensions for each
supported variant, scaling displacements to the captured source.
No weights included in Git or automatically downloaded.

## Native behavior and limitations

Sight's conservative gate is independently authored, not a claim of SDK
shader parity. The worker always computes fresh alpha, then samples
backward flow, warps the previous alpha and RGB, and rejects out-of-range
coordinates, nonfinite data, strong motion/divergence and incompatible
color or alpha. History is reset after model/device changes, retention
changes, incompatible source/capture geometry, failed matting, or capture
gaps over 500 ms. A failed flow execution releases the auxiliary session
and leaves fresh matting available; ordinary stabilization resumes on
subsequent frames. No frame skipping, additional capture, extra worker,
render-thread inference or new runtime. The standalone ONNX graphs have
separate encoders, unlike the upstream shared-encoder SDK, so extra
compute and memory are expected. Backend refresh shows the auxiliary
session's actual provider/state. Inference health includes flow and
warping when enabled, but not the entire OBS rendering pipeline.

## Verification

- RED: deterministic warp test failed because the helper was absent.
  GREEN: backward translation with correct base/capture scaling, RGB
  disagreement, out-of-bounds, invalid/nonfinite flow, excessive motion,
  and zero retention checks passed.
- Real libobs/OBS XS landscape test at 512×512 with matching FP32 flow
  passed on CPU and required CUDA, including nonconstant rendered alpha,
  near-opaque native color preservation and bounded 320×320 shared matte.
- A 32-pixel rightward translated portrait fixture moved the normalized
  matte centroid from 0.469199 to 0.531796 on CPU, then restored to
  0.469203. CUDA produced equivalent results. Switching retention to zero
  reported Flow ready and switching it back reported Flow active.
- The same native test using the matting graph as the auxiliary file
  reported Flow unavailable and rendered the ordinary matte correctly.
- Worker snapshot latency (matting plus flow and blending): 15.1 ms CPU,
  12.8 ms CUDA on the development RTX 2070 SUPER host. These are not
  controlled latency comparisons or full camera frame-rate measurements.
- Full configured build and 64/64 selected regression CTests passed,
  including existing cutout families, device selection, native picker,
  tracking, depth, sketch, inference health and deterministic unit tests.

For repeatable local qualification, configure
`SIGHT_LONGPIPE_MODEL`, `SIGHT_LONGPIPE_FLOW_MODEL`,
`SIGHT_CUTOUT_PORTRAIT`, and optionally
`SIGHT_LONGPIPE_FLOW_MOVED_IMAGE` (512×512 right-shifted fixture) in the
validated build. CTest runs the valid and mismatch paths. Native GPU
verification uses the same smoke executable with required CUDA and the
matching existing NVIDIA runtime libraries. Raw logs:
local development test logs, local development test logs,
local development test logs.

This establishes real graph execution, safe fallback and motion response,
not a ground-truth or live-webcam proof that the stabilizer improves every
sequence. Flow-specific FP16/INT8 conversion, remaining tiers/orientations, full
regression suite, cross-plugin flow-specific tests and GUI visual
inspection are outstanding. No installed plugin or user configuration changed.

## Adaptive stabilization and larger flow tiers

A controlled sequence using the actual C++ processor found that running
flow on every comparable frame was counterproductive on stationary input
with uniform exposure changes. The worker now estimates texture change
on a bounded 64×64 luma grid after fitting global gain and offset. Small
residuals select the existing unwarped stabilization kernel; larger
residuals select the matching optical-flow graph. Both branches share
the same prior alpha and RGB history. Fresh matting is unchanged. This
is a conservative heuristic, not semantic motion detection; subtle or
localized movement may be treated as stationary. The existing RGB/alpha
gates still reject incompatible history. Backend snapshot distinguishes
stationary stabilization from actual flow execution and reports completed
flow runs since model load. Idle/zero-retention resets history, not the
counter. No extra thread or automatic model changes.

### Measured stationary sequence

Actual C++ processors on RTX 2070 SUPER host; 24 frames with alternating
±3.5% gain, first four excluded, 0.6 retention. Paired translation
sequence shifts the image 2 pixels per frame (up to 24 pixels, then
back). MAE compares overlapping aligned pixels to prior output or the
initial prediction; not ground truth. Extra latency excludes matting,
rendering and capture. Same exact local model files and source fixtures
for the before/after comparison.

| Tier/device | Old flow overhead ms | Adaptive overhead ms | Old aligned temporal MAE | Adaptive MAE |
|---|---:|---:|---:|---:|
| XS CPU | 7.098 | 0.976 | .001534 | .000347 |
| XS CUDA | 8.231 | 1.047 | .001533 | .000347 |
| Large CUDA | 7.009 | 1.103 | .000576 | .000042 |
| XL CUDA | 23.531 | 5.051 | .000403 | .000023 |

All stationary sequences performed zero auxiliary graph executions and
matched ordinary stabilization exactly. All moving sequences executed
23 flow graphs and retained previous motion-branch results. Aligned
temporal MAE versus ordinary blending on moving input: XS CPU
.006056 vs .006596; XS CUDA .006055 vs .006593; Large CUDA .002610 vs
.002888; XL CUDA .001692 vs .001796. Raw fresh alpha was competitive:
.006146, .006145, .002578, .001603 respectively. Therefore motion-
compensated blending is useful selectively, not universally superior
to fresh matting. A capture-gap test also confirmed exact fresh alpha
after a two-second discontinuity. These are small synthetic sequences
from existing portrait fixtures, not comprehensive video quality tests.

[Recorded measurements](qualification/longpipe-flow-adaptive-20261126.json).
Manual benchmark source: `tests/benchmark-longpipe-flow.cpp`; it uses
the real segmentation/flow/history classes and packed local RGB fixtures.
Raw logs and fixtures in external cache
`~/loam/.cache/sight-realtime-models/flow-qualification-20261126/`.
Build against the existing libobs and ONNX Runtime headers/libraries
with C++20, `-O3 -DNDEBUG -DUSE_ONNX_MEDIAPIPE -DSIGHT_ANIMEGAN_ONLY`,
link `src/core/segmentation_processor.cpp src/common/sight-log.cpp`.
Arguments: matting ONNX, flow ONNX, cpu or cuda, width, height, packed
RGB fixture. The baseline executable was built before the adaptive
change. No private image bytes or model weights committed.

### Additional real OBS flow qualification

Downloaded official matching Large and XL landscape flow graphs,
verified against publisher manifest:
- Large: 12,594,955 bytes,
  SHA-256 `72ea213e0ea13f72ff602c76b3ac0f2b43bbe09365721c29eb7bb6448e9afa67`.
- XL: 26,353,870 bytes,
  SHA-256 `7991f55d4d6ef48b27a47e4f647fda461598085694963b1401b7ff2e57bbea03`.

[Official exports](https://huggingface.co/Longpipe/longpipe/tree/main/onnx).
Both pairs use original FP32 and remain outside Git. Actual required-CUDA
OBS XS, Large and XL moving-input tests passed: stationary snapshots
reported zero flow runs, translating and restoring fixtures each
triggered actual flow (two total), toggling retention did not trigger
flow on stationary input. XL preserved 1280×768 local rendering and
320×192 shared matte. CPU XS valid/mismatched graph cases and deterministic
motion-gate regressions passed. No other tiers/precision variants or
sustained camera performance qualified in this pass.
