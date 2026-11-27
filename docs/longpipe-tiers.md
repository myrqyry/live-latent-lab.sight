# Longpipe tiers and shared matte compatibility (2026-11-26)

Use Subject Cutout → Longpipe and select the official refined matting
ONNX export, with portrait/landscape orientation matching the input.
The graph accepts RGB float32 [0,1] on a fixed canvas and emits alpha.
Keep upstream resize behavior; no unverified letterbox transform,
automatic orientation switch, or new model runtime. Optional optical-flow
stabilization is described in [flow setup](longpipe-flow.md).
All weights are external and optional.

## Qualified local files

All five additional original FP32 exports matched the publisher manifest.
The previous XS landscape file remains qualified.

| Official export | Width × height | SHA-256 |
|---|---:|---|
| xs/matting_portrait.onnx | 240×384 | ada077894985240f37df9914e220eee963eefb0b375659713a664f0816084612 |
| small/matting_landscape.onnx | 384×224 | e6aca89d9df1cfe26670693eaf138d615b3b74979fe613f8f0195e9f5da14a90 |
| medium/matting_landscape.onnx | 512×320 | a0b385d6690b2d08c1e393ff2258b6f3b3065d8bee0f1ff0f0ce2d68a0ce0cf7 |
| large/matting_landscape.onnx | 640×400 | 2dab0b195cd712477a1549d38465e28dcca4ad05efe407c9406f08caa6c6e4da |
| xl/matting_landscape.onnx | 1280×768 | 9ee02f5a73df90d5b62af667eaf57ca85a4065e7fe676e845d4a28d410f56daa |

[Official model files](https://huggingface.co/Longpipe/longpipe/tree/main/onnx) ·
[Manifest and tensor contract](https://longpipe.dev/docs/onnx/).
The local cache is `~/loam/.cache/sight-realtime-models/longpipe-tiers-20261126/`.
Only XS portrait is qualified; other portrait tier exports have not been
individually tested. Optional locally converted FP16 variants are qualified
on CUDA below; the original FP32 exports remain recommended for CPU.

## Real OBS measurements

Single worker latency snapshots, after about 1.8 seconds of repeated
rendering in a disposable libobs test. The fixtures preserve aspect by
padding the existing square portrait. Not sustained webcam FPS or a
comparative ground-truth quality evaluation. RTX 2070 SUPER host,
CPU for XS through Large, CUDA for XL.

| Variant | Source | Local inference matte | Shared matte | Last inference |
|---|---|---|---|---:|
| XS portrait | 400×640 | 320×512 | 200×320 | 3.9 ms CPU |
| Small landscape | 640×400 | 512×320 | 320×200 | 4.2 ms CPU |
| Medium landscape | 640×400 | 512×320 | 320×200 | 8.3 ms CPU |
| Large landscape | 640×400 | 640×400 | 320×200 | 13.1 ms CPU |
| XL landscape | 1280×768 | 1280×768 | 320×192 | 22.9 ms CUDA |

All tests checked nonconstant actual output alpha and original
near-opaque foreground RGB (mean absolute color error < 0.33 on
the 8-bit scale). Final XL retest passed after the motion fix.
Local rendering retains the higher-resolution matte. Only the shared
ABI publishes an aspect-preserving bounded copy, prepared on the
existing worker. Motion samples a bounded grid directly from the RGB
capture; no full-size second motion buffer. Existing ABI version,
freshness, sequence and source dimensions remain unchanged.

RED: the old 512×512 Longpipe capture rendered successfully but
returned available=false through sight_matte_read_v1. The first
HyperShader integration run then revealed a separate 320-edge guard
in SubjectMotion. The new deterministic 1280×768 motion regression
failed until that guard was replaced by bounded sampling.

GREEN: all three existing model-family portrait CTests now require a
real bounded shared matte. Installed HyperShader module used only
in isolated libobs tests: MODNet and Longpipe each passed actual
R8 consumption, subject geometry, disable/re-enable, filter ordering,
detach/expiry/reattach, subject framing, and still/moving/settled motion
pulse. The installed shader package did not expose the optional
background-defocus test fixture in that run. Full configured build
and 61 selected CTests passed.

No installed plugin, scene, model default, or remote changed.

## FP16 precision comparison — 2026-11-26

Converted the five additional official exports to mixed FP16 internal
precision while retaining float32 input/output, using ONNX Runtime's
float16 converter with keep_io_types, topological sorting and ONNX
checker validation. Original weights retained. Eight warm-ups and 20
timed CUDA session runs on the same RTX 2070 SUPER, six CPU intra-op
threads, heuristic cuDNN convolution selection; includes returned alpha,
excludes preprocessing, OBS capture/compositing and session startup.
Four evaluation views (existing face and hand fixtures, face mirror and
brightness transform); comparisons are against FP32 on CUDA, not
ground-truth quality or an extensive dataset.

| Variant | FP32 MB | FP16 MB | FP32 p50 ms | FP16 p50 ms | Alpha MAE | Edge MAE |
|---|---:|---:|---:|---:|---:|---:|
| xs-portrait | 3.61 | 1.90 | 1.117 | 1.240 | 0.000094 | 0.004974 |
| small-landscape | 3.61 | 1.90 | 1.151 | 1.168 | 0.000075 | 0.004767 |
| medium-landscape | 12.96 | 6.60 | 1.640 | 1.814 | 0.000049 | 0.003032 |
| large-landscape | 14.21 | 7.29 | 2.369 | 2.028 | 0.000044 | 0.003052 |
| xl-landscape | 29.20 | 14.74 | 6.356 | 5.905 | 0.000053 | 0.004008 |

All five FP16 files also passed actual OBS portrait/landscape rendering
with required CUDA, bounded shared matte, original foreground RGB
(mean absolute color error <0.038 on the 8-bit scale), and zero failed
worker results in the short test. Last worker snapshots were
6.6, 4.1, 5.9, 6.1 and 19.6 ms respectively; these are not matched
FP32-vs-FP16 OBS benchmarks. XL retained 1280×768 local matte and
320×192 shared matte. No source/ABI changes needed.

Large and XL show modest CUDA session latency reductions; the smaller
tiers do not show a reliable speed advantage in this short test.
All reduce file size roughly by half. Threshold-mask IoU versus FP32
exceeds 0.9998 across these views. No peak VRAM measurement, CPU FP16
qualification, or sustained webcam benchmark. FP32 defaults remain.
No new INT8 tests (previous QDQ conversion results were unfavorable).

Exact hashes, p90 latency, ranges and fidelity metrics:
[recorded results](qualification/longpipe-tiers-fp16-20261126.json).
The conversion script, derived weights, and raw OBS logs are in the
external tier cache above (compare-fp16.py, *-fp16-io32.onnx,
obs-*-fp16.log). Uses the previous precision comparison helper in
the adjacent precision-20261126 directory. Local derived files are not
publisher releases or approved redistributable assets. No weights in Git.
