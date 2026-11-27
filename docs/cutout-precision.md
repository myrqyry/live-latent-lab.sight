# Subject Cutout precision qualification (2026-11-26)

The existing model-family adapter accepts reduced internal precision when
the graph retains the qualified float32 RGB input and float32 alpha output.
No automatic conversion, download, CPU/GPU preference change, or model
replacement is needed. Choose a matching model family and local file.
Keep FP32 weights for CPU use and fallback; reduced precision is optional.
Use Subject Cutout’s per-instance Inference device selector: CPU for
FP32 Longpipe, or CUDA (required) for a qualified FP16 model. This
does not automatically convert or substitute models.

## Measured trade-offs

RTX 2070 SUPER / Ryzen host, ONNX Runtime 1.30.0, six CPU intra-op threads,
CUDA heuristic convolution selection. Eight warm-ups and twenty timed
session runs with inputs already prepared; includes host output retrieval,
excludes resizing, OBS capture/compositing and model initialization.
Four evaluation views (local face/hand fixtures plus face brightness and
mirror transformations), distinct from the six calibration views
(astronaut and coffee images/transforms). Very small exploratory set.
Errors compare alpha against the corresponding FP32 output on the same
device, not ground-truth segmentation accuracy.

| Family | FP32 CPU p50 ms | INT8 CPU p50 ms | FP32 CUDA p50 ms | FP16 CUDA p50 ms | FP16 alpha MAE | INT8 alpha MAE |
|---|---:|---:|---:|---:|---:|---:|
| U2NetP 320x320 | 80.192 | 92.499 | 13.888 | 9.474 | .001349 | .473135 |
| MODNet 512x512 | 78.042 | 120.950 | 15.926 | 29.265 | .000073 | .364650 |
| Longpipe XS 384x240 | 1.843 | 2.784 | 1.308 | 1.138 | .000236 | .026338 |

FP16 U2NetP: useful CUDA candidate, 2,355,542 bytes. Longpipe XS:
1,897,979 bytes, modest/no robust speed advantage at this small sample
size; very fast FP32 CPU remains attractive. MODNet FP16 (12,980,011
bytes) preserves alpha but is slower; keep FP32 unless storage matters.
These are mixed-precision graph conversions (some ops remain FP32).
Reject these particular INT8 QDQ conversions: all were slower on CPU
and alpha fidelity degraded, including Longpipe soft-edge MAE .323841.
This does not rule out well-calibrated, optimized publisher INT8 exports.

## Native OBS verification

All three local FP16 files passed the existing real OBS portrait test
with an explicitly verified CUDA session, native foreground RGB preserved
and alpha spanning 0..255. These are functional tests, not warmed OBS
frame-rate comparisons. CPU execution of these FP16 variants was not
qualified. No change to defaults or installed module.

Exact model hashes, p90, soft-edge MAE and threshold-mask IoU are in
[recorded results](qualification/cutout-precision-20261126.json).
U2NetP and MODNet source hashes match the prior cached exports; Longpipe
matches the official XS landscape manifest. Local conversions are
derived weights, not publisher exports or redistributable assets.
Consult the original licenses before distribution.

Reproduction on the development host: the script, converted models,
raw logs and JSON are retained in the external model cache at
`~/loam/.cache/sight-realtime-models/precision-20261126/`.
It uses ONNX Runtime's float16 converter with keep_io_types and
topological sorting, checker validation, and static QDQ INT8
Conv/MatMul/Gemm quantization using a small calibration set.
Opset 11 models use per-tensor weights; opset 17 Longpipe uses
per-channel weights. First-pass converter ordering and opset errors
were corrected before collecting the final comparison. No model files
or calibration images are committed.

Additional Longpipe tiers: [FP16 CUDA comparison and OBS qualification](longpipe-tiers.md#fp16-precision-comparison--2026-11-26).
