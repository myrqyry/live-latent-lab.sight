# GPU-resident CUDA texture inference

The useful mechanism from [OBS RTX SuperResolution](https://github.com/Bemjo/OBS-RTX-SuperResolution)
is graphics/compute texture interoperability, rather than its proprietary
Maxine models. Sight's implementation is independent: NVIDIA OpenGL/CUDA
driver interop with the existing Portrait Sketch ONNX model. The Windows
D3D11 implementation, NVIDIA VFX SDK and its models were not copied or
introduced as dependencies.

Enable **GPU-resident CUDA textures** in Portrait Sketch in a build
configured with `SIGHT_ENABLE_CUDA_TEXTURE_INTEROP=ON`. This is an
explicit required-CUDA path. Without the selected build/runtime/device,
the source passes through rather than pretending CPU staging is GPU
interop. The ordinary staged path remains the default and works
independently.

## Data movement and ownership

- The OBS graphics thread draws into a reusable RGBA texture, inserts
  an OpenGL fence and checks it later with a zero-timeout poll.
- The capture and output texture registrations persist at a stable
  resolution. A CUDA device-to-device copy transfers the ready input
  into an owned device buffer; mapped graphics arrays never outlive
  their map/unmap scope.
- The existing bounded inference worker owns the model session,
  dedicated stream, device storage, padded float32 NHWC tensors and
  reusable ONNX Runtime I/O binding. CUDA kernels normalize, replicate
  edges, and crop/repack the model output on the GPU. Inference and
  stream completion waits run on that worker.
- Graphics takes a completed result only when the slot is available,
  copies it directly to the output texture, and retains the last valid
  output while work is pending. It never waits for the worker mutex.
- Resolution changes prepare new storage on the worker without
  reloading the model. Model generations invalidate obsolete pixels;
  native registrations are removed before destroying resized textures.

This removes application-side pixel readbacks and uploads; it is not a
claim of literally zero copies. CUDA graphics mapping/unmapping,
registration and texture drawing may still synchronize with the driver.
ONNX can place small shape calculations on CPU. Only the tested model's
operator placement has been profiled. No hard realtime guarantee under
arbitrary contention, device reset or driver loss is implied.

## Optional build and qualification

Use the existing matched Linux CUDA ONNX Runtime SDK and matching
external CUDA/cuDNN runtime libraries (see [onnx-cuda.md](onnx-cuda.md)).
The optional path additionally requires OpenGL development files and
CUDA toolkit headers, nvcc and the CUDA driver library. Its own kernels
are built as PTX; the default virtual architecture is compute_75, which
can be changed via `SIGHT_CUDA_TEXTURE_ARCH`. No CUDA runtime library
is linked by this interop adapter, and no proprietary GPU model or
runtime is distributed by this change.

```sh
cmake -S . -B build-cuda-texture -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_ORT_ROOT=/absolute/path/to/matched/ort-gpu-sdk \
  -DSIGHT_ENABLE_CUDA_TEXTURE_INTEROP=ON \
  -DSIGHT_ANIMEGAN_MODEL=/absolute/path/to/animegan_v3_portrait.onnx
SIGHT_CUDA_RUNTIME_PATH=/absolute/path/to/cuda13/lib:/absolute/path/to/cudnn/lib \
  tools/verify-cuda-texture.sh build-cuda-texture
```

The qualification must run on a real NVIDIA OpenGL display (not Xvfb's
software renderer). It stages an isolated plugin, creates synthetic
source pixels, executes the actual OBS filter, checks persistent
texture registrations, zero readbacks on the selected path, no failures,
one model load across an inference size change, and a live round trip
through the staged path. The staged and GPU-resident results are
compared pixel by pixel. No existing OBS installation or settings are
modified.

## Local measured evidence (2026-11-27)

RTX 2070 SUPER, NVIDIA driver 615.71.09, CUDA 12.9 driver headers/PTX
compiler, matched ONNX Runtime GPU 1.30.0 with external CUDA 13/cuDNN
libraries. Separate eight-frame texture-to-model-to-texture feasibility
probe at 320x180 (padded inference 320x192): warm median 15.17 ms;
first frame 260.49 ms. CPU reference mean absolute channel error
0.000075, maximum 1/255. ORT profile: 2896 CUDA node executions and
384 CPU shape-node executions across the eight runs; maximum CPU
activation and output each 32 bytes, no Memcpy nodes.

Actual integrated OBS hardware qualification:
- Six GPU outputs, two registrations, zero staging readbacks, zero
  failures, maximum measured texture handoff 453 microseconds.
- Nine GPU outputs after resizing 256x144 to 320x180, four
  registrations total, model loaded once.
- GPU -> staged -> GPU switch: changed pixels, RGB mean absolute
  difference 0.0000115741, maximum 1/255; GPU path resumed without
  further staged reads. Two model loads total across GPU activation
  and reactivation.
- Ordinary CPU build: all 55 model-free cases passed. Technical GPU
  staging passed with 20 files including PTX; vendor runtime
  dependencies remain external.

These results concern the existing Portrait Sketch model only.
Further filters, other GPUs, Windows D3D11 interop, macOS, high
resolutions, multiple simultaneous effects and device-loss recovery
remain unqualified. The optional Linux build is not a public
redistribution approval or installed live plugin.

## Subject Cutout: reuse without changing the matte contract

Cutout's motion, history, Longpipe flow and shared matte ABI consume CPU
RGB/matte data, so the Portrait Sketch GPU-resident pipeline cannot simply
replace its current processing. Preserve those capabilities while removing
unnecessary work:

- Reuse capture, worker RGB, mask, raw/compact matte and upload buffers.
- Fuse bilinear resize and NCHW normalization, retaining the original
  intermediate uint8 rounding and the canvas used by Longpipe flow.
- Graphics uses try-lock submission/publication; copy only the small mask
  into reusable upload storage before calling the graphics driver.
- Preserve published CPU matte bytes for independent consumers. Reject
  obsolete generation tags during capture, publication and rendering.
- Optional **Reuse full-resolution source draw** avoids a second draw on
  ordinary inputs when source and analysis dimensions are identical.
  Default off; downscaled and nested/custom source captures are unchanged.
  This option can change alpha compositing, so it is not silently enabled.

Measured on 2026-11-27 using actual OBS and the RTX 2070 SUPER:
U2NetP, MODNet and Longpipe XS default capture produced byte-identical
320x320 ABI matte snapshots versus the previous build on the same portrait.
Each completed eight frames with no capacity growth after warm-up (six,
eight and eight counted growth requests respectively). A prior exploratory
reuse of downscaled capture changed U2NetP matte edges (mean 0.285/255,
max 29/255), so that behavior is excluded from the final option.
Longpipe XS with its actual flow model and opt-in same-size reuse
completed 36 frames, avoided 35 source draws, retained ten counted
capacity growth requests, and passed motion, disable/enable stabilization,
restored position and shared ABI checks. Full U2Net CPU rendering at
320x180 passed the new buffer regression. The 56 required model-free
tests passed, including a non-waiting input/output contention regression.

Counts measure the explicitly instrumented application pixel buffers,
not every allocation made by OBS, ONNX Runtime or history/flow internals.
This does not eliminate Cutout staging reads or provide a measured
end-to-end speedup. GPU-native model transfers and remaining Depth and
Preprocessors paths require separate qualification.

## Depth Camera frame exchange

Keep the existing depth normalization, temporal blend, change-aware
history and shared CPU depth ABI. Reuse capture, worker RGB, depth,
history and upload workspaces; move smoothing outside the input mutex;
take output snapshots with a non-waiting lock and release it before
texture operations. Source geometry invalidation also uses a try-lock:
if busy, defer the change and pass through rather than wait or display
a mismatched map. Generation checks discard obsolete captures/results
and prevent displaying an obsolete texture.

Actual Depth Anything V2 inference in OBS passed on CPU and CUDA
(320x180 source; 252x140 inference). Eight counted capacity growth
requests remained unchanged across eight completed frames. Changing
inference edge to 336 increased the count to 16, then remained stable
across four more frames; one model load total. Resizing the source to
368x208 immediately passed through unchanged pixels and subsequently
recovered real depth. The CUDA analysis-only shared ABI passed range,
metadata, known sequence, invalid ABI, disable, expiration and recovery.
57 required model-free tests passed, including the new contention
and deferred geometry regression. Counts cover instrumented pixel
vectors, not all ONNX or driver allocations. GPU-native model I/O
binding and staging removal are not implemented for Depth yet.

## Preprocessors frame and model workspaces (2026-11-28)

Preprocessors has no independent CPU image snapshot consumer, so the
graphics thread takes a completed RGBA vector by swapping ownership
rather than copying the frame. Capture and worker result vectors retain
capacity. Input reservation, submission and output exchange use try-locks;
failed capture reservations are cancelled on the next available render
handoff. Upload happens outside publication locks. Texture age,
generation and dimensions remain checked before display. The neural
modes share worker-owned OpenCV resize/color/upsampling storage and
gray output buffers. TEED, PiDiNet and HED retain transactional output
scratch buffers across calls. Existing model normalization, safe-step,
scribble and alpha semantics are retained.

Actual OBS CPU and RTX 2070 SUPER CUDA qualification covered canny,
standard_lineart, pyracanny, luminance, Depth Anything V2, TEED,
PiDiNet and HED. Each mode produced stable counted capture/output
capacity after warm-up. CPU run reached 78 uploads; final CUDA run
reached 81 uploads. Repeated rendered outputs were byte-identical
except CUDA Depth Anything: max difference 1/255, total absolute
difference 27 over 245760 RGBA bytes in the final run. The test permits
only bounded one-level variation for CUDA depth, exact equality otherwise.
Safe and scribble variants, mode switches and missing-model passthrough
are covered by the actual OBS tests. Existing native OBS gate requires
buffer reuse; all 57 required cases passed. The new statistics procedure
is recorded in the native ABI manifest.

Counts cover instrumented capture/output pixel vectors only; OpenCV
internals, ONNX allocations and driver synchronization are not measured.
This is a reusable staged processing optimization, not GPU-resident
tensor binding, zero-readback or a measured end-to-end speedup.

## Depth Anything device tensor I/O (2026-11-28)

Optional Linux build flag `SIGHT_ENABLE_CUDA_DEPTH_IO` defaults to the
CUDA texture-interop flag, but works independently of OpenGL interop.
When built and the depth session selects CUDA, RGB normalization, bound
NCHW input and float depth output, parallel finite-value/min/max reduction
and byte quantization stay on device. The worker uploads compact RGB8 via
reusable pinned memory on its own ordered stream and downloads only an
R8 depth map and four-byte validity flag. ORT provider execution is
ordered explicitly at the binding boundary. CPU temporal history and
shared snapshot behavior remain unchanged. No render-thread CUDA calls
are introduced here. Explicit SIGHT_ONNX_DEVICE=cuda requires CUDA
rather than silently accepting CPU initialization. For qualification,
SIGHT_DEPTH_CUDA_IO=host retains the previous tensor path;
SIGHT_DEPTH_CUDA_IO=device requires the new path and rejects CPU or
builds without the feature. Kernels are compiled into embedded PTX; no
extra runtime asset, model, or CUDA runtime library is distributed.

Real Depth Anything V2 on RTX 2070 SUPER, ORT CUDA 1.30.0:
10 timed iterations per size after three warmups, same provider in
host and device comparisons. Final host -> device medians:
252x140 6.725 -> 5.417 ms;
336x196 9.905 -> 7.787 ms;
518x294 17.525 -> 16.851 ms.
Maximum per-pixel difference 1/255 at each size; mean absolute byte
difference <= 0.000471. Device boundary payloads exactly 3N uploaded
and N+4 downloaded vs nominal 12N and 4N host tensor payloads. The
approximately 75% reduction is logical payload, not a PCIe profiler
measurement; ORT internal copies/CPU partitions are not counted.
Test verified one preparation per size, reuse, resize back, reload,
invalid dimensions and CPU/device conflict. Actual forced device-mode
OBS Depth Camera resize and change-aware history, shared snapshot
expiry/recovery and Preprocessors Depth Anything tests passed. Ordinary
CPU build passed 57 required tests. No end-to-end render FPS claim.
Cutout and the other preprocessors still have staged tensor I/O.

## HED multiscale device tensor I/O (2026-11-28)

Optional Linux flag `SIGHT_ENABLE_CUDA_HED_IO` defaults on when
Preprocessors and CUDA depth I/O are enabled. HED uses the existing
five-output ONNX model, with persistent CUDA input and output tensors,
a worker-owned stream, pinned RGB8 input and an embedded PTX kernel
that performs half-pixel bilinear sampling, float32 averaging, double
sigmoid, optional safe-step and byte conversion. It does not replace
the model or the separate CPU scribble NMS implementation. Explicit
`SIGHT_ONNX_DEVICE=cuda` requires the CUDA provider.
`SIGHT_HED_CUDA_IO=host` selects the original tensor path for
comparison; `device` requires the new path, rejecting CPU or builds
without support. Nonfinite maps reject the result instead of
publishing a partial image.

Real ControlNetHED ONNX, RTX 2070 SUPER, ORT CUDA 1.30.0, same provider
on both sides, eight timed iterations after three warmups:
192x128 4.200 -> 3.350 ms; 320x192 9.177 -> 8.367 ms;
512x320 16.453 -> 15.051 ms. Soft and safe-step maps were byte-identical
at all three sizes. Persistent bindings reused; resize back, invalid
dimensions, reload and CPU/device conflict passed.
Application tensor-boundary payload changes from nominal
12N + 4N*(1+1/4+1/16+1/64+1/256) bytes to 4N+4 bytes
(about 77% less). This is logical payload, not measured PCIe traffic.
Forced device mode passed actual OBS Preprocessors rendering, mode
switches, safe and scribble modes, stable counted buffer capacity and
off passthrough. The normal CPU build passed 57 model-free tests and
the original HED reference hashes. Existing CPU capture, final edge
map download, scribble NMS and texture upload remain. No guarantee of
end-to-end FPS improvement or qualification on other platforms.

## TEED and PiDiNet device tensor I/O (2026-11-28)

Optional Linux flag `SIGHT_ENABLE_CUDA_EDGE_IO` defaults to the HED
CUDA I/O build flag. Both processors share a worker-owned device I/O
adapter, preallocated CUDA tensors, pinned compact input and embedded
PTX. TEED preserves its raw RGB CHW input, four full-size logit maps,
float averaging, double sigmoid and configurable safe steps. PiDiNet
preserves its BGR ordering, /255 float conversion, probability output
and safe-step quantization. Both download only a compact edge map
plus a four-byte validity flag. Explicit CUDA selection requires the
provider, not CPU fallback. Nonfinite output rejects publication.
The C++ device kernels retain the original numeric behavior; CPU
and optional Rust paths remain in place.

TEED selects device I/O automatically when built and using CUDA;
`SIGHT_TEED_CUDA_IO=host` restores original tensor transfer.
PiDiNet remains on its original tensor transfer by default because
the measured latency advantage is inconsistent; explicitly set
`SIGHT_PIDINET_CUDA_IO=device` to use the new adapter. Both processors
reject required device mode when CUDA/build support is unavailable.

Same-provider actual model benchmarks, RTX 2070 SUPER, ORT 1.30.0:
eight timed iterations following three warmups per geometry.
First run, host -> device medians (ms):
TEED 192x128 0.902 -> 0.646; 320x192 1.870 -> 1.498;
512x320 5.388 -> 3.566.
PiDiNet 192x128 3.274 -> 3.549; 320x192 7.007 -> 7.612;
512x320 19.406 -> 19.401.
Second run: TEED 1.364 -> 0.563, 1.870 -> 1.632, 5.280 -> 3.957;
PiDiNet 3.809 -> 3.239, 7.525 -> 7.541, 18.788 -> 17.584.
Soft and safe maps byte-identical across both runs and all sizes.
Nominal tensor-boundary payload 28N (TEED) or 16N (PiDiNet) reduced
to 4N+4 bytes; approximately 86% or 75% respectively, not measured
PCIe traffic. Reuse, resize, reload, invalid dimensions and CPU/device
conflict passed, including the different default policies.
Actual forced-device OBS Preprocessors passed TEED/PiDiNet rendering,
safe and scribble variants, mode switching, stable instrumented
pixel capacity and off passthrough. Normal CPU build 57/57 required
tests, original TEED and PiDiNet reference fingerprints retained.
Capture, CPU scribble NMS and texture upload remain; no guarantee
of frame-rate improvement, simultaneous GPU contention performance,
Rust-enabled CUDA parity or other platform qualification.
