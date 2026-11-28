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
