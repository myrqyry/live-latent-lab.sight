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
