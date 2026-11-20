# Depth Camera validation and provenance

- Model: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)
- License (model card): Apache-2.0
- Revision: `4472b7362082ad9968fee890ca0f1e5aca36b93d`
- Tested artifact: `onnx/model.onnx` (float32, 99,060,839 bytes)
- SHA256: `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`
- Preprocessing: RGB / 255; ImageNet mean/std; NCHW float32; both dimensions multiples of 14.
- Output: `predicted_depth` float32 [1,H,W], normalized by frame to relative [0,255], nearer values generally brighter.
- Runtime: CPU ONNX Runtime 1.17.1, actual GPU execution **not** implemented.

A model with the same shape is *not* automatically compatible. The source's
input/output names are checked; it does not enable every depth model variant.
The backend is intentionally independent of LiteRT and diffusion.

A successful model test verifies nonconstant output and shape, not accuracy.
OBS render smoke separately verifies real shader output under software OpenGL.
Neither test demonstrates stable FPS or true 3-D camera geometry on a live feed.

## Cadence and temporal smoothing

Capture at most once every 1, 2, 4, or 8 **OBS render frames**; the worker never
runs more than one inference concurrently. If inference takes longer than the
cadence, the worker naturally runs slower.

Temporal smoothing blends the last completed depth image into each **new**
inference result. Intermediate OBS frames reuse the most recent texture.
Smoothing reduces short-term flicker but may cause ghosting on fast motion.

An experimental **Reject stale depth at changed pixels** checkbox (off by
default) compares RGB from two successful inference inputs, and attenuates
the prior depth contribution where appearance changed (12–48 mean
8-bit-channel-difference transition). The worker owns this work, not the OBS
graphics callback; history is reset when the model/input generation changes.
The focused C++ test verifies unchanged, changed, partially changed, disabled,
and mismatched-input cases. This is appearance-based history validation,
**not** optical flow or motion compensation. It neither corrects the latency
between model updates nor establishes improved visual quality on live footage.
Because per-frame depth is min/max normalized, temporal smoothing can soften
but cannot eliminate all scale drift.

The filter now also has bounded result freshness. The **Discard depth if no
fresh result for** control defaults to 5 seconds (2/5/10/20 seconds, or
Never). A real successful inference updates the monotonic completion time;
while the worker is stalled, the last result is shown only until it expires,
then the original OBS source is passed through. This guard is not based on
capture timestamps, so it does not eliminate the lag of a long-running
inference. The output-age unit test covers exact boundary, absence of a
successful result, indefinite retention and invalid/reversed clock values.

Changing the target OBS source dimensions also invalidates the previously
inferred depth texture, queued frame and stale in-flight generation, and
immediately passes through the resized source. It does **not** reload an
already initialized model or reset unrelated controls. The geometry policy
unit test covers initial/unchanged/unknown/resized dimensions, and the
isolated native OBS test verifies a real resize without old-mask pixels.

The temporal policy unit test verifies exact blending and frame cadence.
The OBS shader tests verify depth-map and fog modes with real ONNX inference.
Two additional real ONNX + OBS/Xvfb regressions exercise failure cases:

- `sight_depth_obs_expired_passthrough`: a real depth map is first rendered;
  OBS rendering then pauses for 2.9 s with a 2 s freshness setting. The
  first subsequent native render has **zero pixel differences** from the
  original source and logs the expiry decision. The test then resumes OBS
  rendering and confirms real depth output reappears after new inference.
- `sight_depth_obs_resize_passthrough`: after real inference, the OBS source
  changes from 128×128 to 176×156. The immediate filtered render is identical
  to an unfiltered resize reference (**zero pixel differences**) and logs
  geometry invalidation. It also confirms depth rendering returns after
  inference at the new source geometry, rather than staying in passthrough.
  The live video never goes black or retains the old depth shape.

Verification against the clean Loam checkout: **26/26 CTest passed** with
pinned Depth Anything V2 ONNX and other existing models; baseline before
changes was 24/24. The installed OBS plugin was **not** replaced because it
was a different-sized build from the verified checkout, so its feature
provenance needs reconciliation before deployment. The isolated test proves
the built `.so` renders correctly, not that the installed binary changed.
The image-source test also exercises a real Image input and 336-pixel detail.

The next step toward truly modular vision effects is to publish this validated
depth texture as a shared signal instead of running the same depth model once
per effect; no such shared graph is claimed to exist yet.

A separate Depth Anything 3 / C++-ggml feasibility comparison, performed
without changing this working filter, is recorded in
[README-depth-da3-evaluation.md](README-depth-da3-evaluation.md).
