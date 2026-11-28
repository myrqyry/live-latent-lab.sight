# Downloading models for Sight

**No model weights are included in the Sight plugin.** Download the models you
need from their publishers and select the local files in the matching OBS
filter. Sight does not automatically download, mirror, or upload model files.

Suggested location on Linux:

    ~/.config/obs-studio/plugin_config/live-latent-lab-sight/models/

You can use any stable local folder. The OBS file picker does not require a
specific model filename, but the correct model **architecture and format** are
important.

| Filter | Suggested file | Publisher download / reference |
| --- | --- | --- |
| Portrait Sketch | animegan_v3_portrait.onnx | [AnimeGANv3 PortraitSketch ONNX](https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch/blob/main/AnimeGANv3_PortraitSketch.onnx) |
| Subject Cutout | u2net.onnx | [rembg U²-Net model release](https://github.com/danielgatis/rembg/releases/tag/v0.0.0) |
| Depth Camera | depth_anything_v2_small_fp32.onnx | [Depth Anything V2 Small FP32 ONNX](https://huggingface.co/onnx-community/depth-anything-v2-small/blob/main/onnx/model.onnx) |
| Face Motion | face_landmarker.task | [Google MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) |
| Body Motion | pose_landmarker.task | [Google MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) |
| Hand & Gesture Motion | gesture_recognizer.task | [Google MediaPipe Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) |
| Object Tracking | efficientdet_lite0.tflite | [Google EfficientDet-Lite0 float32](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite) |

**Choose the specified ONNX exports or compatible .task variants**, not
arbitrary newer or quantized replacements. The tested Depth Anything V2
FP32 file has SHA256:

    afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c

See [Depth provenance](../tests/README-depth.md) and
[Tasks provenance](../tests/README-tasks.md) for revision, input shape,
and the tested Face/Pose/Gesture checksums. The exact Gesture Recognizer
float16/1 URL is in [Hand provenance](../tests/README-hands.md).

## Subject Cutout models

Select the matching model family; do not infer normalization from filenames.
The default U²-Net family also supports the qualified U²-NetP export.
The native adapter preserves the existing full-resolution foreground RGB
and uses a separate alpha mask. MODNet uses [-1,1] normalization, a
512-pixel maximum inference dimension and multiples of 32. Longpipe
uses [0,1] RGB and the fixed rectangular refined matting graph dimensions
from its official ONNX export. MODNet and smaller Longpipe tiers use a
capture edge up to 512 pixels; Large and XL allow up to their native
640- and 1280-pixel canvas edges, bounded by source resolution.
Select portrait or landscape weights to suit the source orientation.
Longpipe uses the upstream fixed-canvas resize contract, not letterboxing
or automatic orientation switching. See [Longpipe's ONNX guide](https://longpipe.dev/docs/onnx/).
The shared matte and motion interfaces remain bounded to a 320-pixel
analysis grid, independently of the higher-resolution rendered matte.
Longpipe can optionally use a matching optical-flow ONNX model with
Sight's matte stabilization control. See [flow setup and qualification](longpipe-flow.md).
The upstream SDK's shared-encoder pipeline and frame-skipping strategy
are not included.
All five landscape tiers plus XS portrait have passed local native OBS
rendering. See [Longpipe tier qualification](longpipe-tiers.md).

[Precision qualification](cutout-precision.md): locally converted FP16
variants with FP32 I/O passed CUDA rendering. Prefer FP32 for CPU.
The experimental INT8 conversions did not pass the quality/performance
comparison; do not substitute them for the verified weights.

| Qualified local file | SHA-256 | Reference |
|---|---|---|
| u2netp.onnx | 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8 | [U²-NetP](https://huggingface.co/nfa04/u2netp-onnx) |
| modnet-fp32.onnx | 07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9 | [MODNet](https://huggingface.co/Xenova/modnet) |
| longpipe-xs-landscape.onnx | 75749759daa4d7e839e8aa189ce6a4e95edc72253b1f5028e833455ebae2f792 | [Longpipe official exports](https://huggingface.co/Longpipe/longpipe/tree/main/onnx/xs) |

U²-NetP and MODNet hashes match the local prior benchmark cache, not a
fresh publisher checksum verification. Longpipe matches the publisher's
ONNX manifest. Model assets remain outside Git. Consult upstream model
terms before redistribution. Optional CTest inputs are SIGHT_U2NETP_MODEL,
SIGHT_MODNET_MODEL, SIGHT_LONGPIPE_MODEL and SIGHT_CUTOUT_PORTRAIT
(a 512x512 portrait fixture).

## Holistic model (MediaPipe 1.2.3 only)

The candidate build's **lll.sight Tracking → Holistic**
OBS filter uses the [official Holistic float16 bundle](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task).
Download it manually and select it in the model picker. Also select
the verified native MediaPipe 1.2.3 runtime in its separate picker.
One graph emits face, pose and both hands but **no classified gestures**.
See the [SHA-256 and test results](mediapipe-1.2.3.md). No models bundled.

## Object Tracking model (MediaPipe 1.2.3)

Use an object-detection model with compatible MediaPipe/TFLite metadata,
not an arbitrary ONNX or YOLO-World TFLite export. The real
tested Google EfficientDet-Lite0 float32 model is 13,836,895 bytes,
SHA256 40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58.
Save it as efficientdet_lite0.tflite in the persistent Sight models
folder above (or choose it in the filter picker). The model is user-owned
and **never bundled into plugin builds**. The verified native MediaPipe
1.2.3 runtime is selected from the already installed Sight plugin
directory; the legacy MediaPipe C runtime is not compatible.

## First run

1. Download only the models for the Sight filters you want to use, and
   save the files in a stable local directory.
2. Add the relevant Sight filter to your OBS source, then open Properties.
3. Use Browse beside the ONNX or MediaPipe Tasks model path and select
   the file you downloaded.
4. Face/Body/Hand Motion also require a supported **native MediaPipe
   Tasks C runtime**: either the verified legacy build or the official
   [MediaPipe 1.2.3 runtime](mediapipe-1.2.3.md). This shared library is
   separate from the .task model and has its own settings path.
5. A missing or incompatible model must leave video unchanged rather
   than return simulated results; check the OBS log for load errors.

Privacy V2 uses its included face detector. Tracking Signal Receiver,
Value Router, and HyperShaderFX can consume Sight's tracking data without
loading an additional model.

## License and provenance distinctions

**Users download model files; Sight does not distribute them.** Model
usage restrictions still apply to the people downloading and using them.

The [AnimeGANv3 PortraitSketch card](https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch)
labels its repository Apache-2.0, but its description specifically
describes non-commercial use and instructs users to contact its author
for commercial permission. Treat that discrepancy as unresolved for
commercial use. A plugin source license does not automatically grant
rights to all external models it supports.

The native MediaPipe library is a different dependency from the model
weights. Providing model download links does **not** settle rights to
bundle or redistribute the pinned libmediapipe.so runtime.
