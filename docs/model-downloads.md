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

## Holistic model (MediaPipe 1.2.3 only)

The development build's **LLL.sight Holistic Tracking (MediaPipe 1.2.3)**
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
Value Router, and Touch Field can consume Sight's tracking data without
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
