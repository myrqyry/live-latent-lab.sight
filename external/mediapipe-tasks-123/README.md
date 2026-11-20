# MediaPipe v1.2.3 Holistic C result layout

The include/mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h
file is a verbatim, Apache-2.0 licensed header from
google-ai-edge/mediapipe tag v1.2.3. Its copyright and license text
remain intact. This vendored result struct is for offline C++ type checking
and conversion into Sight's already-established TrackingData types.

Sight does not vendor the holistic_landmarker.task model, nor the
native libmediapipe.so runtime. Neither is downloaded by the plugin.

The conversion helper in src/core/holistic_signal_adapter.hpp now
serves a registered, isolated-OBS-tested Holistic filter. One graph
publishes Face, Body and Hand channels. The pointer-valued segmentation
mask is not published; it requires a separately owned GPU/CPU snapshot
contract before MediaPipe closes its result buffers.
