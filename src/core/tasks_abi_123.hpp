#pragma once

// Linux x86_64 ABI overlay for the MediaPipe C API shipped in the official
// mediapipe==1.2.3 PyPI manylinux_2_28_x86_64 wheel (Nov 11, 2026).
//
// ONLY MpBaseOptions changed layout relative to Sight's existing legacy C
// headers; the Face, Pose and Gesture task options' remaining fields are
// identical. These independent structs keep the legacy headers usable by
// already-installed OBS scenes. Do not link this to an unknown libmediapipe.
//
// Upstream source: google-ai-edge/mediapipe tag v1.2.3, Apache-2.0.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"

namespace sight::mp123 {

// The new API uses MP_DELEGATE_DEFAULT=0 and MP_DELEGATE_CPU=1;
// the legacy header's MP_DELEGATE_CPU=0 must NOT be reused.
inline constexpr int kCpuDelegate = 1;
struct BaseOptions {
    const char* model_asset_buffer = nullptr;
    unsigned int model_asset_buffer_count = 0;
    const char* model_asset_path = nullptr;
    int file_descriptor = -1; // 0 is a valid POSIX fd and is not "unset"
    int delegate = kCpuDelegate;
    int host_environment = 0;
    int host_system = 0;
    const char* host_version = nullptr;
    const char* ca_bundle_path = nullptr;
    const char* app_id = nullptr;
    const char* app_version = nullptr;
    const char* npu_dispatch_library_dir = nullptr;
};

struct FaceOptions {
    BaseOptions base_options{};
    MpRunningMode running_mode = MP_RUNNING_MODE_IMAGE;
    int num_faces = 1;
    float min_face_detection_confidence = .5f;
    float min_face_presence_confidence = .5f;
    float min_tracking_confidence = .5f;
    bool output_face_blendshapes = false;
    bool output_facial_transformation_matrixes = false;
    MpFaceLandmarkerOptions::result_callback_fn result_callback = nullptr;
};

struct PoseOptions {
    BaseOptions base_options{};
    MpRunningMode running_mode = MP_RUNNING_MODE_IMAGE;
    int num_poses = 1;
    float min_pose_detection_confidence = .5f;
    float min_pose_presence_confidence = .5f;
    float min_tracking_confidence = .5f;
    bool output_segmentation_masks = false;
    MpPoseLandmarkerOptions::result_callback_fn result_callback = nullptr;
};

struct GestureOptions {
    BaseOptions base_options{};
    MpRunningMode running_mode = MP_RUNNING_MODE_IMAGE;
    int num_hands = 1;
    float min_hand_detection_confidence = .5f;
    float min_hand_presence_confidence = .5f;
    float min_tracking_confidence = .5f;
    MpClassifierOptions canned_gestures_classifier_options{};
    MpClassifierOptions custom_gestures_classifier_options{};
    MpGestureRecognizerOptions::result_callback_fn result_callback = nullptr;
};
} // namespace sight::mp123
