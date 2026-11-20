#ifndef MP_WRAPPER_H
#define MP_WRAPPER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pointer to MediaPipe Runner C++ class */
typedef struct mp_runner mp_runner_t;

/* Supported Graph Types */
typedef enum {
    MP_GRAPH_NONE = 0,
    MP_GRAPH_SELFIE_SEGMENTATION,
    MP_GRAPH_FACE_MESH,
    MP_GRAPH_HAND_TRACKING,
    MP_GRAPH_POSE_ESTIMATION,
    MP_GRAPH_CANNY,
    MP_GRAPH_PRIVACY
} mp_graph_type_t;

/**
 * BlazePose Landmark Indices (33 landmarks)
 * For use with MP_GRAPH_POSE_ESTIMATION
 * 
 * ONNX Models available from:
 *   - https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation/MediaPipePoseDetector.onnx
 *   - https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation/PoseLandmarkDetector_float.onnx
 */
typedef enum {
    MP_BACKEND_NONE = 0,
    MP_BACKEND_ONNX,
    MP_BACKEND_LITERT
} mp_backend_t;
typedef enum {
    MP_POSE_NOSE = 0,
    MP_POSE_LEFT_EYE_INNER = 1,
    MP_POSE_LEFT_EYE = 2,
    MP_POSE_LEFT_EYE_OUTER = 3,
    MP_POSE_RIGHT_EYE_INNER = 4,
    MP_POSE_RIGHT_EYE = 5,
    MP_POSE_RIGHT_EYE_OUTER = 6,
    MP_POSE_LEFT_EAR = 7,
    MP_POSE_RIGHT_EAR = 8,
    MP_POSE_MOUTH_LEFT = 9,
    MP_POSE_MOUTH_RIGHT = 10,
    MP_POSE_LEFT_SHOULDER = 11,
    MP_POSE_RIGHT_SHOULDER = 12,
    MP_POSE_LEFT_ELBOW = 13,
    MP_POSE_RIGHT_ELBOW = 14,
    MP_POSE_LEFT_WRIST = 15,
    MP_POSE_RIGHT_WRIST = 16,
    MP_POSE_LEFT_PINKY = 17,
    MP_POSE_RIGHT_PINKY = 18,
    MP_POSE_LEFT_INDEX = 19,
    MP_POSE_RIGHT_INDEX = 20,
    MP_POSE_LEFT_THUMB = 21,
    MP_POSE_RIGHT_THUMB = 22,
    MP_POSE_LEFT_HIP = 23,
    MP_POSE_RIGHT_HIP = 24,
    MP_POSE_LEFT_KNEE = 25,
    MP_POSE_RIGHT_KNEE = 26,
    MP_POSE_LEFT_ANKLE = 27,
    MP_POSE_RIGHT_ANKLE = 28,
    MP_POSE_LEFT_HEEL = 29,
    MP_POSE_RIGHT_HEEL = 30,
    MP_POSE_LEFT_FOOT_INDEX = 31,
    MP_POSE_RIGHT_FOOT_INDEX = 32,
    MP_POSE_LANDMARK_COUNT = 33
} mp_pose_landmark_t;

/* Face Mesh Landmark Count */
#define MP_FACE_MESH_LANDMARK_COUNT 468
#define MP_FACE_MESH_V2_LANDMARK_COUNT 478  /* Includes iris landmarks */

/* Hand Landmark Count */
#define MP_HAND_LANDMARK_COUNT 21

/* Segmentation Mask Data */
typedef struct {
    uint8_t *buffer;
    int width;
    int height;
    int channels; /* Usually 1 (alpha) or 4 (rgba) */
} mp_mask_data_t;

/* Landmark Data (Generic) */
typedef struct {
    float x;
    float y;
    float z;
    float visibility;
    float presence;
} mp_landmark_t;

typedef struct {
    mp_landmark_t *landmarks;
    int count;
} mp_landmark_list_t;

/* Functions */

/* Lifecycle */
/* Allocate and initialize a runner.
   use_gpu: if true, attempts to use OpenGL/OpenCL for graph execution.
   model_path: path to the directory containing model files (e.g., ONNX, TFLite). */
mp_runner_t* mp_create(bool use_gpu, const char *model_path);

/* Destroy runner */
void mp_destroy(mp_runner_t *runner);

/* Start a specific graph */
bool mp_start_graph(mp_runner_t *runner, mp_graph_type_t type);

/* Process a frame (RGBA, 8-bit) */
/* Returns true if a real backend processed the frame; false when no backend
   is loaded or inference failed (outputs are cleared, nothing simulated) */
bool mp_process_frame(mp_runner_t *runner, const uint8_t *data, int width, int height, int stride);

/* Process a single static image (blocking) */
/* Used for reference image analysis (face puppeteering) */
/* Returns true if landmarks were found */
bool mp_process_image(mp_runner_t *runner, const uint8_t *data, int width, int height, int stride, mp_landmark_list_t *out_landmarks);

/* Retrieve latest results (non-blocking) */
/* Returns true if new data is available */
bool mp_get_segmentation_mask(mp_runner_t *runner, mp_mask_data_t *out_mask);

/* Returns true only while a face is currently detected; otherwise count is
   0 and no landmark data is exposed (never stale or fabricated points) */
bool mp_get_landmarks(mp_runner_t *runner, mp_landmark_list_t *out_landmarks);

// Helper to get backend name (e.g., "ONNX", "LiteRT", "Unavailable")
const char* mp_get_backend_name(void);

// Helper to get current graph type
mp_graph_type_t mp_get_current_graph(mp_runner_t *runner);

// Check availability
bool mp_is_onnx_available(void);

#ifdef __cplusplus
}
#endif

#endif /* MP_WRAPPER_H */


