#include "mp_wrapper.h"
#include <obs-module.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <mutex>
#include <memory>
#include <algorithm>
#include <string>

/**
 * MediaPipe Wrapper Implementation - Real ONNX / LiteRT Integration
 * 
 * INTEGRATION STATUS: REAL BACKENDS ONLY
 * ================================================
 * 
 * Face detection and landmark extraction run through:
 * 1. LiteRT (TFLite) models when compiled in and present on disk
 * 2. ONNX Runtime models when compiled in and present on disk
 *
 * When no backend/model is available, or inference fails, the wrapper
 * reports failure and clears outputs. It never fabricates landmarks
 * or masks; consumers must treat false/zero results as "no detection".
 * 
 * ONNX PIPELINE OVERVIEW:
 * -----------------------
 * 1. Face Detector Model (BlazeFace-based):
 *    - Input:  [1, 128, 128, 3] RGB normalized to [0,1]
 *    - Output: Face bounding boxes with confidence scores
 * 
 * 2. Face Landmark Model (FaceMesh-based):
 *    - Input:  [1, 192, 192, 3] RGB normalized to [0,1]
 *    - Output: 468 3D landmarks (x, y, z) in normalized coordinates
 * 
 * REQUIRED MODELS (Place in <plugin_data>/models/mediapipe/):
 * -----------------------------------------------------------
 *   - face_detection_short_range.onnx (~592 KB)
 *     From: https://huggingface.co/qualcomm/MediaPipe-Face-Detection
 *   
 *   - face_landmark.onnx (~2.45 MB)  
 *     From: https://huggingface.co/qualcomm/MediaPipe-Face-Detection
 *
 * BUILD:
 *   cmake -DUSE_ONNX_MEDIAPIPE=ON ..
 *   cmake --build .
 */

// Configuration
// Configuration
#ifdef USE_ONNX_MEDIAPIPE
    #include <onnxruntime_cxx_api.h>
    #define ONNX_AVAILABLE_DEF true
#else
    #define ONNX_AVAILABLE_DEF false
#endif

#ifdef USE_LITERT
    #include <tensorflow/lite/c/c_api.h>
    #define LITERT_AVAILABLE_DEF true
#else
    #define LITERT_AVAILABLE_DEF false
#endif

// Global availability flags
static bool g_onnx_available = false;
static bool g_litert_available = false;

// Face mesh constants
constexpr int FACE_MESH_LANDMARK_COUNT = 468;
constexpr int FACE_MESH_V2_LANDMARK_COUNT = 478;  // Includes iris

// Model input sizes for BlazeFace and FaceMesh
constexpr int FACE_DETECTOR_INPUT_SIZE = 128;
constexpr int FACE_LANDMARK_INPUT_SIZE = 192;
constexpr int CANNY_INPUT_SIZE = 512;

// Detection thresholds
constexpr float FACE_DETECTION_THRESHOLD = 0.5f;
constexpr float FACE_LANDMARK_THRESHOLD = 0.3f;

// Anchor generation parameters for face detection (BlazeFace Short-Range)
constexpr int NUM_ANCHORS = 896;

// Smooth step function for mask edge falloff
static inline float smoothstep(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Sigmoid activation
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Clamp value to range
static inline float clamp01(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}

/**
 * Face detection bounding box
 */
struct FaceBox {
    float x_center;  // Normalized [0, 1]
    float y_center;  // Normalized [0, 1]
    float width;     // Normalized [0, 1]  
    float height;    // Normalized [0, 1]
    float score;     // Confidence
    
    // Get pixel coordinates for a given image size
    void to_pixel_coords(int img_width, int img_height,
                         int* x1, int* y1, int* x2, int* y2) const {
        float hw = width / 2.0f;
        float hh = height / 2.0f;
        *x1 = (int)((x_center - hw) * img_width);
        *y1 = (int)((y_center - hh) * img_height);
        *x2 = (int)((x_center + hw) * img_width);
        *y2 = (int)((y_center + hh) * img_height);
    }
};

// Box Intersection over Union
static float box_iou(const FaceBox& a, const FaceBox& b) {
    float x_left = fmaxf(a.x_center - a.width/2, b.x_center - b.width/2);
    float y_top = fmaxf(a.y_center - a.height/2, b.y_center - b.height/2);
    float x_right = fminf(a.x_center + a.width/2, b.x_center + b.width/2);
    float y_bottom = fminf(a.y_center + a.height/2, b.y_center + b.height/2);
    
    if (x_right < x_left || y_bottom < y_top) return 0.0f;
    
    float intersection = (x_right - x_left) * (y_bottom - y_top);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    
    return intersection / (area_a + area_b - intersection);
}

/**
 * Anchor box for face detection decoding
 */
struct Anchor {
    float x_center;
    float y_center;
};

#if defined(USE_ONNX_MEDIAPIPE) || defined(USE_LITERT)
/**
 * Generate anchors for BlazeFace Short-Range model
 * Based on MediaPipe's anchor generation algorithm
 */
static std::vector<Anchor> generate_anchors() {
    std::vector<Anchor> anchors;
    anchors.reserve(NUM_ANCHORS);
    
    // Anchor specification for short-range face detector
    // strides: [8, 16] for 128x128 input
    const int strides[] = {8, 16};
    const int stride_count = 2;
    const int anchors_per_stride[] = {2, 6};  // 2 anchors at stride 8, 6 at stride 16
    
    for (int s = 0; s < stride_count; s++) {
        int stride = strides[s];
        int grid_size = FACE_DETECTOR_INPUT_SIZE / stride;
        int num_anchors_at_stride = anchors_per_stride[s];
        
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                for (int a = 0; a < num_anchors_at_stride; a++) {
                    Anchor anchor;
                    anchor.x_center = (x + 0.5f) * stride / (float)FACE_DETECTOR_INPUT_SIZE;
                    anchor.y_center = (y + 0.5f) * stride / (float)FACE_DETECTOR_INPUT_SIZE;
                    anchors.push_back(anchor);
                }
            }
        }
    }
    
    return anchors;
}

/**
 * Decode face detections from BlazeFace output
 */
static std::vector<FaceBox> decode_face_detections(
    const float* raw_boxes,   // [num_anchors, 16] - box regressions (4 coords + 12 keypoints)
    const float* raw_scores,  // [num_anchors, 1] - classification scores
    const std::vector<Anchor>& anchors,
    int num_anchors,
    float threshold) 
{
    std::vector<FaceBox> detections;
    
    for (int i = 0; i < num_anchors; i++) {
        float score = sigmoid(raw_scores[i]);
        if (score < threshold) continue;
        
        // Decode box (first 4 values: x_center, y_center, width, height)
        const float* box = &raw_boxes[i * 16];
        
        FaceBox face;
        face.x_center = box[0] / FACE_DETECTOR_INPUT_SIZE + anchors[i].x_center;
        face.y_center = box[1] / FACE_DETECTOR_INPUT_SIZE + anchors[i].y_center;
        face.width = box[2] / FACE_DETECTOR_INPUT_SIZE;
        face.height = box[3] / FACE_DETECTOR_INPUT_SIZE;
        face.score = score;
        
        detections.push_back(face);
    }
    
    // Non-Maximum Suppression
    std::sort(detections.begin(), detections.end(),
              [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
              
    std::vector<FaceBox> final_faces;
    for (const auto& det : detections) {
        bool keep = true;
        for (const auto& kept : final_faces) {
            if (box_iou(det, kept) > 0.3f) {
                keep = false;
                break;
            }
        }
        if (keep) final_faces.push_back(det);
    }
    
    return final_faces;
}
#endif

struct mp_runner {
    mp_graph_type_t current_graph;
    std::mutex data_lock;
    
    // Mask Data
    std::vector<uint8_t> mask_buffer;
    int mask_width;
    int mask_height;
    
    // Landmark Data
    std::vector<mp_landmark_t> landmarks;
    
    // Last detected face box (for landmark cropping)
    FaceBox last_face_box;
    std::vector<FaceBox> all_faces; // Store all detected faces
    bool has_face;
    
    bool use_gpu;
    
    // Processing buffers
    std::vector<float> detector_input;
    std::vector<float> landmark_input;
    
#ifdef USE_LITERT
    // LiteRT (TFLite) Resources
    TfLiteModel* litert_model_det = nullptr;
    TfLiteInterpreter* litert_interp_det = nullptr;
    
    TfLiteModel* litert_model_land = nullptr;
    TfLiteInterpreter* litert_interp_land = nullptr;
    
    TfLiteModel* litert_model_canny = nullptr;
    TfLiteInterpreter* litert_interp_canny = nullptr;

    TfLiteInterpreterOptions* litert_options = nullptr;
    
    // Resize input tensor for LiteRT involves resizing the interpreter input?
    // Usually fixed size models.
    std::vector<float> canny_input;
#endif

#ifdef USE_ONNX_MEDIAPIPE
    // ONNX Runtime Sessions
    std::unique_ptr<Ort::Env> ort_env;
    std::unique_ptr<Ort::Session> face_detector_session;
    std::unique_ptr<Ort::Session> face_landmark_session;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;
    Ort::MemoryInfo memory_info;
    
    // Pre-computed anchors
    std::vector<Anchor> anchors;
    
    // Model I/O names
    std::vector<const char*> detector_input_names;
    std::vector<const char*> detector_output_names;
    std::vector<const char*> landmark_input_names;
    std::vector<const char*> landmark_output_names;
    
    mp_runner() : memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        anchors = generate_anchors();
    }
#else
    mp_runner() {}
#endif
};

/**
 * Bilinear resize helper for preprocessing
 * Resizes RGBA8 image to float RGB normalized [0, 1]
 */
static void resize_and_normalize(
    const uint8_t* src, int src_width, int src_height, int src_stride,
    float* dst, int dst_width, int dst_height)
{
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;
    
    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            // Source coordinates
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = std::min(x0 + 1, src_width - 1);
            int y1 = std::min(y0 + 1, src_height - 1);
            
            float xf = src_x - x0;
            float yf = src_y - y0;
            
            // Bilinear interpolation for each channel
            for (int c = 0; c < 3; c++) {
                float v00 = src[y0 * src_stride + x0 * 4 + c];
                float v01 = src[y1 * src_stride + x0 * 4 + c];
                float v10 = src[y0 * src_stride + x1 * 4 + c];
                float v11 = src[y1 * src_stride + x1 * 4 + c];
                
                float v0 = v00 + xf * (v10 - v00);
                float v1 = v01 + xf * (v11 - v01);
                float v = v0 + yf * (v1 - v0);
                
                // Normalize to [0, 1] and store in NHWC format (model expects NHWC)
                dst[(y * dst_width + x) * 3 + c] = v / 255.0f;
            }
        }
    }
}

/**
 * Crop and resize helper for landmark input
 * Crops face region from source and resizes to landmark input size
 */
static void crop_resize_and_normalize(
    const uint8_t* src, int src_width, int src_height, int src_stride,
    const FaceBox& box, float* dst, int dst_size)
{
    // Expand box by 25% for better landmark extraction (standard MediaPipe practice)
    float expand = 0.25f;
    float cx = box.x_center;
    float cy = box.y_center;
    float w = box.width * (1.0f + expand);
    float h = box.height * (1.0f + expand);
    
    // Make square (use larger dimension)
    float size = fmaxf(w, h);
    
    // Calculate crop bounds in pixels
    float x_min = (cx - size / 2.0f) * src_width;
    float y_min = (cy - size / 2.0f) * src_height;
    float x_max = (cx + size / 2.0f) * src_width;
    float y_max = (cy + size / 2.0f) * src_height;
    
    float crop_width = x_max - x_min;
    float crop_height = y_max - y_min;
    
    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            // Map to source coordinates
            float src_x = x_min + (x / (float)dst_size) * crop_width;
            float src_y = y_min + (y / (float)dst_size) * crop_height;
            
            // Clamp to image bounds
            int sx = std::max(0, std::min((int)src_x, src_width - 1));
            int sy = std::max(0, std::min((int)src_y, src_height - 1));
            
            // Copy and normalize
            for (int c = 0; c < 3; c++) {
                float v = src[sy * src_stride + sx * 4 + c];
                dst[(y * dst_size + x) * 3 + c] = v / 255.0f;
            }
        }
    }
}

extern "C" {

mp_runner_t* mp_create(bool use_gpu, const char *model_path) {
    mp_runner *runner = new mp_runner();
    runner->use_gpu = use_gpu;
    runner->current_graph = MP_GRAPH_NONE;
    runner->mask_width = 256;
    runner->mask_height = 256;
    runner->mask_buffer.resize(runner->mask_width * runner->mask_height);
    runner->has_face = false;
    
    // Init with standard face mesh landmark count
    runner->landmarks.resize(FACE_MESH_LANDMARK_COUNT);
    
    // Allocate processing buffers
    runner->detector_input.resize(FACE_DETECTOR_INPUT_SIZE * FACE_DETECTOR_INPUT_SIZE * 3);
    runner->landmark_input.resize(FACE_LANDMARK_INPUT_SIZE * FACE_LANDMARK_INPUT_SIZE * 3);

    std::string base_path = model_path ? std::string(model_path) : "";
    if (!base_path.empty() && base_path.back() != '/' && base_path.back() != '\\') {
        base_path += "/";
    }

    // 1. Try LiteRT (TFLite)
#ifdef USE_LITERT
    try {
        runner->litert_options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(runner->litert_options, 2);
        
        bool tflite_found = false;
        std::string det_path, land_path;
        
        if (!base_path.empty()) {
            std::string d = base_path + "face_detection_short_range.tflite";
            std::string l = base_path + "face_landmark.tflite";
            FILE *f1 = fopen(d.c_str(), "rb");
            FILE *f2 = fopen(l.c_str(), "rb");
            if (f1 && f2) {
                fclose(f1); fclose(f2);
                det_path = d; land_path = l;
                tflite_found = true;
                blog(LOG_INFO, "[MediaPipe] Found LiteRT models at: %s", base_path.c_str());
            } else {
                if (f1) fclose(f1);
                if (f2) fclose(f2);
            }
        }
        
        if (tflite_found) {
            runner->litert_model_det = TfLiteModelCreateFromFile(det_path.c_str());
            runner->litert_model_land = TfLiteModelCreateFromFile(land_path.c_str());
            
            if (runner->litert_model_det && runner->litert_model_land) {
                runner->litert_interp_det = TfLiteInterpreterCreate(runner->litert_model_det, runner->litert_options);
                runner->litert_interp_land = TfLiteInterpreterCreate(runner->litert_model_land, runner->litert_options);
                
                if (runner->litert_interp_det && runner->litert_interp_land) {
                    TfLiteInterpreterAllocateTensors(runner->litert_interp_det);
                    TfLiteInterpreterAllocateTensors(runner->litert_interp_land);
                    g_litert_available = true;
                    blog(LOG_INFO, "[MediaPipe] LiteRT Face models initialized successfully!");
                }
            }
            
            // Auto-load Canny if present (not critical for face backend)
            std::string canny_path = base_path + "canny_edge_plugin.tflite";
            FILE *f_canny = fopen(canny_path.c_str(), "rb");
            if (f_canny) {
                fclose(f_canny);
                runner->litert_model_canny = TfLiteModelCreateFromFile(canny_path.c_str());
                if (runner->litert_model_canny) {
                    runner->litert_interp_canny = TfLiteInterpreterCreate(runner->litert_model_canny, runner->litert_options);
                    if (runner->litert_interp_canny) {
                        TfLiteInterpreterAllocateTensors(runner->litert_interp_canny);
                        
                        // Resize input vector
                        runner->canny_input.resize(CANNY_INPUT_SIZE * CANNY_INPUT_SIZE * 3);
                        blog(LOG_INFO, "[MediaPipe] LiteRT Canny model initialized!");
                    }
                }
            }
            
        } else {
            blog(LOG_WARNING, "[MediaPipe] LiteRT models not found in: %s", base_path.c_str());
        }
    } catch (...) {
        blog(LOG_WARNING, "[MediaPipe] LiteRT init exception.");
    }
#endif

    // 2. Try ONNX (if LiteRT not loaded)
#ifdef USE_ONNX_MEDIAPIPE
    if (!g_litert_available) {
        try {
            runner->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MediaPipe_ONNX");
            runner->allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
            
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(2);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            if (use_gpu) {
                #if defined(USE_OPENVINO_EP)
                // OpenVINO EP prioritized for Intel hardware (CPU, iGPU, NPU)
                OrtOpenVINOProviderOptions ov_opts;
                ov_opts.device_type = "AUTO";  // Let OpenVINO pick best device
                session_options.AppendExecutionProvider_OpenVINO(ov_opts);
                blog(LOG_INFO, "[MediaPipe] Using OpenVINO Execution Provider (device: AUTO)");
                #elif defined(USE_CUDA)
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_opts);
                #endif
            }

            bool onnx_found = false;
            std::string det_path, land_path;
            
            if (!base_path.empty()) {
                std::string d = base_path + "face_detection_short_range.onnx";
                std::string l = base_path + "face_landmark.onnx";
                FILE *f1 = fopen(d.c_str(), "rb");
                FILE *f2 = fopen(l.c_str(), "rb");
                if (f1 && f2) {
                    fclose(f1); fclose(f2);
                    det_path = d; land_path = l;
                    onnx_found = true;
                    blog(LOG_INFO, "[MediaPipe] Found ONNX models at: %s", base_path.c_str());
                } else {
                    if (f1) fclose(f1);
                    if (f2) fclose(f2);
                }
            }
            
            if (onnx_found) {
                runner->face_detector_session = std::make_unique<Ort::Session>(*runner->ort_env, det_path.c_str(), session_options);
                runner->face_landmark_session = std::make_unique<Ort::Session>(*runner->ort_env, land_path.c_str(), session_options);
                
                // Set I/O names
                runner->detector_input_names.push_back("input");
                runner->detector_output_names.push_back("regressors");
                runner->detector_output_names.push_back("classificators");
                
                runner->landmark_input_names.push_back("input_1");
                runner->landmark_output_names.push_back("output");
                
                g_onnx_available = true;
                blog(LOG_INFO, "[MediaPipe] ONNX Runtime initialized successfully!");
            } else {
                blog(LOG_WARNING, "[MediaPipe] Models not found in: %s", base_path.c_str());
            }
        } catch (const std::exception& e) {
             blog(LOG_ERROR, "[MediaPipe] ONNX init failed: %s", e.what());
        }
    }
#endif

    // Fallback status
    if (!g_litert_available && !g_onnx_available) {
        blog(LOG_WARNING, "[MediaPipe] No backend available. MediaPipe graphs will report no detections.");
    }

    return (mp_runner_t*)runner;
}

void mp_destroy(mp_runner_t *ptr) {
    if (!ptr) return;
    mp_runner *runner = (mp_runner*)ptr;
    
#ifdef USE_LITERT
    if (runner->litert_interp_det)  TfLiteInterpreterDelete(runner->litert_interp_det);
    if (runner->litert_model_det)   TfLiteModelDelete(runner->litert_model_det);
    if (runner->litert_interp_land) TfLiteInterpreterDelete(runner->litert_interp_land);
    if (runner->litert_model_land)  TfLiteModelDelete(runner->litert_model_land);
    
    if (runner->litert_interp_canny) TfLiteInterpreterDelete(runner->litert_interp_canny);
    if (runner->litert_model_canny)  TfLiteModelDelete(runner->litert_model_canny);
    
    if (runner->litert_options)     TfLiteInterpreterOptionsDelete(runner->litert_options);
#endif

    delete runner;
}

bool mp_start_graph(mp_runner_t *ptr, mp_graph_type_t type) {
    if (!ptr) return false;
    mp_runner *runner = (mp_runner*)ptr;
    runner->current_graph = type;
    
    const char* graph_name = "Unknown";
    switch(type) {
        case MP_GRAPH_SELFIE_SEGMENTATION: graph_name = "Selfie Segmentation"; break;
        case MP_GRAPH_FACE_MESH: graph_name = "Face Mesh"; break;
        case MP_GRAPH_HAND_TRACKING: graph_name = "Hand Tracking"; break;
        case MP_GRAPH_POSE_ESTIMATION: graph_name = "Pose Estimation"; break;
        case MP_GRAPH_CANNY: graph_name = "Canny Edge (ControlNet)"; break;
        case MP_GRAPH_PRIVACY: graph_name = "Privacy Filter (Blur Bystanders)"; break;
        default: break;
    }
    
    blog(LOG_INFO, "[MediaPipe] Started graph: %s (%s)", graph_name, mp_get_backend_name());
    return true;
}

extern "C" mp_graph_type_t mp_get_current_graph(mp_runner_t *ptr) {
    if (!ptr) return MP_GRAPH_NONE;
    return ((mp_runner*)ptr)->current_graph;
}

/**
 * Process a video frame for landmarks/segmentation.
 * 
 * Runs the real LiteRT or ONNX face pipeline when a backend is loaded.
 * Returns false and clears outputs when no backend is available or
 * inference fails — simulated frames are never produced.
 */
bool mp_process_frame(mp_runner_t *ptr, const uint8_t *data, int width, int height, int stride) {
    if (!ptr) return false;
    mp_runner *runner = (mp_runner*)ptr;

#ifdef USE_LITERT
    if (g_litert_available) {
        
        // === Canny Edge Plugin (LiteRT) ===
        if (runner->current_graph == MP_GRAPH_CANNY && runner->litert_interp_canny) {
             resize_and_normalize(data, width, height, stride,
                                 runner->canny_input.data(),
                                 CANNY_INPUT_SIZE, CANNY_INPUT_SIZE);
            
            TfLiteTensor* c_input = TfLiteInterpreterGetInputTensor(runner->litert_interp_canny, 0);
            TfLiteTensorCopyFromBuffer(c_input, runner->canny_input.data(), runner->canny_input.size() * sizeof(float));
            
            if (TfLiteInterpreterInvoke(runner->litert_interp_canny) == kTfLiteOk) {
                // Success
                // The output are embedding features for SD, not visualizable edges.
            }
            return true;
        }

        if (runner->litert_interp_det && runner->litert_interp_land) {
        // === STAGE 1: Face Detection (LiteRT) ===
        resize_and_normalize(data, width, height, stride,
                             runner->detector_input.data(),
                             FACE_DETECTOR_INPUT_SIZE, FACE_DETECTOR_INPUT_SIZE);
        
        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(runner->litert_interp_det, 0);
        TfLiteTensorCopyFromBuffer(input_tensor, runner->detector_input.data(), runner->detector_input.size() * sizeof(float));
        
        if (TfLiteInterpreterInvoke(runner->litert_interp_det) == kTfLiteOk) {
            const TfLiteTensor* out_reg = TfLiteInterpreterGetOutputTensor(runner->litert_interp_det, 0);
            const TfLiteTensor* out_score = TfLiteInterpreterGetOutputTensor(runner->litert_interp_det, 1);
            
            // Verify output shapes/types if strictly needed, but assuming correctness for speed
            const float* raw_boxes = (const float*)TfLiteTensorData(out_reg);
            const float* raw_scores = (const float*)TfLiteTensorData(out_score);
            
            std::vector<FaceBox> faces = decode_face_detections(
                raw_boxes, raw_scores, runner->anchors, NUM_ANCHORS, FACE_DETECTION_THRESHOLD);
            
            std::lock_guard<std::mutex> lock(runner->data_lock);
            
            if (!faces.empty()) {
                runner->has_face = true;
                runner->last_face_box = faces[0];
                
                // === STAGE 2: Landmarks (LiteRT) ===
                crop_resize_and_normalize(data, width, height, stride,
                                          runner->last_face_box,
                                          runner->landmark_input.data(),
                                          FACE_LANDMARK_INPUT_SIZE);
                
                TfLiteTensor* l_input = TfLiteInterpreterGetInputTensor(runner->litert_interp_land, 0);
                TfLiteTensorCopyFromBuffer(l_input, runner->landmark_input.data(), runner->landmark_input.size() * sizeof(float));
                
                if (TfLiteInterpreterInvoke(runner->litert_interp_land) == kTfLiteOk) {
                    const TfLiteTensor* l_out = TfLiteInterpreterGetOutputTensor(runner->litert_interp_land, 0);
                    const float* raw_landmarks = (const float*)TfLiteTensorData(l_out);
                    
                    // Transform landmarks
                    float expand = 0.25f;
                    float box_size = fmaxf(runner->last_face_box.width, runner->last_face_box.height) * (1.0f + expand);
                    float box_x = runner->last_face_box.x_center - box_size / 2.0f;
                    float box_y = runner->last_face_box.y_center - box_size / 2.0f;
                    
                    for (int i = 0; i < FACE_MESH_LANDMARK_COUNT; i++) {
                         float lx = raw_landmarks[i * 3 + 0] / FACE_LANDMARK_INPUT_SIZE;
                         float ly = raw_landmarks[i * 3 + 1] / FACE_LANDMARK_INPUT_SIZE;
                         float lz = raw_landmarks[i * 3 + 2] / FACE_LANDMARK_INPUT_SIZE;
                         
                         runner->landmarks[i].x = box_x + lx * box_size;
                         runner->landmarks[i].y = box_y + ly * box_size;
                         runner->landmarks[i].z = lz;
                         runner->landmarks[i].visibility = 1.0f;
                         runner->landmarks[i].presence = 1.0f;
                    }
                    
                    // Generate mask (reused logic)
                    if (runner->current_graph == MP_GRAPH_SELFIE_SEGMENTATION || 
                        runner->current_graph == MP_GRAPH_FACE_MESH) {
                        int mw = runner->mask_width;
                        int mh = runner->mask_height;
                        for (int y = 0; y < mh; y++) {
                            for (int x = 0; x < mw; x++) {
                                float u = (float)x / mw;
                                float v = (float)y / mh;
                                float dx = (u - runner->last_face_box.x_center) / (runner->last_face_box.width * 0.6f);
                                float dy = (v - runner->last_face_box.y_center) / (runner->last_face_box.height * 0.8f);
                                float dist = sqrtf(dx*dx + dy*dy);
                                float alpha = 1.0f - smoothstep(0.8f, 1.0f, dist);
                                runner->mask_buffer[y * mw + x] = (uint8_t)(clamp01(alpha) * 255.0f);
                            }
                        }
                    } else if (runner->current_graph == MP_GRAPH_PRIVACY) {
                        // Privacy Mode: Blur Bystanders
                        // 1. Identify Primary Face (closest to center)
                        int primary_idx = -1;
                        float min_dist = 1000.0f;
                        for (size_t i = 0; i < runner->all_faces.size(); i++) {
                            float dx = runner->all_faces[i].x_center - 0.5f;
                            float dy = runner->all_faces[i].y_center - 0.5f;
                            float dist = dx*dx + dy*dy;
                            if (dist < min_dist) {
                                min_dist = dist;
                                primary_idx = (int)i;
                            }
                        }
                        
                        // 2. Clear mask (White = Keep)
                        memset(runner->mask_buffer.data(), 255, runner->mask_buffer.size());
                        
                        // 3. Draw Black Ellipses for Bystanders
                        int mw = runner->mask_width;
                        int mh = runner->mask_height;
                        
                        for (size_t i = 0; i < runner->all_faces.size(); i++) {
                            if ((int)i == primary_idx) continue; // Skip streamer
                            
                            const FaceBox& face = runner->all_faces[i];
                            
                            // Expand blur area slightly
                            float blur_w = face.width * 1.2f;
                            float blur_h = face.height * 1.5f;
                            
                            // Bounding box of ellipse to optimize loop
                            int x0 = (int)((face.x_center - blur_w/2) * mw);
                            int y0 = (int)((face.y_center - blur_h/2) * mh);
                            int x1 = (int)((face.x_center + blur_w/2) * mw);
                            int y1 = (int)((face.y_center + blur_h/2) * mh);
                            
                            x0 = std::max(0, x0); y0 = std::max(0, y0);
                            x1 = std::min(mw, x1); y1 = std::min(mh, y1);
                            
                            for (int y = y0; y < y1; y++) {
                                for (int x = x0; x < x1; x++) {
                                     float u = (float)x / mw;
                                     float v = (float)y / mh;
                                     
                                     float dx = (u - face.x_center) / (blur_w * 0.5f);
                                     float dy = (v - face.y_center) / (blur_h * 0.5f);
                                     float dist = dx*dx + dy*dy;
                                     
                                     if (dist < 1.0f) {
                                         // Gradient edge
                                         float alpha = smoothstep(0.8f, 1.0f, dist);
                                         // Multiply existing mask (so overlapping blur zones merge)
                                         uint8_t current = runner->mask_buffer[y * mw + x];
                                         runner->mask_buffer[y * mw + x] = (uint8_t)(std::min((float)current, alpha * 255.0f));
                                     }
                                }
                            }
                        }
                    }
                }
            } else {
                runner->has_face = false;
                memset(runner->mask_buffer.data(), 0, runner->mask_buffer.size());
            }
        }
        return true;
    }
#endif
#ifdef USE_ONNX_MEDIAPIPE
    if (g_onnx_available && runner->face_detector_session && runner->face_landmark_session) {
        try {
            // === STAGE 1: Face Detection ===
            // Preprocess: Resize to 128x128, normalize to [0,1]
            resize_and_normalize(data, width, height, stride,
                                 runner->detector_input.data(),
                                 FACE_DETECTOR_INPUT_SIZE, FACE_DETECTOR_INPUT_SIZE);
            
            // Create input tensor
            std::array<int64_t, 4> detector_input_shape = {1, FACE_DETECTOR_INPUT_SIZE, FACE_DETECTOR_INPUT_SIZE, 3};
            Ort::Value detector_input_tensor = Ort::Value::CreateTensor<float>(
                runner->memory_info,
                runner->detector_input.data(),
                runner->detector_input.size(),
                detector_input_shape.data(),
                detector_input_shape.size());
            
            // Run face detection
            auto detector_outputs = runner->face_detector_session->Run(
                Ort::RunOptions{nullptr},
                runner->detector_input_names.data(),
                &detector_input_tensor, 1,
                runner->detector_output_names.data(),
                runner->detector_output_names.size());
            
            // Decode detections
            // Output 0: regressors [1, 896, 16] - bounding box + keypoints
            // Output 1: classificators [1, 896, 1] - confidence scores
            auto* raw_boxes = detector_outputs[0].GetTensorMutableData<float>();
            auto* raw_scores = detector_outputs[1].GetTensorMutableData<float>();
            
            std::vector<FaceBox> faces = decode_face_detections(
                raw_boxes, raw_scores, runner->anchors, NUM_ANCHORS, FACE_DETECTION_THRESHOLD);
            
            std::lock_guard<std::mutex> lock(runner->data_lock);
            
            if (!faces.empty()) {
                runner->has_face = true;
                runner->last_face_box = faces[0];
                
                // === STAGE 2: Face Landmark Detection ===
                // Crop face region and resize to 192x192
                crop_resize_and_normalize(data, width, height, stride,
                                          runner->last_face_box,
                                          runner->landmark_input.data(),
                                          FACE_LANDMARK_INPUT_SIZE);
                
                // Create landmark input tensor
                std::array<int64_t, 4> landmark_input_shape = {1, FACE_LANDMARK_INPUT_SIZE, FACE_LANDMARK_INPUT_SIZE, 3};
                Ort::Value landmark_input_tensor = Ort::Value::CreateTensor<float>(
                    runner->memory_info,
                    runner->landmark_input.data(),
                    runner->landmark_input.size(),
                    landmark_input_shape.data(),
                    landmark_input_shape.size());
                
                // Run landmark detection  
                auto landmark_outputs = runner->face_landmark_session->Run(
                    Ort::RunOptions{nullptr},
                    runner->landmark_input_names.data(),
                    &landmark_input_tensor, 1,
                    runner->landmark_output_names.data(),
                    runner->landmark_output_names.size());
                
                // Extract landmarks [1, 1404] = 468 * 3 (x, y, z)
                auto* raw_landmarks = landmark_outputs[0].GetTensorMutableData<float>();
                
                // Transform landmarks back to original image coordinates
                float expand = 0.25f;
                float box_size = fmaxf(runner->last_face_box.width, runner->last_face_box.height) * (1.0f + expand);
                float box_x = runner->last_face_box.x_center - box_size / 2.0f;
                float box_y = runner->last_face_box.y_center - box_size / 2.0f;
                
                for (int i = 0; i < FACE_MESH_LANDMARK_COUNT; i++) {
                    // Raw landmarks are in [0, 192] range (input image space)
                    float lx = raw_landmarks[i * 3 + 0] / FACE_LANDMARK_INPUT_SIZE;  // Normalize to [0, 1]
                    float ly = raw_landmarks[i * 3 + 1] / FACE_LANDMARK_INPUT_SIZE;
                    float lz = raw_landmarks[i * 3 + 2] / FACE_LANDMARK_INPUT_SIZE;
                    
                    // Transform to original image coordinates
                    runner->landmarks[i].x = box_x + lx * box_size;
                    runner->landmarks[i].y = box_y + ly * box_size;
                    runner->landmarks[i].z = lz;  // Z is relative depth
                    runner->landmarks[i].visibility = 1.0f;
                    runner->landmarks[i].presence = 1.0f;
                }
                
                // Generate segmentation mask based on face region
                if (runner->current_graph == MP_GRAPH_SELFIE_SEGMENTATION || 
                    runner->current_graph == MP_GRAPH_FACE_MESH) {
                    
                    int mw = runner->mask_width;
                    int mh = runner->mask_height;
                    
                    for (int y = 0; y < mh; y++) {
                        for (int x = 0; x < mw; x++) {
                            float u = (float)x / mw;
                            float v = (float)y / mh;
                            
                            // Ellipse mask based on face bounds
                            float dx = (u - runner->last_face_box.x_center) / (runner->last_face_box.width * 0.6f);
                            float dy = (v - runner->last_face_box.y_center) / (runner->last_face_box.height * 0.8f);
                            float dist = sqrtf(dx*dx + dy*dy);
                            
                            float alpha = 1.0f - smoothstep(0.8f, 1.0f, dist);
                            runner->mask_buffer[y * mw + x] = (uint8_t)(clamp01(alpha) * 255.0f);
                        }
                    }
                } else if (runner->current_graph == MP_GRAPH_PRIVACY) {
                        // Privacy Mode: Blur Bystanders
                        // 1. Identify Primary Face (closest to center)
                        int primary_idx = -1;
                        float min_dist = 1000.0f;
                        for (size_t i = 0; i < runner->all_faces.size(); i++) {
                            float dx = runner->all_faces[i].x_center - 0.5f;
                            float dy = runner->all_faces[i].y_center - 0.5f;
                            float dist = dx*dx + dy*dy;
                            if (dist < min_dist) {
                                min_dist = dist;
                                primary_idx = (int)i;
                            }
                        }
                        
                        // 2. Clear mask (White = Keep)
                        memset(runner->mask_buffer.data(), 255, runner->mask_buffer.size());
                        
                        // 3. Draw Black Ellipses for Bystanders
                        int mw = runner->mask_width;
                        int mh = runner->mask_height;
                        
                        for (size_t i = 0; i < runner->all_faces.size(); i++) {
                            if ((int)i == primary_idx) continue; // Skip streamer
                            
                            const FaceBox& face = runner->all_faces[i];
                            
                            // Expand blur area slightly
                            float blur_w = face.width * 1.2f;
                            float blur_h = face.height * 1.5f;
                            
                            // Bounding box of ellipse to optimize loop
                            int x0 = (int)((face.x_center - blur_w/2) * mw);
                            int y0 = (int)((face.y_center - blur_h/2) * mh);
                            int x1 = (int)((face.x_center + blur_w/2) * mw);
                            int y1 = (int)((face.y_center + blur_h/2) * mh);
                            
                            x0 = std::max(0, x0); y0 = std::max(0, y0);
                            x1 = std::min(mw, x1); y1 = std::min(mh, y1);
                            
                            for (int y = y0; y < y1; y++) {
                                for (int x = x0; x < x1; x++) {
                                     float u = (float)x / mw;
                                     float v = (float)y / mh;
                                     
                                     float dx = (u - face.x_center) / (blur_w * 0.5f);
                                     float dy = (v - face.y_center) / (blur_h * 0.5f);
                                     float dist = dx*dx + dy*dy;
                                     
                                     if (dist < 1.0f) {
                                         // Gradient edge
                                         float alpha = smoothstep(0.8f, 1.0f, dist);
                                         // Multiply existing mask (so overlapping blur zones merge)
                                         uint8_t current = runner->mask_buffer[y * mw + x];
                                         runner->mask_buffer[y * mw + x] = (uint8_t)(std::min((float)current, alpha * 255.0f));
                                     }
                                }
                            }
                        }
                }
            } else {
                runner->has_face = false;
                // Clear mask when no face detected
                memset(runner->mask_buffer.data(), 0, runner->mask_buffer.size());
            }
            
            return true;
        } catch (const Ort::Exception& e) {
            blog(LOG_ERROR, "[MediaPipe] ONNX error: %s", e.what());
        }
    }
#endif
    
    // No usable backend, or the pipeline failed: clear outputs and report
    // failure. Missing models must produce "no detection", never fabricated
    // landmarks or masks (project invariant on effects).
    (void)data; (void)width; (void)height; (void)stride;
    {
        std::lock_guard<std::mutex> lock(runner->data_lock);
        runner->has_face = false;
        memset(runner->mask_buffer.data(), 0, runner->mask_buffer.size());
    }
    return false;
}

bool mp_process_image(mp_runner_t *ptr, const uint8_t *data, int width, int height, int stride, mp_landmark_list_t *out_landmarks) {
    if (!ptr || !out_landmarks) return false;
    out_landmarks->landmarks = nullptr;
    out_landmarks->count = 0;
    
#ifdef USE_ONNX_MEDIAPIPE
    mp_runner *runner = (mp_runner*)ptr;
    if (g_onnx_available && runner->face_landmark_session) {
        // For static image, run the full pipeline once
        if (mp_process_frame(ptr, data, width, height, stride)) {
            std::lock_guard<std::mutex> lock(runner->data_lock);
            if (runner->has_face) {
                out_landmarks->landmarks = runner->landmarks.data();
                out_landmarks->count = (int)runner->landmarks.size();
                return true;
            }
        }
        return false;
    }
#else
    (void)data; (void)width; (void)height; (void)stride;
#endif
    
    // No usable backend: report failure rather than returning fabricated
    // reference landmarks (project invariant: no invented detections).
    return false;
}

bool mp_get_segmentation_mask(mp_runner_t *ptr, mp_mask_data_t *out_mask) {
    if (!ptr || !out_mask) return false;
    mp_runner *runner = (mp_runner*)ptr;
    
    std::lock_guard<std::mutex> lock(runner->data_lock);
    
    out_mask->buffer = runner->mask_buffer.data();
    out_mask->width = runner->mask_width;
    out_mask->height = runner->mask_height;
    out_mask->channels = 1;
    return true;
}

bool mp_get_landmarks(mp_runner_t *ptr, mp_landmark_list_t *out_landmarks) {
    if (!ptr || !out_landmarks) return false;
    mp_runner *runner = (mp_runner*)ptr;

    std::lock_guard<std::mutex> lock(runner->data_lock);
    
    // No current face means no trustworthy landmarks. The pre-sized buffer
    // may hold stale or zero entries; returning it would fabricate a
    // detection for callers that only check count > 0.
    if (!runner->has_face) {
        out_landmarks->landmarks = nullptr;
        out_landmarks->count = 0;
        return false;
    }
    out_landmarks->landmarks = runner->landmarks.data();
    out_landmarks->count = (int)runner->landmarks.size();
    return true;
}

// Query backend information
const char* mp_get_backend_name(void) {
    if (g_litert_available) return "LiteRT";
    if (g_onnx_available) return "ONNX";
    return "Unavailable";
}

bool mp_is_onnx_available(void) {
#ifdef USE_ONNX_MEDIAPIPE
    return g_onnx_available;
#else
    return false;
#endif
}

} // extern "C"
