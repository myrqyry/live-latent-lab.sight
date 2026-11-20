/**
 * Latent Virtual Dolly - 2.5D Parallax Filter
 * 
 * Creates depth-based parallax effects for a "virtual camera" feel.
 * Uses depth estimation (neural or luminance) to displace pixels
 * based on virtual camera movement.
 * 
 * Effect: Foreground moves more than background, creating 3D illusion.
 */

#include <obs-module.h>
#include <graphics/graphics.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "core/depth_estimator.hpp"

#include "core/head_pose_estimator.hpp"
// facedetectcnn.h is in src/3rdparty/libfacedetection/
#include "3rdparty/libfacedetection/facedetectcnn.h"

#define S_DOLLY_X "dolly_x"
#define S_DOLLY_Y "dolly_y"
#define S_DOLLY_Z "dolly_z"
#define S_DOLLY_STRENGTH "dolly_strength"
#define S_DEPTH_SOURCE "depth_source"
#define S_DEPTH_MODEL_PATH "depth_model_path"
#define S_INPAINT_MODE "inpaint_mode"
#define S_HEAD_TRACKING "head_tracking"
#define S_POSE_MODEL_PATH "pose_model_path"

struct parallax_filter_data {
    obs_source_t* context;
    
    // Dolly parameters
    std::atomic<float> dolly_x = 0.0f;      // -1 to 1 (left/right)
    std::atomic<float> dolly_y = 0.0f;      // -1 to 1 (up/down)
    std::atomic<float> dolly_z = 0.0f;      // -1 to 1 (zoom in/out)
    std::atomic<float> strength = 50.0f;   // Parallax strength in pixels
    
    // Automation
    std::atomic<bool> head_tracking{false};
    float current_x = 0.0f;    // Smoothed values
    float current_y = 0.0f;
    
    // Depth source: 0 = Luminance, 1 = Neural (Depth Anything), 2 = MiDaS
    std::atomic<int> depth_source{0};
    std::atomic<int> inpaint_mode = 1;      // 0 = None, 1 = Neighbor average
    
    // Estimators
    std::unique_ptr<lll::DepthEstimator> depth_estimator;
    std::string depth_model_path;
    
    std::unique_ptr<lll::HeadPoseEstimator> pose_estimator;
    std::string pose_model_path;
    
    // Processing state
    std::vector<float> depth_buffer;
    std::vector<uint8_t> input_buffer;
    std::vector<uint8_t> output_buffer;
    
    struct InferenceJob {
        std::vector<uint8_t> rgba;
        uint32_t width = 0;
        uint32_t height = 0;
        bool head_tracking = false;
        bool neural_depth = false;
    };

    struct InferenceResult {
        std::vector<float> depth;
        uint32_t width = 0;
        uint32_t height = 0;
        float target_x = 0.0f;
        float target_y = 0.0f;
        bool pose_valid = false;
        bool depth_valid = false;
    };

    std::mutex inference_mutex;
    std::condition_variable inference_cv;
    std::optional<InferenceJob> pending_job;
    InferenceResult completed_result;
    bool completed_result_ready = false;
    bool stop_worker = false;
    std::thread inference_worker;
    std::mutex model_mutex;
    
    // OBS resources
    gs_texture_t* output_texture = nullptr;
    gs_texrender_t* texrender = nullptr;
    gs_stagesurf_t* stagesurf = nullptr;
    
    uint32_t width = 0;
    uint32_t height = 0;
    bool ready = false;
};

static const char* parallax_get_name(void* unused) {
    UNUSED_PARAMETER(unused);
    return "Latent Virtual Dolly";
}

static void parallax_inference_loop(parallax_filter_data* filter)
{
    std::vector<uint8_t> face_buffer(0x20000);

    for (;;) {
        parallax_filter_data::InferenceJob job;
        {
            std::unique_lock lock(filter->inference_mutex);
            filter->inference_cv.wait(lock, [filter] {
                return filter->stop_worker || filter->pending_job.has_value();
            });
            if (filter->stop_worker)
                return;
            job = std::move(*filter->pending_job);
            filter->pending_job.reset();
        }

        parallax_filter_data::InferenceResult result;
        result.width = job.width;
        result.height = job.height;

        std::vector<uint8_t> rgb(job.width * job.height * 3);
        for (size_t i = 0; i < job.width * job.height; ++i) {
            rgb[i * 3 + 0] = job.rgba[i * 4 + 0];
            rgb[i * 3 + 1] = job.rgba[i * 4 + 1];
            rgb[i * 3 + 2] = job.rgba[i * 4 + 2];
        }

        std::scoped_lock model_lock(filter->model_mutex);

        if (job.head_tracking && filter->pose_estimator &&
            filter->pose_estimator->is_ready()) {
            int* detections = facedetect_cnn(
                face_buffer.data(), rgb.data(), job.width, job.height,
                job.width * 3);
            if (detections && *detections > 0) {
                short* face = reinterpret_cast<short *>(detections + 1);
                const int x = face[1];
                const int y = face[2];
                const int w = face[3];
                const int h = face[4];
                if (x >= 0 && y >= 0 && w > 32 && h > 32 &&
                    x + w <= static_cast<int>(job.width) &&
                    y + h <= static_cast<int>(job.height)) {
                    std::vector<uint8_t> crop(w * h * 3);
                    for (int fy = 0; fy < h; ++fy) {
                        const uint8_t *src = rgb.data() +
                            (static_cast<size_t>(y + fy) * job.width + x) * 3;
                        std::memcpy(crop.data() + static_cast<size_t>(fy) * w * 3,
                                    src, static_cast<size_t>(w) * 3);
                    }
                    const auto pose = filter->pose_estimator->estimate(
                        crop.data(), w, h);
                    result.target_x = std::clamp(pose.yaw / 45.0f, -1.0f, 1.0f);
                    result.target_y = std::clamp(pose.pitch / 30.0f, -1.0f, 1.0f);
                    result.pose_valid = true;
                }
            }
        }

        if (job.neural_depth && filter->depth_estimator &&
            filter->depth_estimator->is_ready()) {
            result.depth.resize(static_cast<size_t>(job.width) * job.height);
            result.depth_valid = filter->depth_estimator->estimate(
                rgb.data(), job.width, job.height, result.depth.data());
            if (!result.depth_valid)
                result.depth.clear();
        }

        {
            std::scoped_lock lock(filter->inference_mutex);
            filter->completed_result = std::move(result);
            filter->completed_result_ready = true;
        }
    }
}

static void parallax_destroy(void* data) {
    parallax_filter_data* filter = static_cast<parallax_filter_data*>(data);
    if (!filter) return;
    
    {
        std::scoped_lock lock(filter->inference_mutex);
        filter->stop_worker = true;
    }
    filter->inference_cv.notify_one();
    if (filter->inference_worker.joinable())
        filter->inference_worker.join();

    obs_enter_graphics();
    if (filter->output_texture) gs_texture_destroy(filter->output_texture);
    if (filter->texrender) gs_texrender_destroy(filter->texrender);
    if (filter->stagesurf) gs_stagesurface_destroy(filter->stagesurf);
    obs_leave_graphics();
    
    delete filter;
}

static void* parallax_create(obs_data_t* settings, obs_source_t* context) {
    parallax_filter_data* filter = new parallax_filter_data();
    filter->context = context;
    blog(LOG_INFO, "DEBUG_ACTIVE_PARALLAX_7KQ29");
    
    filter->dolly_x.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_X)),
                          std::memory_order_release);
    filter->dolly_y.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_Y)),
                          std::memory_order_release);
    filter->dolly_z.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_Z)),
                          std::memory_order_release);
    filter->strength.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_STRENGTH)),
                           std::memory_order_release);
    filter->depth_source.store(static_cast<int>(obs_data_get_int(settings, S_DEPTH_SOURCE)),
                               std::memory_order_release);
    filter->inpaint_mode.store(static_cast<int>(obs_data_get_int(settings, S_INPAINT_MODE)),
                               std::memory_order_release);
    filter->head_tracking.store(obs_data_get_bool(settings, S_HEAD_TRACKING),
                                std::memory_order_release);
    
    const char* model_path = obs_data_get_string(settings, S_DEPTH_MODEL_PATH);
    if (model_path && *model_path) {
        filter->depth_model_path = model_path;
        filter->depth_estimator = std::make_unique<lll::DepthEstimator>();
        filter->depth_estimator->load(model_path,
                                      lll::ModelType::DepthAnythingV2, true);
    }
    
    const char* pose_path = obs_data_get_string(settings, S_POSE_MODEL_PATH);
    if (pose_path && *pose_path) {
        filter->pose_model_path = pose_path;
        filter->pose_estimator = std::make_unique<lll::HeadPoseEstimator>();
        filter->pose_estimator->load(pose_path, true);
    }
    
    filter->inference_worker = std::thread(parallax_inference_loop, filter);
    
    return filter;
}

static void parallax_update(void* data, obs_data_t* settings) {
    parallax_filter_data* filter = static_cast<parallax_filter_data*>(data);
    
    filter->dolly_x.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_X)),
                          std::memory_order_release);
    filter->dolly_y.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_Y)),
                          std::memory_order_release);
    filter->dolly_z.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_Z)),
                          std::memory_order_release);
    filter->strength.store(static_cast<float>(obs_data_get_double(settings, S_DOLLY_STRENGTH)),
                           std::memory_order_release);
    filter->depth_source.store(static_cast<int>(obs_data_get_int(settings, S_DEPTH_SOURCE)),
                               std::memory_order_release);
    filter->inpaint_mode.store(static_cast<int>(obs_data_get_int(settings, S_INPAINT_MODE)),
                               std::memory_order_release);
    filter->head_tracking.store(obs_data_get_bool(settings, S_HEAD_TRACKING),
                                std::memory_order_release);
    
    const char* model_path = obs_data_get_string(settings, S_DEPTH_MODEL_PATH);
    {
        std::scoped_lock lock(filter->model_mutex);
        if (model_path && *model_path && filter->depth_model_path != model_path) {
            filter->depth_model_path = model_path;
            if (!filter->depth_estimator)
                filter->depth_estimator = std::make_unique<lll::DepthEstimator>();
            filter->depth_estimator->load(model_path,
                                          lll::ModelType::DepthAnythingV2, true);
        }
    }
    
    const char* pose_path = obs_data_get_string(settings, S_POSE_MODEL_PATH);
    {
        std::scoped_lock lock(filter->model_mutex);
        if (pose_path && *pose_path && filter->pose_model_path != pose_path) {
            filter->pose_model_path = pose_path;
            if (!filter->pose_estimator)
                filter->pose_estimator = std::make_unique<lll::HeadPoseEstimator>();
            filter->pose_estimator->load(pose_path, true);
        }
    }
}

static void parallax_defaults(obs_data_t* settings) {
    obs_data_set_default_double(settings, S_DOLLY_X, 0.0);
    obs_data_set_default_double(settings, S_DOLLY_Y, 0.0);
    obs_data_set_default_double(settings, S_DOLLY_Z, 0.0);
    obs_data_set_default_double(settings, S_DOLLY_STRENGTH, 50.0);
    obs_data_set_default_int(settings, S_DEPTH_SOURCE, 0);
    obs_data_set_default_int(settings, S_INPAINT_MODE, 1);
    obs_data_set_default_bool(settings, S_HEAD_TRACKING, false);
}

static obs_properties_t* parallax_properties(void* data) {
    UNUSED_PARAMETER(data);
    obs_properties_t* props = obs_properties_create();
    
    obs_properties_add_float_slider(props, S_DOLLY_X, "Dolly X (Left/Right)", -1.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, S_DOLLY_Y, "Dolly Y (Up/Down)", -1.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, S_DOLLY_Z, "Dolly Z (Zoom)", -1.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, S_DOLLY_STRENGTH, "Parallax Strength", 0.0, 200.0, 1.0);
    
    obs_property_t* depth_src = obs_properties_add_list(props, S_DEPTH_SOURCE, "Depth Source",
                                                         OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(depth_src, "Luminance (Fast)", 0);
    obs_property_list_add_int(depth_src, "Neural Depth (Accurate)", 1);
    
     obs_properties_add_path(props, S_DEPTH_MODEL_PATH, "Depth Model (LiteRT)",
                            OBS_PATH_FILE, "LiteRT Models (*.tflite)", nullptr);
                           
    obs_properties_add_bool(props, S_HEAD_TRACKING, "Auto Parallax (Head Tracking)");
     obs_properties_add_path(props, S_POSE_MODEL_PATH, "Pose Model (LiteRT)",
                            OBS_PATH_FILE, "LiteRT Models (*.tflite)", nullptr);
    
    obs_property_t* inpaint = obs_properties_add_list(props, S_INPAINT_MODE, "Hole Filling",
                                                       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(inpaint, "None", 0);
    obs_property_list_add_int(inpaint, "Neighbor Average", 1);
    
    return props;
}

// Estimate depth from luminance (fast approximation)
static void estimate_depth_luminance(const uint8_t* rgba, float* depth, int w, int h) {
    for (int i = 0; i < w * h; ++i) {
        // Assume brighter = closer (inverse depth)
        float luma = 0.299f * rgba[i * 4 + 0] +
                    0.587f * rgba[i * 4 + 1] +
                    0.114f * rgba[i * 4 + 2];
        depth[i] = luma / 255.0f;
    }
}

// Apply parallax displacement
static void apply_parallax(const uint8_t* src, uint8_t* dst, 
                          const float* depth,
                          int w, int h,
                          float dx, float dy, float dz, float strength,
                          int inpaint_mode) {
    // Clear output to transparent
    std::memset(dst, 0, w * h * 4);
    
    // Track which pixels are filled (for inpainting)
    std::vector<bool> filled(w * h, false);
    
    // Forward warp: push pixels based on depth
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            float d = depth[idx];
            
            // Displacement based on depth and dolly
            // Higher depth (closer) = more displacement
            float disp_x = dx * d * strength;
            float disp_y = dy * d * strength;
            float scale = 1.0f + dz * d * 0.2f;  // Zoom effect
            
            // Target position after displacement
            int tx = static_cast<int>(x + disp_x);
            int ty = static_cast<int>(y + disp_y);
            
            // Apply zoom around center
            if (std::abs(dz) > 0.01f) {
                float cx = w / 2.0f;
                float cy = h / 2.0f;
                tx = static_cast<int>(cx + (tx - cx) * scale);
                ty = static_cast<int>(cy + (ty - cy) * scale);
            }
            
            // Bounds check
            if (tx >= 0 && tx < w && ty >= 0 && ty < h) {
                int tidx = ty * w + tx;
                // Z-buffer style: only write if this pixel is closer
                if (!filled[tidx] || depth[idx] > depth[tidx]) {
                    dst[tidx * 4 + 0] = src[idx * 4 + 0];
                    dst[tidx * 4 + 1] = src[idx * 4 + 1];
                    dst[tidx * 4 + 2] = src[idx * 4 + 2];
                    dst[tidx * 4 + 3] = 255;
                    filled[tidx] = true;
                }
            }
        }
    }
    
    // Inpainting for holes
    if (inpaint_mode == 1) {
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int idx = y * w + x;
                if (!filled[idx]) {
                    // Average neighbors
                    int count = 0;
                    int r = 0, g = 0, b = 0;
                    
                    for (int dy_o = -1; dy_o <= 1; ++dy_o) {
                        for (int dx_o = -1; dx_o <= 1; ++dx_o) {
                            if (dx_o == 0 && dy_o == 0) continue;
                            int nidx = (y + dy_o) * w + (x + dx_o);
                            if (filled[nidx]) {
                                r += dst[nidx * 4 + 0];
                                g += dst[nidx * 4 + 1];
                                b += dst[nidx * 4 + 2];
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0) {
                        dst[idx * 4 + 0] = r / count;
                        dst[idx * 4 + 1] = g / count;
                        dst[idx * 4 + 2] = b / count;
                        dst[idx * 4 + 3] = 255;
                    }
                }
            }
        }
    }
}

static void parallax_video_render(void* data, gs_effect_t* effect) {
    parallax_filter_data* filter = static_cast<parallax_filter_data*>(data);
    if (!filter) {
        return;
    }
    
    obs_source_t* target = obs_filter_get_target(filter->context);
    obs_source_t* parent = obs_filter_get_parent(filter->context);
    
    if (!target || !parent) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    uint32_t cx = obs_source_get_base_width(target);
    uint32_t cy = obs_source_get_base_height(target);
    
    if (cx == 0 || cy == 0) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Auto-animate dolly based on head pose
    const float configured_dolly_x = filter->dolly_x.load(std::memory_order_acquire);
    const float configured_dolly_y = filter->dolly_y.load(std::memory_order_acquire);
    const float configured_dolly_z = filter->dolly_z.load(std::memory_order_acquire);
    const float configured_strength = filter->strength.load(std::memory_order_acquire);
    const int configured_inpaint_mode = filter->inpaint_mode.load(std::memory_order_acquire);
    float active_dolly_x = configured_dolly_x;
    float active_dolly_y = configured_dolly_y;
    
    // Resize buffers if needed
    if (filter->width != cx || filter->height != cy) {
        filter->width = cx;
        filter->height = cy;
        filter->depth_buffer.resize(cx * cy);
        filter->input_buffer.resize(cx * cy * 4);
        filter->output_buffer.resize(cx * cy * 4);
        
        if (filter->stagesurf) {
            gs_stagesurface_destroy(filter->stagesurf);
            filter->stagesurf = nullptr;
        }
        if (filter->texrender) {
            gs_texrender_destroy(filter->texrender);
            filter->texrender = nullptr;
        }
        if (filter->output_texture) {
            gs_texture_destroy(filter->output_texture);
            filter->output_texture = nullptr;
        }
    }
    
    // Create resources if needed
    if (!filter->texrender) {
        filter->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    }
    if (!filter->stagesurf) {
        filter->stagesurf = gs_stagesurface_create(cx, cy, GS_RGBA);
    }
    
    // Capture input
    if (gs_texrender_begin(filter->texrender, cx, cy)) {
        gs_ortho(0.0f, (float)cx, 0.0f, (float)cy, -100.0f, 100.0f);
        obs_source_video_render(target);
        gs_texrender_end(filter->texrender);
    }
    
    gs_texture_t* src_tex = gs_texrender_get_texture(filter->texrender);
    gs_stage_texture(filter->stagesurf, src_tex);
    
    uint8_t* ptr;
    uint32_t linesize;
    if (gs_stagesurface_map(filter->stagesurf, &ptr, &linesize)) {
        // Copy to input buffer (handle linesize)
        if (linesize == cx * 4) {
            std::memcpy(filter->input_buffer.data(), ptr, cx * cy * 4);
        } else {
            for (uint32_t y = 0; y < cy; ++y) {
                std::memcpy(filter->input_buffer.data() + y * cx * 4, 
                           ptr + y * linesize, cx * 4);
            }
        }
        gs_stagesurface_unmap(filter->stagesurf);
        
        parallax_filter_data::InferenceJob job;
        job.rgba = filter->input_buffer;
        job.width = cx;
        job.height = cy;
         job.head_tracking = filter->head_tracking.load(std::memory_order_acquire);
         job.neural_depth = filter->depth_source.load(std::memory_order_acquire) == 1;
        {
            std::scoped_lock lock(filter->inference_mutex);
            filter->pending_job = std::move(job);
        }
        filter->inference_cv.notify_one();

        parallax_filter_data::InferenceResult result;
        {
            std::scoped_lock lock(filter->inference_mutex);
            if (filter->completed_result_ready &&
                filter->completed_result.width == cx &&
                filter->completed_result.height == cy) {
                result = std::move(filter->completed_result);
                filter->completed_result_ready = false;
            }
        }

        if (result.pose_valid) {
            filter->current_x = filter->current_x * 0.9f +
                                result.target_x * 0.1f;
            filter->current_y = filter->current_y * 0.9f +
                                result.target_y * 0.1f;
            active_dolly_x = filter->current_x;
            active_dolly_y = filter->current_y;
        }
        
        // Skip if no dolly movement (and not tracking)
        if (!filter->head_tracking.load(std::memory_order_acquire) &&
            std::abs(active_dolly_x) < 0.01f && 
            std::abs(active_dolly_y) < 0.01f && 
             std::abs(configured_dolly_z) < 0.01f) {
            // Passthrough original input (efficiency)
            // But we already captured it to input_buffer. Need to output it.
            // Actually, best to just render the child source directly in the effect pass if we skip?
            // "obs_source_skip_video_filter" does that.
            obs_source_skip_video_filter(filter->context);
            return;
        }

        // Estimate depth
        if (filter->depth_source.load(std::memory_order_acquire) == 1 &&
            result.depth_valid &&
            result.depth.size() == static_cast<size_t>(cx) * cy) {
            filter->depth_buffer = std::move(result.depth);
        } else {
            // Luminance depth
            estimate_depth_luminance(filter->input_buffer.data(), 
                                    filter->depth_buffer.data(), cx, cy);
        }
        
        // Apply parallax
        apply_parallax(filter->input_buffer.data(), filter->output_buffer.data(),
                      filter->depth_buffer.data(),
                      cx, cy,
                       active_dolly_x, active_dolly_y, configured_dolly_z,
                       configured_strength, configured_inpaint_mode);
        
        // Update output texture
        if (filter->output_texture) {
            gs_texture_destroy(filter->output_texture);
        }
        const uint8_t* data_ptr = filter->output_buffer.data();
        filter->output_texture = gs_texture_create(cx, cy, GS_RGBA, 1,
                                                   &data_ptr,
                                                   GS_DYNAMIC);
    }
    
    // Render output
    if (filter->output_texture) {
        gs_effect_t *draw_effect = effect ? effect : obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if (!draw_effect) {
            obs_source_skip_video_filter(filter->context);
            return;
        }

        gs_effect_set_texture(gs_effect_get_param_by_name(draw_effect, "image"),
                              filter->output_texture);
        while (gs_effect_loop(draw_effect, "Draw")) {
            gs_draw_sprite(filter->output_texture, 0, cx, cy);
        }
    } else {
        obs_source_skip_video_filter(filter->context);
    }
}

struct obs_source_info parallax_filter_info = {
    .id = "parallax_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = parallax_get_name,
    .create = parallax_create,
    .destroy = parallax_destroy,
    .get_defaults = parallax_defaults,
    .get_properties = parallax_properties,
    .update = parallax_update,
    .video_render = parallax_video_render,
};
