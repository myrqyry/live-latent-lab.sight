#include <obs-module.h>
#include <graphics/graphics.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <util/platform.h>
#include <util/dstr.h>
#include <QDesktopServices>
#include <QUrl>

/**
 * lll.sight: AI Style Transfer Filter for OBS Studio
 * 
 * Architecture Overview:
 * - Main Thread: Handles OBS rendering pipeline (video_render, video_tick)
 * - Worker Thread: Runs AI inference asynchronously (ai_worker_loop)
 * - Synchronization: Uses mutexes and condition variables for thread-safe communication
 * 
 * Performance Optimizations:
 * - Circular buffer frame cache (pre-allocated, zero runtime allocations)
 * - Thread-safe fixed-size status buffer (lock-free reads via atomic operations)
 * - Buffer size validation prevents memory corruption
 * - Exception handling ensures worker thread never terminates unexpectedly
 * 
 * Thread Safety:
 * - input_mutex: Protects input frame data and pending frame flags
 * - output_mutex: Protects output texture and processed frame data
 * - status_mutex: Protects status message buffer
 */

extern "C" {
    #include "stable-diffusion.h"
}
#include "common/image-utils.hpp"
#include "pose-renderer.hpp"
#include "common/image-utils.hpp"
#include "pose-renderer.hpp"
#include "core/latent-engine.hpp"
#include "core/scene_manager.hpp"
#include "core/depth_estimator.hpp"
#include "core/scene_manager.hpp"
#include "core/depth_estimator.hpp"
#include "core/litert_keypoint_estimator.hpp"
#include "core/litert_model_catalog.hpp"
#include "core/litert_model_manager.hpp"
#include "core/nsfw_detector.hpp"
#include "core/fast_style_transfer.hpp"
#include "common/mesh_renderer.hpp"
#include "mediapipe/mp_wrapper.h"

// Dock panel integration (defined in ai-style-dock.cpp when ENABLE_DOCK_PANEL is defined)
#ifdef ENABLE_DOCK_PANEL
extern "C" {
    void ai_dock_update_preview(const uint8_t* data, int width, int height);
    void ai_dock_update_stats(float fps, int dropped, float vram_mb);
    void ai_dock_update_progress(float progress);
    void ai_dock_set_status(const char* status);
    void ai_dock_connect_source(obs_source_t *source);
}
#else
// No-op stubs when dock panel is disabled
static inline void ai_dock_update_preview(const uint8_t*, int, int) {}
static inline void ai_dock_update_stats(float, int, float) {}
static inline void ai_dock_update_progress(float) {}
static inline void ai_dock_set_status(const char*) {}
static inline void ai_dock_connect_source(obs_source_t*) {}
#endif

#define S_MODEL_PATH "model_path"
#define S_VAE_PATH "vae_path"
#define S_PROMPT "prompt"
#define S_NEGATIVE_PROMPT "negative_prompt"
#define S_STRENGTH "strength"
#define S_STEPS "steps"
#define S_SCALE "processing_scale"
#define S_FRAMERATE "frame_rate"
#define S_STATUS "status"
#define S_CACHE_FRAMES "cache_frames"
#define S_SAMPLER "sampler_method"
#define S_SCHEDULER "scheduler"
#define S_CFG_SCALE "cfg_scale"
#define S_SEED "seed"
#define S_RANDOM_SEED "random_seed"
#define S_THREAD_COUNT "thread_count"
#define S_VAE_TILING "vae_tiling"
#define S_TAESD_PATH "taesd_path"
#define S_PRESET "style_preset"
#define S_LORA_PATH "lora_path"
#define S_LORA_STRENGTH "lora_strength"
#define S_FLASH_ATTN "flash_attn"
#define S_EASYCACHE "easycache_enabled"
#define S_SIMILARITY_THRESHOLD "similarity_threshold"
#define S_MOTION_ADAPTIVE "motion_adaptive_steps"
#define S_TEMPORAL_SMOOTHING "temporal_smoothing"
#define S_CONTROLNET_PATH "controlnet_path"
#define S_CONTROLNET_STRENGTH "controlnet_strength"
#define S_CONTROLNET_MODE "controlnet_mode"
#define S_VAE_ON_CPU "vae_on_cpu"
#define S_TAESD_PREVIEW "taesd_preview_only"
#define S_VAE_CONV_DIRECT "vae_conv_direct"
#define S_PURE_AI "pure_ai_mode"
#define S_TRANSITION_DURATION "pure_ai_transition_duration"
#define S_MP_BACKEND "mp_backend_type"
#define S_MP_SHOW_MASK "mp_show_mask"
#define S_PRIVACY_BLUR "privacy_blur_strength"

#define S_PRIVACY_BLUR "privacy_blur_strength"
#define S_NSFW_CHECK "nsfw_check"
#define S_FAST_STYLE_ENABLED "fast_style_enabled"
#define S_FAST_STYLE_MODEL "fast_style_model"

#define PRESET_LCM_NAME "LCM / Turbo (Fast)"
#define PRESET_ZIMAGE_NAME "Z-Image Turbo"
#define STATUS_UPDATE_INTERVAL 300 // ~5 seconds at 60fps

// Recommended models for documentation
// Z-Image-Turbo: https://huggingface.co/leejet/Z-Image-Turbo-GGUF
// LCM-LoRA SD1.5: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
// LCM-LoRA SDXL: https://huggingface.co/latent-consistency/lcm-lora-sdxl
// ControlNet Canny: https://huggingface.co/lllyasviel/sd-controlnet-canny
// ControlNet Depth: https://huggingface.co/lllyasviel/sd-controlnet-depth

struct StylePreset {
    const char* name;
    const char* prompt;
    const char* negative_prompt;
    float strength;
    float cfg;
    int steps;
    int sampler;
    int scheduler;
};

struct ControlNetParams {
    float canny_high_threshold = 100.0f;
    float canny_low_threshold = 50.0f;
    float silhouette_threshold = 30.0f;
    float soft_edge_sensitivity = 4.0f;
    bool auto_threshold = true;
};

struct style_transfer_data {
    obs_source_t *context;
    
    // Configuration - Basic
    std::string current_model_path;
    std::string vae_path;
    std::string prompt;
    std::string negative_prompt;
    float strength = 0.75f;
    int steps = 20;
    float scale_factor = 0.5f;
    float frame_interval = 0.5f;
    bool cache_frames = true;
    
    // Configuration - Advanced (LCM/Turbo support)
    int sampler_method = EULER_A_SAMPLE_METHOD;
    int scheduler = DISCRETE_SCHEDULER;
    float cfg_scale = 7.0f;
    int64_t seed = -1;  // -1 = random
    bool random_seed = true;
    int thread_count = 0;  // 0 = auto
    bool vae_tiling = false;
    std::string taesd_path;
    std::string current_taesd_path;
    std::string lora_path;
    float lora_strength = 1.0f;
    bool is_lcm_lora = false; // Auto-detected for optimal settings
    
    // Performance Features
    bool flash_attn = true;
    bool easycache_enabled = true;
    float similarity_threshold = 1.0f; // MSE threshold
    
    // ComfyUI-Inspired Optimizations
    float easycache_reuse_threshold = 0.5f;  // Higher = more aggressive caching
    float easycache_start_percent = 0.0f;    // When to start caching (0-1)
    float easycache_end_percent = 1.0f;      // When to stop caching (0-1)
    bool rcfg_enabled = true;                // Residual CFG - use lower CFG for early steps
    bool adaptive_steps = true;              // Reduce steps for similar frames
    int min_adaptive_steps = 2;              // Minimum steps when frame is very similar
    
    // VAE/Latent Space Optimizations
    bool vae_on_cpu = false;        // Offload VAE to CPU to save VRAM
    bool taesd_preview_only = false; // Use TAESD for preview only (faster but lower quality)
    bool vae_conv_direct = true;    // Use direct convolution for VAE (faster)
    bool pure_ai_mode = false;      // If true, hide original input and show only AI output
    bool show_preview = false;      // Show intermediate diffusion steps
    
    // State
    std::vector<uint8_t> last_input_small;
    float last_diff = 0.0f;
    
    // V2 Improvements
    bool motion_adaptive = true;
    float temporal_smoothing = 0.0f; // 0.0 = off, 1.0 = full (previous frame)
    std::vector<uint8_t> last_output_full;
    
    // ControlNet Support
    std::string controlnet_path;
    std::string current_controlnet_path;
    float controlnet_strength = 0.8f;
    int controlnet_mode = 0; // 0=Off, 1=Canny, 2=Luminance, 3=Tile, 4=Silhouette, 5=Scribble, 6=MotionPose, 7=NeuralDepth
    std::vector<uint8_t> control_image_data;
    ControlNetParams cn_params;
    
    // Depth Anything v2 estimator
    std::unique_ptr<lll::DepthEstimator> depth_estimator;
    std::string depth_model_path;

    // LiteRT RTMPose-17 is loaded lazily by the AI worker for ControlNet pose.
    std::unique_ptr<lll::LiteRtKeypointEstimator> pose_estimator;
    std::atomic<bool> pose_model_reset_requested{false};
    bool pose_model_attempted = false;
    
    // Safety Filter
    bool nsfw_check = false;
    std::unique_ptr<lll::NSFWDetector> nsfw_detector;
    
    // Fast Style Transfer
    bool fast_style_enabled = false;
    std::string fast_style_model_path;
    std::string current_fast_style_path;
    std::unique_ptr<lll::FastStyleTransfer> fast_style_transfer;

    // MediaPipe Integration (Internal use for ControlNet)
    int mp_backend_type = 0; // 0=None, see mp_graph_type_t
    bool mp_show_mask = false;
    std::mutex mp_mutex;
    
    // Threading
    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};
    std::atomic<bool> model_ready{false};
    std::atomic<bool> processing{false};
    
    // Synchronization
    std::mutex input_mutex;
    std::mutex output_mutex;
    std::mutex engine_load_mutex;  // Protects filter state during model load
    std::condition_variable frame_cv;
    
    // Frame buffers
    std::vector<uint8_t> input_full_res;
    std::vector<uint8_t> input_small;
    std::vector<uint8_t> output_small;
    std::vector<uint8_t> output_full_res;
    
    // Scene (EnTT AR)
    std::unique_ptr<lll::SceneManager> scene_manager;
    std::mutex scene_mutex;
    
    // Frame caching - circular buffer (pre-allocated to avoid runtime allocations)
    struct FrameCache {
        std::vector<std::vector<uint8_t>> frames;
        size_t write_index = 0;
        size_t max_size = 3;
        bool initialized = false;
        
        bool init(size_t frame_size, size_t cache_size) {
            if (frame_size == 0 || cache_size == 0) return false;

            // Limit maximum memory usage (512MB)
            const size_t MAX_CACHE_MEMORY = 512ULL * 1024 * 1024;
            double total_mem_mb = (double)frame_size * cache_size / (1024.0 * 1024.0);

            if (total_mem_mb > 512.0) {
                size_t new_size = MAX_CACHE_MEMORY / frame_size;
                if (new_size < 1) new_size = 1;
                blog(LOG_WARNING, "[FrameCache] Requested %.2f MB exceeds limit. Reduced cache from %zu to %zu frames", 
                     total_mem_mb, cache_size, new_size);
                cache_size = new_size;
            }

            try {
                max_size = cache_size;
                frames.resize(max_size);
                for (auto& frame : frames) {
                    frame.reserve(frame_size);
                    frame.resize(frame_size);
                }
                write_index = 0;
                initialized = true;
                return true;
            } catch (const std::bad_alloc& e) {
                blog(LOG_ERROR, "[FrameCache] Allocation failed: %s", e.what());
                frames.clear();
                initialized = false;
                return false;
            }
        }
        
        void push(const std::vector<uint8_t>& data) {
            if (data.empty()) return;
            
            if (!initialized || frames.empty() || frames[0].size() != data.size()) {
                if (!init(data.size(), max_size)) {
                    return; // Failed to init
                }
            }
            
            if (frames.empty()) return; // Safety check
            
            std::memcpy(frames[write_index].data(), data.data(), data.size());
            write_index = (write_index + 1) % max_size;
        }
        
        const std::vector<uint8_t>* get_latest() const {
            if (!initialized || frames.empty()) return nullptr;
            size_t idx = (write_index == 0) ? (max_size - 1) : (write_index - 1);
            return &frames[idx];
        }
    } frame_cache;
    
    // Texture management (Output)
    gs_texture_t *output_texture = nullptr;
    gs_texture_t *prev_output_texture = nullptr; // For smooth transition
    float transition_timer = 0.0f;
    float transition_duration = 0.5f; // Seconds
    bool is_transitioning = false;
    
    uint32_t texture_width = 0;
    uint32_t texture_height = 0;
    std::atomic<bool> new_frame_ready{false};

    // Texture management (Input Capture)
    gs_texrender_t *capture_texrender = nullptr;
    gs_stagesurf_t *capture_staging = nullptr;
    
    // Dimensions
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t process_w = 0;
    uint32_t process_h = 0;
    uint32_t last_process_w = 0;
    uint32_t last_process_h = 0;
    
    // Timing and Triggers
    float capture_elapsed = 0.0f;
    bool trigger_capture = false;
    bool has_pending_frame = false;
    int status_update_counter = 0;          // For throttled status updates
    bool initial_status_shown = false;       // Track if initial message shown
    
    // Model loading
    bool model_load_requested = false;
    std::string requested_model_path;
    std::string requested_vae_path;
    int requested_thread_count = 0;
    
    // Status feedback - thread-safe fixed-size buffer
    mutable std::mutex status_mutex;
    std::string status;
    std::atomic<int> frames_processed{0};
    uint64_t generation_attempts = 0;
    uint64_t successful_generations = 0;
    uint64_t failed_generations = 0;
    uint64_t skipped_frame_count = 0;
    uint64_t consecutive_skips = 0;
    uint64_t max_consecutive_skips = 0;
    double first_generation_ms = 0.0;
    double generation_ms_total = 0.0;
    
    // Hotkeys
    obs_hotkey_id hotkey_pure_ai = OBS_INVALID_HOTKEY_ID;
    obs_hotkey_id hotkey_trigger = OBS_INVALID_HOTKEY_ID;
    
    void set_status(const char* msg) {
        std::lock_guard<std::mutex> lock(status_mutex);
        status = msg;
        // Duplicate status to log so we can debug "UI not updating" issues
        if (std::string(msg) != "Active" && std::string(msg) != "Processing frame...") {
            blog(LOG_INFO, "[lll.sight] Status: %s", status.c_str());
        }
    }
    
    // Non-blocking status update for high-frequency callbacks
    void set_status_fast(const char* msg) {
        // Try-lock pattern to avoid blocking the inference thread if UI is reading status
        if (status_mutex.try_lock()) {
            status = msg;
            status_mutex.unlock();
        }
    }
    
    std::string get_status() const {
        std::lock_guard<std::mutex> lock(status_mutex);
        return status;
    }
    std::atomic<int> frames_dropped{0};
    std::chrono::steady_clock::time_point last_fps_update;
    std::atomic<float> current_fps{0.0f};
    
    // Status update throttling
    std::chrono::steady_clock::time_point last_status_update;
    float status_update_interval = 3.0f; // Seconds
};

// Helper function to validate model file format
// Helper to get default model paths
static const char* get_default_model_paths() {
    static std::string found_path;
    static std::vector<std::string> paths = {
        #ifdef _WIN32
        "C:/AI/models/stable-diffusion/",
        #elif __APPLE__
        "/Users/Shared/ai-models/",
        #else
        "/usr/share/obs-ai-models/",
        #endif
    };
    
    // Add standard OBS data paths
    char *data_path = obs_module_file("models");
    if (data_path) {
        paths.push_back(data_path);
        bfree(data_path);
    }
    
    // Scan for existing directory
    for (const auto& path : paths) {
        if (os_file_exists(path.c_str())) { // os_file_exists usually checks for file, but here valid for dir? 
             // Actually OBS API mostly checks files. We might want os_opendir check or just return first one.
             // For safety, let's just return a platform appropriate default if none found.
             return path.c_str();
        }
    }
    
    #ifdef _WIN32
    return "C:/AI/models/";
    #else
    return "/usr/share/obs-ai-models/";
    #endif
}

// Improved model validation
static bool is_valid_model_path(const char *path) {
    if (!path || !*path) return false;
    
    // Check file exists and is readable
    FILE *f = fopen(path, "rb");
    if (!f) {
        blog(LOG_WARNING, "[lll.sight] Cannot open model file: %s", path);
        return false;
    }
    
    // Validate file size (should be at least 1MB)
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (file_size < 1024 * 1024) {
        blog(LOG_WARNING, "[lll.sight] Model file too small (%ld bytes): %s", file_size, path);
        fclose(f);
        return false;
    }
    
    // Check magic numbers
    uint8_t header[8];
    if (fread(header, 1, 8, f) != 8) {
        fclose(f);
        return false;
    }
    fclose(f);
    
    // GGUF magic: "GGUF" (0x46475547)
    if (memcmp(header, "GGUF", 4) == 0) {
        // blog(LOG_INFO, "[lll.sight] Detected valid GGUF model");
        return true;
    }
    
    // SafeTensors/CKPT via extension as usage fallback (magic numbers vary)
    const char *ext = strrchr(path, '.');
    if (ext && (strcmp(ext, ".ckpt") == 0 || strcmp(ext, ".safetensors") == 0)) {
        return true;
    }
    
    // blog(LOG_ERROR, "[lll.sight] Unrecognized model format for: %s", path);
    return false;
}

// Progress callback
static void on_progress(int step, int steps, float time, void* data) {
    auto* filter = static_cast<style_transfer_data*>(data);
    
    // Throttling: Only update status every few steps or on last step
    static int update_counter = 0;
    bool should_update = (step == steps) || (step == 0) || (++update_counter % 5 == 0);
    
    if (should_update) {
        char status[64];
        snprintf(status, sizeof(status), "Step %d/%d (%.2fs)", step, steps, time);
        // Use non-blocking update to avoid stalling inference
        filter->set_status_fast(status);
        
        // Update dock panel progress
        if (steps > 0) {
            float progress = (float)step / (float)steps;
            ai_dock_update_progress(progress);
        }
    }
}

static void load_model(style_transfer_data *filter, const char *path, const char *vae_path, int thread_count) {
    filter->model_ready = false;
    
    filter->set_status("Loading Model...");
    
    // Updated to use new EngineParams alias and structure
    lll::EngineParams params;
    params.model_path = path;
    params.num_threads = thread_count; // Changed from n_threads
    params.use_gpu = true; // Use GPU by default
    // params.flash_attn = filter->flash_attn; // VisionEngine config might store this in backend-specific way or we add it to generic config
    // We added extra fields to EngineConfig in VisionEngine for this reason
    params.flash_attn = filter->flash_attn;
    params.vae_path = vae_path ? vae_path : "";
    params.taesd_path = filter->taesd_path;
    params.controlnet_path = filter->controlnet_path;
    params.lora_path = filter->lora_path;
    
    // Note: optimization params (easycache etc) are not in generic EngineConfig yet
    // We might need to pass them via generation params or extending config
    
    if (lll::LatentEngine::get().load_model(params)) {
        filter->current_model_path = path;
        filter->last_input_small.clear();
        filter->last_diff = 0.0f;
        filter->generation_attempts = 0;
        filter->successful_generations = 0;
        filter->failed_generations = 0;
        filter->skipped_frame_count = 0;
        filter->consecutive_skips = 0;
        filter->max_consecutive_skips = 0;
        filter->first_generation_ms = 0.0;
        filter->generation_ms_total = 0.0;
        // filter->thread_count = thread_count; // Not in struct anymore?
        filter->model_ready = true;
        filter->set_status("Model loaded");
    } else {
        filter->set_status("Failed to load model");
    }
}

static void apply_privacy_blur(uint8_t* rgb, int w, int h, const uint8_t* mask, int mw, int mh, float strength) {
    // Moved to privacy-filter.cpp
}

static bool run_fast_style(style_transfer_data *filter, const uint8_t* input, int w, int h, uint8_t* output) {
    if (!filter->fast_style_transfer) {
        filter->fast_style_transfer = std::make_unique<lll::FastStyleTransfer>();
    }
    
    if (filter->fast_style_model_path != filter->current_fast_style_path) {
        filter->set_status("Loading Fast Style Model...");
        if (filter->fast_style_transfer->load(filter->fast_style_model_path)) {
            filter->current_fast_style_path = filter->fast_style_model_path;
            filter->set_status("Fast Model Loaded");
        } else {
            filter->set_status("Failed to load Fast Model");
            return false;
        }
    }
    
    if (!filter->fast_style_transfer->is_ready()) return false;

    // Fast style runs on RGB packed, our I/O is RGBA.
    // Convert input RGBA -> RGB
    if (w <= 0 || h <= 0 ||
        static_cast<size_t>(w) > std::numeric_limits<size_t>::max() / h) {
        return false;
    }
    const size_t pixels = static_cast<size_t>(w) * h;
    if (pixels > std::numeric_limits<size_t>::max() / 3) return false;
    const size_t rgb_size = pixels * 3;
    std::vector<uint8_t> rgb_in(rgb_size);
    lll::convert_rgba_to_rgb(input, rgb_in.data(), pixels);
    
    std::vector<uint8_t> rgb_out(rgb_size);
    
    bool success = filter->fast_style_transfer->process(rgb_in.data(), w, h, rgb_out.data(), w, h);
    
    if (success) {
        // Convert RGB -> RGBA output
        lll::convert_rgb_to_rgba(rgb_out.data(), output, pixels);
        return true;
    }
    return false;
}

static bool checked_image_sizes(uint32_t width, uint32_t height, size_t channels,
                                size_t& pixels, size_t& bytes) {
    if (width == 0 || height == 0 ||
        static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height) {
        return false;
    }
    pixels = static_cast<size_t>(width) * height;
    if (channels == 0 || channels > std::numeric_limits<size_t>::max() / pixels) {
        return false;
    }
    bytes = pixels * channels;
    return true;
}

static bool run_style_transfer(style_transfer_data *filter) {
    // === Fast Style Transfer Bypass ===
    if (filter->fast_style_enabled && !filter->fast_style_model_path.empty()) {
        uint32_t target_w = filter->process_w;
        uint32_t target_h = filter->process_h;
        size_t target_pixels = 0;
        size_t target_rgba_size = 0;
        size_t full_pixels = 0;
        size_t full_rgba_size = 0;
        if (!checked_image_sizes(target_w, target_h, 4, target_pixels, target_rgba_size) ||
            !checked_image_sizes(filter->width, filter->height, 4, full_pixels, full_rgba_size)) {
            return false;
        }
        
        // Ensure output buffer size
        {
            std::lock_guard<std::mutex> out_lock(filter->output_mutex);
            if (filter->output_small.size() != target_rgba_size) {
                filter->output_small.resize(target_rgba_size);
            }
            if (filter->output_full_res.size() != full_rgba_size) {
                 filter->output_full_res.resize(full_rgba_size);
            }
        }
        
        if (run_fast_style(filter, filter->input_small.data(), target_w, target_h, filter->output_small.data())) {
             // Upscale to full res
             resize_bilinear(
                 filter->output_small.data(), filter->output_small.size(), target_w, target_h,
                 filter->output_full_res.data(), filter->output_full_res.size(), filter->width, filter->height
             );

             // Check NSFW even for fast style (safety first)
              if (filter->nsfw_check && filter->output_small.size() == target_rgba_size) {
                 if (!filter->nsfw_detector) {
                     filter->nsfw_detector = std::make_unique<lll::NSFWDetector>();
                     char* model_path = obs_module_file("models/open_nsfw.onnx");
                     if (model_path) {
                         filter->nsfw_detector->load(model_path);
                         bfree(model_path);
                     } else {
                         filter->nsfw_detector->load("/usr/share/obs-ai-models/open_nsfw.onnx");
                     }
                 }
                 if (filter->nsfw_detector->is_ready()) {
                      std::vector<uint8_t> rgb_check(target_pixels * 3);
                      lll::convert_rgba_to_rgb(filter->output_small.data(), rgb_check.data(), target_pixels);
                     float score = filter->nsfw_detector->predict(rgb_check.data(), target_w, target_h);
                     if (score > 0.8f) {
                         // Apply Stack Blur (Fast Gaussian)
                         stack_blur_rgba(filter->output_full_res.data(), filter->width, filter->height, 50); // Strong blur
                         filter->set_status("Safety Filter Triggered");
                     }
                 }
            }
             
             std::lock_guard<std::mutex> out_lock(filter->output_mutex);
             filter->new_frame_ready = true;
             return true;
        } else {
            // Drop back to SD if fails? Or just fail.
            return false;
        }
    }

    if (!lll::LatentEngine::get().is_ready()) return false;

    uint32_t target_w = filter->process_w;
    uint32_t target_h = filter->process_h;
    size_t target_pixels = 0;
    size_t target_rgb_size = 0;
    size_t target_rgba_size = 0;
    size_t full_pixels = 0;
    size_t full_rgba_size = 0;
    if (!checked_image_sizes(target_w, target_h, 3, target_pixels, target_rgb_size) ||
        !checked_image_sizes(target_w, target_h, 4, target_pixels, target_rgba_size) ||
        !checked_image_sizes(filter->width, filter->height, 4, full_pixels, full_rgba_size)) {
        return false;
    }

    // Convert RGBA -> RGB
    sd_image_t sd_input_img = {0};
    sd_input_img.width = target_w;
    sd_input_img.height = target_h;
    sd_input_img.channel = 3;
    sd_input_img.data = new uint8_t[target_rgb_size];

    // Optimized RGBA -> RGB conversion
    lll::convert_rgba_to_rgb(filter->input_small.data(), sd_input_img.data, target_pixels);
    
    // === MediaPipe Integration ===
    if (filter->mp_backend_type != 0 && lll::LatentEngine::get().get_mp_runner()) {
         // Run Preprocess (e.g. Privacy Mask, Face Mesh, etc.)
         lll::LatentEngine::get().run_mediapipe_preprocess(
             filter->input_small.data(), target_w, target_h, (mp_graph_type_t)filter->mp_backend_type);
             
         // Note: Privacy Blur removed from here.
         // This block now primarily ensures the MP graph is run so that *if* we add logic to 
         // extract Canny/Pose from MP, the data is ready.
    }
    
    // ControlNet: Generate control image based on mode
    sd_image_t control_img = {0};
    
    // RAII for control image buffer to prevent leaks
    std::unique_ptr<uint8_t[]> control_buffer;
    
    bool using_controlnet = (filter->controlnet_mode > 0 && !filter->controlnet_path.empty());
    
    if (using_controlnet) {
        control_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[target_rgb_size]);
        
        control_img.width = target_w;
        control_img.height = target_h;
        control_img.channel = 3;
        control_img.data = control_buffer.get();
        
        // Copy input image data for preprocessing
        std::memcpy(control_img.data, sd_input_img.data, target_rgb_size);
        
        if (filter->controlnet_mode == 1) {
            // Canny edge detection mode
            float high_thresh = filter->cn_params.canny_high_threshold;
            float low_thresh = filter->cn_params.canny_low_threshold;
            
            // Adaptive thresholding based on image histogram/intensity
            if (filter->cn_params.auto_threshold) {
                 double total_intensity = 0.0;
                 for (size_t i = 0; i < target_pixels; ++i) {
                     total_intensity += (control_img.data[i*3] + control_img.data[i*3+1] + control_img.data[i*3+2]) / 3.0;
                 }
                 float avg_intensity = (float)(total_intensity / target_pixels);
                 
                 high_thresh = 50.0f + (avg_intensity / 255.0f) * 100.0f;
                 low_thresh = high_thresh * 0.5f;
            }

            preprocess_canny(control_img, high_thresh, low_thresh, 255.0f, 255.0f, false);
            blog(LOG_DEBUG, "[lll.sight] Generated Canny control image");
        } else if (filter->controlnet_mode == 2) {
            // Luminance (pseudo-depth) mode - Fast approximation using grayscale
            // This is much faster than actual depth estimation and works well for ControlNet
             for (size_t i = 0; i < target_pixels; ++i) {
                // ITU-R BT.601 luma formula: Y = 0.299R + 0.587G + 0.114B
                uint8_t luma = (uint8_t)(
                    0.299f * control_img.data[i * 3 + 0] +
                    0.587f * control_img.data[i * 3 + 1] +
                    0.114f * control_img.data[i * 3 + 2]
                );
                control_img.data[i * 3 + 0] = luma;
                control_img.data[i * 3 + 1] = luma;
                control_img.data[i * 3 + 2] = luma;
            }
            blog(LOG_DEBUG, "[lll.sight] Generated Luminance control image");
        } else if (filter->controlnet_mode == 3) {
            // Pass-through mode - Use input image directly as control
            // Useful for lineart, scribble, or tile ControlNets
            blog(LOG_DEBUG, "[lll.sight] Using pass-through control image");
        } else if (filter->controlnet_mode == 4) {
            // Silhouette/High-Contrast mode - Creates edge-enhanced silhouette
            // Works well with lineart, scribble, and some pose ControlNets
            // Uses Sobel-like edge detection followed by thresholding
            
            // First pass: convert to grayscale
             std::vector<float> gray(target_pixels);
             for (size_t i = 0; i < target_pixels; ++i) {
                gray[i] = 0.299f * control_img.data[i * 3 + 0] +
                          0.587f * control_img.data[i * 3 + 1] +
                          0.114f * control_img.data[i * 3 + 2];
            }
            
            // Second pass: simple edge detection (gradient magnitude)
            for (size_t y = 1; y < target_h - 1; ++y) {
                for (size_t x = 1; x < target_w - 1; ++x) {
                    size_t idx = y * target_w + x;
                    
                    // Sobel-like gradient
                    float gx = gray[(y-1)*target_w + (x+1)] - gray[(y-1)*target_w + (x-1)]
                             + 2.0f * (gray[y*target_w + (x+1)] - gray[y*target_w + (x-1)])
                             + gray[(y+1)*target_w + (x+1)] - gray[(y+1)*target_w + (x-1)];
                    
                    float gy = gray[(y+1)*target_w + (x-1)] - gray[(y-1)*target_w + (x-1)]
                             + 2.0f * (gray[(y+1)*target_w + x] - gray[(y-1)*target_w + x])
                             + gray[(y+1)*target_w + (x+1)] - gray[(y-1)*target_w + (x+1)];
                    
                    float magnitude = std::sqrt(gx*gx + gy*gy);
                    uint8_t edge = (magnitude > 30.0f) ? 255 : 0;  // Threshold
                    
                    control_img.data[idx * 3 + 0] = edge;
                    control_img.data[idx * 3 + 1] = edge;
                    control_img.data[idx * 3 + 2] = edge;
                }
            }
            blog(LOG_DEBUG, "[lll.sight] Generated Silhouette control image");
        } else if (filter->controlnet_mode == 5) {
            // Soft Edge mode - Smoother edges for more artistic control
            // Good for scribble and sketch ControlNets
            
            // Convert to grayscale then blur slightly and find edges
            for (size_t y = 2; y < target_h - 2; ++y) {
                for (size_t x = 2; x < target_w - 2; ++x) {
                    size_t idx = y * target_w + x;
                    
                    // 5x5 Gaussian-weighted difference
                    float center = 0.0f;
                    float surround = 0.0f;
                    
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            size_t sidx = (y + dy) * target_w + (x + dx);
                            float luma = 0.299f * control_img.data[sidx * 3 + 0] +
                                        0.587f * control_img.data[sidx * 3 + 1] +
                                        0.114f * control_img.data[sidx * 3 + 2];
                            
                            if (std::abs(dx) <= 1 && std::abs(dy) <= 1) {
                                center += luma;
                            } else {
                                surround += luma;
                            }
                        }
                    }
                    
                    center /= 9.0f;  // 3x3 center
                    surround /= 16.0f;  // Outer ring
                    
                    float diff = std::abs(center - surround);
                    uint8_t edge = std::min(255, (int)(diff * 4.0f));
                    
                    control_img.data[idx * 3 + 0] = edge;
                    control_img.data[idx * 3 + 1] = edge;
                    control_img.data[idx * 3 + 2] = edge;
                }
            }
            blog(LOG_DEBUG, "[lll.sight] Generated Soft Edge control image");
        } else if (filter->controlnet_mode == 6) {
            // RTMPose-17 is top-down and expects a centered person crop. The
            // estimator runs here on the worker thread, never in video_render.
            if (filter->pose_model_reset_requested.exchange(false)) {
                filter->pose_estimator.reset();
                filter->pose_model_attempted = false;
            }

            if (!filter->pose_estimator && !filter->pose_model_attempted) {
                filter->pose_model_attempted = true;
                char *model_dir = obs_module_config_path("models/litert");
                if (model_dir) {
                    lll::LiteRtModelManager manager(model_dir);
                    std::optional<std::filesystem::path> model_path = manager.resolve("rtmpose-pose17");
                    std::string error;
                    if (!model_path) {
                        const auto *spec = lll::find_litert_model("rtmpose-pose17");
                        if (spec)
                            model_path = manager.install(*spec, &error);
                    }
                    if (model_path) {
                        auto estimator = std::make_unique<lll::LiteRtKeypointEstimator>();
                        if (estimator->load(*model_path, lll::LiteRtKeypointModel::Pose17))
                            filter->pose_estimator = std::move(estimator);
                    }
                    if (!filter->pose_estimator && !error.empty())
                        blog(LOG_WARNING, "[lll.sight] RTMPose model unavailable: %s", error.c_str());
                    bfree(model_dir);
                }
            }

            PoseKeypoints pose;
            std::vector<lll::LiteRtKeypoint> keypoints;
            if (filter->pose_estimator &&
                filter->pose_estimator->estimate(control_img.data, target_w, target_h, keypoints,
                                                lll::LiteRtDevice::Gpu)) {
                const float inv_width = target_w > 0 ? 1.0f / target_w : 0.0f;
                const float inv_height = target_h > 0 ? 1.0f / target_h : 0.0f;
                for (size_t i = 0; i < keypoints.size() && i < COCO_KEYPOINT_COUNT; ++i) {
                    const auto &keypoint = keypoints[i];
                    pose.set(static_cast<int>(i), keypoint.x * inv_width,
                             keypoint.y * inv_height, keypoint.confidence);
                }
            }

            render_openpose_skeleton(control_img.data, target_w, target_h, pose, true, 4, 6);
            blog(LOG_DEBUG, "[lll.sight] Generated RTMPose control image");
        } else if (filter->controlnet_mode == 7) {
            // Neural Depth mode - Use Depth Anything v2 for real depth estimation
            if (!filter->depth_estimator) {
                filter->depth_estimator = std::make_unique<lll::DepthEstimator>();
                // Try to load model from standard location
                char* model_dir = obs_module_file("models/depth_anything_v2_small.onnx");
                if (model_dir) {
                    filter->depth_estimator->load(model_dir, lll::ModelType::DepthAnythingV2, true);
                    bfree(model_dir);
                }
            }
            
            if (filter->depth_estimator && filter->depth_estimator->is_ready()) {
                // Estimate depth and convert to grayscale control image
                 std::vector<uint8_t> depth_gray(target_pixels);
                if (filter->depth_estimator->estimate_u8(control_img.data, target_w, target_h, depth_gray.data())) {
                    // Copy grayscale depth to RGB control image
                 for (size_t i = 0; i < target_pixels; ++i) {
                        control_img.data[i * 3 + 0] = depth_gray[i];
                        control_img.data[i * 3 + 1] = depth_gray[i];
                        control_img.data[i * 3 + 2] = depth_gray[i];
                    }
                    blog(LOG_DEBUG, "[lll.sight] Generated Neural Depth control image");
                } else {
                    // Fallback to luminance if depth estimation fails
                 for (size_t i = 0; i < target_pixels; ++i) {
                        uint8_t luma = (uint8_t)(
                            0.299f * control_img.data[i * 3 + 0] +
                            0.587f * control_img.data[i * 3 + 1] +
                            0.114f * control_img.data[i * 3 + 2]
                        );
                        control_img.data[i * 3 + 0] = luma;
                        control_img.data[i * 3 + 1] = luma;
                        control_img.data[i * 3 + 2] = luma;
                    }
                    blog(LOG_WARNING, "[lll.sight] Neural depth failed, using luminance fallback");
                }
            } else {
                // Fallback to luminance if model not loaded
                 for (size_t i = 0; i < target_pixels; ++i) {
                    uint8_t luma = (uint8_t)(
                        0.299f * control_img.data[i * 3 + 0] +
                        0.587f * control_img.data[i * 3 + 1] +
                        0.114f * control_img.data[i * 3 + 2]
                    );
                    control_img.data[i * 3 + 0] = luma;
                    control_img.data[i * 3 + 1] = luma;
                    control_img.data[i * 3 + 2] = luma;
                }
                blog(LOG_WARNING, "[lll.sight] Depth model not ready, using luminance fallback");
            }
        } else if (filter->controlnet_mode == 8) {
            // Face Mesh mode - Render MediaPipe landmarks as control image
            mp_runner_t* mp = lll::LatentEngine::get().get_mp_runner();
            if (mp) {
                // Ensure MediaPipe is running face mesh
                mp_graph_type_t current_graph = mp_get_current_graph(mp);
                if (current_graph != MP_GRAPH_FACE_MESH) {
                    mp_start_graph(mp, MP_GRAPH_FACE_MESH);
                }
                
                // Get landmarks
                mp_landmark_t landmarks[468];
                mp_landmark_list_t list;
                list.landmarks = landmarks;
                list.count = 468;
                bool success = mp_get_landmarks(mp, &list);
                int count = success ? list.count : 0;
                
                if (count > 0) {
                    // Render mesh as control image (mode 1 = wireframe)
                    lll::render_face_mesh_control(
                        control_img.data, target_w, target_h,
                        landmarks, count,
                        1,  // Wireframe mode
                        0, 0, 0,        // Black background
                        255, 255, 255   // White lines
                    );
                    blog(LOG_DEBUG, "[lll.sight] Generated Face Mesh control image with %d landmarks", count);
                } else {
                    // No face detected - use black control image
                     std::memset(control_img.data, 0, target_rgb_size);
                    blog(LOG_DEBUG, "[lll.sight] No face detected for mesh control");
                }
            } else {
                // MediaPipe not available - fallback to pass-through
                blog(LOG_WARNING, "[lll.sight] MediaPipe not available for Face Mesh mode");
            }
        }
    }

    lll::framework::VisionEngine::GenerationParams gen_params;
    gen_params.prompt = filter->prompt;
    gen_params.negative_prompt = filter->negative_prompt;
    gen_params.width = target_w;
    gen_params.height = target_h;
    
    // ComfyUI-Inspired Adaptive Steps
    int target_steps = filter->steps;
    if (filter->adaptive_steps && filter->last_diff >= 0.0f) {
        float similarity_ratio = filter->last_diff / std::max(filter->similarity_threshold, 0.1f);
        if (similarity_ratio < 0.3f) {
            target_steps = filter->min_adaptive_steps;
        } else if (filter->motion_adaptive) {
            float motion_factor = std::clamp(similarity_ratio, 0.5f, 1.5f);
            target_steps = (int)(filter->steps * motion_factor);
        }
        target_steps = std::clamp(target_steps, filter->min_adaptive_steps, filter->steps * 2);
    }
    
    gen_params.steps = target_steps;
    gen_params.cfg_scale = filter->cfg_scale;
    gen_params.strength = filter->strength;
    
    if (filter->random_seed || filter->seed < 0) {
        gen_params.seed = std::chrono::system_clock::now().time_since_epoch().count() % INT64_MAX;
    } else {
        gen_params.seed = filter->seed;
    }
    
    gen_params.strength = filter->strength;
    gen_params.sample_method = (sample_method_t)filter->sampler_method;
    gen_params.scheduler = (scheduler_t)filter->scheduler;
    
    // Pass raw pointers (VisionEngine expects const uint8_t*)
    gen_params.init_image = sd_input_img.data;
    
    if (using_controlnet) {
        gen_params.control_image = control_img.data;
        gen_params.control_strength = filter->controlnet_strength;
    }
    
    blog(LOG_DEBUG, "[lll.sight] Inference starting: %dx%d, steps=%d, seed=%ld", target_w, target_h, target_steps, gen_params.seed);
    
    // Generate with new API
    auto result = lll::LatentEngine::get().generate(gen_params);
    
    // Cleanup input buffers
    delete[] sd_input_img.data;
    sd_input_img.data = nullptr;
    // control_buffer handles control_img.data automatically
    
    bool success = false;
    
    if (!result.data.empty() && result.width == target_w && result.height == target_h &&
        (result.channels == 3 || result.channels == 4)) {
        std::lock_guard<std::mutex> out_lock(filter->output_mutex);
        
        if (filter->output_small.size() != target_rgba_size) {
            filter->output_small.resize(target_rgba_size);
        }

        // Convert result to RGBA
        if (result.channels == 3 && result.data.size() >= target_rgb_size) {
            lll::convert_rgb_to_rgba(result.data.data(), filter->output_small.data(), target_pixels);
        } else if (result.channels == 4 && result.data.size() >= target_rgba_size) {
            std::memcpy(filter->output_small.data(), result.data.data(), target_rgba_size);
        } else {
            return false;
        }
        
        // Upscale to full res
        uint32_t width = filter->width;
        uint32_t height = filter->height;
        if (filter->output_full_res.size() != full_rgba_size) {
            filter->output_full_res.resize(full_rgba_size);
        }
        
        bool resize_ok = resize_bilinear(
            filter->output_small.data(), filter->output_small.size(), target_w, target_h,
            filter->output_full_res.data(), filter->output_full_res.size(), width, height
        );
        
        if (!resize_ok) {
            blog(LOG_ERROR, "[lll.sight] Output resize failed: buffer size mismatch");
            success = false;
        } else {
            filter->new_frame_ready = true;
            success = true;
            
            if (filter->cache_frames) {
                filter->frame_cache.push(filter->output_full_res);
            }

            // === NSFW Check ===
            if (filter->nsfw_check && filter->output_small.size() == target_rgba_size) {
                 if (!filter->nsfw_detector) {
                     filter->nsfw_detector = std::make_unique<lll::NSFWDetector>();
                     // Helper to find model path
                     char* model_path = obs_module_file("models/open_nsfw.onnx");
                     if (model_path) {
                         filter->nsfw_detector->load(model_path);
                         bfree(model_path);
                     } else {
                         // Fallback attempt
                         filter->nsfw_detector->load("/usr/share/obs-ai-models/open_nsfw.onnx");
                     }
                 }
                 
                 if (filter->nsfw_detector->is_ready()) {
                     // Convert small RGBA output to packed RGB for detector
                     // Ideally we would optimize this, but for safety check speed is acceptable on small image
                     std::vector<uint8_t> rgb_check(target_rgb_size);
                     lll::convert_rgba_to_rgb(filter->output_small.data(), rgb_check.data(), target_pixels);

                     float score = filter->nsfw_detector->predict(rgb_check.data(), target_w, target_h);
                     if (score > 0.8f) {
                         blog(LOG_WARNING, "[lll.sight] NSFW Content Detected (Score: %.2f) - Blocking Frame", score);
                         
                         // Block content: Nice Effective Blur (Stack Blur)
                         // We blur the full resolution output directly
                         stack_blur_rgba(filter->output_full_res.data(), filter->width, filter->height, 64); // Radius 64 is very strong
                         
                         filter->set_status("Safety Filter Triggered");
                     }
                 }
            }

        }
    } else {
        // Generation failed
        success = false;
    }

    return success;
}

static void ai_worker_loop(style_transfer_data *filter) {
    while (!filter->stop_thread) {
        try {
            std::unique_lock<std::mutex> lock(filter->input_mutex);
            filter->frame_cv.wait(lock, [filter] {
                return filter->has_pending_frame || 
                       filter->model_load_requested || 
                       filter->stop_thread;
            });
            
            if (filter->stop_thread) break;
            
            if (filter->model_load_requested) {
            std::string model_path = filter->requested_model_path;
            std::string vae_path = filter->requested_vae_path;
            int thread_count = filter->requested_thread_count;
            filter->model_load_requested = false;
            lock.unlock();
            
            // Lock engine state for the duration of model loading
            {
                std::lock_guard<std::mutex> engine_lock(filter->engine_load_mutex);
                
                filter->set_status("Loading model...");
                filter->model_ready = false;  // Prevent rendering from using old model
                
                load_model(filter, model_path.c_str(), vae_path.c_str(), thread_count);
                
                // model_ready is set inside load_model(), but ensure it matches state here if needed
                // actually load_model sets it.
            }
            continue;
        }
            
            if (!filter->model_ready || !lll::LatentEngine::get().is_ready() || filter->input_full_res.empty()) {
                filter->has_pending_frame = false;
                filter->frames_dropped++;  // Track dropped frames
                continue;
            }
            
            uint32_t width = filter->width;
            uint32_t height = filter->height;
            std::vector<uint8_t> input_copy = filter->input_full_res;
            filter->has_pending_frame = false;
            lock.unlock();
            
            filter->processing = true;
            filter->set_status("Processing frame...");
            
            uint32_t target_w = std::max(64u, (uint32_t)(width * filter->scale_factor) / 64 * 64);
            uint32_t target_h = std::max(64u, (uint32_t)(height * filter->scale_factor) / 64 * 64);
            
            // Robust Buffer Validation & Resizing
        
        // 1. Validate dimensions
        if (target_w == 0 || target_h == 0 || width == 0 || height == 0) {
            blog(LOG_ERROR, "[lll.sight] Invalid dimensions: target=%ux%u, source=%ux%u",
                 target_w, target_h, width, height);
            filter->processing = false;
            filter->frames_dropped++;
            continue;
        }

        const size_t required_size = static_cast<size_t>(target_w) * target_h * 4;
        
        // 2. Check for integer overflow
        if (target_w > SIZE_MAX / 4 / target_h) {
            blog(LOG_ERROR, "[lll.sight] Buffer size overflow: %ux%u", target_w, target_h);
            filter->set_status("Memory Error");
            filter->processing = false;
            filter->frames_dropped++;
            continue;
        }

        // 3. Resize with exception handling
        if (target_w != filter->last_process_w || target_h != filter->last_process_h) {
            try {
                filter->input_small.resize(required_size);
                filter->process_w = target_w;
                filter->process_h = target_h;
                filter->last_process_w = target_w;
                filter->last_process_h = target_h;
            } catch (const std::bad_alloc& e) {
                blog(LOG_ERROR, "[lll.sight] Failed to allocate input buffer: %s", e.what());
                filter->set_status("Out of Memory");
                filter->processing = false;
                filter->frames_dropped++;
                continue;
            }
        }
        
        // 4. Pre-validate buffer sizes match requirements
        if (input_copy.size() != static_cast<size_t>(width) * height * 4) {
             blog(LOG_ERROR, "[lll.sight] Input buffer size mismatch: %zu (expected %zu)",
                  input_copy.size(), static_cast<size_t>(width) * height * 4);
             continue;
        }
        
        if (filter->input_small.size() != required_size) {
             // Should have been resized above, but double check
             blog(LOG_ERROR, "[lll.sight] Target buffer mismatch: %zu (expected %zu)",
                  filter->input_small.size(), required_size);
             try {
                filter->input_small.resize(required_size);
             } catch (...) {
                filter->processing = false;
                continue;
             }
        }

        bool resize_ok = resize_bilinear(
            input_copy.data(), input_copy.size(), width, height,
            filter->input_small.data(), filter->input_small.size(), target_w, target_h
        );
        
        if (!resize_ok) {
            blog(LOG_ERROR, "[lll.sight] Bilinear resize failed");
            filter->processing = false;
            continue;
        }
            
            // Check for frame similarity (Technique from ScreenDiffusion)
            float diff = 0.0f;
            if (!filter->last_input_small.empty()) {
                diff = calculate_image_difference(
                    filter->input_small.data(), 
                    filter->last_input_small.data(), 
                    filter->input_small.size()
                );
                filter->last_diff = diff;
                
                if (filter->similarity_threshold > 0.0f && diff < filter->similarity_threshold) {
                    // Skip inference
                    // Skip inference, but we still need to signal that we "processed" a frame 
                    // or just reuse the previous output already in the cache.
                    filter->has_pending_frame = false;
                    filter->processing = false;
                    filter->skipped_frame_count++;
                    filter->consecutive_skips++;
                    filter->max_consecutive_skips = std::max(
                        filter->max_consecutive_skips, filter->consecutive_skips);
                    // We don't increment frames_processed here to keep FPS accurate for AI work,
                    // but we might want to still trigger a render update if the UI expects it.
                    // However, if we skip, the video_render will just keep showing the last output_texture.
                    continue; 
                }
            }
            
            // Save current input for next similarity check
            if (filter->last_input_small.size() != filter->input_small.size()) {
                filter->last_input_small.resize(filter->input_small.size());
            }
            std::memcpy(filter->last_input_small.data(), filter->input_small.data(), filter->input_small.size());

            auto generation_start = std::chrono::steady_clock::now();
            bool success = run_style_transfer(filter);
            auto generation_elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - generation_start).count();
            filter->generation_attempts++;
            filter->generation_ms_total += generation_elapsed;
            if (filter->generation_attempts == 1) {
                filter->first_generation_ms = generation_elapsed;
            }
            filter->consecutive_skips = 0;
            
            if (success) {
                filter->successful_generations++;
                filter->frames_processed++;
                filter->set_status("Active");
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - filter->last_fps_update).count();
                if (elapsed >= 1000) {
                    filter->current_fps = filter->frames_processed.load() / (elapsed / 1000.0f);
                    filter->frames_processed = 0;
                    filter->last_fps_update = now;

                    const double average_generation_ms = filter->generation_attempts > 0
                        ? filter->generation_ms_total / filter->generation_attempts
                        : 0.0;
                    blog(LOG_INFO,
                         "[lll.sight] Profiling: attempts=%llu success=%llu failed=%llu "
                         "skipped=%llu max_skip_streak=%llu first_pipeline=%.1fms average_pipeline=%.1fms",
                         static_cast<unsigned long long>(filter->generation_attempts),
                         static_cast<unsigned long long>(filter->successful_generations),
                         static_cast<unsigned long long>(filter->failed_generations),
                         static_cast<unsigned long long>(filter->skipped_frame_count),
                         static_cast<unsigned long long>(filter->max_consecutive_skips),
                         filter->first_generation_ms,
                         average_generation_ms);
                    
                    // Update dock panel stats
                    ai_dock_update_stats(
                        filter->current_fps.load(),
                        filter->frames_dropped.load(),
                        0.0f  // VRAM tracking not implemented yet
                    );
                }
            } else {
                filter->failed_generations++;
                filter->set_status("Processing failed");
            }
            
            filter->processing = false;
            
        } catch (const std::exception& e) {
            blog(LOG_ERROR, "[lll.sight] Worker thread exception: %s", e.what());
            filter->set_status("Error: processing exception");
            filter->processing = false;
            filter->frames_dropped++;
            // Continue running despite error
        } catch (...) {
            blog(LOG_ERROR, "[lll.sight] Worker thread unknown exception");
            filter->set_status("Error: unknown exception");
            filter->processing = false;
            filter->frames_dropped++;
        }
    }
}

// Tick only handles timing logic, NEVER graphics
static void video_tick(void *data, float seconds) {
    auto *filter = static_cast<style_transfer_data *>(data);
    
    // Update transition timer
    if (filter->is_transitioning) {
        filter->transition_timer += seconds;
        if (filter->transition_timer >= filter->transition_duration) {
            filter->is_transitioning = false;
            filter->transition_timer = 0.0f;
        }
    }

    if (filter->scene_manager) {
        std::lock_guard<std::mutex> lock(filter->scene_mutex);
        filter->scene_manager->sync_mediapipe(lll::LatentEngine::get().get_mp_runner());
        filter->scene_manager->update_physics(16.0f);
    }

    // Capture timing logic
    if (!filter->model_ready.load()) {
        // Show initial message immediately
        if (!filter->initial_status_shown) {
            filter->set_status("Configure model path in filter properties");
            filter->initial_status_shown = true;
        }
        
        // Update periodically if model is being loaded
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - filter->last_status_update).count();
            
        if (elapsed >= filter->status_update_interval * 1000) {
            std::string current_status = filter->get_status();
            // Only update if not currently loading
            if (current_status.find("Loading") == std::string::npos) {
                filter->set_status("Waiting for model (configure in properties)");
            }
            filter->last_status_update = now;
        }
        return;
    }

    filter->capture_elapsed += seconds;
    if (filter->capture_elapsed >= filter->frame_interval) {
        filter->trigger_capture = true;
        filter->capture_elapsed = 0.0f;
    }
}

static void video_render(void *data, gs_effect_t *effect) {
    style_transfer_data *filter = (style_transfer_data *)data;
    obs_source_t *target = obs_filter_get_target(filter->context);
    if (!target) {
        blog(LOG_WARNING, "[lll.sight] No filter target");
        obs_source_skip_video_filter(filter->context);
        return;
    }

    uint32_t width = obs_source_get_base_width(target);
    uint32_t height = obs_source_get_base_height(target);
    if (!width || !height) {
        blog(LOG_WARNING, "[lll.sight] Invalid source dimensions: %ux%u", width, height);
        obs_source_skip_video_filter(filter->context);
        return;
    }

    std::vector<float> scene_render_data;
    if (filter->scene_manager) {
        std::lock_guard<std::mutex> lock(filter->scene_mutex);
        filter->scene_manager->get_render_data(scene_render_data, 64);
    }

    // --- DIAGNOSTIC COUNTERS (remove after debugging) ---
    static int render_counter = 0;
    static int capture_counter = 0;
    static int texture_update_counter = 0;
    render_counter++;

    // --- 1. CAPTURE PHASE ---
    if (filter->trigger_capture && filter->model_ready && !filter->processing) {
        if (!filter->capture_texrender) {
            filter->capture_texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
            blog(LOG_INFO, "[lll.sight] Created texrender");
        }
        
        if (gs_texrender_begin(filter->capture_texrender, width, height)) {
            vec4 clear_color;
            vec4_zero(&clear_color);
            gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
            
            gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);
            obs_source_video_render(target);
            gs_texrender_end(filter->capture_texrender);
            
            gs_texture_t *tex = gs_texrender_get_texture(filter->capture_texrender);
            if (tex) {
                 if (!filter->capture_staging) {
                      filter->capture_staging = gs_stagesurface_create(width, height, GS_RGBA);
                      blog(LOG_INFO, "[lll.sight] Created staging surface %ux%u", width, height);
                 }
                 
                 if (gs_stagesurface_get_width(filter->capture_staging) != width ||
                     gs_stagesurface_get_height(filter->capture_staging) != height) {
                     gs_stagesurface_destroy(filter->capture_staging);
                     filter->capture_staging = gs_stagesurface_create(width, height, GS_RGBA);
                     blog(LOG_INFO, "[lll.sight] Recreated staging surface %ux%u", width, height);
                 }
                 
                 if (filter->capture_staging) {
                    gs_stage_texture(filter->capture_staging, tex);
                     uint8_t *mapped_data;
                      uint32_t linesize;

                     if (gs_stagesurface_map(filter->capture_staging, &mapped_data, &linesize)) {
                         size_t input_pixels = 0;
                         size_t input_size = 0;
                         if (!checked_image_sizes(width, height, 4, input_pixels, input_size)) {
                             blog(LOG_ERROR, "[lll.sight] Input dimensions overflow: %ux%u", width, height);
                             gs_stagesurface_unmap(filter->capture_staging);
                             filter->frames_dropped++;
                         } else {
                         (void)input_pixels;
                         const size_t input_row_size = static_cast<size_t>(width) * 4;
                         std::lock_guard<std::mutex> lock(filter->input_mutex);
                         filter->width = width;
                         filter->height = height;
                         if (filter->input_full_res.size() != input_size) {
                              filter->input_full_res.resize(input_size);
                         }
                        
                         // Copy row by row
                         for (uint32_t y = 0; y < height; y++) {
                             memcpy(&filter->input_full_res[static_cast<size_t>(y) * input_row_size],
                                    mapped_data + static_cast<size_t>(y) * linesize, input_row_size);
                        }
                        filter->has_pending_frame = true;
                        
                        gs_stagesurface_unmap(filter->capture_staging);
                        
                        capture_counter++;
                         if (capture_counter <= 5 || capture_counter % 100 == 0) {
                            blog(LOG_INFO, "[lll.sight] Captured frame #%d (%ux%u), notifying worker",
                                 capture_counter, width, height);
                         }
                         }
                     } else {
                        blog(LOG_ERROR, "[lll.sight] Failed to map staging surface");
                    }
                    filter->frame_cv.notify_one();
                 }
            } else {
                blog(LOG_ERROR, "[lll.sight] Failed to get texture from texrender");
            }
        } else {
            blog(LOG_ERROR, "[lll.sight] Failed to begin texrender");
        }
        filter->trigger_capture = false;
    }

            // --- 2. UPDATE OUTPUT TEXTURE ---
    {
        std::lock_guard<std::mutex> lock(filter->output_mutex);
        
        if (filter->new_frame_ready && !filter->output_full_res.empty()) {
            const bool needs_recreation = !filter->output_texture ||
                                           filter->texture_width != width ||
                                           filter->texture_height != height;
            
            if (needs_recreation) {
                blog(LOG_INFO, "[lll.sight] Resolution change detected: %ux%u -> %ux%u",
                     filter->texture_width, filter->texture_height, width, height);

                // Keep old textures for one frame to allow smooth transition
                gs_texture_t* temp_old_output = filter->output_texture;
                gs_texture_t* temp_old_prev = filter->prev_output_texture;
                
                // Create new textures
                filter->output_texture = gs_texture_create(
                    width, height, GS_RGBA, 1, nullptr, GS_DYNAMIC
                );
                filter->prev_output_texture = nullptr;
                
                if (!filter->output_texture) {
                    blog(LOG_ERROR, "[lll.sight] Failed to create output texture %ux%u", width, height);
                    // Restore old textures as fallback
                    filter->output_texture = temp_old_output;
                    filter->prev_output_texture = temp_old_prev;
                    filter->new_frame_ready = false;
                } else {
                    // Successfully created new texture, destroy old ones
                    if (temp_old_output) gs_texture_destroy(temp_old_output);
                    if (temp_old_prev) gs_texture_destroy(temp_old_prev);
                    
                    filter->texture_width = width;
                    filter->texture_height = height;
                    
                    // Clear cache on resolution change
                    filter->frame_cache.initialized = false;
                    filter->last_input_small.clear();
                }
            } else {
                // If we have an existing output texture, move it to prev for transition
                if (filter->pure_ai_mode && filter->transition_duration > 0.0f && filter->output_texture) {
                    
                    // Validate dimensions before creating transition texture
                    bool prev_valid = filter->prev_output_texture && 
                                      gs_texture_get_width(filter->prev_output_texture) == width &&
                                      gs_texture_get_height(filter->prev_output_texture) == height;
                                      
                    if (!prev_valid) {
                         if (filter->prev_output_texture) {
                             gs_texture_destroy(filter->prev_output_texture);
                             filter->prev_output_texture = nullptr;
                         }
                         filter->prev_output_texture = gs_texture_create(width, height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
                         
                         if (!filter->prev_output_texture) {
                             blog(LOG_ERROR, "[lll.sight] Failed to create transition texture");
                             filter->is_transitioning = false;
                         }
                    }

                    if (filter->prev_output_texture) {
                        gs_copy_texture(filter->prev_output_texture, filter->output_texture);
                        filter->is_transitioning = true;
                        filter->transition_timer = 0.0f;
                    }
                }
            }
            
            if (filter->output_texture) {
                gs_texture_set_image(
                    filter->output_texture,
                    filter->output_full_res.data(),
                    width * 4,
                    false
                );
                filter->new_frame_ready = false;
            }
        }
    }

    // --- 3. RENDER PHASE ---
    if (filter->output_texture && filter->model_ready) {
        // Render processed output using default effect
        gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if (!effect) {
            obs_source_skip_video_filter(filter->context);
            return;
        }
        
        enum gs_color_space space = gs_get_color_space();
        obs_source_process_filter_begin_with_color_space(filter->context, GS_RGBA, space, OBS_ALLOW_DIRECT_RENDERING);
        
        gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), filter->output_texture);
        while (gs_effect_loop(effect, "Draw")) {
            gs_draw_sprite(filter->output_texture, 0, width, height);
        }

        // Visual Feedback for Processing
        if (filter->model_ready && filter->processing) {
             gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
             if (solid) {
                 gs_eparam_t *color_param = gs_effect_get_param_by_name(solid, "color");
                 
                 // Pulse effect
                 float pulse = 0.5f + 0.5f * sinf((float)os_gettime_ns() / 500000000.0f);
                 struct vec4 indicator_color;
                 vec4_set(&indicator_color, 0.2f, 0.8f, 0.2f, pulse); // Green pulse
                 
                 gs_effect_set_vec4(color_param, &indicator_color);
                 
                 while (gs_effect_loop(solid, "Solid")) {
                     // Small circle/square in top-right corner
                     // We are in the filter context, coordinates are 0..width, 0..height
                     int size = 10;
                     gs_draw_sprite(nullptr, 0, size, size); // Just draws a quad?
                     // gs_draw_sprite requires a texture usually, or we use gs_draw using a technique
                     // But solid effect draws a solid color quad.
                     // We need to set matrix to position it.
                     
                     gs_matrix_push();
                     gs_matrix_translate3f((float)(width - 20), 20.0f, 0.0f);
                     gs_matrix_scale3f((float)size, (float)size, 1.0f);
                     gs_draw_sprite(nullptr, 0, 1, 1); // Draw 1x1 rect scaled
                     gs_matrix_pop();
                 }
              }
        }

        obs_source_process_filter_end(filter->context, effect, width, height);

    } else {
        // Pass-through mode or Pure AI blanking
        if (filter->pure_ai_mode) {
             // Handle Smooth Transition logic
             // If we are transitioning, we draw prev_texture then blend output_texture on top.
             // If not, we just draw output_texture.
             // If we have no output_texture yet (first frame), we draw nothing (handled by clearing below).

             enum gs_color_space space = gs_get_color_space();
             obs_source_process_filter_begin_with_color_space(filter->context, GS_RGBA, space, OBS_ALLOW_DIRECT_RENDERING);
             
             // Clear background to black/transparent first
             vec4 clear_color;
             vec4_zero(&clear_color);
             gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);

             if (filter->output_texture) {
                 gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
                 if (effect) {
                     gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), filter->output_texture);
                     while (gs_effect_loop(effect, "Draw")) {
                         gs_draw_sprite(filter->output_texture, 0, width, height);
                     }
                 }
             }
             
             // Draw AR Overlay
             if (!scene_render_data.empty()) {
                  gs_matrix_push();
                  gs_matrix_scale3f((float)width, (float)height, 1.0f);

                  for (size_t i = 0; i + 3 < scene_render_data.size(); i += 4) {
                       const float x = scene_render_data[i];
                       const float y = scene_render_data[i + 1];
                       const float scale = scene_render_data[i + 2];
                       if (x < 0.0f || y < 0.0f) continue;
                       gs_matrix_push();
                       gs_matrix_translate3f(x, y, 0.0f);
                        gs_matrix_scale3f(scale, scale, 1.0f);
                        gs_draw_sprite(nullptr, 0, 1, 1);
                       gs_matrix_pop();
                 }
                 gs_matrix_pop();
            }

             obs_source_process_filter_end(filter->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);

        } else {
            // Normal Pass-through with AR Overlay
            enum gs_color_space space = gs_get_color_space();
            obs_source_process_filter_begin_with_color_space(filter->context, GS_RGBA, space, OBS_ALLOW_DIRECT_RENDERING);
            
            // 1. Draw Source
            obs_source_video_render(target);
            
            // 2. Draw AR Overlay
             if (!scene_render_data.empty()) {
                  gs_matrix_push();
                  gs_matrix_scale3f((float)width, (float)height, 1.0f);

                  for (size_t i = 0; i + 3 < scene_render_data.size(); i += 4) {
                       const float x = scene_render_data[i];
                       const float y = scene_render_data[i + 1];
                       const float scale = scene_render_data[i + 2];
                       if (x < 0.0f || y < 0.0f) continue;
                       gs_matrix_push();
                       gs_matrix_translate3f(x, y, 0.0f);
                        gs_matrix_scale3f(scale, scale, 1.0f);

                      // Draw simple white box (could be textured sprite)
                       gs_draw_sprite(nullptr, 0, 1, 1);

                      gs_matrix_pop();
                 }
                 gs_matrix_pop();
            }

            obs_source_process_filter_end(filter->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);
        }
    }
}

static void update(void *data, obs_data_t *settings) {
    style_transfer_data *filter = (style_transfer_data *)data;
    const char *path = obs_data_get_string(settings, S_MODEL_PATH);
    
    // Path validation
    if (path && *path) {
        if (strstr(path, "..") || path[0] == '/' || path[0] == '\\' || (path[0] && path[1] == ':')) {
            blog(LOG_ERROR, "[lll.sight] Invalid model path: %s", path);
            filter->set_status("Invalid model path");
            return;
        }
    }
    
    // Basic settings
    filter->prompt = obs_data_get_string(settings, S_PROMPT);
    filter->negative_prompt = obs_data_get_string(settings, S_NEGATIVE_PROMPT);
    filter->strength = (float)obs_data_get_double(settings, S_STRENGTH);
    filter->steps = (int)obs_data_get_int(settings, S_STEPS);
    filter->scale_factor = (float)obs_data_get_double(settings, S_SCALE);
    filter->cache_frames = obs_data_get_bool(settings, S_CACHE_FRAMES);
    
    // Advanced (LCM/Turbo) settings
    filter->sampler_method = (int)obs_data_get_int(settings, S_SAMPLER);
    filter->scheduler = (int)obs_data_get_int(settings, S_SCHEDULER);
    filter->cfg_scale = (float)obs_data_get_double(settings, S_CFG_SCALE);
    filter->seed = obs_data_get_int(settings, S_SEED);
    filter->random_seed = obs_data_get_bool(settings, S_RANDOM_SEED);
    int new_thread_count = (int)obs_data_get_int(settings, S_THREAD_COUNT);
    std::string old_vae_path = filter->vae_path;
    const char* vae_path = obs_data_get_string(settings, S_VAE_PATH);
    if (vae_path) filter->vae_path = vae_path;
    
    // Phase 2 Settings
    filter->vae_tiling = obs_data_get_bool(settings, S_VAE_TILING);
    const char* taesd = obs_data_get_string(settings, S_TAESD_PATH);
    if (taesd) filter->taesd_path = taesd;
    
    // LoRA
    filter->lora_path = obs_data_get_string(settings, S_LORA_PATH);
    filter->lora_strength = (float)obs_data_get_double(settings, S_LORA_STRENGTH);
    // Frame rate
    float fps = (float)obs_data_get_double(settings, S_FRAMERATE);
    if (fps < 0.1f) fps = 0.1f;
    filter->frame_interval = 1.0f / fps;
    
    // Performance
    bool old_flash_attn = filter->flash_attn;
    filter->flash_attn = obs_data_get_bool(settings, S_FLASH_ATTN);
    filter->easycache_enabled = obs_data_get_bool(settings, S_EASYCACHE);
    filter->similarity_threshold = (float)obs_data_get_double(settings, S_SIMILARITY_THRESHOLD);
    filter->motion_adaptive = obs_data_get_bool(settings, S_MOTION_ADAPTIVE);
    filter->temporal_smoothing = (float)obs_data_get_double(settings, S_TEMPORAL_SMOOTHING);
    
    // ComfyUI-Inspired Optimizations
    filter->easycache_reuse_threshold = (float)obs_data_get_double(settings, "easycache_reuse_threshold");
    filter->easycache_start_percent = (float)obs_data_get_double(settings, "easycache_start_percent");
    filter->easycache_end_percent = (float)obs_data_get_double(settings, "easycache_end_percent");
    filter->adaptive_steps = obs_data_get_bool(settings, "adaptive_steps");
    filter->min_adaptive_steps = (int)obs_data_get_int(settings, "min_adaptive_steps");
    
    // VAE/Latent Space Optimizations
    filter->vae_on_cpu = obs_data_get_bool(settings, S_VAE_ON_CPU);
    filter->taesd_preview_only = obs_data_get_bool(settings, S_TAESD_PREVIEW);
    filter->vae_conv_direct = obs_data_get_bool(settings, S_VAE_CONV_DIRECT);
    filter->pure_ai_mode = obs_data_get_bool(settings, S_PURE_AI);
    filter->show_preview = obs_data_get_bool(settings, "show_preview");
    filter->transition_duration = (float)obs_data_get_double(settings, S_TRANSITION_DURATION);
    
    // ControlNet
    const char* controlnet = obs_data_get_string(settings, S_CONTROLNET_PATH);
    if (controlnet) filter->controlnet_path = controlnet;
    filter->controlnet_strength = (float)obs_data_get_double(settings, S_CONTROLNET_STRENGTH);
    filter->controlnet_mode = (int)obs_data_get_int(settings, S_CONTROLNET_MODE);
    filter->pose_model_reset_requested = true;
    
    // ControlNet Advanced
    filter->cn_params.auto_threshold = obs_data_get_bool(settings, "cn_auto_threshold");
    filter->cn_params.canny_high_threshold = (float)obs_data_get_double(settings, "cn_canny_high");
    filter->cn_params.canny_low_threshold = (float)obs_data_get_double(settings, "cn_canny_low");
    
    // MediaPipe (Internal for ControlNet mostly)
    int old_mp_graph = filter->mp_backend_type;
    filter->mp_backend_type = (int)obs_data_get_int(settings, S_MP_BACKEND);
    filter->mp_show_mask = obs_data_get_bool(settings, S_MP_SHOW_MASK);
    
    // Check if we need to init MP for ControlNet features
    if (filter->mp_backend_type != 0) { 
        if (!lll::LatentEngine::get().get_mp_runner()) {
             lll::LatentEngine::get().init_mediapipe(true); 
        }
    }

    // Safety Filter
    filter->nsfw_check = obs_data_get_bool(settings, S_NSFW_CHECK);
    
    // Fast Style
    filter->fast_style_enabled = obs_data_get_bool(settings, S_FAST_STYLE_ENABLED);
    const char* fast_model = obs_data_get_string(settings, S_FAST_STYLE_MODEL);
    if (fast_model) filter->fast_style_model_path = fast_model;

    // Clamp values
    if (filter->scale_factor < 0.1f) filter->scale_factor = 0.1f;

    // Trigger model reload if model/VAE path, thread count, flash_attn, or ControlNet changed
    bool need_reload = (path && filter->current_model_path != path) ||
                       (vae_path && old_vae_path != vae_path) ||
                       (new_thread_count != filter->thread_count && lll::LatentEngine::get().is_ready()) ||
                       (taesd && filter->current_taesd_path != taesd) ||
                       (filter->flash_attn != old_flash_attn) ||
                       (controlnet && filter->current_controlnet_path != controlnet);
    
    if (need_reload && path && *path) {
        {
            std::lock_guard<std::mutex> lock(filter->input_mutex);
            filter->requested_model_path = path;
            filter->requested_vae_path = filter->vae_path;
            filter->requested_thread_count = new_thread_count;
            if (taesd) filter->current_taesd_path = taesd;
            if (controlnet) filter->current_controlnet_path = controlnet;
            filter->model_load_requested = true;
        }
        filter->frame_cv.notify_one();
    }
}

static void destroy(void *data) {
    auto *filter = static_cast<style_transfer_data *>(data);
    blog(LOG_INFO, "Shutting down filter...");
    
    filter->stop_thread = true;
    filter->model_ready = false;
    
    {
        std::lock_guard<std::mutex> lock(filter->input_mutex);
        filter->has_pending_frame = false;
        filter->model_load_requested = false;
    }
    
    filter->frame_cv.notify_all();
    if (filter->worker_thread.joinable()) {
        filter->worker_thread.join();
    }
    
    // Unregister hotkeys
    if (filter->hotkey_pure_ai != OBS_INVALID_HOTKEY_ID) {
        obs_hotkey_unregister(filter->hotkey_pure_ai);
        filter->hotkey_pure_ai = OBS_INVALID_HOTKEY_ID;
    }
    if (filter->hotkey_trigger != OBS_INVALID_HOTKEY_ID) {
        obs_hotkey_unregister(filter->hotkey_trigger);
        filter->hotkey_trigger = OBS_INVALID_HOTKEY_ID;
    }
    
    // Thread-safe context cleanup
    {
        std::lock_guard<std::mutex> ctx_lock(filter->engine_load_mutex);
        // LatentEngine is shared, we don't unload it here to avoid breaking other sources
    }
    
    // Graphics cleanup
    obs_enter_graphics();
    if (filter->output_texture) {
        gs_texture_destroy(filter->output_texture);
        filter->output_texture = nullptr;
    }
    if (filter->prev_output_texture) {
        gs_texture_destroy(filter->prev_output_texture);
        filter->prev_output_texture = nullptr;
    }
    if (filter->capture_texrender) {
        gs_texrender_destroy(filter->capture_texrender);
        filter->capture_texrender = nullptr;
    }
    if (filter->capture_staging) {
        gs_stagesurface_destroy(filter->capture_staging);
        filter->capture_staging = nullptr;
    }
    obs_leave_graphics();
    
    blog(LOG_INFO, "Filter destroyed successfully");
    delete filter;
}

// Hotkey Callbacks
static void pure_ai_hotkey_callback(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed) return;
    auto *filter = static_cast<style_transfer_data *>(data);
    
    // Toggle property
    filter->pure_ai_mode = !filter->pure_ai_mode;
    
    // Logic to update OBS property UI if open? 
    // This is hard to do from here without the properties object, but the internal state is what matters for rendering.
    // We can just log it.
    blog(LOG_INFO, "[lll.sight] Hotkey: Pure AI Mode %s", filter->pure_ai_mode ? "Enabled" : "Disabled");
}

static void trigger_hotkey_callback(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed) return;
    auto *filter = static_cast<style_transfer_data *>(data);
    
    blog(LOG_INFO, "[lll.sight] Hotkey: Trigger Generation (Clear Cache)");
    
    // Clear cache to force new frame generation
    std::lock_guard<std::mutex> lock(filter->output_mutex);
    filter->new_frame_ready = false;
    filter->frames_processed = 0; // Reset counter maybe?
    // Optionally we could reset model_ready or invoke a generation if manual trigger logic existed.
    // For now, clearing the ready flag usually prompts the system to want a new frame.
}

static void *create(obs_data_t *settings, obs_source_t *context) {
    style_transfer_data *filter = new style_transfer_data();
    filter->context = context;
    
    // Register Hotkeys
    filter->hotkey_pure_ai = obs_hotkey_register_source(context, "obs_sd_cpp.pure_ai_toggle", 
                                                       "Pure AI: Toggle Mode",
                                                       pure_ai_hotkey_callback, filter);
                                                       
    filter->hotkey_trigger = obs_hotkey_register_source(context, "obs_sd_cpp.trigger_gen",
                                                       "Pure AI: Trigger Generation",
                                                       trigger_hotkey_callback, filter);
    
    filter->worker_thread = std::thread(ai_worker_loop, filter);
    
    filter->scene_manager = std::make_unique<lll::SceneManager>();
    
    update(filter, settings);
    
    // Connect to dock panel for preset integration
    ai_dock_connect_source(filter->context);
    
    return filter;
}

static void get_defaults(obs_data_t *settings) {
    // Basic settings
    obs_data_set_default_string(settings, S_MODEL_PATH, "");
    obs_data_set_default_string(settings, S_VAE_PATH, "");
    obs_data_set_default_string(settings, S_PROMPT, "artistic style");
    obs_data_set_default_string(settings, S_NEGATIVE_PROMPT, "blurry, distorted");
    obs_data_set_default_double(settings, S_STRENGTH, 0.75);
    obs_data_set_default_int(settings, S_STEPS, 20);
    obs_data_set_default_double(settings, S_SCALE, 0.5);
    obs_data_set_default_double(settings, S_FRAMERATE, 2.0);
    obs_data_set_default_bool(settings, S_CACHE_FRAMES, true);
    
    // Advanced (LCM/Turbo) settings
    obs_data_set_default_int(settings, S_SAMPLER, EULER_A_SAMPLE_METHOD);
    obs_data_set_default_int(settings, S_SCHEDULER, DISCRETE_SCHEDULER);
    obs_data_set_default_double(settings, S_CFG_SCALE, 7.0);
    obs_data_set_default_int(settings, S_SEED, -1);
    obs_data_set_default_bool(settings, S_RANDOM_SEED, true);
    obs_data_set_default_int(settings, S_THREAD_COUNT, 0);  // 0 = auto
    obs_data_set_default_bool(settings, S_NSFW_CHECK, false);
    
    // Fast Style Defaults
    obs_data_set_default_bool(settings, S_FAST_STYLE_ENABLED, false);
    obs_data_set_default_string(settings, S_FAST_STYLE_MODEL, "");
    
    // Phase 2 Settings
    obs_data_set_default_string(settings, S_PRESET, "None");
    obs_data_set_default_bool(settings, S_VAE_TILING, false);
    obs_data_set_default_string(settings, S_TAESD_PATH, "");
    obs_data_set_default_string(settings, S_LORA_PATH, "");
    obs_data_set_default_double(settings, S_LORA_STRENGTH, 1.0);
    
    // Performance Defaults
    obs_data_set_default_bool(settings, S_FLASH_ATTN, true);
    obs_data_set_default_bool(settings, S_EASYCACHE, true);
    obs_data_set_default_double(settings, S_SIMILARITY_THRESHOLD, 0.5);
    obs_data_set_default_bool(settings, S_MOTION_ADAPTIVE, true);
    obs_data_set_default_double(settings, S_TEMPORAL_SMOOTHING, 0.0);
    
    // ComfyUI-Inspired Optimization Defaults
    obs_data_set_default_double(settings, "easycache_reuse_threshold", 0.5);
    obs_data_set_default_double(settings, "easycache_start_percent", 0.0);
    obs_data_set_default_double(settings, "easycache_end_percent", 1.0);
    obs_data_set_default_bool(settings, "adaptive_steps", true);
    obs_data_set_default_int(settings, "min_adaptive_steps", 2);
    
    // ControlNet Defaults
    obs_data_set_default_string(settings, S_CONTROLNET_PATH, "");
    obs_data_set_default_double(settings, S_CONTROLNET_STRENGTH, 0.8);
    obs_data_set_default_int(settings, S_CONTROLNET_MODE, 0); // Off by default
    
    obs_data_set_default_bool(settings, "cn_auto_threshold", true);
    obs_data_set_default_double(settings, "cn_canny_high", 100.0);
    obs_data_set_default_double(settings, "cn_canny_low", 50.0);
    
    // VAE/Latent Space Defaults
    obs_data_set_default_bool(settings, S_VAE_ON_CPU, false);
    obs_data_set_default_bool(settings, S_TAESD_PREVIEW, false);
    obs_data_set_default_bool(settings, S_VAE_CONV_DIRECT, true); // Enabled by default for speed
    obs_data_set_default_bool(settings, S_PURE_AI, false);
    obs_data_set_default_double(settings, S_TRANSITION_DURATION, 0.5);
}

static const StylePreset PRESETS[] = {
    {"None", nullptr, nullptr, 0, 0, 0, 0, 0},
    {"Anime", "anime style, cel shaded, vibrant", "photorealistic, 3d render", 0.7f, 7.0f, 20, EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER},
    {"Cinematic", "cinematic lighting, dramatic, 8k", "cartoon, sketch", 0.8f, 7.0f, 25, DPMPP2M_SAMPLE_METHOD, DISCRETE_SCHEDULER},
    {"Cyberpunk", "cyberpunk, neon lights, futuristic", "natural, rustic", 0.75f, 7.5f, 20, EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER},
    {"Oil Painting", "oil painting, thick brushstrokes", "digital, sharp", 0.8f, 7.0f, 25, EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER},
    {"Sketch", "pencil sketch, rough lines", "color, photo", 0.9f, 9.0f, 15, EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER},
    {PRESET_LCM_NAME, "high quality", "blurry", 0.75f, 1.5f, 5, LCM_SAMPLE_METHOD, LCM_SCHEDULER},
    // Z-Image Turbo: Optimized for 8 steps, photorealistic output, Apache 2.0 license
    {PRESET_ZIMAGE_NAME, "photorealistic, high quality, detailed, sharp", "blurry, distorted, low quality, artifacts", 
     0.8f, 2.0f, 8, EULER_SAMPLE_METHOD, DISCRETE_SCHEDULER}
};

static bool preset_changed(obs_properties_t *props, obs_property_t *p, obs_data_t *settings) {
    const char *name = obs_data_get_string(settings, S_PRESET);
    
    for (const auto& preset : PRESETS) {
        if (strcmp(preset.name, name) == 0 && preset.prompt) {
            obs_data_set_string(settings, S_PROMPT, preset.prompt);
            obs_data_set_string(settings, S_NEGATIVE_PROMPT, preset.negative_prompt);
            obs_data_set_double(settings, S_STRENGTH, preset.strength);
            obs_data_set_double(settings, S_CFG_SCALE, preset.cfg);
            obs_data_set_int(settings, S_STEPS, preset.steps);
            obs_data_set_int(settings, S_SAMPLER, preset.sampler);
            obs_data_set_int(settings, S_SCHEDULER, preset.scheduler);
            return true;
        }
    }
    return true;
}

static bool model_path_changed(obs_properties_t *props, obs_property_t *p, obs_data_t *settings) {
    const char *path = obs_data_get_string(settings, S_MODEL_PATH);
    if (!path || !*path) return false;
    
    std::string path_str = path;
    std::transform(path_str.begin(), path_str.end(), path_str.begin(), ::tolower);
    
    // Auto-detect Z-Image Turbo models FIRST (more specific, before generic turbo check)
    if (path_str.find("z-image") != std::string::npos || 
        path_str.find("z_image") != std::string::npos ||
        path_str.find("zimage") != std::string::npos) {
        
        FILE *f = fopen(path, "rb");
        if (!f) {
            blog(LOG_WARNING, "[lll.sight] Selected model file not found: %s", path);
            return false;
        }
        fclose(f);

        blog(LOG_INFO, "[lll.sight] Detected Z-Image Turbo model, applying optimal settings...");
        
        obs_data_set_string(settings, S_PRESET, PRESET_ZIMAGE_NAME);
        
        // Apply Z-Image optimal settings
        for (const auto& preset : PRESETS) {
            if (strcmp(preset.name, PRESET_ZIMAGE_NAME) == 0) {
                obs_data_set_double(settings, S_STRENGTH, preset.strength);
                obs_data_set_double(settings, S_CFG_SCALE, preset.cfg);
                obs_data_set_int(settings, S_STEPS, preset.steps);
                obs_data_set_int(settings, S_SAMPLER, preset.sampler);
                obs_data_set_int(settings, S_SCHEDULER, preset.scheduler);
                break;
            }
        }
        return true;
    }
    
    // Auto-detect generic LCM/Turbo models (SDXL-Turbo, etc.)
    if (path_str.find("lcm") != std::string::npos || path_str.find("turbo") != std::string::npos) {
        
        // Validate file exists before auto-configuring
        FILE *f = fopen(path, "rb");
        if (!f) {
            blog(LOG_WARNING, "[lll.sight] Selected model file not found: %s", path);
            return false;
        }
        fclose(f);

        blog(LOG_INFO, "[lll.sight] Detected LCM/Turbo model, applying recommended settings...");
        
        // Find LCM preset
        obs_data_set_string(settings, S_PRESET, PRESET_LCM_NAME);
        
        // Apply settings manually since changing the preset drop-down value 
        // doesn't automatically trigger its callback from here
        for (const auto& preset : PRESETS) {
            if (strcmp(preset.name, PRESET_LCM_NAME) == 0) {
                 obs_data_set_double(settings, S_STRENGTH, preset.strength);
                 obs_data_set_double(settings, S_CFG_SCALE, preset.cfg);
                 obs_data_set_int(settings, S_STEPS, preset.steps);
                 obs_data_set_int(settings, S_SAMPLER, preset.sampler);
                 obs_data_set_int(settings, S_SCHEDULER, preset.scheduler);
                 break;
            }
        }
        return true;
    }
    
    return true;
}

static bool test_inference_callback(obs_properties_t *props, obs_property_t *property, void *data) {
    auto *filter = static_cast<style_transfer_data *>(data);
    
    blog(LOG_INFO, "[lll.sight] TEST: model_ready=%d, processing=%d, has_pending=%d",
         filter->model_ready.load(), filter->processing.load(), filter->has_pending_frame);
    blog(LOG_INFO, "[lll.sight] TEST: output_texture=%p, frame_cache_init=%d",
         filter->output_texture, filter->frame_cache.initialized);
        filter->set_status("Test button clicked - check logs");
    return true;
}

static bool export_preset_callback(obs_properties_t *props, obs_property_t *p, void *data) {
    auto *filter = static_cast<style_transfer_data*>(data);
    obs_data_t* settings = obs_source_get_settings(filter->context);
    
    if (!settings) return false;
    
    const char* json = obs_data_get_json(settings);
    
    // Create presets directory in plugin config path
    char path[512];
    char *config_dir = obs_module_get_config_path(obs_current_module(), "");
    if (config_dir) {
        snprintf(path, sizeof(path), "%s/presets", config_dir);
        os_mkdirs(path);
        bfree(config_dir);
    }
    
    // Generate filename with timestamp
    time_t now = time(nullptr);
    char filename[512];
    // Re-get config path to append filename (inefficient but safe)
    config_dir = obs_module_get_config_path(obs_current_module(), "");
    if (config_dir) {
        snprintf(filename, sizeof(filename), "%s/presets/preset_%ld.json", config_dir, now);
        bfree(config_dir);
        
        FILE* f = fopen(filename, "w");
        if (f) {
            fputs(json, f);
            fclose(f);
            blog(LOG_INFO, "[lll.sight] Preset exported to: %s", filename);
            filter->set_status("Preset exported successfully");
        } else {
            filter->set_status("Failed to export preset");
        }
    }
    
    obs_data_release(settings);
    return false;
}

static bool import_preset_callback(obs_properties_t *props, obs_property_t *p, void *data) {
    auto *filter = static_cast<style_transfer_data*>(data);
    obs_data_t *settings = obs_source_get_settings(filter->context);
    
    if (!settings) return false;
    
    const char *path = obs_data_get_string(settings, "preset_import_path");
    
    if (path && *path) {
        obs_data_t* imported = obs_data_create_from_json_file(path);
        if (imported) {
            obs_source_update(filter->context, imported);
            obs_data_release(imported);
            filter->set_status("Preset imported successfully");
        } else {
             filter->set_status("Failed to load preset file");
        }
    } else {
        filter->set_status("No preset file selected");
    }
    
    obs_data_release(settings);
    return false;
}

static bool open_models_url_callback(obs_properties_t *props, obs_property_t *p, void *data) {
    UNUSED_PARAMETER(props);
    UNUSED_PARAMETER(p);
    UNUSED_PARAMETER(data);
    QDesktopServices::openUrl(QUrl("https://huggingface.co/leejet/Z-Image-Turbo-GGUF"));
    return false;
}

static obs_properties_t *get_properties(void *data) {
    auto *filter = static_cast<style_transfer_data *>(data);
    obs_properties_t *props = obs_properties_create();

    // === Model and Runtime ===
    obs_properties_add_text(props, "section_model", obs_module_text("StyleTransfer.Section.Model"), OBS_TEXT_INFO);
    obs_properties_add_button(props, "dl_all", obs_module_text("StyleTransfer.DownloadModels"),
        open_models_url_callback);

    obs_properties_add_button(props, "open_model_folder", obs_module_text("StyleTransfer.OpenModelFolder"),
    [](obs_properties_t*, obs_property_t*, void*) {
        const char* path = get_default_model_paths();
        os_mkdirs(path); // Create if doesn't exist
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
        return false;
    });

    // Status display (read-only)
    obs_property_t *status_prop = obs_properties_add_text(props, S_STATUS, obs_module_text("StyleTransfer.Status"), OBS_TEXT_INFO);
    obs_property_set_enabled(status_prop, false);
    
    // === Model and presets ===
    obs_property_t *p = obs_properties_add_path(props, S_MODEL_PATH, "Model", 
                                                 OBS_PATH_FILE, "*.gguf *.ckpt *.safetensors", NULL);
    obs_property_set_long_description(p, "Stable Diffusion model (GGUF recommended for speed)");
    obs_property_set_modified_callback(p, model_path_changed);

    obs_property_t *vae = obs_properties_add_path(props, S_VAE_PATH, "Standalone VAE (Optional)",
                                                  OBS_PATH_FILE, "*.safetensors *.ckpt *.pt", NULL);
    obs_property_set_long_description(vae, "Optional standalone VAE. Leave empty to use the checkpoint's built-in VAE.");
    
    obs_property_t *lora_p = obs_properties_add_path(props, S_LORA_PATH, "LoRA Model",
                                                      OBS_PATH_FILE, "*.safetensors *.ckpt", NULL);
    obs_property_set_long_description(lora_p, "Optional LoRA model (e.g. LCM or style)");
    
    obs_property_t *lora_s = obs_properties_add_float_slider(props, S_LORA_STRENGTH, "LoRA Strength", 0.0, 2.0, 0.05);
    
    obs_property_t *preset_list = obs_properties_add_list(props, S_PRESET, "Style Preset", 
                                                         OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    for (const auto& preset : PRESETS) {
        obs_property_list_add_string(preset_list, preset.name, preset.name);
    }
    obs_property_set_modified_callback(preset_list, preset_changed);
    
    obs_properties_add_button(props, "export_preset", "💾 Export Settings", export_preset_callback);
    
    obs_properties_add_path(props, "preset_import_path", "Preset File", OBS_PATH_FILE, "*.json", NULL);
    obs_properties_add_button(props, "import_preset", "📂 Load Preset", import_preset_callback);
    
    // === Prompt and generation ===
    obs_properties_add_text(props, "section_generation", obs_module_text("StyleTransfer.Section.Generation"), OBS_TEXT_INFO);
    obs_property_t *prompt_prop = obs_properties_add_text(props, S_PROMPT, "Prompt", OBS_TEXT_MULTILINE);
    obs_property_set_long_description(prompt_prop, "Describe the artistic style to apply");
    
    obs_property_t *neg_prompt_prop = obs_properties_add_text(props, S_NEGATIVE_PROMPT, "Negative Prompt", OBS_TEXT_MULTILINE);
    obs_property_set_long_description(neg_prompt_prop, "What to avoid in the output");
    
    // === Input and capture ===
    obs_properties_add_text(props, "section_input", obs_module_text("StyleTransfer.Section.Input"), OBS_TEXT_INFO);
    obs_property_t *fps_prop = obs_properties_add_float_slider(props, S_FRAMERATE, "Processing FPS", 0.5, 30.0, 0.5);
    obs_property_set_long_description(fps_prop, "Frames per second to process (lower = better performance)");
    
    obs_property_t *scale_prop = obs_properties_add_float_slider(props, S_SCALE, "Processing Scale", 0.1, 1.0, 0.05);
    obs_property_set_long_description(scale_prop, "Resolution scale (lower = faster)");
    
    obs_property_t *strength_prop = obs_properties_add_float_slider(props, S_STRENGTH, "Effect Strength", 0.0, 1.0, 0.01);
    // Thread-safe status check for UI
    if (filter) {
        if (filter->engine_load_mutex.try_lock()) {
            filter->engine_load_mutex.unlock();
        }
    }
    
    obs_property_set_long_description(strength_prop, "How much style affects the image");
    
    obs_property_t* nsfw = obs_properties_add_bool(props, S_NSFW_CHECK, "Safety Filter (NSFW)");
    obs_property_set_long_description(nsfw, "Blocks unsafe generated content with a blur");

    // === Control and guidance ===
    obs_properties_add_text(props, "section_control", obs_module_text("StyleTransfer.Section.Control"), OBS_TEXT_INFO);
    // Grouping removed for compatibility
    
    obs_property_t* f_enable = obs_properties_add_bool(props, S_FAST_STYLE_ENABLED, "Enable Fast Style (Bypass SD)");
    
    obs_property_t* f_model = obs_properties_add_path(props, S_FAST_STYLE_MODEL, "Fast Style Model (.onnx)", 
                                                       OBS_PATH_FILE, "*.onnx", NULL);
    obs_property_set_long_description(f_model, "Select a PINTO/ONNX style transfer model");
    
    // === Output and runtime ===
    obs_properties_add_text(props, "section_output", obs_module_text("StyleTransfer.Section.Output"), OBS_TEXT_INFO);

    // === Sampler and scheduler ===
    obs_property_t *sampler = obs_properties_add_list(props, S_SAMPLER, "Sampler", 
                                                       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(sampler, "Euler A", EULER_A_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "Euler", EULER_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "LCM", LCM_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "DPM++ 2M", DPMPP2M_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "DPM++ 2S Ancestral", DPMPP2S_A_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "Heun", HEUN_SAMPLE_METHOD);
    obs_property_list_add_int(sampler, "DDIM Trailing", DDIM_TRAILING_SAMPLE_METHOD);
    obs_property_set_long_description(sampler, "LCM for LCM/Turbo models, Euler A for standard");
    
    obs_property_t *scheduler = obs_properties_add_list(props, S_SCHEDULER, "Scheduler",
                                                         OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(scheduler, "Discrete", DISCRETE_SCHEDULER);
    obs_property_list_add_int(scheduler, "Karras", KARRAS_SCHEDULER);
    obs_property_list_add_int(scheduler, "LCM", LCM_SCHEDULER);
    obs_property_list_add_int(scheduler, "SGM Uniform", SGM_UNIFORM_SCHEDULER);
    obs_property_list_add_int(scheduler, "Simple", SIMPLE_SCHEDULER);
    obs_property_set_long_description(scheduler, "LCM for LCM models, Discrete for standard");
    
    obs_property_t *cfg_prop = obs_properties_add_float_slider(props, S_CFG_SCALE, "CFG Scale", 1.0, 20.0, 0.5);
    obs_property_set_long_description(cfg_prop, "Guidance strength (LCM: 1-2, Standard: 5-10)");
    
    // === Hardware and performance ===
    obs_properties_add_bool(props, S_VAE_TILING, "Low VRAM Mode (VAE Tiling)");
    
    obs_property_t *taesd = obs_properties_add_path(props, S_TAESD_PATH, "TAESD Path (Optional)", 
                                                 OBS_PATH_FILE, "*.safetensors *.pth", NULL);
    obs_property_set_long_description(taesd, "Tiny AutoEncoder for faster decoding");
    
    obs_property_t *threads_prop = obs_properties_add_int_slider(props, S_THREAD_COUNT, "CPU Threads", 0, 32, 1);
    obs_property_set_long_description(threads_prop, "0 = auto-detect optimal count");
    
    obs_properties_add_bool(props, S_CACHE_FRAMES, "Enable Frame Caching");

    // Performance & Optimization Group (Techniques from ScreenDiffusion)
    obs_properties_t *perf_group = obs_properties_create();
    obs_properties_add_bool(perf_group, S_FLASH_ATTN, "Enable Flash Attention (High Speed)");
    obs_properties_add_bool(perf_group, S_EASYCACHE, "Enable EasyCache (Internal Optimization)");
    obs_property_t *sim_prop = obs_properties_add_float_slider(perf_group, S_SIMILARITY_THRESHOLD, "Skip Fragment Similarity Threshold", 0.0, 10.0, 0.1);
    obs_property_set_long_description(sim_prop, "Skip processing if frame is similar to last one. 0 = disabled, 0.5-1.0 is recommended.");
    
    obs_properties_add_bool(perf_group, S_MOTION_ADAPTIVE, "Motion-Adaptive Steps (SDV2)");
    obs_property_t *smooth_prop = obs_properties_add_float_slider(perf_group, S_TEMPORAL_SMOOTHING, "Temporal Smoothing Alpha", 0.0, 0.95, 0.05);
    obs_property_set_long_description(smooth_prop, "Blends output with previous frame to reduce flicker. (0.0=off, 0.5=high)");
    
    // ComfyUI-Inspired Optimizations
    obs_properties_add_bool(perf_group, "adaptive_steps", "Adaptive Steps (ComfyUI-style)");
    obs_property_t *min_steps = obs_properties_add_int_slider(perf_group, "min_adaptive_steps", "Min Adaptive Steps", 1, 10, 1);
    obs_property_set_long_description(min_steps, "Minimum steps when frame is very similar to previous. Lower = faster but may reduce quality.");
    
    obs_property_t *ec_thresh = obs_properties_add_float_slider(perf_group, "easycache_reuse_threshold", "EasyCache Aggressiveness", 0.1, 0.9, 0.1);
    obs_property_set_long_description(ec_thresh, "Higher = more aggressive caching (faster, but may miss details)");
    
    // VAE/Latent Space Optimizations
    obs_property_t *vae_conv = obs_properties_add_bool(perf_group, S_VAE_CONV_DIRECT, "VAE Direct Convolution (Faster)");
    obs_property_set_long_description(vae_conv, "Uses optimized convolution for VAE encoding/decoding. Recommended ON.");
    
    obs_property_t *vae_cpu = obs_properties_add_bool(perf_group, S_VAE_ON_CPU, "VAE on CPU (Save VRAM)");
    obs_property_set_long_description(vae_cpu, "Offload VAE to CPU. Slower but saves ~1GB VRAM for larger models.");
    
    obs_property_t *taesd_preview = obs_properties_add_bool(perf_group, S_TAESD_PREVIEW, "TAESD Preview Mode");
    obs_property_set_long_description(taesd_preview, "Use TAESD for decoding only (faster but slightly lower quality). Requires TAESD path set.");

    obs_property_t *pure_ai = obs_properties_add_bool(perf_group, S_PURE_AI, "Pure AI Mode (Hide Input)");
    obs_property_set_long_description(pure_ai, "Hides the original camera feed while generating. Effect is a 'slideshow' of AI frames only.");

    obs_properties_add_bool(perf_group, "show_preview", "Show Live Preview (Slower)");
    
    obs_property_t *trans_prop = obs_properties_add_float_slider(perf_group, S_TRANSITION_DURATION, "Pure AI Transition (s)", 0.0, 2.0, 0.1);
    obs_property_set_long_description(trans_prop, "Cross-fade duration between generated frames in Pure AI Mode.");

    obs_properties_add_group(props, "perf_group", obs_module_text("StyleTransfer.Group.Performance"), OBS_GROUP_NORMAL, perf_group);

    // MediaPipe Vision Group (For debug/ControlNet)
    obs_properties_t *mp_group = obs_properties_create();
    obs_property_t *mp_mode = obs_properties_add_list(mp_group, S_MP_BACKEND, "Vision Preprocessor", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(mp_mode, "None", 0); 
    obs_property_list_add_int(mp_mode, "Selfie Segmentation (Mask)", 1);
    obs_property_list_add_int(mp_mode, "Face Mesh (Landmarks)", 2);
    // Removed Privacy Filter from here
    
    obs_properties_add_bool(mp_group, S_MP_SHOW_MASK, "Visualize Preprocessor Output");
    
    obs_properties_add_group(props, "mp_group", obs_module_text("StyleTransfer.Group.MediaPipe"), OBS_GROUP_NORMAL, mp_group);

    // ControlNet Group
    obs_properties_t *controlnet_group = obs_properties_create();
    
    obs_property_t *cn_mode = obs_properties_add_list(controlnet_group, S_CONTROLNET_MODE, "ControlNet Mode",
                                                       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(cn_mode, "Off", 0);
    obs_property_list_add_int(cn_mode, "Canny Edge", 1);
    obs_property_list_add_int(cn_mode, "Luminance (Depth)", 2);
    obs_property_list_add_int(cn_mode, "Pass-through (Tile)", 3);
    obs_property_list_add_int(cn_mode, "Silhouette (Lineart)", 4);
    obs_property_list_add_int(cn_mode, "Soft Edge (Scribble)", 5);
    obs_property_list_add_int(cn_mode, "Motion Pose (OpenPose)", 6);
    obs_property_list_add_int(cn_mode, "Neural Depth (Depth Anything)", 7);
    obs_property_list_add_int(cn_mode, "Face Mesh (MediaPipe)", 8);
    obs_property_set_long_description(cn_mode, 
        "Canny=edges, Luminance=fast depth, Tile=direct, Lineart=silhouette, "
        "Scribble=soft edges, Motion Pose=skeleton, Neural Depth=AI depth, "
        "Face Mesh=MediaPipe landmarks. Use matching ControlNet model.");
    
    obs_property_t *cn_path = obs_properties_add_path(controlnet_group, S_CONTROLNET_PATH, "ControlNet Model",
                                                       OBS_PATH_FILE, "*.gguf *.safetensors *.ckpt", NULL);
    obs_property_set_long_description(cn_path, "Path to ControlNet model (e.g., control_v11p_sd15_canny.gguf)");
    
    obs_property_t *cn_strength = obs_properties_add_float_slider(controlnet_group, S_CONTROLNET_STRENGTH, "ControlNet Strength", 0.0, 2.0, 0.05);
    obs_property_set_long_description(cn_strength, "How strongly ControlNet guides the output. 0.8-1.0 recommended.");
    
    obs_properties_add_group(props, "controlnet_group", obs_module_text("StyleTransfer.Group.ControlNet"), OBS_GROUP_NORMAL, controlnet_group);
    
    obs_properties_t *cn_advanced = obs_properties_create();
    obs_properties_add_bool(cn_advanced, "cn_auto_threshold", "Auto-Adjust Thresholds");
    obs_properties_add_float_slider(cn_advanced, "cn_canny_high", "Canny High Threshold", 10.0, 300.0, 5.0);
    obs_properties_add_float_slider(cn_advanced, "cn_canny_low", "Canny Low Threshold", 5.0, 150.0, 5.0);
    obs_properties_add_group(props, "cn_advanced", obs_module_text("StyleTransfer.Group.EdgeSettings"), OBS_GROUP_NORMAL, cn_advanced);

    // === Diagnostics ===
    obs_properties_add_text(props, "section_diagnostics", obs_module_text("StyleTransfer.Section.Diagnostics"), OBS_TEXT_INFO);
    obs_properties_add_button(props, "test_button", obs_module_text("StyleTransfer.TestInference"), test_inference_callback);
    
    
    // === Seed Control ===
    obs_properties_add_bool(props, S_RANDOM_SEED, "Random Seed Each Frame");
    obs_property_t *seed_prop = obs_properties_add_int(props, S_SEED, "Seed", -1, INT32_MAX, 1);
    obs_property_set_long_description(seed_prop, "Fixed seed (-1 or Random enabled = random)");

    // Thread-safe status update
    if (filter) {
        std::string status = filter->get_status();
        float fps = filter->current_fps.load();
        int dropped = filter->frames_dropped.load();
        
        char status_text[256];
        if (fps > 0.0f) {
            snprintf(status_text, sizeof(status_text), "%s | %.1f FPS | Dropped: %d", 
                    status.c_str(), fps, dropped);
        } else {
            snprintf(status_text, sizeof(status_text), "%s | Dropped: %d", 
                    status.c_str(), dropped);
        }
        
        obs_data_t *settings = obs_source_get_settings(filter->context);
        obs_data_set_string(settings, S_STATUS, status_text);
        obs_data_release(settings);
    }
    return props;
}

static const char *get_name(void *unused) { return "lll.sight Latent Style Filter"; }

static enum gs_color_space get_video_color_space(void *data, size_t count,
                                              const enum gs_color_space *preferred_spaces)
{
    UNUSED_PARAMETER(data);
    UNUSED_PARAMETER(count);
    UNUSED_PARAMETER(preferred_spaces);
    
    // AI models typically operate in sRGB space.
    // By returning GS_CS_SRGB, we ensure OBS performs tonemapping if the source is HDR.
    return GS_CS_SRGB;
}

struct obs_source_info style_transfer_filter_info = {
    .id = "style_transfer_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = get_name,
    .create = create,
    .destroy = destroy,
    .get_width = NULL,
    .get_height = NULL,
    .get_defaults = get_defaults,
    .get_properties = get_properties,
    .update = update,
    .activate = NULL,
    .deactivate = NULL,
    .show = NULL,
    .hide = NULL,
    .video_tick = video_tick,
    .video_render = video_render,
    .video_get_color_space = get_video_color_space
};
