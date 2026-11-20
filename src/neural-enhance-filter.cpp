#include <obs-module.h>
#include <graphics/graphics.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <condition_variable>
#include <cstring>
#include <algorithm>
#include <util/platform.h>
#include <util/dstr.h>
#include "stable-diffusion.h"
#include "common/image-utils.hpp"
#include "core/latent-engine.hpp"
#include "core/enhancement_processor.hpp"

#define S_MODEL_PATH "model_path"
#define S_STRENGTH "strength"
#define S_DETAIL "detail_level"
#define S_MODE "enhancement_mode"
#define S_THREAD_COUNT "thread_count"
#define S_CACHE_FRAMES "cache_frames"
#define S_STATUS "status"

// New Modes
#define MODE_GEN (0)
#define MODE_UPSCALE (1)
#define MODE_STYLE (2)
#define MODE_RESTORE (3)

struct neural_enhance_data {
    obs_source_t *context;
    
    // Config
    std::string model_path;
    float strength = 0.25f;
    std::string detail_level = "Medium";
    std::string mode = "Balanced";
    int thread_count = 0;
    bool cache_frames = true;
    
    // Status (exposed via properties)
    std::string current_status = "Unloaded";

    // State
    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};
    std::atomic<bool> model_ready{false};
    std::atomic<bool> processing{false};
    
    // Processors
    std::unique_ptr<lll::EnhancementProcessor> enhance_proc;
    int current_mode_idx = 0; // 0=Gen, 1=Upscale, 2=Style, 3=Restore
    
    std::mutex input_mutex;
    std::mutex output_mutex;
    std::mutex engine_load_mutex;
    std::condition_variable frame_cv;
    
    bool has_pending_frame = false;
    bool model_load_requested = false;
    std::string requested_model_path;
    
    // Buffers
    std::vector<uint8_t> input_small; // For inference resolution
    std::vector<uint8_t> output_full_res; // For display
    std::vector<uint8_t> temp_rgb; // Generic rgb buffer
    
    // Caching
    struct FrameCache {
        std::vector<std::vector<uint8_t>> frames;
        size_t write_index = 0;
        size_t max_size = 3; 
        bool initialized = false;
        
        void push(const std::vector<uint8_t>& data) {
            if (data.empty()) return;
            if (!initialized) {
                 frames.resize(max_size); 
                 for(auto& f : frames) f.resize(data.size());
                 initialized = true;
            }
            if (frames[write_index].size() != data.size()) frames[write_index].resize(data.size());
            std::memcpy(frames[write_index].data(), data.data(), data.size());
            write_index = (write_index + 1) % max_size;
        }
        const std::vector<uint8_t>* get_latest() const {
            if (!initialized || frames.empty()) return nullptr;
            size_t idx = (write_index == 0) ? (max_size - 1) : (write_index - 1);
            return &frames[idx];
        }
    } frame_cache;

    // OBS Render State
    gs_texture_t *output_texture = nullptr;
    gs_texrender_t *texrender = nullptr;
    gs_stagesurf_t *stagesurf = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t process_w = 0;
    uint32_t process_h = 0;
    bool new_frame_ready = false;
};

// --- Helpers ---

static const char* get_enhance_trigger_prompt(const char* detail_level) {
    if (strcmp(detail_level, "High") == 0) return "noise free, sharp edges, high fidelity, 8k, clear, authentic";
    if (strcmp(detail_level, "Low") == 0) return "clean, smooth, denoise, soft, artifact free";
    return "balanced quality, clear image, noise reduction, sharp"; // Medium
}

static void load_model_deferred(neural_enhance_data *filter, const char *path) {
    std::lock_guard<std::mutex> lock(filter->engine_load_mutex);
    filter->model_ready = false;
    
    // Check mode
    if (filter->current_mode_idx > 0) {
        // ONNX Models
        if (!filter->enhance_proc) {
            filter->enhance_proc = std::make_unique<lll::EnhancementProcessor>();
        }
        
        lll::EnhanceModelType type = lll::EnhanceModelType::RealESRGAN;
        const char* actual_path = path;
        char* module_path = nullptr;
        
        // Auto-select internal model if path is simple name
        if (filter->current_mode_idx == MODE_UPSCALE) {
             type = lll::EnhanceModelType::RealESRGAN;
             if (strstr(path, ".onnx") == nullptr) {
                 module_path = obs_module_file("models/realesrgan_x4.onnx");
                 actual_path = module_path;
             }
        } else if (filter->current_mode_idx == MODE_STYLE) {
             type = lll::EnhanceModelType::AnimeGANv3;
             if (strstr(path, ".onnx") == nullptr) {
                 module_path = obs_module_file("models/animegan_v3_portrait.onnx");
                 actual_path = module_path;
             }
        } else if (filter->current_mode_idx == MODE_RESTORE) {
             type = lll::EnhanceModelType::ZeroDCE;
             if (strstr(path, ".onnx") == nullptr) {
                 module_path = obs_module_file("models/zero_dce.onnx");
                 actual_path = module_path;
             }
        }
        
        if (actual_path && filter->enhance_proc->load(actual_path, type, true)) {
            filter->model_path = path;
            filter->model_ready = true;
            filter->current_status = "Active: " + std::string(actual_path);
        } else {
            filter->current_status = "Error: Failed to load ONNX model";
        }
        
        if (module_path) bfree(module_path);
        
    } else {
        // GGUF / Stable Diffusion
        lll::EngineParams params;
        params.model_path = path;
        params.num_threads = filter->thread_count;
        params.use_gpu = true;
        params.flash_attn = true; 
        
        if (lll::LatentEngine::get().load_model(params)) {
            filter->model_path = path;
            filter->model_ready = true;
            filter->current_status = "Active: " + std::string(path);
        } else {
            filter->model_ready = false;
            filter->current_status = "Error: Failed to load model";
        }
    }
    
    // Trigger property update
    obs_data_t *settings = obs_source_get_settings(filter->context);
    obs_data_set_string(settings, S_STATUS, filter->current_status.c_str());
    obs_source_update(filter->context, settings);
    obs_data_release(settings);
}

static void worker_loop(neural_enhance_data *filter) {
    while (!filter->stop_thread) {
        std::unique_lock<std::mutex> lock(filter->input_mutex);
        filter->frame_cv.wait(lock, [filter] {
            return filter->has_pending_frame || filter->model_load_requested || filter->stop_thread;
        });

        if (filter->stop_thread) break;

        if (filter->model_load_requested) {
            std::string path = filter->requested_model_path;
            filter->model_load_requested = false;
            lock.unlock();
            load_model_deferred(filter, path.c_str());
            continue;
        }

        if (!filter->has_pending_frame || !filter->model_ready) {
             continue;
        }

        // Process Frame
        std::vector<uint8_t> input_copy = filter->input_small;
        uint32_t w = filter->process_w;
        uint32_t h = filter->process_h;
        filter->has_pending_frame = false;
        lock.unlock();

        // Check if we are using ONNX processor
        if (filter->current_mode_idx > 0) {
            // ONNX Mode
            if (!filter->enhance_proc || !filter->enhance_proc->is_ready()) {
                filter->processing = false;
                continue;
            }
            
            filter->processing = true;
            
            // Convert to RGB
            size_t pixel_count = w * h;
            if (filter->temp_rgb.size() != pixel_count * 3) filter->temp_rgb.resize(pixel_count * 3);
            lll::convert_rgba_to_rgb(input_copy.data(), filter->temp_rgb.data(), pixel_count);
            
            // Determine output size
            int out_w = w;
            int out_h = h;
            if (filter->enhance_proc->get_type() == lll::EnhanceModelType::RealESRGAN) {
                out_w = w * 4;
                out_h = h * 4;
            }
            
            std::vector<uint8_t> rgb_out(out_w * out_h * 3);
            
            if (filter->enhance_proc->process(filter->temp_rgb.data(), w, h, rgb_out.data(), out_w, out_h)) {
                
                std::lock_guard<std::mutex> out_lock(filter->output_mutex);
                
                // Convert to RGBA
                std::vector<uint8_t> rgba_out(out_w * out_h * 4);
                lll::convert_rgb_to_rgba(rgb_out.data(), rgba_out.data(), out_w * out_h);
                
                // If Upscale 4x, we might want to keep it big, or resize back to canvas.
                // If process_w was small (512), 4x is 2048. 
                // We write to output_full_res (which is filter->width * height).
                // So we resize result to filter output.
                
                if (filter->output_full_res.size() != filter->width * filter->height * 4) {
                    filter->output_full_res.resize(filter->width * filter->height * 4);
                }
                
                resize_bilinear(rgba_out.data(), rgba_out.size(), out_w, out_h,
                                filter->output_full_res.data(), filter->output_full_res.size(),
                                filter->width, filter->height);
                                
                if (filter->cache_frames) {
                    filter->frame_cache.push(filter->output_full_res);
                }
                filter->new_frame_ready = true;
            }
            
            filter->processing = false;
            continue;
        }

        if (!lll::LatentEngine::get().is_ready()) continue;

        filter->processing = true;

        lll::framework::VisionEngine::GenerationParams params;
        params.prompt = get_enhance_trigger_prompt(filter->detail_level.c_str());
        params.negative_prompt = "blur, noise, artifacts, low quality, distortion, grain, ugly";
        params.width = w;
        params.height = h;
        params.cfg_scale = 7.0f; // Standard for img2img
        params.seed = -1;
        
        // Map mode to steps
        if (filter->mode == "Speed") params.steps = 2; // LCM/Turbo territory
        else if (filter->mode == "Quality") params.steps = 8;
        else params.steps = 4; // Balanced

        params.strength = filter->strength; // Critical param
        
        // Prepare Init Image (RGB)
        std::unique_ptr<uint8_t[]> rgb_input(new uint8_t[w * h * 3]);
        lll::convert_rgba_to_rgb(input_copy.data(), rgb_input.get(), w * h);
        params.init_image = rgb_input.get();

        auto result = lll::LatentEngine::get().generate(params);
        
        if (!result.data.empty()) {
            std::lock_guard<std::mutex> out_lock(filter->output_mutex);
            
            // Output is likely RGB 
            std::vector<uint8_t> rgba_small(w * h * 4);
            if (result.channels == 3) {
                 lll::convert_rgb_to_rgba(result.data.data(), rgba_small.data(), w * h);
            } else {
                 std::memcpy(rgba_small.data(), result.data.data(), result.data.size());
            }

            // Upscale to full res
            if (filter->output_full_res.size() != filter->width * filter->height * 4) {
                filter->output_full_res.resize(filter->width * filter->height * 4);
            }
            
            resize_bilinear(rgba_small.data(), rgba_small.size(), w, h,
                            filter->output_full_res.data(), filter->output_full_res.size(),  
                            filter->width, filter->height);
                            
            if (filter->cache_frames) {
                filter->frame_cache.push(filter->output_full_res);
            }
            filter->new_frame_ready = true;
        }
        
        filter->processing = false;
    }
}

// --- OBS Callbacks ---

static const char *enhance_get_name(void *unused) {
    UNUSED_PARAMETER(unused);
    return "LLL.sight Neural Enhance";
}

static void enhance_destroy(void *data) {
    neural_enhance_data *filter = (neural_enhance_data*)data;
    if (filter) {
        filter->stop_thread = true;
        filter->frame_cv.notify_one();
        if (filter->worker_thread.joinable()) {
            filter->worker_thread.join();
        }
        
        std::lock_guard<std::mutex> lock(filter->output_mutex);
        if (filter->output_texture) {
            gs_texture_destroy(filter->output_texture);
        }
        if (filter->texrender) {
            gs_texrender_destroy(filter->texrender);
        }
        if (filter->stagesurf) {
            gs_stagesurface_destroy(filter->stagesurf);
        }
        
        delete filter;
    }
}

static void *enhance_create(obs_data_t *settings, obs_source_t *context) {
    neural_enhance_data *filter = new neural_enhance_data();
    filter->context = context;

    const char* model_path = obs_data_get_string(settings, S_MODEL_PATH);
    filter->strength = (float)obs_data_get_double(settings, S_STRENGTH);
    filter->detail_level = obs_data_get_string(settings, S_DETAIL);
    filter->mode = obs_data_get_string(settings, S_MODE);
    filter->thread_count = (int)obs_data_get_int(settings, S_THREAD_COUNT);
    filter->cache_frames = obs_data_get_bool(settings, S_CACHE_FRAMES);
    if (filter->mode == "Upscale") filter->current_mode_idx = MODE_UPSCALE;
    else if (filter->mode == "Style") filter->current_mode_idx = MODE_STYLE;
    else if (filter->mode == "Restore") filter->current_mode_idx = MODE_RESTORE;

    // Start worker
    filter->worker_thread = std::thread(worker_loop, filter);
    
    // Trigger model load if path exists
    if (model_path && *model_path) {
        std::lock_guard<std::mutex> lock(filter->input_mutex);
        filter->requested_model_path = model_path;
        filter->model_load_requested = true;
        filter->frame_cv.notify_one();
    }
    
    return filter;
}

static void enhance_update(void *data, obs_data_t *settings) {
    neural_enhance_data *filter = (neural_enhance_data*)data;
    
    const char* new_path = obs_data_get_string(settings, S_MODEL_PATH);
    float new_strength = (float)obs_data_get_double(settings, S_STRENGTH);
    const char* new_detail = obs_data_get_string(settings, S_DETAIL);
    const char* new_mode = obs_data_get_string(settings, S_MODE);
    
    if (strcmp(new_mode, "Upscale") == 0) filter->current_mode_idx = MODE_UPSCALE;
    else if (strcmp(new_mode, "Style") == 0) filter->current_mode_idx = MODE_STYLE;
    else if (strcmp(new_mode, "Restore") == 0) filter->current_mode_idx = MODE_RESTORE;
    else filter->current_mode_idx = MODE_GEN;
    
    // Auto-reload if switching between GEN and ONNX modes, or ONNX types
    // Simplified: Check if path or mode change implies reload
    bool mode_changed = (filter->mode != new_mode);
    filter->mode = new_mode;
    
    if (filter->model_path != new_path || mode_changed) {
        std::lock_guard<std::mutex> lock(filter->input_mutex);
        filter->requested_model_path = new_path ? new_path : "auto";
        filter->model_load_requested = true;
        filter->frame_cv.notify_one();
    }
    
    filter->strength = new_strength;
    filter->detail_level = new_detail;
    filter->thread_count = (int)obs_data_get_int(settings, S_THREAD_COUNT);
    filter->cache_frames = obs_data_get_bool(settings, S_CACHE_FRAMES);
}

static obs_properties_t *enhance_properties(void *data) {
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_path(props, S_MODEL_PATH, "Model (GGUF/ONNX)", OBS_PATH_FILE, "Models (*.gguf *.onnx);;All Files (*.*)", nullptr);
    
    obs_property_t *p_strength = obs_properties_add_float_slider(props, S_STRENGTH, "Enhancement Strength", 0.0, 0.5, 0.01);
    obs_property_set_long_description(p_strength, "How much to hallucinate details. Lower is safer.");
    
    obs_property_t *p_mode = obs_properties_add_list(props, S_MODE, "Mode", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(p_mode, "Generative (Stable Diffusion)", "Balanced");
    obs_property_list_add_string(p_mode, "Generative (Turbo)", "Speed");
    obs_property_list_add_string(p_mode, "Generic Upscale (Real-ESRGAN)", "Upscale");
    obs_property_list_add_string(p_mode, "Anime Style (AnimeGANv3)", "Style");
    obs_property_list_add_string(p_mode, "Restore Lighting (Raidas/Zero-DCE)", "Restore");
    
    obs_property_t *p_detail = obs_properties_add_list(props, S_DETAIL, "Target Detail", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(p_detail, "Low (Clean/Smooth)", "Low");
    obs_property_list_add_string(p_detail, "Medium (Balanced)", "Medium");
    obs_property_list_add_string(p_detail, "High (Sharp/Texture)", "High");

    obs_properties_add_bool(props, S_CACHE_FRAMES, "Cache Frames (Smoother Playback)");
    
    obs_property_t *p_status = obs_properties_add_text(props, S_STATUS, "Status", OBS_TEXT_DEFAULT);
    obs_property_set_enabled(p_status, false); // Read-only

    return props;
}

static void enhance_video_render(void *data, gs_effect_t *effect) {
    neural_enhance_data *filter = (neural_enhance_data*)data;
    if (!filter) return;

    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);
    
    if (!target || !parent) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // 1. Capture Input
    uint32_t cx = obs_source_get_base_width(target);
    uint32_t cy = obs_source_get_base_height(target);
    
    if (cx == 0 || cy == 0) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    if (filter->width != cx || filter->height != cy) {
        filter->width = cx;
        filter->height = cy;
        // Logic to determine inference resolution (e.g. 512x512 aspect ratio)
        // For simplicity, fixed 512 wide, preserve aspect
        float aspect = (float)cx / (float)cy;
        filter->process_w = 512;
        filter->process_h = (uint32_t)(512.0f / aspect);
        // Ensure mult of 8
        filter->process_h = (filter->process_h + 7) & ~7;
    }
    
    // Only capture if not currently processing and we are ready
    // Actually we should capture every frame but only dispatch if worker is free, 
    // OR we just render the last cached frame if processing.
    
    if (!filter->processing && filter->model_ready) {
        // Capture logic: Render downscaled to texrender, then copy to staging
        if (!filter->texrender) {
            filter->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        }
        
        if (!filter->stagesurf) {
            filter->stagesurf = gs_stagesurface_create(filter->process_w, filter->process_h, GS_RGBA);
        }
        
        if (filter->texrender && filter->stagesurf) {
            // 1. Render Source to Low-Res Texture
            if (gs_texrender_begin(filter->texrender, filter->process_w, filter->process_h)) {
                gs_ortho(0.0f, (float)filter->process_w, 0.0f, (float)filter->process_h, -100.0f, 100.0f);
                
                obs_source_video_render(target);
                
                gs_texrender_end(filter->texrender);
            }
            
            // 2. Copy Low-Res Texture to Staging Surface
            gs_texture_t *rendered_tex = gs_texrender_get_texture(filter->texrender);
            gs_stage_texture(filter->stagesurf, rendered_tex);
            
            // 3. Map and Read (Synchronous for now, but small 512px buffer is fast)
            uint8_t *ptr;
            uint32_t linesize;
            if (gs_stagesurface_map(filter->stagesurf, &ptr, &linesize)) {
                std::lock_guard<std::mutex> lock(filter->input_mutex);
                
                // Copy to input buffer (handle potential padding/linesize)
                // We want tight RGB/RGBA for inference
                // For simplicity, let's copy to a tight RGBA buffer first
                size_t expected_size = filter->process_w * filter->process_h * 4;
                if (filter->input_small.size() != expected_size) {
                    filter->input_small.resize(expected_size);
                }
                
                // Copy row by row if linesize differs, or bulk if matches
                if (linesize == filter->process_w * 4) {
                    std::memcpy(filter->input_small.data(), ptr, expected_size);
                } else {
                    for (uint32_t y = 0; y < filter->process_h; y++) {
                         std::memcpy(filter->input_small.data() + y * filter->process_w * 4, 
                                     ptr + y * linesize, 
                                     filter->process_w * 4);
                    }
                }
                
                filter->has_pending_frame = true;
                filter->frame_cv.notify_one();
                
                gs_stagesurface_unmap(filter->stagesurf);
            }
        }
    }
    
    // Render the LATEST output
    std::lock_guard<std::mutex> out_lock(filter->output_mutex);
    
    if (filter->new_frame_ready) {
        if (filter->output_texture) {
            gs_texture_destroy(filter->output_texture);
            filter->output_texture = nullptr;
        }
        
        const std::vector<uint8_t>* frame_data = nullptr;
        if (filter->frame_cache.initialized) {
            frame_data = filter->frame_cache.get_latest();
        } else {
             frame_data = &filter->output_full_res;
        }
        
        if (frame_data && frame_data->size() == static_cast<size_t>(cx) * cy * 4) {
            const uint8_t *pixels = frame_data->data();
            filter->output_texture = gs_texture_create(cx, cy, GS_RGBA, 1, &pixels, GS_DYNAMIC);
            filter->new_frame_ready = false;
        }
    }
    
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
        // Passthrough if not ready
        obs_source_skip_video_filter(filter->context);
    }
}

static void enhance_tick(void *data, float seconds) {
    neural_enhance_data *filter = (neural_enhance_data*)data;
    if (!filter) return;
    
    // We need to capture the frame on the video thread, usually in render or via a tick 
    // if we had a texrender target.
    // Real capture logic needs to happen in render context.
    
    // Minimal logic: if worker is free, signal we want a frame in render
    // filter->trigger_capture = true;
}

struct obs_source_info neural_enhance_filter_info = {
    .id = "neural_enhance_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = enhance_get_name,
    .create = enhance_create,
    .destroy = enhance_destroy,
    .get_properties = enhance_properties,
    .update = enhance_update,
    .video_tick = enhance_tick,
    .video_render = enhance_video_render
};
