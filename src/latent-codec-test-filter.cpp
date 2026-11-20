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
#include <chrono>
#include <util/platform.h>
#include <util/dstr.h>
#include "core/latent-engine.hpp"
#include "framework/vision_engine.hpp"

#define S_MODEL_PATH "model_path"
#define S_VAE_PATH "vae_path" // Optional override

struct latent_codec_data {
    obs_source_t *context;
    
    // Config
    std::string model_path;
    std::string vae_path;

    // State
    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};
    std::atomic<bool> model_ready{false};
    
    std::mutex input_mutex;
    std::mutex output_mutex;
    std::condition_variable frame_cv;
    
    bool has_pending_frame = false;
    bool model_load_requested = false;
    
    // Buffers
    lll::framework::VisionEngine::ImageData input_frame;
    lll::framework::VisionEngine::ImageData output_frame;
    bool new_output_ready = false;
    
    // Metrics
    std::atomic<double> last_latency_ms{0.0};
    
    // OBS Render State
    gs_texture_t *output_texture = nullptr;
    gs_texrender_t *texrender = nullptr;
    gs_stagesurf_t *stagesurf = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t process_w = 0;
    uint32_t process_h = 0;
};

static const char *codec_get_name(void *unused) {
    UNUSED_PARAMETER(unused);
    return "LLL.sight Latent Codec Test";
}

static void load_model_deferred(latent_codec_data *filter) {
    lll::EngineParams params;
    params.model_path = filter->model_path;
    params.vae_path = filter->vae_path;
    params.use_gpu = true; 
    
    blog(LOG_INFO, "[Latent Codec] Loading model...");
    if (lll::LatentEngine::get().load_model(params)) {
        filter->model_ready = true;
        blog(LOG_INFO, "[Latent Codec] Model loaded.");
    } else {
        blog(LOG_ERROR, "[Latent Codec] Failed to load model.");
    }
}

static void worker_loop(latent_codec_data *filter) {
    while (!filter->stop_thread) {
        std::unique_lock<std::mutex> lock(filter->input_mutex);
        filter->frame_cv.wait(lock, [filter] { 
            return filter->has_pending_frame || filter->stop_thread || filter->model_load_requested; 
        });
        
        if (filter->stop_thread) break;
        
        if (filter->model_load_requested) {
            filter->model_load_requested = false;
            lock.unlock();
            load_model_deferred(filter);
            continue;
        }
        
        if (filter->has_pending_frame && filter->model_ready) {
            // Copy input
            auto current_input = filter->input_frame;
            filter->has_pending_frame = false;
            lock.unlock();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 1. Encode
            auto latent = lll::LatentEngine::get().encode(current_input);
            
            // 2. Decode
            if (latent.data.size() > 0) {
                auto decoded = lll::LatentEngine::get().decode(latent);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end_time - start_time;
                filter->last_latency_ms = duration.count();
                
                if (decoded.data.size() > 0) {
                    std::lock_guard<std::mutex> out_lock(filter->output_mutex);
                    filter->output_frame = decoded;
                    filter->new_output_ready = true;
                }
            }
        } else {
            lock.unlock(); // Should not happen often
        }
    }
}

static void codec_update(void *data, obs_data_t *settings) {
    latent_codec_data *filter = (latent_codec_data *)data;
    
    const char *path = obs_data_get_string(settings, S_MODEL_PATH);
    const char *vae = obs_data_get_string(settings, S_VAE_PATH);
    
    std::string new_path = path ? path : "";
    std::string new_vae = vae ? vae : "";
    
    if (new_path != filter->model_path || new_vae != filter->vae_path) {
        filter->model_path = new_path;
        filter->vae_path = new_vae;
        
        std::lock_guard<std::mutex> lock(filter->input_mutex);
        filter->model_load_requested = true;
        filter->frame_cv.notify_one();
    }
}

static void *codec_create(obs_data_t *settings, obs_source_t *context) {
    latent_codec_data *filter = new latent_codec_data();
    filter->context = context;
    
    codec_update(filter, settings);
    
    filter->worker_thread = std::thread(worker_loop, filter);
    
    return filter;
}

static void codec_destroy(void *data) {
    latent_codec_data *filter = (latent_codec_data *)data;
    
    filter->stop_thread = true;
    filter->frame_cv.notify_all();
    if (filter->worker_thread.joinable()) {
        filter->worker_thread.join();
    }
    
    if (filter->texrender) gs_texrender_destroy(filter->texrender);
    if (filter->stagesurf) gs_stagesurface_destroy(filter->stagesurf);
    if (filter->output_texture) gs_texture_destroy(filter->output_texture);
    
    delete filter;
}

static void codec_video_render(void *data, gs_effect_t *effect) {
    latent_codec_data *filter = (latent_codec_data *)data;
    
    obs_source_t *target = obs_filter_get_target(filter->context);
    if (!target) return;

    uint32_t width = obs_source_get_base_width(target);
    uint32_t height = obs_source_get_base_height(target);
    
    if (width == 0 || height == 0) return;
    
    if (filter->width != width || filter->height != height) {
        filter->width = width;
        filter->height = height;
        // Process at reduced resolution for speed? Or full? 
        // VAE is slow. Let's start with full but maybe scale down if needed.
        // For SD 1.5, 512x512 is sweet spot.
        filter->process_w = 512; 
        filter->process_h = 512;
        
        if (filter->texrender) gs_texrender_destroy(filter->texrender);
        filter->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        
        if (filter->stagesurf) gs_stagesurface_destroy(filter->stagesurf);
        filter->stagesurf = gs_stagesurface_create(filter->process_w, filter->process_h, GS_RGBA);
        
        if (filter->output_texture) gs_texture_destroy(filter->output_texture);
        filter->output_texture = gs_texture_create(filter->process_w, filter->process_h, GS_RGBA, 1, nullptr, GS_DYNAMIC);
    }
    
    // Capture Frame to Texture
    if (gs_texrender_begin(filter->texrender, filter->process_w, filter->process_h)) {
        gs_ortho(0.0f, (float)filter->process_w, 0.0f, (float)filter->process_h, -100.0f, 100.0f);
        obs_source_video_render(target);
        gs_texrender_end(filter->texrender);
    }
    
    // Readback to CPU
    gs_texture_t *rendered_tex = gs_texrender_get_texture(filter->texrender);
    gs_stage_texture(filter->stagesurf, rendered_tex);
    
    uint8_t *ptr;
    uint32_t linesize;
    if (gs_stagesurface_map(filter->stagesurf, &ptr, &linesize)) {
        // Copy to input buffer for worker
        std::unique_lock<std::mutex> lock(filter->input_mutex, std::try_to_lock);
        if (lock.owns_lock()) {
             // Only update if worker is ready to take it? or always update?
             // Always update.
             filter->input_frame.width = filter->process_w;
             filter->input_frame.height = filter->process_h;
             filter->input_frame.channels = 4;
             // Input needs to be tightly packed for SD usually?
             // VisionEngine::ImageData is vector<uint8_t>.
             // We need to copy line by line if stride differs.
             size_t size = filter->process_w * filter->process_h * 4;
             if (filter->input_frame.data.size() != size) filter->input_frame.data.resize(size);
             
             for(uint32_t y=0; y<filter->process_h; y++) {
                 std::memcpy(filter->input_frame.data.data() + y * filter->process_w * 4,
                             ptr + y * linesize,
                             filter->process_w * 4);
             }
             
             filter->has_pending_frame = true;
             filter->frame_cv.notify_one();
        }
        gs_stagesurface_unmap(filter->stagesurf);
    }
    
    // Draw Output
    {
        std::lock_guard<std::mutex> lock(filter->output_mutex);
        if (filter->new_output_ready && filter->output_frame.data.size() > 0) {
            // Update output texture
            gs_texture_set_image(filter->output_texture, 
                                 filter->output_frame.data.data(), 
                                 filter->output_frame.width * 4, 
                                 false);
            filter->new_output_ready = false;
        }
    }
    
    // Draw the texture at the source dimensions.
    if (filter->output_texture) {
        gs_effect_t *draw_effect = effect ? effect : obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if (!draw_effect) {
            obs_source_skip_video_filter(filter->context);
            return;
        }
        gs_effect_set_texture(gs_effect_get_param_by_name(draw_effect, "image"),
                              filter->output_texture);
        while (gs_effect_loop(draw_effect, "Draw")) {
            gs_draw_sprite(filter->output_texture, 0, width, height);
        }
    } else {
        obs_source_skip_video_filter(filter->context);
    }
}

static obs_properties_t *codec_properties(void *data) {
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_path(props, S_MODEL_PATH, "Model Path", OBS_PATH_FILE, "SD Models (*.safetensors *.ckpt)", nullptr);
    obs_properties_add_path(props, S_VAE_PATH, "VAE Path (Optional)", OBS_PATH_FILE, "VAE Models (*.safetensors *.ckpt *.pt)", nullptr);

    return props;
}

struct obs_source_info latent_codec_test_filter_info = {
    .id = "latent_codec_test_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = codec_get_name,
    .create = codec_create,
    .destroy = codec_destroy,
    .get_properties = codec_properties,
    .update = codec_update,
    .video_render = codec_video_render,
};
