#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/dstr.h>
#include <util/platform.h>
#include <util/threading.h>
#include <media-io/video-io.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "core/latent-engine.hpp"
#include "common/image-utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "common/stb_image.h"

#define S_MODEL_PATH "model_path"
#define S_PROMPT "prompt"
#define S_NEGATIVE_PROMPT "negative_prompt"
#define S_INIT_IMAGE "init_image"
#define S_STEPS "steps"
#define S_CFG_SCALE "cfg_scale"
#define S_SEED "seed"
#define S_WIDTH "width"
#define S_HEIGHT "height"
#define S_BUTTON_GENERATE "generate_button"

#define S_ENDLESS_MODE "endless_mode"
#define S_ZOOM "zoom_factor"
#define S_PAN_X "pan_x"
#define S_PAN_Y "pan_y"
#define S_SAVE_EVERY "save_every_frame"
#define S_OUTPUT_DIR "output_dir"
#define S_STRENGTH "denoising_strength"

struct image_gen_data {
    obs_source_t *context;
    
    // Properties
    std::string model_path;
    std::string prompt;
    std::string negative_prompt;
    std::string init_image_path;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    int width = 512;
    int height = 512;
    
    // Endless mode properties
    bool endless_mode = false;
    float zoom_factor = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    bool save_every_frame = false;
    std::string output_dir;
    float strength = 0.7f;

    // State
    gs_texture_t *texture = nullptr;
    std::vector<uint8_t> texture_data;
    uint32_t texture_width = 0;
    uint32_t texture_height = 0;
    bool new_texture_ready = false;
    
    // Endless mode state
    std::vector<uint8_t> endless_feedback_buffer; // Stores the previous frame for loopback

    // Worker
    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};
    std::atomic<bool> trigger_generation{false};
    std::mutex curr_mutex; // Protects data exchanged with worker (only path/prompt/etc copies)
    std::mutex param_mutex; // Protects worker_job parameters
    
    // Internal copies for the worker to avoid locking the main struct too long
    struct {
         std::string model_path;
         std::string prompt;
         std::string negative_prompt;
         std::string init_image_path;
         int steps;
         float cfg_scale;
         int64_t seed;
         int width;
         int height;

         bool endless_mode;
         TransformParams transform;
         bool save_every_frame;
         std::string output_dir;
         float strength;
    } worker_job;
    
    obs_hotkey_id hotkey_gen = OBS_INVALID_HOTKEY_ID;
    
    mutable std::mutex status_mutex;
    std::string status_buffer;
};

// -------------------------------------------------------------------------------------------------
// WORKER THREAD
// -------------------------------------------------------------------------------------------------

static void set_status(image_gen_data *data, const char *msg) {
    std::lock_guard<std::mutex> lock(data->status_mutex);
    data->status_buffer = msg ? msg : "";
}

// Simple PPM Writer (Uncompressed RGB)
static bool save_ppm(const std::string& path, int width, int height, const uint8_t* data) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    // P6 header
    ofs << "P6\n" << width << " " << height << "\n255\n";

    // Write pixel data (convert RGBA to RGB)
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgb[i*3+0] = data[i*4+0];
        rgb[i*3+1] = data[i*4+1];
        rgb[i*3+2] = data[i*4+2];
    }

    ofs.write((const char*)rgb.data(), rgb.size());
    return true;
}

static void worker_func(image_gen_data *data) {
    blog(LOG_INFO, "[Live Latent Lab: Sight] Generator worker thread started");
    
    int frame_count = 0;

    while (!data->stop_thread) {
        if (data->trigger_generation) {
            
            // Lock and copy params safely
            decltype(data->worker_job) job;
            {
                std::lock_guard<std::mutex> lock(data->param_mutex);
                job = data->worker_job;
            }

            if (!job.endless_mode) {
                 data->trigger_generation = false;
            }

            set_status(data, "Generating...");
            
            // Check if model needs loading
            lll::EngineParams engine_params;
            engine_params.model_path = job.model_path;
            // engine_params.rng_type = CUDA_RNG; // Not in generic interface, uses backend default
            
            if (!lll::LatentEngine::get().load_model(engine_params)) {
                set_status(data, "Failed to load model");
                data->trigger_generation = false; // Stop endless loop on error
                continue;
            }
            
            set_status(data, "Generating Image...");
            
            lll::framework::VisionEngine::GenerationParams gen_params;
            gen_params.prompt = job.prompt;
            gen_params.negative_prompt = job.negative_prompt;
            gen_params.width = job.width;
            gen_params.height = job.height;
            gen_params.steps = job.steps;
            gen_params.cfg_scale = job.cfg_scale;
            gen_params.seed = job.seed;
            gen_params.strength = job.strength;
            
            if (gen_params.seed < 0) {
                gen_params.seed = std::chrono::system_clock::now().time_since_epoch().count() % INT64_MAX;
            }
            
            // Init Image Logic
            std::vector<uint8_t> init_image_rgb;
            uint8_t* stbi_data = nullptr;

            if (job.endless_mode && !data->endless_feedback_buffer.empty()) {
                // Use feedback buffer (already transformed)
                gen_params.init_image = data->endless_feedback_buffer.data();
            } else if (!job.init_image_path.empty()) {
                int iw, ih, ic;
                // Force 4 channels for consistency with our resize tools, though we need 3 for SD
                stbi_data = stbi_load(job.init_image_path.c_str(), &iw, &ih, &ic, 4);

                if (stbi_data) {
                    if (iw != job.width || ih != job.height) {
                        // Resize needed
                        std::vector<uint8_t> resized(job.width * job.height * 4);
                        if (resize_bilinear(stbi_data, iw * ih * 4, iw, ih, resized.data(), resized.size(), job.width, job.height)) {
                            // Convert RGBA to RGB for Stable Diffusion
                            init_image_rgb.resize(job.width * job.height * 3);
                            lll::convert_rgba_to_rgb(resized.data(), init_image_rgb.data(), job.width * job.height);
                            gen_params.init_image = init_image_rgb.data();
                        } else {
                            blog(LOG_WARNING, "[Sight] Resize failed for init image");
                        }
                    } else {
                        // Use directly (convert to RGB)
                        init_image_rgb.resize(job.width * job.height * 3);
                        lll::convert_rgba_to_rgb(stbi_data, init_image_rgb.data(), job.width * job.height);
                        gen_params.init_image = init_image_rgb.data();
                    }

                    stbi_image_free(stbi_data);
                } else {
                    blog(LOG_WARNING, "[Sight] Failed to load init image: %s", job.init_image_path.c_str());
                }
            }

            auto result = lll::LatentEngine::get().generate(gen_params);
                           
            if (!result.data.empty()) {
                  {
                      std::lock_guard<std::mutex> lock(data->curr_mutex);
                      // Convert RGB to RGBA
                      data->texture_data.resize(result.width * result.height * 4);
                      
                      const uint8_t* src = result.data.data();
                      uint8_t* dst = data->texture_data.data();
                      size_t pixel_count = result.width * result.height;
                      
                      if (result.channels == 3) {
                          for (size_t i=0; i < pixel_count; ++i) {
                               dst[0] = src[0];
                               dst[1] = src[1];
                               dst[2] = src[2];
                               dst[3] = 255;
                               src += 3;
                               dst += 4;
                          }
                      } else { // Assume 4 channels (RGBA)
                          std::memcpy(dst, src, pixel_count * 4);
                      }
                      
                      data->texture_width = result.width;
                      data->texture_height = result.height;
                      data->new_texture_ready = true;
                  }
                  
                  // Endless Mode Logic
                  if (job.endless_mode) {
                      // 1. Save if requested
                      if (job.save_every_frame && !job.output_dir.empty()) {
                          std::stringstream ss;
                          ss << job.output_dir << "/frame_" << std::setfill('0') << std::setw(5) << frame_count++ << ".ppm";
                          save_ppm(ss.str(), result.width, result.height, data->texture_data.data());
                      }

                      // 2. Prepare Feedback Buffer
                      if (data->endless_feedback_buffer.size() != data->texture_data.size()) {
                          data->endless_feedback_buffer.resize(data->texture_data.size());
                      }

                      // 3. Apply Transform (Zoom/Pan)
                      transform_image(
                          data->texture_data.data(), data->texture_data.size(),
                          result.width, result.height,
                          data->endless_feedback_buffer.data(), data->endless_feedback_buffer.size(),
                          job.transform
                      );

                      // 4. Continue loop (implicitly via trigger_generation staying true)
                  } else {
                      data->trigger_generation = false;
                  }

            } else {
                // Generation failed
                data->trigger_generation = false;
            }
            
            if (!job.endless_mode) {
                set_status(data, "Idle");
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// OBS CALLBACKS
// -------------------------------------------------------------------------------------------------

static const char *get_name(void *unused) { return "LLL.sight Generator Source"; }

static void *create(obs_data_t *settings, obs_source_t *context) {
    image_gen_data *data = new image_gen_data();
    data->context = context;
    
    // Init worker
    data->worker_thread = std::thread(worker_func, data);
    
    // Hotkey
    auto hotkey_cb = [](void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
        if (pressed) {
            auto *d = (image_gen_data*)data;
            d->trigger_generation = true;
        }
    };
    data->hotkey_gen = obs_hotkey_register_source(context, "ai_image_gen.trigger", "Generate AI Image", hotkey_cb, data);
    
    obs_source_update(context, settings);
    return data;
}

static void destroy(void *data) {
    image_gen_data *d = (image_gen_data*)data;
    d->stop_thread = true;
    if (d->worker_thread.joinable()) d->worker_thread.join();
    // LatentEngine handles its own cleanup since it's shared
    obs_enter_graphics();
    if (d->texture) gs_texture_destroy(d->texture);
    obs_leave_graphics();
    delete d;
}

static void update(void *data, obs_data_t *settings) {
    image_gen_data *d = (image_gen_data*)data;
    
    std::string model_path = obs_data_get_string(settings, S_MODEL_PATH);
    std::string init_image_path = obs_data_get_string(settings, S_INIT_IMAGE);
    std::string output_dir = obs_data_get_string(settings, S_OUTPUT_DIR);
    
    // Path validation
    auto validate_path = [](const std::string& path) -> bool {
        if (path.empty()) return true; // Allow empty
        if (path.find("..") != std::string::npos) return false;
        if (path[0] == '/' || path[0] == '\\') return false;
        return true;
    };
    
    if (!validate_path(model_path)) {
        blog(LOG_WARNING, "[Live Latent Lab: Sight] Invalid model path: %s", model_path.c_str());
        model_path.clear();
    }
    if (!validate_path(init_image_path)) {
        blog(LOG_WARNING, "[Live Latent Lab: Sight] Invalid init image path: %s", init_image_path.c_str());
        init_image_path.clear();
    }
    if (!validate_path(output_dir)) {
        blog(LOG_WARNING, "[Live Latent Lab: Sight] Invalid output dir: %s", output_dir.c_str());
        output_dir.clear();
    }
    
    d->model_path = model_path;
    d->prompt = obs_data_get_string(settings, S_PROMPT);
    d->negative_prompt = obs_data_get_string(settings, S_NEGATIVE_PROMPT);
    d->init_image_path = init_image_path;
    d->steps = obs_data_get_int(settings, S_STEPS);
    d->cfg_scale = (float)obs_data_get_double(settings, S_CFG_SCALE);
    d->seed = obs_data_get_int(settings, S_SEED);
    d->width = obs_data_get_int(settings, S_WIDTH);
    d->height = obs_data_get_int(settings, S_HEIGHT);
    
    // Endless properties
    d->endless_mode = obs_data_get_bool(settings, S_ENDLESS_MODE);
    d->zoom_factor = (float)obs_data_get_double(settings, S_ZOOM);
    d->pan_x = (float)obs_data_get_double(settings, S_PAN_X);
    d->pan_y = (float)obs_data_get_double(settings, S_PAN_Y);
    d->save_every_frame = obs_data_get_bool(settings, S_SAVE_EVERY);
    d->output_dir = output_dir;
    d->strength = (float)obs_data_get_double(settings, S_STRENGTH);

    // Update worker job
    std::lock_guard<std::mutex> lock(d->param_mutex);

    d->worker_job.model_path = d->model_path;
    d->worker_job.prompt = d->prompt;
    d->worker_job.negative_prompt = d->negative_prompt;
    d->worker_job.init_image_path = d->init_image_path;
    d->worker_job.steps = d->steps;
    d->worker_job.cfg_scale = d->cfg_scale;
    d->worker_job.seed = d->seed;
    d->worker_job.width = d->width;
    d->worker_job.height = d->height;

    d->worker_job.endless_mode = d->endless_mode;
    d->worker_job.transform.zoom_factor = d->zoom_factor;
    d->worker_job.transform.pan_x = d->pan_x;
    d->worker_job.transform.pan_y = d->pan_y;
    d->worker_job.save_every_frame = d->save_every_frame;
    d->worker_job.output_dir = d->output_dir;
    d->worker_job.strength = d->strength;
}

static bool generate_clicked(obs_properties_t *props, obs_property_t *p, void *data) {
    image_gen_data *d = (image_gen_data*)data;
    d->trigger_generation = true;
    return false;
}

static obs_properties_t *get_properties(void *data) {
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_path(props, S_MODEL_PATH, "Model Path", OBS_PATH_FILE, "*.safetensors *.gguf", NULL);
    obs_properties_add_text(props, S_PROMPT, "Prompt", OBS_TEXT_MULTILINE);
    obs_properties_add_text(props, S_NEGATIVE_PROMPT, "Negative Prompt", OBS_TEXT_DEFAULT);
    obs_properties_add_path(props, S_INIT_IMAGE, "Init Image (Optional)", OBS_PATH_FILE, "*.png *.jpg *.jpeg", NULL);
    
    obs_properties_add_int(props, S_WIDTH, "Width", 256, 1024, 64);
    obs_properties_add_int(props, S_HEIGHT, "Height", 256, 1024, 64);
    obs_properties_add_int(props, S_STEPS, "Steps", 1, 50, 1);
    obs_properties_add_float_slider(props, S_CFG_SCALE, "CFG Scale", 1.0, 20.0, 0.5);
    obs_properties_add_int(props, S_SEED, "Seed (-1 = random)", -1, 2147483647, 1);
    
    // Endless Mode Group
    obs_properties_t *grp = obs_properties_create();
    obs_properties_add_group(props, "endless_group", "Endless Morphing Mode", OBS_GROUP_NORMAL, grp);

    obs_properties_add_bool(grp, S_ENDLESS_MODE, "Enable Endless Mode");
    obs_properties_add_float_slider(grp, S_ZOOM, "Zoom Factor (1.0 = None)", 0.9, 1.1, 0.001);
    obs_properties_add_float(grp, S_PAN_X, "Pan X (Pixels)", -100.0, 100.0, 0.1);
    obs_properties_add_float(grp, S_PAN_Y, "Pan Y (Pixels)", -100.0, 100.0, 0.1);

    obs_properties_add_float_slider(grp, S_STRENGTH, "Denoising Strength (0.1=Little Change, 0.9=Big Change)", 0.0, 1.0, 0.05);

    obs_properties_add_bool(grp, S_SAVE_EVERY, "Save Every Frame");
    obs_properties_add_path(grp, S_OUTPUT_DIR, "Output Directory", OBS_PATH_DIRECTORY, NULL, NULL);

    obs_properties_add_button(props, S_BUTTON_GENERATE, "Start / Generate", generate_clicked);
    
    return props;
}

static uint32_t get_width(void *data) { return ((image_gen_data*)data)->width; }
static uint32_t get_height(void *data) { return ((image_gen_data*)data)->height; }

static void video_render(void *data, gs_effect_t *effect) {
    image_gen_data *d = (image_gen_data*)data;
    
    std::lock_guard<std::mutex> lock(d->curr_mutex);
    if (d->new_texture_ready) {
        // Note: video_render is already inside graphics context
        if (d->texture) gs_texture_destroy(d->texture);
        d->texture = gs_texture_create(d->texture_width, d->texture_height, GS_RGBA, 1, (const uint8_t**)d->texture_data.data(), GS_DYNAMIC);
        if (!d->texture) {
            blog(LOG_ERROR, "[Live Latent Lab: Sight] Failed to create texture");
        }
        d->new_texture_ready = false;
    }
    
    if (d->texture) {
        if (!effect) effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), d->texture);
        while (gs_effect_loop(effect, "Draw")) {
             gs_matrix_push();
             gs_matrix_identity();
             gs_draw_sprite(d->texture, 0, d->width, d->height);
             gs_matrix_pop();
        }
    } else {
        // Draw placeholder?
        vec4 color;
        vec4_set(&color, 0.2f, 0.2f, 0.2f, 1.0f);
        gs_clear(GS_CLEAR_COLOR, &color, 0.0f, 0);
    }
}

struct obs_source_info image_gen_source_info = {
    .id = "ai_image_generator",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = get_name,
    .create = create,
    .destroy = destroy,
    .get_width = get_width,
    .get_height = get_height,
    .get_properties = get_properties,
    .update = update,
    .video_render = video_render,
};
