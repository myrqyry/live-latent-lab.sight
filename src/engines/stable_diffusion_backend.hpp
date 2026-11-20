#pragma once
#include "framework/vision_engine.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
extern "C" {
    #include "stable-diffusion.h"
}

namespace lll {
namespace engines {

class StableDiffusionBackend : public framework::VisionEngine {
public:
    StableDiffusionBackend() = default;
    ~StableDiffusionBackend() override { shutdown(); }
    
    bool initialize(const EngineConfig& config) override {
        shutdown(); // Ensure clean state
        
        // Use struct-based init
        sd_ctx_params_t sd_params;
        sd_ctx_params_init(&sd_params);
        
        sd_params.model_path = config.model_path.c_str();
        sd_params.vae_path = config.vae_path.empty() ? nullptr : config.vae_path.c_str();
        sd_params.taesd_path = config.taesd_path.empty() ? nullptr : config.taesd_path.c_str();
        sd_params.control_net_path = config.controlnet_path.empty() ? nullptr : config.controlnet_path.c_str();
        sd_params.n_threads = config.num_threads;
        sd_params.wtype = SD_TYPE_F16;
        sd_params.rng_type = CUDA_RNG;
        sd_params.diffusion_flash_attn = config.flash_attn;
        
        sd_ctx_ = new_sd_ctx(&sd_params);
        
        return (sd_ctx_ != nullptr);
    }
    
    void shutdown() override {
        sd_set_progress_callback(nullptr, nullptr);
        if (sd_ctx_) {
            free_sd_ctx(sd_ctx_);
            sd_ctx_ = nullptr;
        }
        progress_callback_ = {};
    }
    
    bool is_ready() const override {
        return sd_ctx_ != nullptr;
    }
    
    ImageData generate(const GenerationParams& params) override {
        ImageData result{};
        if (!sd_ctx_ || !valid_generation_params(params)) return result;
        
        sd_image_t* img = nullptr;
        
        // Map GenerationParams to sd_img_gen_params_t
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        gen_params.prompt = params.prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.width = params.width;
        gen_params.height = params.height;
        gen_params.strength = params.strength;
        gen_params.seed = params.seed;
        gen_params.sample_params.sample_steps = params.steps;
        gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
        gen_params.sample_params.sample_method = (params.sample_method >= 0) ? (sample_method_t)params.sample_method : EULER_A_SAMPLE_METHOD;
        gen_params.sample_params.scheduler = (params.scheduler >= 0) ? (scheduler_t)params.scheduler : DISCRETE_SCHEDULER;

        
        if (params.init_image) {
             // Map to gen_params.init_image
             gen_params.init_image.width = params.width;
             gen_params.init_image.height = params.height;
             gen_params.init_image.channel = 3; 
             gen_params.init_image.data = const_cast<uint8_t*>(params.init_image);
        }
        
        if (params.control_image) {
             gen_params.control_image.width = params.width;
             gen_params.control_image.height = params.height;
             gen_params.control_image.channel = 3;
             gen_params.control_image.data = const_cast<uint8_t*>(params.control_image);
             gen_params.control_strength = params.control_strength;
        }

        // Use generate_image which handles both txt2img and img2img
        img = generate_image(sd_ctx_, &gen_params);
        
        if (img) {
            size_t output_size = 0;
            if (img->data && checked_size(img->width, img->height, img->channel, output_size)) {
                result.width = img->width;
                result.height = img->height;
                result.channels = img->channel;
                result.data.resize(output_size);
                std::memcpy(result.data.data(), img->data, result.data.size());
            }

            free(img->data);
            free(img);
        }
        
        return result;
    }
    
    LatentData encode_latent(const ImageData& image) override {
        LatentData result{};
        size_t image_size = 0;
        if (!sd_ctx_ || !valid_image(image) ||
            !checked_size(image.width, image.height, image.channels, image_size) ||
            image.data.size() < image_size) {
            return result;
        }
        
        sd_image_t sd_img;
        sd_img.width = image.width;
        sd_img.height = image.height;
        sd_img.channel = image.channels;
        sd_img.data = const_cast<uint8_t*>(image.data.data()); // Unsafe const cast but SD API expects non-const
        
        sd_latent_t* lat = sd_vae_encode(sd_ctx_, &sd_img);
        if (lat) {
            size_t latent_size = 0;
            if (!lat->data || !checked_size(lat->width, lat->height, lat->channel, latent_size)) {
                sd_free_latent(lat);
                return result;
            }
            result.width = lat->width;
            result.height = lat->height;
            result.channels = lat->channel;
            result.data.resize(latent_size);
            std::memcpy(result.data.data(), lat->data, result.data.size() * sizeof(float));
            sd_free_latent(lat);
        }
        return result;
    }

    ImageData decode_latent(const LatentData& latent) override {
        ImageData result{};
        size_t latent_size = 0;
        if (!sd_ctx_ || !valid_latent(latent) ||
            !checked_size(latent.width, latent.height, latent.channels, latent_size) ||
            latent.data.size() < latent_size) {
            return result;
        }
        
        sd_latent_t sd_lat;
        sd_lat.width = latent.width;
        sd_lat.height = latent.height;
        sd_lat.channel = latent.channels;
        sd_lat.data = const_cast<float*>(latent.data.data());
        
        sd_image_t* img = sd_vae_decode(sd_ctx_, &sd_lat);
        if (img) {
            size_t output_size = 0;
            if (img->data && checked_size(img->width, img->height, img->channel, output_size)) {
                result.width = img->width;
                result.height = img->height;
                result.channels = img->channel;
                result.data.resize(output_size);
                std::memcpy(result.data.data(), img->data, result.data.size());
            }
            free(img->data); // sd_vae_decode allocates data with calloc/tensor conversion
            free(img);
        }
        return result;
    }
    
    std::string backend_name() const override {
        return "stable-diffusion.cpp";
    }
    
    size_t get_vram_usage_mb() const override {
        // stable-diffusion.cpp does not expose portable VRAM telemetry.
        return 0;
    }

    void set_progress_callback(ProgressCallback callback) override {
        progress_callback_ = std::move(callback);
        sd_set_progress_callback(progress_callback_ ? &progress_trampoline : nullptr,
                                 progress_callback_ ? this : nullptr);
    }

private:
    static bool checked_size(uint32_t width, uint32_t height, uint32_t channels, size_t& size) {
        if (width == 0 || height == 0 || channels == 0 ||
            static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height) {
            return false;
        }
        size_t pixels = static_cast<size_t>(width) * height;
        if (static_cast<size_t>(channels) > std::numeric_limits<size_t>::max() / pixels) {
            return false;
        }
        size = pixels * channels;
        return true;
    }

    static bool valid_image(const ImageData& image) {
        return image.channels == 3 || image.channels == 4;
    }

    static bool valid_latent(const LatentData& latent) {
        return latent.channels > 0;
    }

    static bool valid_generation_params(const GenerationParams& params) {
        constexpr uint32_t max_dimension = static_cast<uint32_t>(std::numeric_limits<int>::max());
        return params.width > 0 && params.height > 0 &&
               params.width <= max_dimension && params.height <= max_dimension &&
               params.steps > 0 && std::isfinite(params.cfg_scale) &&
               std::isfinite(params.strength) &&
               (!params.init_image || params.width <= max_dimension) &&
               (!params.control_image || params.width <= max_dimension);
    }

    static void progress_trampoline(int step, int steps, float time, void* data) {
        auto* backend = static_cast<StableDiffusionBackend*>(data);
        if (backend && backend->progress_callback_) {
            backend->progress_callback_(step, steps, time);
        }
    }

    sd_ctx_t* sd_ctx_ = nullptr;
    ProgressCallback progress_callback_;
};

} // namespace engines
} // namespace lll
