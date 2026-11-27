#pragma once
#include "framework/vision_engine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
extern "C" {
#include "stable-diffusion.h"
}

namespace lll::engines {

// This adapter uses stable-diffusion.cpp's opaque C API (2026). It supports
// both prompt-only and image-conditioned generation. It never owns OBS
// textures; callers must run expensive work away from OBS's render thread.
class StableDiffusionBackend final : public framework::VisionEngine {
public:
    ~StableDiffusionBackend() override { shutdown(); }

    bool initialize(const EngineConfig& config) override {
        shutdown();
        if (config.model_path.empty() && config.diffusion_model_path.empty())
            return fail("A model or diffusion component path is required");
        Options options(sd_ctx_opt_create(), &sd_opt_free);
        if (!options) return fail("Unable to allocate context options");
        if (!config.model_path.empty() && !set(options.get(), "model_path", config.model_path)) return false;
        if (!config.diffusion_model_path.empty() && !set(options.get(), "diffusion_model_path", config.diffusion_model_path)) return false;
        if (!config.high_noise_diffusion_model_path.empty() && !set(options.get(), "high_noise_diffusion_model_path", config.high_noise_diffusion_model_path)) return false;
        if (!config.clip_l_path.empty() && !set(options.get(), "clip_l_path", config.clip_l_path)) return false;
        if (!config.clip_g_path.empty() && !set(options.get(), "clip_g_path", config.clip_g_path)) return false;
        if (!config.t5xxl_path.empty() && !set(options.get(), "t5xxl_path", config.t5xxl_path)) return false;
        if (!config.llm_path.empty() && !set(options.get(), "llm_path", config.llm_path)) return false;
        if (!config.vae_path.empty() && !set(options.get(), "vae_path", config.vae_path)) return false;
        if (!config.taesd_path.empty() && !set(options.get(), "taesd_path", config.taesd_path)) return false;
        if (!config.controlnet_path.empty() && !set(options.get(), "control_net_path", config.controlnet_path)) return false;
        if (config.num_threads > 0 && !set(options.get(), "n_threads", config.num_threads)) return false;
        if (config.flash_attn && !set(options.get(), "diffusion_flash_attn", true)) return false;
        if (!config.backend_device.empty()) {
            if (!set(options.get(), "backend", config.backend_device)) return false;
        } else if (!config.use_gpu && !set(options.get(), "backend", std::string("cpu"))) return false;

        if (!config.weight_type.empty() &&
            !set(options.get(),"wtype",config.weight_type)) return false;
        if (!config.conv_direct_mode.empty()) {
            if(config.conv_direct_mode!="auto" && config.conv_direct_mode!="on" &&
               config.conv_direct_mode!="off") return fail("Invalid conv_direct_mode");
            if(!set(options.get(),"conv_direct_mode",config.conv_direct_mode)) return false;
        }
        sd_ctx_t* created = nullptr;
        if (sd_ctx_create(options.get(), &created) != SD_STATUS_OK || !created) {
            return fail(upstream_error("Failed to load diffusion model"));
        }
        ctx_ = created;
        if (!config.controlnet_path.empty() && !sd_ctx_has_control_net(ctx_)) {
            shutdown();
            return fail("Requested ControlNet is not active");
        }
        lora_path_ = config.lora_path;
        if (progress_callback_)
            sd_set_generation_progress_callback(ctx_, &progress_trampoline, this);
        last_error_.clear();
        return true;
    }

    void shutdown() override {
        // The caller must serialize this with generate() and model reload.
        if (ctx_) {
            sd_set_generation_progress_callback(ctx_, nullptr, nullptr);
            sd_ctx_free(ctx_);
            ctx_ = nullptr;
        }
        lora_path_.clear();
    }

    bool is_ready() const override { return ctx_ != nullptr; }

    ImageData generate(const GenerationParams& params) override {
        ImageData output{};
        if (!ctx_ || !validate(params)) return output;
        Options options(sd_img_gen_opt_create(), &sd_opt_free);
        if (!options) { fail("Unable to allocate image options"); return output; }
        if (!set(options.get(), "prompt", params.prompt) ||
            !set(options.get(), "negative_prompt", params.negative_prompt) ||
            !set(options.get(), "width", params.width) ||
            !set(options.get(), "height", params.height) ||
            !set(options.get(), "seed", params.seed) ||
            !set(options.get(), "sample.steps", params.steps) ||
            !set(options.get(), "sample.guidance.txt_cfg", params.cfg_scale)) return output;
        if (params.init_image && !set_image(options.get(), "init_image", params.init_image, params.width, params.height)) return output;
        if (params.init_image) {
            // CompVis maps noise level n to sigma=n/(1-n). Bound the
            // explicit noise level below 1 and near its trained range.
            const float strength=params.continuous_denoise?std::min(params.strength,.93f):params.strength;
            if (!set(options.get(),"strength",strength)) return output;
            if(params.continuous_denoise &&
                !set(options.get(),"sample.args",std::string("strength_as_noise_level=true,force_first_sigma=true"))) return output;
        }
        if (params.control_image) {
            if (!set_image(options.get(), "control_image", params.control_image, params.width, params.height) ||
                !set(options.get(), "control_strength", params.control_strength)) return output;
        }
        if (!set_sampler(options.get(), params.sample_method, sampler_names(), "sample.method", params.sample_method_name) ||
            !set_sampler(options.get(), params.scheduler, scheduler_names(), "sample.scheduler", params.scheduler_name)) return output;
        if (params.distilled_guidance >= 0 && !set(options.get(), "sample.guidance.distilled_guidance", params.distilled_guidance)) return output;
        if (!params.cache_mode.empty() && !set(options.get(), "cache.mode", params.cache_mode)) return output;
        if (!lora_path_.empty()) {
            sd_lora_t lora{false, 1.0f, lora_path_.c_str()};
            if (sd_opt_set_loras(options.get(), "loras", &lora, 1) != SD_STATUS_OK) {
                fail(upstream_error("Invalid LoRA")); return output;
            }
        }

        sd_result_t* raw = nullptr;
        if (sd_generate_image(ctx_, options.get(), &raw) != SD_STATUS_OK || !raw) {
            fail(upstream_error("Image generation failed"));
            if (raw) sd_result_free(raw);
            return output;
        }
        std::unique_ptr<sd_result_t, decltype(&sd_result_free)> result(raw, &sd_result_free);
        if (sd_result_get_kind(result.get()) != SD_RESULT_IMAGE || sd_result_get_image_count(result.get()) == 0) {
            fail("Generation returned no image"); return output;
        }
        const sd_tensor_t* tensor = sd_result_get_image(result.get(), 0);
        if (!tensor || sd_tensor_type(tensor) != SD_TENSOR_U8 || sd_tensor_rank(tensor) != 3) {
            fail("Generation returned an unsupported image tensor"); return output;
        }
        const auto channels = sd_tensor_dim(tensor, 0);
        const auto width = sd_tensor_dim(tensor, 1);
        const auto height = sd_tensor_dim(tensor, 2);
        size_t length = 0;
        if ((channels != 3 && channels != 4) || width <= 0 || height <= 0 ||
            width > std::numeric_limits<int>::max() || height > std::numeric_limits<int>::max() ||
            !checked_size(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                          static_cast<uint32_t>(channels), length) ||
            sd_tensor_byte_size(tensor) != length || !sd_tensor_data(tensor)) {
            fail("Generation returned invalid image dimensions or pixels"); return output;
        }
        output.width = static_cast<uint32_t>(width);
        output.height = static_cast<uint32_t>(height);
        output.channels = static_cast<uint32_t>(channels);
        output.data.resize(length);
        std::memcpy(output.data.data(), sd_tensor_data(tensor), length);
        last_error_.clear();
        return output;
    }

    // The current upstream C API does not expose standalone VAE encode/decode.
    // Explicitly unavailable; do not construct a pretend latent or image.
    LatentData encode_latent(const ImageData&) override {
        fail("Standalone VAE encoding is unavailable in the current stable-diffusion.cpp C API");
        return {};
    }
    ImageData decode_latent(const LatentData&) override {
        fail("Standalone VAE decoding is unavailable in the current stable-diffusion.cpp C API");
        return {};
    }
    std::string backend_name() const override { return "stable-diffusion.cpp"; }
    size_t get_vram_usage_mb() const override { return 0; } // no portable telemetry
    const std::string& last_error() const { return last_error_; }
    // Use only from the serialized owner/progress callback. The caller must
    // never race context teardown with this operation.
    void cancel_current_generation() { if(ctx_)sd_cancel_generation(ctx_,SD_CANCEL_ALL); }
    std::string model_version() const { return ctx_?sd_get_model_version_name(ctx_):"Unknown"; }

    void set_progress_callback(ProgressCallback callback) override {
        progress_callback_ = std::move(callback);
        if (ctx_)
            sd_set_generation_progress_callback(ctx_,
                progress_callback_ ? &progress_trampoline : nullptr,
                progress_callback_ ? this : nullptr);
    }

private:
    using Options = std::unique_ptr<sd_opt_t, decltype(&sd_opt_free)>;
    static constexpr std::array<const char*, 12> sampler_names() {
        return {"euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
                "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd"};
    }
    static constexpr std::array<const char*, 10> scheduler_names() {
        return {"discrete", "karras", "exponential", "ays", "gits",
                "sgm_uniform", "simple", "smoothstep", "kl_optimal", "lcm"};
    }
    static bool checked_size(uint32_t w, uint32_t h, uint32_t c, size_t& size) {
        if (!w || !h || !c || w > std::numeric_limits<size_t>::max() / h) return false;
        const size_t pixels = static_cast<size_t>(w) * h;
        if (c > std::numeric_limits<size_t>::max() / pixels) return false;
        size = pixels * c;
        return true;
    }
    bool validate(const GenerationParams& p) {
        size_t length = 0;
        if (p.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            p.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            !checked_size(p.width, p.height, 3, length) || p.steps < 1 ||
            !std::isfinite(p.cfg_scale) || !std::isfinite(p.strength) ||
            !std::isfinite(p.control_strength) || !std::isfinite(p.distilled_guidance) ||
            (p.init_image && (p.strength < 0 || p.strength > 1))) {
            return fail("Invalid image generation parameters");
        }
        return true;
    }
    bool set_image(sd_opt_t* opt, const char* key, const uint8_t* rgb, uint32_t w, uint32_t h) {
        size_t bytes = 0;
        if (!rgb || !checked_size(w, h, 3, bytes)) return fail("Invalid source image");
        const int64_t shape[] = {3, static_cast<int64_t>(w), static_cast<int64_t>(h)};
        std::unique_ptr<sd_tensor_t, decltype(&sd_tensor_free)> tensor(
            sd_tensor_create(SD_TENSOR_U8, 3, shape, rgb, bytes), &sd_tensor_free);
        if (!tensor || sd_opt_set_tensor(opt, key, tensor.get()) != SD_STATUS_OK)
            return fail(upstream_error("Invalid input image"));
        return true;
    }
    template <typename T> bool set(sd_opt_t* opt, const char* key, T value) {
        sd_status_t status;
        if constexpr (std::is_same_v<T, std::string>) status = sd_opt_set_string(opt, key, value.c_str());
        else if constexpr (std::is_same_v<T, bool>) status = sd_opt_set_bool(opt, key, value);
        else if constexpr (std::is_floating_point_v<T>) status = sd_opt_set_float(opt, key, value);
        else status = sd_opt_set_int(opt, key, static_cast<int64_t>(value));
        return status == SD_STATUS_OK || fail(upstream_error(std::string("Invalid option ") + key));
    }
    template <size_t N> bool set_sampler(sd_opt_t* opt, int index,
                                          const std::array<const char*, N>& names, const char* key,
                                          const std::string& explicit_name) {
        if (!explicit_name.empty()) return set(opt, key, explicit_name);
        if (index == -1) return true; // current upstream model-selected default
        if (index < 0 || static_cast<size_t>(index) >= names.size()) return fail("Invalid legacy sampling choice");
        return sd_opt_set_string(opt, key, names[static_cast<size_t>(index)]) == SD_STATUS_OK ||
               fail(upstream_error(std::string("Invalid option ") + key));
    }
    static std::string upstream_error(const std::string& fallback) {
        const char* diagnostic = sd_get_last_error();
        return diagnostic && *diagnostic ? fallback + ": " + diagnostic : fallback;
    }
    bool fail(std::string message) { last_error_ = std::move(message); return false; }
    static void progress_trampoline(const sd_generation_progress_t* event, void* data) {
        auto* self = static_cast<StableDiffusionBackend*>(data);
        if (!self || !event || !self->progress_callback_) return;
        const auto clamp = [](int64_t n) -> int {
            return static_cast<int>(std::clamp<int64_t>(n, 0, std::numeric_limits<int>::max()));
        };
        // The context-local API reports work units, not elapsed seconds.
        self->progress_callback_(clamp(event->completed), clamp(event->total), 0.0f);
    }
    sd_ctx_t* ctx_ = nullptr;
    std::string lora_path_;
    std::string last_error_;
    ProgressCallback progress_callback_;
};

} // namespace lll::engines
