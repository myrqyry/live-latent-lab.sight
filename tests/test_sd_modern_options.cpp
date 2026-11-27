#include "stable-diffusion.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

int main() {
    using Opt = std::unique_ptr<sd_opt_t, decltype(&sd_opt_free)>;
    Opt model(sd_ctx_opt_create(), &sd_opt_free);
    Opt img(sd_img_gen_opt_create(), &sd_opt_free);
    assert(model && img);
    // Modern split model components must be accepted on context options.
    assert(sd_opt_set_string(model.get(), "diffusion_model_path", "/does-not-exist.gguf") == SD_STATUS_OK);
    assert(sd_opt_set_string(model.get(), "clip_l_path", "/does-not-exist-clip.safetensors") == SD_STATUS_OK);
    assert(sd_opt_set_string(model.get(), "vae_path", "/does-not-exist-vae.safetensors") == SD_STATUS_OK);
    assert(sd_opt_set_string(img.get(), "sample.method", "er_sde") == SD_STATUS_OK);
    assert(sd_opt_set_string(img.get(), "sample.scheduler", "flux2") == SD_STATUS_OK);
    assert(sd_opt_set_float(img.get(), "sample.guidance.distilled_guidance", 3.5) == SD_STATUS_OK);
    assert(sd_opt_set_string(img.get(), "cache.mode", "easycache") == SD_STATUS_OK);
    assert(sd_opt_set_string(model.get(),"wtype","q8_0")==SD_STATUS_OK);
    assert(sd_opt_set_string(model.get(),"conv_direct_mode","off")==SD_STATUS_OK);
    assert(sd_opt_set_string(img.get(),"sample.args","strength_as_noise_level=true,force_first_sigma=true")==SD_STATUS_OK);
    assert(sd_opt_set_string(img.get(), "a_nonexistent_feature", "nope") == SD_STATUS_UNKNOWN_KEY);

    // Real upstream Canny API operates on typed RGB tensors, not the old struct.
    constexpr int W = 64, H = 64;
    std::vector<uint8_t> pixels(W * H * 3, 0);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        if (x >= W / 2) {
            auto index = static_cast<size_t>((y * W + x) * 3);
            pixels[index] = pixels[index + 1] = pixels[index + 2] = 255;
        }
    }
    const int64_t shape[] = {3, W, H};
    std::unique_ptr<sd_tensor_t, decltype(&sd_tensor_free)> input(
        sd_tensor_create(SD_TENSOR_U8, 3, shape, pixels.data(), pixels.size()), &sd_tensor_free);
    assert(input);
    sd_tensor_t* canny_raw = nullptr;
    assert(sd_preprocess_canny(input.get(), 120, 60, 255, 255, false, &canny_raw) == SD_STATUS_OK);
    std::unique_ptr<sd_tensor_t, decltype(&sd_tensor_free)> canny(canny_raw, &sd_tensor_free);
    assert(canny && sd_tensor_type(canny.get()) == SD_TENSOR_U8);
    assert(sd_tensor_rank(canny.get()) == 3);
    assert(sd_tensor_dim(canny.get(), 0) == 3);
    assert(sd_tensor_dim(canny.get(), 1) == W);
    assert(sd_tensor_dim(canny.get(), 2) == H);
    assert(sd_tensor_byte_size(canny.get()) == pixels.size());
    assert(sd_tensor_data(canny.get()));
    return 0;
}
