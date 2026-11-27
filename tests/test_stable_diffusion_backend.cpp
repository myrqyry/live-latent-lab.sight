#include "engines/stable_diffusion_backend.hpp"

#include <cassert>

using Backend = lll::engines::StableDiffusionBackend;
using ImageData = lll::framework::VisionEngine::ImageData;
using LatentData = lll::framework::VisionEngine::LatentData;

int main() {
    Backend backend;

    ImageData invalid_image{};
    invalid_image.width = 64;
    invalid_image.height = 64;
    invalid_image.channels = 3;
    assert(backend.encode_latent(invalid_image).data.empty());

    LatentData invalid_latent{};
    invalid_latent.width = 8;
    invalid_latent.height = 8;
    invalid_latent.channels = 4;
    assert(backend.decode_latent(invalid_latent).data.empty());

    lll::framework::VisionEngine::GenerationParams invalid_generation{};
    invalid_generation.width = 0;
    assert(backend.generate(invalid_generation).data.empty());

    assert(!backend.initialize({}));
    assert(!backend.last_error().empty());

    // Missing weights must fail closed through the real upstream loader.
    Backend::EngineConfig missing_model{};
    missing_model.model_path = "/definitely-not-a-real-sight-checkpoint-20261121.safetensors";
    missing_model.use_gpu = false;
    assert(!backend.initialize(missing_model));
    assert(!backend.is_ready());
    assert(!backend.last_error().empty());

    // Invalid experimental execution options must fail before attempting
    // to load a model, and must not silently fall back to auto.
    missing_model.weight_type = "invalid";
    assert(!backend.initialize(missing_model));
    assert(backend.last_error().find("wtype") != std::string::npos);
    missing_model.weight_type.clear();
    missing_model.conv_direct_mode = "invalid";
    assert(!backend.initialize(missing_model));
    assert(backend.last_error().find("conv_direct_mode") != std::string::npos);
    missing_model.conv_direct_mode.clear();

    // Modern optional split-model configuration is forwarded to upstream.
    missing_model.model_path.clear();
    missing_model.diffusion_model_path = "/definitely-not-a-real-diffusion-component.gguf";
    assert(!backend.initialize(missing_model));
    assert(!backend.is_ready());

    assert(backend.encode_latent(invalid_image).data.empty());
    assert(backend.last_error().find("unavailable") != std::string::npos);
    backend.set_progress_callback([](int, int, float) {});
    assert(backend.get_vram_usage_mb() == 0);
    backend.shutdown();
    return 0;
}
