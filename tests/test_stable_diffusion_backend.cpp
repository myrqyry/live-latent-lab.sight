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

    backend.set_progress_callback([](int, int, float) {});
    assert(backend.get_vram_usage_mb() == 0);
    backend.shutdown();
    return 0;
}
