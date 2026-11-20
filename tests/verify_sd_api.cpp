#include <iostream>
#include <vector>

// Wrap C header to prevent C++ name mangling issues
extern "C" {
    #include "../libs/stable-diffusion.cpp/stable-diffusion.h"
}

int main() {
    std::cout << "Verifying Stable Diffusion API..." << std::endl;

    // Test Context Creation
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = "test_model.gguf";
    ctx_params.wtype = SD_TYPE_F16;
    ctx_params.rng_type = STD_DEFAULT_RNG;
    
    // We won't actually run this, just compile check
    // sd_ctx_t* ctx = new_sd_ctx(&ctx_params); 

    // Test Image Generation Params
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    
    gen_params.width = 512;
    gen_params.height = 512;
    gen_params.batch_count = 1;
    gen_params.prompt = "test prompt";
    
    gen_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    gen_params.sample_params.sample_steps = 20;
    gen_params.sample_params.guidance.txt_cfg = 7.0f;
    
    // Test Img2Img fields
    gen_params.init_image.width = 512;
    gen_params.init_image.height = 512;
    gen_params.init_image.channel = 3;
    gen_params.init_image.data = nullptr;
    gen_params.strength = 0.5f;

    // If this compiles, our struct usage matches the header's expectation.
    // generate_image(ctx, &gen_params);

    std::cout << "API structures match!" << std::endl;
    return 0;
}
