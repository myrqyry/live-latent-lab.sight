#include <obs.h>
#include <util/bmem.h>
#include <iostream>
#include <cassert>

// Extern declarations for sources/filters
extern struct obs_source_info style_transfer_filter_info;
extern struct obs_source_info image_gen_source_info;
extern struct obs_source_info neural_enhance_filter_info;
extern struct obs_source_info latent_codec_test_filter_info;
extern struct obs_source_info privacy_filter_info;
extern struct obs_source_info parallax_filter_info;
extern struct obs_source_info mediapipe_filter_info;

void register_my_sources() {
    obs_register_source(&style_transfer_filter_info);
    obs_register_source(&image_gen_source_info);
    obs_register_source(&neural_enhance_filter_info);
    obs_register_source(&latent_codec_test_filter_info);
    obs_register_source(&privacy_filter_info);
    obs_register_source(&parallax_filter_info);
    obs_register_source(&mediapipe_filter_info);
}

void test_lifecycle() {
    std::cout << "Starting Memory Leak Test..." << std::endl;

    // 1. Initialize libobs core
    if (!obs_startup("en-US", NULL, NULL)) {
        std::cerr << "Failed to start OBS context" << std::endl;
        exit(1);
    }

    // 2. Register your plugin components manually
    register_my_sources();

    // 3. Create and Destroy your object (Test Image Gen Source)
    {
        obs_source_t *source = obs_source_create("ai_image_generator", "Test Source", NULL, NULL);
        if (!source) {
             std::cerr << "Failed to create source: ai_image_generator" << std::endl;
        } else {
            // Simulate some usage
            obs_data_t *settings = obs_data_create();
            obs_data_set_string(settings, "prompt", "test prompt");
            obs_source_update(source, settings);
            obs_data_release(settings);

            // Destroy
            obs_source_release(source);
        }
    }

    // Test Style Transfer Filter
    {
         obs_source_t *filter = obs_source_create("style_transfer_filter", "Test Filter", NULL, NULL);
         if(filter) {
             obs_source_release(filter);
         }
    }

    // 4. Shutdown OBS
    obs_shutdown();

    // 5. Verification: Check the leak counter
    long leaks = bnum_allocs();
    if (leaks > 0) {
        std::cerr << "TEST FAILED: " << leaks << " memory allocations leaked!" << std::endl;
        exit(1);
    } else {
        std::cout << "TEST PASSED: 0 Leaks detected." << std::endl;
    }
}

int main() {
    test_lifecycle();
    return 0;
}
