#include <obs.h>
#include <util/bmem.h>
#include <cassert>
#include <iostream>

// Externs for plugin sources (simulating module load)
extern struct obs_source_info image_gen_source_info;
extern struct obs_source_info style_transfer_filter_info;
extern struct obs_source_info neural_enhance_filter_info;
extern struct obs_source_info latent_codec_test_filter_info;
extern struct obs_source_info privacy_filter_info;
extern struct obs_source_info parallax_filter_info;
extern struct obs_source_info mediapipe_filter_info;

void register_sources() {
    obs_register_source(&image_gen_source_info);
    obs_register_source(&style_transfer_filter_info);
    obs_register_source(&neural_enhance_filter_info);
    obs_register_source(&latent_codec_test_filter_info);
    obs_register_source(&privacy_filter_info);
    obs_register_source(&parallax_filter_info);
    obs_register_source(&mediapipe_filter_info);
}

void test_lifecycle() {
    // 1. Initialize libobs core
    if (!obs_startup("en-US", NULL, NULL)) {
        std::cerr << "Failed to start OBS context" << std::endl;
        exit(1);
    }

    // 2. Register sources manually
    register_sources();

    // 3. Create and Destroy object
    // Test: Image Generator Source
    {
        obs_source_t *source = obs_source_create("ai_image_generator", "Test Source", NULL, NULL);
        if (!source) {
             std::cerr << "Failed to create ai_image_generator source" << std::endl;
             obs_shutdown();
             exit(1); // Fail the test if source cannot be created
        } else {
            // Update settings
            obs_data_t *settings = obs_data_create();
            obs_data_set_string(settings, "prompt", "test prompt");
            obs_source_update(source, settings);
            obs_data_release(settings);

            obs_source_release(source);
        }
    }

    // 4. Shutdown OBS
    obs_shutdown();

    // 5. Check for leaks
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
