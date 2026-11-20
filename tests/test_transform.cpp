#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../src/common/image-utils.hpp"

// Helper to create a test image (checkerboard)
std::vector<uint8_t> create_test_image(int width, int height) {
    std::vector<uint8_t> img(width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool white = ((x / 32) + (y / 32)) % 2 == 0;
            int idx = (y * width + x) * 4;
            img[idx+0] = white ? 255 : 0; // R
            img[idx+1] = white ? 255 : 0; // G
            img[idx+2] = white ? 255 : 0; // B
            img[idx+3] = 255;             // A
        }
    }
    return img;
}

// Helper to check pixel value
bool check_pixel(const std::vector<uint8_t>& img, int width, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    int idx = (y * width + x) * 4;
    return img[idx+0] == r && img[idx+1] == g && img[idx+2] == b;
}

int main() {
    std::cout << "Running Transform Logic Tests..." << std::endl;

    int w = 512;
    int h = 512;
    auto src = create_test_image(w, h);
    std::vector<uint8_t> dst(src.size());

    // Test 1: Identity (No Zoom, No Pan)
    {
        TransformParams params;
        params.zoom_factor = 1.0f;
        params.pan_x = 0.0f;
        params.pan_y = 0.0f;

        bool ok = transform_image(src.data(), src.size(), w, h, dst.data(), dst.size(), params);
        assert(ok);

        float diff = calculate_image_difference(src.data(), dst.data(), src.size());
        std::cout << "Test 1 (Identity): Diff = " << diff << std::endl;
        assert(diff == 0.0f);
    }

    // Test 2: Pan Right (Shift content to right, so we see left part of image moving right)
    // Actually, pan_x positive means "Camera moves Right", which means image content moves LEFT.
    // Wait, let's verify the logic in image-utils.hpp:
    // float src_x = (x - cx - pan_x) / scale + cx;
    // If pan_x is +100: src_x = (x - cx - 100) ...
    // So for destination pixel x=center, we sample source pixel (center - 100).
    // This means we are "pulling" from the left.
    // So if we have content at x=100, and we want it at x=200 (shift right),
    // dst(200) pulls from src(100).
    // 200 - cx - pan_x + cx = 100 => 200 - pan_x = 100 => pan_x = 100.
    // So positive Pan X shifts content to the RIGHT.

    {
        TransformParams params;
        params.zoom_factor = 1.0f;
        params.pan_x = 32.0f; // Shift content right by 32px (1 block)
        params.pan_y = 0.0f;

        // Clear dst
        std::fill(dst.begin(), dst.end(), 0);

        transform_image(src.data(), src.size(), w, h, dst.data(), dst.size(), params);

        // At (0,0) in source is WHITE.
        // At (32,0) in dst, we should see what was at (0,0) -> WHITE.
        // At (0,0) in dst, we pull from (-32,0). Clamp to edge -> (0,0) -> WHITE.

        // Let's check a transition.
        // Source: 0-31 White, 32-63 Black.
        // Dst (shifted right 32): 0-31 White (clamped), 32-63 White (shifted), 64-95 Black.

        if (check_pixel(dst, w, 40, 0, 255, 255, 255)) std::cout << "Pixel at 40 is White (Correct)" << std::endl;
        else std::cout << "Pixel at 40 is NOT White (Error)" << std::endl;

        if (check_pixel(dst, w, 80, 0, 0, 0, 0)) std::cout << "Pixel at 80 is Black (Correct)" << std::endl;
        else std::cout << "Pixel at 80 is NOT Black (Error)" << std::endl;

        assert(check_pixel(dst, w, 40, 0, 255, 255, 255));
        assert(check_pixel(dst, w, 80, 0, 0, 0, 0));
        std::cout << "Test 2 (Pan Right): Passed" << std::endl;
    }

    // Test 3: Zoom In (Center stays same, outer parts pushed out)
    {
        TransformParams params;
        params.zoom_factor = 2.0f; // 2x Zoom
        params.pan_x = 0.0f;

        std::fill(dst.begin(), dst.end(), 0);
        transform_image(src.data(), src.size(), w, h, dst.data(), dst.size(), params);

        // Center pixel should match center pixel of source
        int cx = w/2;
        int cy = h/2;

        // Source center (256, 256). 256/32 = 8. 8+8=16 (even) -> White.
        // Let's verify source center.
        if (check_pixel(src, w, cx, cy, 255, 255, 255)) {
             // Destination center should also be white
             assert(check_pixel(dst, w, cx, cy, 255, 255, 255));
        }

        std::cout << "Test 3 (Zoom In): Passed" << std::endl;
    }

    std::cout << "All Tests Passed!" << std::endl;
    return 0;
}
