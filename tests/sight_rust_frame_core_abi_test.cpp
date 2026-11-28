#include "core/rust_frame_core.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <utility>

int main()
{
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    // Match the existing Depth Anything V2 C++ normalization operation
    // at multiple sizes, including a real 196x252 model input.
    for (const auto [w, h] : {
            std::pair{1u, 1u}, std::pair{7u, 5u},
            std::pair{196u, 252u}}) {
        const size_t plane = size_t(w) * h;
        std::vector<uint8_t> rgb(3 * plane);
        for (size_t i = 0; i < rgb.size(); ++i)
            rgb[i] = static_cast<uint8_t>((i * 73 + 19) & 255);
        std::vector<float> output(3 * plane, -999.f);
        const int32_t ok = sight_normalize_rgb_to_chw_f32_v1(
            rgb.data(), rgb.size(), w, h, mean, stddev,
            output.data(), output.size());
        assert(ok == 1);
        for (size_t p = 0; p < plane; ++p)
            for (size_t c = 0; c < 3; ++c) {
                const float expected =
                    (static_cast<float>(rgb[p * 3 + c]) / 255.0f
                       - mean[c]) / stddev[c];
                const float actual = output[c * plane + p];
                assert(std::abs(expected - actual) < 1e-6f);
            }
    }

    const uint8_t input[6] = {255, 0, 127, 64, 90, 200};
    float output[6] = {-8, -8, -8, -8, -8, -8};
    auto rejects = [&](const uint8_t *rgb, size_t rgb_len,
                       uint32_t width, uint32_t height, const float *m,
                       const float *s, float *dst, size_t dst_len) {
        assert(sight_normalize_rgb_to_chw_f32_v1(
            rgb, rgb_len, width, height, m, s, dst, dst_len) == 0);
        for (float value : output) assert(value == -8);
    };
    rejects(nullptr, sizeof(input), 2, 1, mean, stddev, output, 6);
    rejects(input, 5, 2, 1, mean, stddev, output, 6);
    rejects(input, sizeof(input), 2, 1, mean, stddev, output, 5);
    rejects(input, sizeof(input), 0, 1, mean, stddev, output, 6);
    rejects(input, sizeof(input), UINT32_MAX, UINT32_MAX, mean, stddev, output, 6);
    rejects(input, sizeof(input), 2, 1, nullptr, stddev, output, 6);
    constexpr float bad_std[3] = {0.229f, 0.0f, 0.225f};
    rejects(input, sizeof(input), 2, 1, mean, bad_std, output, 6);
    constexpr float bad_mean[3] = {0.485f, 0.456f,
                                   std::numeric_limits<float>::quiet_NaN()};
    rejects(input, sizeof(input), 2, 1, bad_mean, stddev, output, 6);
    // Overlap check: the output points into the RGB allocation.
    alignas(float) uint8_t overlapping[32]{};
    assert(sight_normalize_rgb_to_chw_f32_v1(
        overlapping, 6, 2, 1, mean, stddev,
        reinterpret_cast<float *>(overlapping), 6) == 0);
    // TEED requires raw (not 1/255-scaled) RGB channels in CHW order.
    constexpr uint8_t teed_rgb[] = {10, 20, 30, 40, 50, 60};
    const float zero[3] = {0.f, 0.f, 0.f};
    const float one[3] = {1.f, 1.f, 1.f};
    float teed_chw[6]{};
    assert(sight_rgb8_to_chw_f32_v2(teed_rgb, 6, 2, 1, 1.f,
                                    zero, one, teed_chw, 6) == 1);
    const float expected_chw[] = {10, 40, 20, 50, 30, 60};
    for (size_t i = 0; i < 6; ++i)
        assert(teed_chw[i] == expected_chw[i]);
    float invalid_chw[6] = {-9, -9, -9, -9, -9, -9};
    assert(sight_rgb8_to_chw_f32_v2(teed_rgb, 6, 2, 1,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    zero, one, invalid_chw, 6) == 0);
    for (float f : invalid_chw) assert(f == -9.f);

    const float teed_logits[4] = {0.f, -20.f, 8.f, 0.f};
    uint8_t teed_gray[4] = {99, 99, 99, 99};
    assert(sight_teed_edges_to_gray_v1(teed_logits, teed_logits,
                                       teed_logits, teed_logits,
                                       4, 2, teed_gray, 4) == 1);
    const uint8_t expected_edges[] = {127, 0, 255, 127};
    for (size_t i = 0; i < 4; ++i)
        assert(teed_gray[i] == expected_edges[i]);
    assert(sight_teed_edges_to_gray_v1(teed_logits, teed_logits,
                                       teed_logits, teed_logits,
                                       4, 11, teed_gray, 4) == 0);
    for (size_t i = 0; i < 4; ++i)
        assert(teed_gray[i] == expected_edges[i]);
    // PiDiNet single sigmoid-probability map, optionally safe_step(2).
    const float pidinet_probs[] = {0.f, 0.1f, 0.25f, 0.5f, 0.75f, 0.99f, 1.f};
    uint8_t pidinet_pixels[7] = {88, 88, 88, 88, 88, 88, 88};
    assert(sight_pidinet_probs_to_gray_v1(
        pidinet_probs, 7, 0, pidinet_pixels, 7) == 1);
    const uint8_t natural[] = {0, 25, 63, 127, 191, 252, 255};
    for (size_t i = 0; i < 7; ++i) assert(pidinet_pixels[i] == natural[i]);
    assert(sight_pidinet_probs_to_gray_v1(
        pidinet_probs, 7, 1, pidinet_pixels, 7) == 1);
    const uint8_t safe_values[] = {0, 0, 0, 127, 255, 255, 255};
    for (size_t i = 0; i < 7; ++i) assert(pidinet_pixels[i] == safe_values[i]);
    assert(sight_pidinet_probs_to_gray_v1(
        pidinet_probs, 7, 2, pidinet_pixels, 7) == 0);
    for (size_t i = 0; i < 7; ++i) assert(pidinet_pixels[i] == safe_values[i]);
    const float nan_probs[] = {
        0.f, 0.1f, 0.25f, 0.5f, 0.75f,
        std::numeric_limits<float>::quiet_NaN(), 1.f
    };
    assert(sight_pidinet_probs_to_gray_v1(
        nan_probs, 7, 0, pidinet_pixels, 7) == 0);
    for (size_t i = 0; i < 7; ++i) assert(pidinet_pixels[i] == safe_values[i]);
    std::cout << "PASS: Rust staticlib/C++ ABI, tensor parity, and fail-closed buffer validation\n";
}
