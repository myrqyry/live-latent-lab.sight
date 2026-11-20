#include "core/onnx_depth_processor.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2)
        return 2;
    constexpr uint32_t W = 196, H = 252;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 3;
            rgb[i + 0] = static_cast<uint8_t>((x * 3 + y) & 255);
            rgb[i + 1] = static_cast<uint8_t>((y * 4) & 255);
            rgb[i + 2] = static_cast<uint8_t>((x + y * 2) & 255);
        }
    lll::OnnxDepthProcessor model;
    std::vector<uint8_t> output;
    if (model.estimate(rgb.data(), W, H, output) || model.load("") ||
        !model.load(argv[1]) ||
        model.estimate(rgb.data(), W - 1, H, output) ||
        !model.estimate(rgb.data(), W, H, output) || output.size() != W * H)
        return 1;
    const auto [min, max] = std::minmax_element(output.begin(), output.end());
    std::cout << "actual_model_depth_min=" << +*min
              << " max=" << +*max
              << " pixels=" << output.size() << "\n";
    if (*max < *min + 32)
        return 1;
    std::cout << "PASS: real depth model changed across a structured RGB input\n";
}
