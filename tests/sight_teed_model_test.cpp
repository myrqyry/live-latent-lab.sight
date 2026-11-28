#include "core/onnx_teed_processor.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    lll::OnnxTeedProcessor model;
    std::vector<uint8_t> initial;
    if (model.estimate(nullptr, 128, 128, 2, initial) ||
        model.load("") || !model.load(argv[1])) {
        std::cerr << "FAIL: TEED load or premature inference: " << model.last_error() << "\n"; return 1;
    }
    constexpr uint32_t W = 192, H = 128;
    std::vector<uint8_t> rgb(size_t(W) * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t p = (size_t(y) * W + x) * 3;
            rgb[p] = (x > 38 && x < 155) ? 245 : 35;
            rgb[p + 1] = (y > 20 && y < 110) ? 25 : 210;
            rgb[p + 2] = static_cast<uint8_t>((x + y) % 256);
        }
    std::vector<uint8_t> out;
    if (model.estimate(rgb.data(), W - 1, H, 2, out) ||
        model.estimate(rgb.data(), W, H, 11, out) ||
        !model.estimate(rgb.data(), W, H, 2, out) || out.size() != size_t(W) * H) {
        std::cerr << "FAIL: native ONNX TEED inference " << model.last_error() << "\n";
        return 1;
    }
    const auto [mn, mx] = std::minmax_element(out.begin(), out.end());
    std::cout << "teed_real_model_range=" << int(*mn) << ".." << int(*mx)
              << " pixels=" << out.size() << "\n";
    if (int(*mx) - int(*mn) < 16) {
        std::cerr << "FAIL: TEED output flat or non-informative\n"; return 1;
    }
    std::vector<uint8_t> no_steps;
    if (!model.estimate(rgb.data(), W, H, 0, no_steps) ||
        no_steps == out || no_steps.size() != out.size()) {
        std::cerr << "FAIL: TEED safe_step not reflected in real model output\n"; return 1;
    }
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t value : out) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    std::cout << "teed_final_edge_hash=" << std::hex << hash << std::dec << "\n";
    std::cout << "PASS: actual TEED four-output ONNX model inference and postprocessing\n";
}
