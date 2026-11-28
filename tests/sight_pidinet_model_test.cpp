#include "core/onnx_pidinet_processor.hpp"
#include "core/pidinet_scribble.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    lll::OnnxPidiNetProcessor model;
    std::vector<uint8_t> output;
    if (model.estimate(nullptr, 128, 128, false, output) ||
        model.load("") || !model.load(argv[1])) {
        std::cerr << "FAIL: PiDiNet model load or premature inference: "
                  << model.last_error() << "\n";
        return 1;
    }
    constexpr uint32_t W = 192, H = 128;
    std::vector<uint8_t> bgr(size_t(W) * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t p = (size_t(y) * W + x) * 3;
            bgr[p] = (x > 38 && x < 155) ? 245 : 35;
            bgr[p + 1] = (y > 20 && y < 110) ? 25 : 210;
            bgr[p + 2] = static_cast<uint8_t>((x + y) % 256);
        }
    if (model.estimate(bgr.data(), W - 1, H, false, output) ||
        !model.estimate(bgr.data(), W, H, false, output) ||
        output.size() != size_t(W) * H) {
        std::cerr << "FAIL: native PiDiNet real inference: "
                  << model.last_error() << "\n";
        return 1;
    }
    const auto [low, high] = std::minmax_element(output.begin(), output.end());
    if (int(*high) - int(*low) < 20) {
        std::cerr << "FAIL: PiDiNet output lacks useful edges: "
                  << int(*low) << ".." << int(*high) << "\n";
        return 1;
    }
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t v : output) {
        hash ^= v;
        hash *= 1099511628211ull;
    }
    std::cout << "pidinet_final_edge_hash=" << std::hex << hash << std::dec
              << " range=" << int(*low) << ".." << int(*high) << "\n";
    std::vector<uint8_t> safe_output;
    if (!model.estimate(bgr.data(), W, H, true, safe_output) ||
        safe_output == output || safe_output.size() != output.size()) {
        std::cerr << "FAIL: PiDiNet safe-step transform ineffective\n";
        return 1;
    }
    uint64_t safe_hash = 14695981039346656037ull;
    for (uint8_t value : safe_output) {
        safe_hash ^= value;
        safe_hash *= 1099511628211ull;
    }
    std::cout << "pidinet_safe_hash=" << std::hex << safe_hash
              << std::dec << "\n";
    std::vector<uint8_t> scribble = output;
    if (!lll::pidinet_scribble_edges(scribble, W, H)) {
        std::cerr << "FAIL: PiDiNet NMS rejected actual edge model output\n";
        return 1;
    }
    uint64_t scribble_hash = 14695981039346656037ull;
    for (uint8_t v : scribble) {
        if (v != 0 && v != 255) {
            std::cerr << "FAIL: PiDiNet Scribble is not binary\n";
            return 1;
        }
        scribble_hash ^= v;
        scribble_hash *= 1099511628211ull;
    }
    std::cout << "pidinet_scribble_hash=" << std::hex << scribble_hash
              << std::dec << "\n";
    if (scribble_hash != 0x89630336a29f82d9ull) {
        std::cerr << "FAIL: PiDiNet Scribble differs from reference NMS\n";
        return 1;
    }
    std::cout << "PASS: native PiDiNet ONNX probability inference and safe-step\n";
}
