#include "core/onnx_hed_processor.hpp"
#include "core/pidinet_scribble.hpp"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

uint64_t fingerprint(const std::vector<uint8_t> &image)
{
    uint64_t hash = 14695981039346656037ull;
    for (const uint8_t b : image) {
        hash ^= b;
        hash *= 1099511628211ull;
    }
    return hash;
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    lll::OnnxHedProcessor processor;
    std::vector<uint8_t> initial;
    if (processor.estimate(nullptr, 192, 128, false, initial) ||
        processor.load("") || !processor.load(argv[1])) {
        std::cerr << "FAIL: HED load/premature inference: "
                  << processor.last_error() << "\n";
        return 1;
    }
    constexpr uint32_t W = 192, H = 128;
    std::vector<uint8_t> rgb(size_t(W) * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = (size_t(y) * W + x) * 3;
            rgb[i] = x > 38 && x < 155 ? 245 : 35;
            rgb[i + 1] = y > 20 && y < 110 ? 25 : 210;
            rgb[i + 2] = uint8_t((x + y) % 256);
        }
    std::vector<uint8_t> soft;
    if (processor.estimate(rgb.data(), W - 1, H, false, soft) ||
        !processor.estimate(rgb.data(), W, H, false, soft) ||
        soft.size() != size_t(W) * H) {
        std::cerr << "FAIL: HED native inference: " << processor.last_error() << "\n";
        return 1;
    }
    std::vector<uint8_t> safe;
    if (!processor.estimate(rgb.data(), W, H, true, safe) || safe == soft) {
        std::cerr << "FAIL: HED safe-step output\n";
        return 1;
    }
    std::vector<uint8_t> scribble = soft;
    if (!lll::pidinet_scribble_edges(scribble, W, H)) {
        std::cerr << "FAIL: native HED directional NMS\n";
        return 1;
    }
    const auto [low, high] = std::minmax_element(soft.begin(), soft.end());
    if (int(*high) - int(*low) < 16) {
        std::cerr << "FAIL: HED output flat\n"; return 1;
    }
    std::cout << "hed_soft_hash=" << std::hex << fingerprint(soft) << "\n"
              << "hed_safe_hash=" << fingerprint(safe) << "\n"
              << "hed_scribble_hash=" << fingerprint(scribble) << std::dec << "\n";
    if (fingerprint(soft) != 0xfe20f4b9a51aa7f8ull ||
        fingerprint(safe) != 0xa34f2134fa16052eull ||
        fingerprint(scribble) != 0xbe9f923fb79992e0ull) {
        std::cerr << "FAIL: native HED output differs from original PyTorch/cv2\n";
        return 1;
    }
    std::cout << "PASS: real HED five-scale ONNX parity and Scribble NMS\n";
    return 0;
}
