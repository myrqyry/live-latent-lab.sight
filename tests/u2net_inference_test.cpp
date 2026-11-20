#include "core/segmentation_processor.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: u2net_inference_test /path/to/u2net.onnx\n";
        return 2;
    }
    constexpr uint32_t W = 320, H = 320;
    std::vector<uint8_t> rgb(W * H * 3);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 3;
            const bool fg = x > 82 && x < 242 && y > 45 && y < 275;
            rgb[i] = fg ? 224 : 60;
            rgb[i+1] = fg ? 125 : 160;
            rgb[i+2] = fg ? 66 : 210;
        }
    }
    std::vector<uint8_t> alpha(W * H, 0);
    lll::SegmentationProcessor model;
    if (model.process(rgb.data(), W, H, alpha.data())) {
        std::cerr << "Segmentation ran before model load\n";
        return 1;
    }
    if (!model.load(argv[1], lll::SegModelType::U2Net, false)) {
        std::cerr << "Failed to load U2Net\n";
        return 1;
    }
    if (!model.process(rgb.data(), W, H, alpha.data())) {
        std::cerr << "U2Net real inference failed\n";
        return 1;
    }
    const auto [mn, mx] = std::minmax_element(alpha.begin(), alpha.end());
    uint64_t sum = 0;
    size_t mid = 0;
    for (auto a : alpha) {
        sum += a;
        mid += a > 25 && a < 230;
    }
    std::cout << "U2Net mask: min=" << +*mn << " max=" << +*mx
              << " mean=" << static_cast<double>(sum) / alpha.size()
              << " midtones=" << mid << "/" << alpha.size() << "\n";
    if (*mx <= *mn + 20) {
        std::cerr << "FAIL: U2Net matte nearly constant\n";
        return 1;
    }
    std::cout << "PASS: real U2Net produced nontrivial alpha\n";
    return 0;
}
