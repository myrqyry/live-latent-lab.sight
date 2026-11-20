#include "core/enhancement_processor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: animegan_inference_test /path/to/model.onnx\n";
        return 2;
    }
    constexpr int width = 128;
    constexpr int height = 128;
    const size_t bytes = static_cast<size_t>(width) * height * 3;
    std::vector<uint8_t> input(bytes);
    std::vector<uint8_t> output(bytes, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto idx = (static_cast<size_t>(y) * width + x) * 3;
            input[idx] = static_cast<uint8_t>(x * 2);
            input[idx + 1] = static_cast<uint8_t>(y * 2);
            input[idx + 2] = static_cast<uint8_t>((x + y) % 256);
        }
    }

    lll::EnhancementProcessor processor;
    if (processor.process(input.data(), width, height, output.data(), width, height)) {
        std::cerr << "Unexpected inference without a model\n";
        return 1;
    }
    if (!processor.load(argv[1], lll::EnhanceModelType::AnimeGANv3, false)) {
        std::cerr << "Failed to load actual AnimeGAN ONNX model\n";
        return 1;
    }
    if (processor.process(input.data(), width, height, output.data(), width + 1, height)) {
        std::cerr << "Accepted mismatched output dimensions\n";
        return 1;
    }
    if (!processor.process(input.data(), width, height, output.data(), width, height)) {
        std::cerr << "AnimeGAN inference failed on a real input frame\n";
        return 1;
    }

    size_t changed = 0;
    uint8_t lo = 255, hi = 0;
    uint64_t checksum = 0;
    for (size_t i = 0; i < bytes; ++i) {
        changed += (input[i] != output[i]);
        lo = std::min(lo, output[i]);
        hi = std::max(hi, output[i]);
        checksum += output[i];
    }
    std::cout << "AnimeGAN NHWC inference: changed=" << changed
              << "/" << bytes << " output_min=" << +lo << " output_max=" << +hi
              << " output_sum=" << checksum << '\n';
    if (changed < bytes / 10 || lo == hi) {
        std::cerr << "Output not distinguishable from pass-through/flat frame\n";
        return 1;
    }
    std::cout << "PASS: actual model transformed a frame\n";
    return 0;
}
