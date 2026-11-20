#pragma once

#include <string>
#include <vector>
#include <memory>
#include <obs-module.h>

// Forward declare ONNX types
namespace Ort {
    class Session;
    class Env;
    class MemoryInfo;
    class Value;
}

namespace lll {

class FastStyleTransfer {
public:
    FastStyleTransfer();
    ~FastStyleTransfer();

    // Check if model is loaded
    bool is_ready() const;

    // Load model from path
    bool load(const std::string& model_path);

    // Process image
    // Input/Output: Packed RGB data (width * height * 3)
    // Note: Models may require fixed resolutions (e.g. 224x224 or 512x512)
    // This function handles resizing if necessary.
    // returns true on success.
    bool process(const uint8_t* input_rgb, int width, int height, 
                 uint8_t* output_rgb, int out_width, int out_height);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lll
