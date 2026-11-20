#pragma once

#include <string>
#include <vector>
#include <memory>
#include <obs-module.h>

// Forward declare ONNX types to avoid heavy includes in header
namespace Ort {
    class Session;
    class Env;
    class MemoryInfo;
    class Value;
}

namespace lll {

class NSFWDetector {
public:
    NSFWDetector();
    ~NSFWDetector();

    // Check if model is loaded and ready
    bool is_ready() const;

    // Load model from path
    bool load(const std::string& model_path);

    // Run detection on RGB image data
    // Returns prediction score (0.0 = Safe, 1.0 = NSFW)
    float predict(const uint8_t* rgb_data, int width, int height);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lll
