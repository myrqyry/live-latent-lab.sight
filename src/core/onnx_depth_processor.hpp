#pragma once

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lll {
// A deliberately narrow backend: validated Depth Anything V2 NCHW float32
// inference. No LiteRT dependency, synthetic map, or luminance fallback.
class OnnxDepthProcessor {
public:
    bool load(const std::string &model_path);
    bool estimate(const uint8_t *rgb, uint32_t width, uint32_t height,
                  std::vector<uint8_t> &depth);
    bool ready() const { return static_cast<bool>(session_); }
    const std::string &last_error() const { return last_error_; }
private:
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "SightDepth"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string input_name_, output_name_, last_error_;
    std::vector<float> normalized_;
};
}
