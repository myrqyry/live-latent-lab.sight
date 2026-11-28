#pragma once
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lll {
// Native HED: five multiscale raw-logit edge projections from local ONNX.
// RGB input is float32, raw 0..255. No Python/ComfyUI runtime dependency.
class OnnxHedProcessor {
public:
    bool load(const std::string &path);
    bool estimate(const uint8_t *rgb, uint32_t width, uint32_t height,
                  bool safe, std::vector<uint8_t> &gray);
    bool ready() const { return bool(session_); }
    const std::string &last_error() const { return last_error_; }
private:
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "SightHED"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string last_error_;
    std::vector<uint8_t> output_scratch_;
    std::vector<float> input_;
};
} // namespace lll
