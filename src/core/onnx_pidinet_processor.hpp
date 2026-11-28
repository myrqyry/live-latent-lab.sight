#pragma once

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lll {
// Native PiDiNet model contract: BGR8 -> BGR float32 [1,3,H,W]/255,
// single sigmoid probability output [1,1,H,W], optional safe_step(2).
class OnnxPidiNetProcessor {
public:
    bool load(const std::string &path);
    bool estimate(const uint8_t *bgr, uint32_t width, uint32_t height,
                  bool safe, std::vector<uint8_t> &gray);
    bool ready() const { return static_cast<bool>(session_); }
    const std::string &last_error() const { return last_error_; }
private:
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "SightPiDiNet"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string last_error_;
    std::vector<uint8_t> output_scratch_;
    std::vector<float> input_;
};
} // namespace lll
