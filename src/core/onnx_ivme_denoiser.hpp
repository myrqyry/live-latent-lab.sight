#pragma once
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace lll {
class OnnxIvmeDenoiser {
public:
    bool load(const std::string& model);
    // Model-native RGB uint8 in/out at the same resolution.
    bool denoise(const uint8_t* rgb, uint32_t w, uint32_t h, float sigma,
                 std::vector<uint8_t>& cleaned);
    bool ready() const {return static_cast<bool>(session_);}
    const std::string& last_error() const {return last_error_;}
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"SightIvme"};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session> session_;
    std::string last_error_;
    std::vector<float> input_;
};
}
