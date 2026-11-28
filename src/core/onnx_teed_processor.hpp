#pragma once

#include <onnxruntime_cxx_api.h>
#include "edge_tensor_io_stats.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lll {
#ifdef SIGHT_WITH_CUDA_EDGE_IO
class EdgeDeviceIO;
#endif
// Native TEED four-logit ONNX runtime. Pre- and postprocessing follow the
// original TEDDetector semantics (RGB 0..255, sigmoid, optional safe_step).
class OnnxTeedProcessor {
public:
    OnnxTeedProcessor();
    ~OnnxTeedProcessor();
    EdgeTensorIOStats tensor_io_stats() const;
    bool load(const std::string &path);
    bool estimate(const uint8_t *rgb, uint32_t width, uint32_t height,
                  uint32_t safe_steps, std::vector<uint8_t> &gray);
    bool ready() const { return bool(session_); }
    const std::string &last_error() const { return last_error_; }

private:
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "SightTEED"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string last_error_;
    std::vector<uint8_t> output_scratch_;
    std::vector<float> input_;
#ifdef SIGHT_WITH_CUDA_EDGE_IO
    std::unique_ptr<EdgeDeviceIO> device_io_;
#endif
};
} // namespace lll
