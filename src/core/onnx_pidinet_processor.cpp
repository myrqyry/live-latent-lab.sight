#include "onnx_pidinet_processor.hpp"
#include "onnx_session.hpp"

#ifdef SIGHT_HAVE_RUST_FRAME_CORE
#include "rust_frame_core.h"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef SIGHT_WITH_CUDA_EDGE_IO
#include "edge-cuda-io.hpp"
#endif
namespace lll {
OnnxPidiNetProcessor::OnnxPidiNetProcessor()=default;
OnnxPidiNetProcessor::~OnnxPidiNetProcessor()=default;
EdgeTensorIOStats OnnxPidiNetProcessor::tensor_io_stats() const {
#ifdef SIGHT_WITH_CUDA_EDGE_IO
    if(device_io_)return device_io_->stats;
#endif
    return {};
}
bool OnnxPidiNetProcessor::load(const std::string &path)
{
#ifdef SIGHT_WITH_CUDA_EDGE_IO
    device_io_.reset();
#endif
    session_.reset();
    input_.clear();
    last_error_.clear();
    if (path.empty()) {
        last_error_ = "Missing local PiDiNet ONNX file";
        return false;
    }
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* request=std::getenv("SIGHT_ONNX_DEVICE");
        const char* provider=nullptr;
        const bool require_cuda=request&&std::string_view(request)=="cuda";
        auto candidate=open_onnx_session(environment_,path.c_str(),options,true,&provider,require_cuda);
        if (candidate->GetInputCount() != 1 || candidate->GetOutputCount() != 1)
            throw std::runtime_error("PiDiNet requires one BGR input and one edge output");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto in_name = candidate->GetInputNameAllocated(0, allocator);
        const auto out_name = candidate->GetOutputNameAllocated(0, allocator);
        if (std::string(in_name.get()) != "bgr" ||
            std::string(out_name.get()) != "edge")
            throw std::runtime_error("PiDiNet ONNX must have bgr -> edge tensor names");
        auto input_type = candidate->GetInputTypeInfo(0);
        const auto info = input_type.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) ||
            shape[1] != 3)
            throw std::runtime_error("Expected PiDiNet [1,3,H,W] float32 input");
        auto output_type = candidate->GetOutputTypeInfo(0);
        const auto out_info = output_type.GetTensorTypeAndShapeInfo();
        const auto out_shape = out_info.GetShape();
        if (out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            out_shape.size() != 4 ||
            (out_shape[0] != 1 && out_shape[0] != -1) ||
            out_shape[1] != 1)
            throw std::runtime_error("Expected PiDiNet [1,1,H,W] float32 output");
        const char* io=std::getenv("SIGHT_PIDINET_CUDA_IO");
        const bool force_device=io&&std::string_view(io)=="device";
#ifdef SIGHT_WITH_CUDA_EDGE_IO
        const bool cuda=provider&&std::string_view(provider).find("CUDA")!=std::string_view::npos;
        if(force_device&&!cuda)throw std::runtime_error("Device edge I/O requires CUDA provider");
        // Single-map PiDiNet benefits less from device-side conversion.
        // Keep the measured faster default until further qualification.
        if(cuda&&force_device){
            auto device=std::make_unique<EdgeDeviceIO>(true);device->init();
            device_io_=std::move(device);
        }
#else
        if(force_device)throw std::runtime_error("Device edge I/O unavailable in this build");
#endif
        session_ = std::move(candidate);
        return true;
    } catch (const std::exception &error) {
        last_error_ = error.what();
        return false;
    }
}

bool OnnxPidiNetProcessor::estimate(const uint8_t *bgr,
                                    uint32_t w, uint32_t h, bool safe,
                                    std::vector<uint8_t> &gray)
{
    last_error_.clear();
    if (!ready() || !bgr || w < 8 || h < 8 || w > 1024 || h > 1024 ||
        w % 8 != 0 || h % 8 != 0) {
        last_error_ = "Invalid PiDiNet model state or BGR frame dimensions";
        return false;
    }
    try {
#ifdef SIGHT_WITH_CUDA_EDGE_IO
        if(device_io_){device_io_->run(*session_,bgr,w,h,safe,gray);return true;}
#endif
        const size_t pixels = size_t(w) * h;
        static constexpr float mean[3] = {0.f, 0.f, 0.f};
        static constexpr float stddev[3] = {1.f, 1.f, 1.f};
        input_.resize(pixels * 3);
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        // Pixel order is BGR by design, matching reference's [:,:,::-1].
        if (sight_rgb8_to_chw_f32_v2(bgr, pixels * 3, w, h, 1.f / 255.f,
                                     mean, stddev, input_.data(), input_.size()) != 1) {
            last_error_ = "Rust PiDiNet BGR tensor preparation rejected input";
            return false;
        }
#else
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c)
                input_[c * pixels + p] = float(bgr[p * 3 + c]) / 255.f;
#endif
        const int64_t shape[] = {1, 3, int64_t(h), int64_t(w)};
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_, input_.data(), input_.size(), shape, 4);
        static constexpr const char *inputs[] = {"bgr"};
        static constexpr const char *outputs[] = {"edge"};
        auto result = session_->Run(
            Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        if (result.size() != 1 || !result[0].IsTensor())
            throw std::runtime_error("Missing PiDiNet edge probabilities");
        auto info = result[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{1, 1, int64_t(h), int64_t(w)})
            throw std::runtime_error("Unexpected PiDiNet edge tensor dimensions");
        const auto *probabilities = result[0].GetTensorData<float>();
        output_scratch_.resize(pixels);
        auto& next=output_scratch_;
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        if (sight_pidinet_probs_to_gray_v1(
                probabilities, pixels, safe ? 1 : 0,
                next.data(), next.size()) != 1) {
            last_error_ = "Rust PiDiNet output conversion rejected probabilities";
            return false;
        }
#else
        for (size_t p = 0; p < pixels; ++p) {
            const float probability = probabilities[p];
            if (!std::isfinite(probability)) {
                last_error_ = "PiDiNet probability map contains nonfinite values";
                return false;
            }
        }
        for (size_t p = 0; p < pixels; ++p) {
            const float value = probabilities[p];
            const float adjusted = safe
                ? float(int32_t(std::trunc(value * 3.f))) / 2.f
                : value;
            next[p] = uint8_t(std::clamp(adjusted * 255.f, 0.f, 255.f));
        }
#endif
        gray.swap(next);
        return true;
    } catch (const std::exception &error) {
        last_error_ = error.what();
        return false;
    }
}
} // namespace lll
