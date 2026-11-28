#include "onnx_depth_processor.hpp"
#include "onnx_session.hpp"
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
#include "rust_frame_core.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <iostream>

#ifdef SIGHT_WITH_CUDA_DEPTH_IO
#include "depth-cuda-io.inc"
#endif

namespace lll {
OnnxDepthProcessor::OnnxDepthProcessor()=default;
OnnxDepthProcessor::~OnnxDepthProcessor()=default;
DepthTensorIOStats OnnxDepthProcessor::tensor_io_stats() const {
#ifdef SIGHT_WITH_CUDA_DEPTH_IO
    if(device_io_)return device_io_->stats;
#endif
    return {};
}

bool OnnxDepthProcessor::load(const std::string &model_path, bool prefer_gpu)
{
#ifdef SIGHT_WITH_CUDA_DEPTH_IO
    device_io_.reset();
#endif
    session_.reset();
    provider_=nullptr;
    last_error_.clear();
    input_name_.clear();
    output_name_.clear();
    if (model_path.empty())
        return false;

    try {
        Ort::SessionOptions options;
        // Leave resources for OBS graphics, video, and other filters.
        const unsigned logical = std::thread::hardware_concurrency();
        options.SetIntraOpNumThreads(static_cast<int>(
            std::min(4u, std::max(1u, logical / 2u))));
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const char* requested=std::getenv("SIGHT_ONNX_DEVICE");
        const bool require_cuda=prefer_gpu&&requested&&std::string_view(requested)=="cuda";
        auto candidate = open_onnx_session(environment_, model_path.c_str(), options, prefer_gpu, &provider_,require_cuda);
        if (candidate->GetInputCount() != 1 || candidate->GetOutputCount() != 1)
            throw std::runtime_error("Depth Anything needs one input and one output");
        auto in_type = candidate->GetInputTypeInfo(0);
        auto out_type = candidate->GetOutputTypeInfo(0);
        const auto in = in_type.GetTensorTypeAndShapeInfo();
        const auto out = out_type.GetTensorTypeAndShapeInfo();
        const auto a = in.GetShape();
        const auto b = out.GetShape();
        if (in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            a.size() != 4 || (a[0] != -1 && a[0] != 1) ||
            a[1] != 3 || b.size() != 3 ||
            (b[0] != -1 && b[0] != 1))
            throw std::runtime_error("Not an expected float32 Depth Anything NCHW model");
        Ort::AllocatorWithDefaultOptions allocator;
        auto input = candidate->GetInputNameAllocated(0, allocator);
        auto output = candidate->GetOutputNameAllocated(0, allocator);
        input_name_ = input.get();
        output_name_ = output.get();
        if (input_name_ != "pixel_values" || output_name_ != "predicted_depth")
            throw std::runtime_error("Expected Depth Anything pixel_values -> predicted_depth");
#ifdef SIGHT_WITH_CUDA_DEPTH_IO
        const char* io=std::getenv("SIGHT_DEPTH_CUDA_IO");
        const bool force_device=io&&std::string_view(io)=="device";
        const bool host=io&&std::string_view(io)=="host";
        const bool cuda=provider_&&std::string_view(provider_).find("CUDA")!=std::string_view::npos;
        if(force_device&&!cuda)throw std::runtime_error("Device depth I/O requires CUDA provider");
        if(cuda&&!host){
            auto device=std::make_unique<DeviceIO>();device->init();
            device_io_=std::move(device);
            provider_="ONNX Runtime CUDA (bound depth tensors)";
        }
#else
        const char* io=std::getenv("SIGHT_DEPTH_CUDA_IO");
        if(io&&std::string_view(io)=="device")throw std::runtime_error("Device depth I/O unavailable in this build");
#endif
        session_ = std::move(candidate);
        return true;
    } catch (const std::exception &e) {
        provider_=nullptr;
        last_error_ = e.what();
        return false;
    }
}

bool OnnxDepthProcessor::estimate(const uint8_t *rgb, uint32_t width, uint32_t height,
                                  std::vector<uint8_t> &depth)
{
    last_error_.clear();
    if (!ready() || !rgb || width < 14 || height < 14 ||
        width % 14 || height % 14 || width > 560 || height > 560)
        return false;
    try {
#ifdef SIGHT_WITH_CUDA_DEPTH_IO
        if(device_io_){
            device_io_->run(*session_,input_name_,output_name_,rgb,width,height,depth);
            return true;
        }
#endif
        const size_t plane = static_cast<size_t>(width) * height;
        static constexpr float mean[] = {0.485f, 0.456f, 0.406f};
        static constexpr float stdv[] = {0.229f, 0.224f, 0.225f};
        normalized_.resize(3 * plane);
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        if (sight_normalize_rgb_to_chw_f32_v1(
                rgb, plane * 3, width, height, mean, stdv,
                normalized_.data(), normalized_.size()) != 1) {
            last_error_ = "Rust tensor preparation rejected RGB frame dimensions";
            return false;
        }
#else
        for (size_t p = 0; p < plane; ++p) {
            for (size_t c = 0; c < 3; ++c)
                normalized_[c * plane + p] =
                    (static_cast<float>(rgb[p * 3 + c]) / 255.0f - mean[c]) / stdv[c];
        }
#endif
        const int64_t dimensions[] = {1, 3, static_cast<int64_t>(height),
                                         static_cast<int64_t>(width)};
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_, normalized_.data(), normalized_.size(), dimensions, 4);
        const char *inputs[] = {input_name_.c_str()};
        const char *outputs[] = {output_name_.c_str()};
        auto results = session_->Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1,
                                     outputs, 1);
        if (results.size() != 1 || !results[0].IsTensor())
            return false;
        const auto shape_info = results[0].GetTensorTypeAndShapeInfo();
        const auto shape = shape_info.GetShape();
        if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape != std::vector<int64_t>{1, static_cast<int64_t>(height),
                                                static_cast<int64_t>(width)})
            return false;
        const float *values = results[0].GetTensorData<float>();
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t p = 0; p < plane; ++p) {
            if (!std::isfinite(values[p]))
                return false;
            mn = std::min(mn, values[p]);
            mx = std::max(mx, values[p]);
        }
        depth.resize(plane);
        const float span = mx - mn;
        if (span < 1e-6f) {
            std::fill(depth.begin(), depth.end(), uint8_t{0});
            return true;
        }
        for (size_t p = 0; p < plane; ++p)
            depth[p] = static_cast<uint8_t>(std::clamp(
                (values[p] - mn) * (255.f / span), 0.f, 255.f));
        return true;
    } catch (const std::exception &e) {
        last_error_ = e.what();
        return false;
    }
}
}
