#include "onnx_hed_processor.hpp"
#include "onnx_session.hpp"
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
#include "rust_frame_core.h"
#endif
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lll {
bool OnnxHedProcessor::load(const std::string &path)
{
    session_.reset();
    input_.clear();
    last_error_.clear();
    if (path.empty()) {
        last_error_ = "Missing local HED ONNX path";
        return false;
    }
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        auto candidate = open_onnx_session(
            environment_, path.c_str(), options);
        if (candidate->GetInputCount() != 1 || candidate->GetOutputCount() != 5)
            throw std::runtime_error("HED requires one RGB input and five edge outputs");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto name = candidate->GetInputNameAllocated(0, allocator);
        if (std::string(name.get()) != "rgb")
            throw std::runtime_error("HED input must be named rgb");
        auto input_type = candidate->GetInputTypeInfo(0);
        auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_info.GetShape();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input_shape.size() != 4 || (input_shape[0] != 1 && input_shape[0] != -1) ||
            input_shape[1] != 3)
            throw std::runtime_error("HED input must be float32 [1,3,H,W]");
        for (size_t i = 0; i < 5; ++i) {
            const auto output_name = candidate->GetOutputNameAllocated(i, allocator);
            if (std::string(output_name.get()) != "scale" + std::to_string(i + 1))
                throw std::runtime_error("HED outputs must be scale1..scale5");
            auto type = candidate->GetOutputTypeInfo(i);
            auto info = type.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) ||
                shape[1] != 1)
                throw std::runtime_error("HED scale output must be float32 [1,1,h,w]");
        }
        session_ = std::move(candidate);
        return true;
    } catch (const std::exception &e) {
        last_error_ = e.what();
        return false;
    }
}

bool OnnxHedProcessor::estimate(const uint8_t *rgb, uint32_t w, uint32_t h,
                                bool safe, std::vector<uint8_t> &gray)
{
    last_error_.clear();
    if (!ready() || !rgb || w < 16 || h < 16 ||
        w > 1024 || h > 1024 || w % 16 || h % 16) {
        last_error_ = "HED requires loaded model and frame divisible by sixteen";
        return false;
    }
    try {
        const size_t pixels = size_t(w) * h;
        input_.resize(pixels * 3);
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        static constexpr float means[3] = {0.f, 0.f, 0.f};
        static constexpr float stddev[3] = {1.f, 1.f, 1.f};
        if (sight_rgb8_to_chw_f32_v2(
                rgb, pixels * 3, w, h, 1.f,
                means, stddev, input_.data(), input_.size()) != 1) {
            last_error_ = "Rust HED RGB8/CHW conversion rejected input";
            return false;
        }
#else
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c)
                input_[c * pixels + p] = float(rgb[p * 3 + c]);
#endif
        const int64_t dims[] = {1, 3, int64_t(h), int64_t(w)};
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_, input_.data(), input_.size(), dims, 4);
        static constexpr const char *input_names[] = {"rgb"};
        static constexpr const char *output_names[] = {
            "scale1", "scale2", "scale3", "scale4", "scale5"
        };
        auto results = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 5);
        if (results.size() != 5)
            throw std::runtime_error("HED returned fewer than five output maps");
        std::array<cv::Mat, 5> upsampled{};
        for (size_t i = 0; i < 5; ++i) {
            if (!results[i].IsTensor())
                throw std::runtime_error("HED output is not a tensor");
            const auto info = results[i].GetTensorTypeAndShapeInfo();
            const auto iw = w >> i, ih = h >> i;
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                info.GetShape() != std::vector<int64_t>{1, 1, int64_t(ih), int64_t(iw)})
                throw std::runtime_error("Unexpected HED multi-scale output shape");
            cv::Mat scale(int(ih), int(iw), CV_32FC1,
                          const_cast<float *>(results[i].GetTensorData<float>()));
            cv::resize(scale, upsampled[i], cv::Size(int(w), int(h)), 0, 0,
                       cv::INTER_LINEAR);
            if (!upsampled[i].isContinuous())
                throw std::runtime_error("HED result has unsupported row stride");
        }
        std::vector<uint8_t> output(pixels);
        for (size_t p = 0; p < pixels; ++p) {
            float sum = 0.f;
            for (const auto &map : upsampled) {
                const auto v = map.ptr<float>()[p];
                if (!std::isfinite(v))
                    throw std::runtime_error("HED returned nonfinite logits");
                sum += v;
            }
            // Reference: np.mean(float32 edge projections), sigmoid(float64),
            // optional safe_step(x, 2), clip, truncate to uint8.
            const float average = sum / 5.f;
            const double sigmoid = 1.0 / (1.0 + std::exp(-double(average)));
            const double value = safe
                ? double(float(int32_t(float(sigmoid) * 3.f)) / 2.f)
                : sigmoid;
            output[p] = uint8_t(std::clamp(value * 255.0, 0.0, 255.0));
        }
        gray.swap(output);
        return true;
    } catch (const std::exception &e) {
        last_error_ = e.what();
        return false;
    }
}
} // namespace lll
