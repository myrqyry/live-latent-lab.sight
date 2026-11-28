#include "onnx_teed_processor.hpp"
#include "onnx_session.hpp"

#ifdef SIGHT_HAVE_RUST_FRAME_CORE
#include "rust_frame_core.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lll {
bool OnnxTeedProcessor::load(const std::string &path)
{
    session_.reset();
    last_error_.clear();
    input_.clear();
    if (path.empty()) {
        last_error_ = "Missing local TEED ONNX path";
        return false;
    }
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        auto candidate = open_onnx_session(environment_, path.c_str(), options);
        if (candidate->GetInputCount() != 1 || candidate->GetOutputCount() != 4)
            throw std::runtime_error("Expected TEED one RGB input and four edge outputs");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = candidate->GetInputNameAllocated(0, allocator);
        if (std::string(input_name.get()) != "rgb")
            throw std::runtime_error("TEED expects input name rgb");
        auto input_type = candidate->GetInputTypeInfo(0);
        auto input_info = input_type.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_info.GetShape();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input_shape.size() != 4 || (input_shape[0] != 1 && input_shape[0] != -1) ||
            input_shape[1] != 3)
            throw std::runtime_error("TEED input must be [1,3,H,W] float32");

        for (size_t i = 0; i < 4; ++i) {
            auto output_name = candidate->GetOutputNameAllocated(i, allocator);
            if (std::string(output_name.get()) != "edge" + std::to_string(i))
                throw std::runtime_error("Expected TEED edge0..edge3 outputs");
            auto output_type = candidate->GetOutputTypeInfo(i);
            auto info = output_type.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) ||
                shape[1] != 1)
                throw std::runtime_error("TEED edge outputs must be [1,1,H,W] float32");
        }
        session_ = std::move(candidate);
        return true;
    } catch (const std::exception &error) {
        last_error_ = error.what();
        return false;
    }
}

bool OnnxTeedProcessor::estimate(const uint8_t *rgb,
                                 uint32_t w, uint32_t h, uint32_t safe_steps,
                                 std::vector<uint8_t> &gray)
{
    last_error_.clear();
    if (!ready() || !rgb || w < 8 || h < 8 || w > 1024 || h > 1024 ||
        w % 8 || h % 8 || safe_steps > 10) {
        last_error_ = "Invalid TEED input size, model state or safe_steps";
        return false;
    }
    try {
        const size_t pixels = size_t(w) * h;
        static constexpr float mean[] = {0.f, 0.f, 0.f};
        static constexpr float stddev[] = {1.f, 1.f, 1.f};
        input_.resize(pixels * 3);
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        if (sight_rgb8_to_chw_f32_v2(rgb, pixels * 3, w, h, 1.f,
                                     mean, stddev, input_.data(), input_.size()) != 1) {
            last_error_ = "Rust TEED tensor preparation rejected RGB buffer";
            return false;
        }
#else
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c)
                input_[c * pixels + p] = static_cast<float>(rgb[p * 3 + c]);
#endif
        const int64_t dims[] = {1, 3, int64_t(h), int64_t(w)};
        auto input = Ort::Value::CreateTensor<float>(
            memory_, input_.data(), input_.size(), dims, 4);
        static constexpr const char *inputs[] = {"rgb"};
        static constexpr const char *outputs[] = {"edge0", "edge1", "edge2", "edge3"};
        auto results = session_->Run(Ort::RunOptions{nullptr},
                                     inputs, &input, 1, outputs, 4);
        if (results.size() != 4)
            throw std::runtime_error("Wrong number of TEED edge maps");
        std::array<const float *, 4> maps{};
        for (size_t i = 0; i < results.size(); ++i) {
            if (!results[i].IsTensor())
                throw std::runtime_error("TEED edge output is not a tensor");
            const auto info = results[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                info.GetShape() != std::vector<int64_t>{1, 1, int64_t(h), int64_t(w)})
                throw std::runtime_error("TEED output tensor shape changed");
            maps[i] = results[i].GetTensorData<float>();
        }
        output_scratch_.resize(pixels);
        auto& pixels_out=output_scratch_;
#ifdef SIGHT_HAVE_RUST_FRAME_CORE
        if (sight_teed_edges_to_gray_v1(maps[0], maps[1], maps[2], maps[3],
                                        pixels, safe_steps, pixels_out.data(),
                                        pixels_out.size()) != 1) {
            last_error_ = "Rust TEED four-map postprocessing rejected invalid tensor";
            return false;
        }
#else
        for (size_t p = 0; p < pixels; ++p) {
            for (const float *map : maps) {
                if (!std::isfinite(map[p])) {
                    last_error_ = "TEED produced nonfinite edges";
                    return false;
                }
            }
        }
        for (size_t p = 0; p < pixels; ++p) {
            const float logits = ((maps[0][p] + maps[1][p]) +
                                  maps[2][p] + maps[3][p]) * 0.25f;
            const double sigmoid = 1.0 / (1.0 + std::exp(-double(logits)));
            const double safe = safe_steps == 0 ? sigmoid :
                double(float(int32_t(float(sigmoid) * float(safe_steps + 1))) /
                       float(safe_steps));
            pixels_out[p] = uint8_t(std::clamp(safe * 255.0, 0.0, 255.0));
        }
#endif
        gray.swap(pixels_out);
        return true;
    } catch (const std::exception &error) {
        last_error_ = error.what();
        return false;
    }
}
} // namespace lll
