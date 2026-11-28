#include "enhancement_processor.hpp"
#ifdef SIGHT_ANIMEGAN_ONLY
#include "onnx_session.hpp"
#endif
#include "../common/sight-log.h"
#include <util/bmem.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <thread>

namespace lll {

#ifdef USE_ONNX_MEDIAPIPE

EnhancementProcessor::EnhancementProcessor() 
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

EnhancementProcessor::~EnhancementProcessor() {
    for(char* p : input_names_ptr_) bfree(p);
    for(char* p : output_names_ptr_) bfree(p);
}

bool EnhancementProcessor::load(const char* model_path, EnhanceModelType type, bool use_gpu) {
    ready_ = false;
    provider_=nullptr;
    last_error_.clear();
    session_.reset();
    if (!model_path || !*model_path) return false;
    type_ = type;

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EnhanceProcessor");
        
        Ort::SessionOptions session_opts;
        // Portrait Sketch is CPU-only in the validated build. One thread
        // underutilizes the available cores, but using all cores can starve
        // OBS. Reserve at least half the logical threads, capped at 4 workers.
        const unsigned logical = std::thread::hardware_concurrency();
        const int threads = type == EnhanceModelType::AnimeGANv3
            ? static_cast<int>(std::min(4u, std::max(1u, logical / 2)))
            : 1;
        session_opts.SetIntraOpNumThreads(threads);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef SIGHT_ANIMEGAN_ONLY
        session_ = open_onnx_session(*env_, model_path, session_opts, use_gpu, &provider_);
#else
        if (use_gpu) {
#if defined(USE_OPENVINO_EP)
            OrtOpenVINOProviderOptions ov_opts;
            ov_opts.device_type = "AUTO";
            session_opts.AppendExecutionProvider_OpenVINO(ov_opts);
#elif defined(USE_CUDA)
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            session_opts.AppendExecutionProvider_CUDA(cuda_opts);
#endif
        }

        session_ = std::make_unique<Ort::Session>(*env_, model_path, session_opts);
#endif
        // AnimeGANv3's actual ONNX contract is float32 NHWC -> NHWC.
        // Reject a different model rather than silently scrambling its pixels.
        if (type == EnhanceModelType::AnimeGANv3) {
            if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
                throw std::runtime_error("AnimeGAN requires one input and one output");
            auto input_type = session_->GetInputTypeInfo(0);
            auto output_type = session_->GetOutputTypeInfo(0);
            const auto input_info = input_type.GetTensorTypeAndShapeInfo();
            const auto output_info = output_type.GetTensorTypeAndShapeInfo();
            const auto input_shape = input_info.GetShape();
            const auto output_shape = output_info.GetShape();
            if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                input_shape.size() != 4 || output_shape.size() != 4 ||
                input_shape[3] != 3 || output_shape[3] != 3)
                throw std::runtime_error("AnimeGAN model must use float32 NHWC RGB tensors");
        }
        Ort::AllocatorWithDefaultOptions allocator;

        // Clear previous names
        for(char* p : input_names_ptr_) bfree(p); input_names_ptr_.clear();
        for(char* p : output_names_ptr_) bfree(p); output_names_ptr_.clear();
        input_names_.clear();
        output_names_.clear();

        size_t num_inputs = session_->GetInputCount();
        for(size_t i=0; i<num_inputs; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            char* copy = bstrdup(name.get());
            input_names_ptr_.push_back(copy);
            input_names_.push_back(copy);
        }

        size_t num_outputs = session_->GetOutputCount();
        for(size_t i=0; i<num_outputs; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            char* copy = bstrdup(name.get());
            output_names_ptr_.push_back(copy);
            output_names_.push_back(copy);
        }
        
        sight_logf(LOG_INFO, "[Enhance] Loaded %s (CPU intra-op threads: %d)",
                   model_path, threads);
        ready_ = true;
        return true;

    } catch (const std::exception& e) {
        last_error_=e.what();
        sight_logf(LOG_ERROR, "[Enhance] Failed to load: %s", e.what());
        ready_ = false;
        return false;
    }
}

bool EnhancementProcessor::process(const uint8_t* rgb_in, int w, int h, uint8_t* rgb_out, int out_w, int out_h) {
    if (!ready_ || !rgb_in || !rgb_out || w <= 0 || h <= 0 ||
        out_w <= 0 || out_h <= 0) return false;
    const int scale = type_ == EnhanceModelType::RealESRGAN ? 4 : 1;
    if (static_cast<int64_t>(w) * scale != out_w ||
        static_cast<int64_t>(h) * scale != out_h) return false;

    try {
        if (type_ == EnhanceModelType::RealESRGAN) {
            process_esrgan(rgb_in, w, h, rgb_out);
        } else {
            process_generic(rgb_in, w, h, rgb_out);
        }
        return true;
    } catch (const std::exception& e) {
        sight_logf(LOG_ERROR, "[Enhance] Process failed: %s", e.what());
        return false;
    }
}

void EnhancementProcessor::process_esrgan(const uint8_t* in, int w, int h, uint8_t* out) {
    // Real-ESRGAN: Usually [1, 3, H, W], input [0, 1]
    std::vector<float> input_data(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        input_data[i] = in[i*3+0] / 255.0f;
        input_data[w*h + i] = in[i*3+1] / 255.0f;
        input_data[2*w*h + i] = in[i*3+2] / 255.0f;
    }

    std::vector<int64_t> input_shape = {1, 3, h, w};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), output_names_.size());

    // Output [1, 3, 4H, 4W]
    const float* out_data = outputs[0].GetTensorData<float>();
    int ow = w * 4;
    int oh = h * 4;
    
    for (int i = 0; i < ow * oh; i++) {
        float r = std::clamp(out_data[i], 0.0f, 1.0f);
        float g = std::clamp(out_data[ow*oh + i], 0.0f, 1.0f);
        float b = std::clamp(out_data[2*ow*oh + i], 0.0f, 1.0f);
        
        out[i*3+0] = (uint8_t)(r * 255.0f);
        out[i*3+1] = (uint8_t)(g * 255.0f);
        out[i*3+2] = (uint8_t)(b * 255.0f);
    }
}

void EnhancementProcessor::process_generic(const uint8_t* in, int w, int h, uint8_t* out) {
    // This AnimeGANv3 ONNX export is NHWC. Zero-DCE retains its old NCHW path.
    const bool anime = type_ == EnhanceModelType::AnimeGANv3;
    const size_t pixels = static_cast<size_t>(w) * h;
    if (input_data_.capacity() < pixels * 3) ++input_growths_;
    input_data_.resize(pixels * 3);
    auto &input_data = input_data_;

    for (size_t i = 0; i < pixels; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            float v = in[i * 3 + channel] / 255.0f;
            if (anime) v = v * 2.0f - 1.0f;
            input_data[anime ? i * 3 + channel : channel * pixels + i] = v;
        }
    }

    const std::array<int64_t,4> shape = anime
        ? std::array<int64_t,4>{1, h, w, 3}
        : std::array<int64_t,4>{1, 3, h, w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(),
        shape.data(), shape.size());

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());
    if (outputs.size() != 1 || !outputs[0].IsTensor())
        throw std::runtime_error("Enhancement model did not produce one tensor");
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != pixels * 3)
        throw std::runtime_error("Unexpected enhancement model output type or size");
    if (anime) {
        const auto dimensions = info.GetShape();
        if (dimensions.size() != 4 || dimensions[0] != 1 ||
            dimensions[1] != h || dimensions[2] != w || dimensions[3] != 3)
            throw std::runtime_error("AnimeGAN output must match input NHWC dimensions");
    }

    const float* result = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            float v = result[anime ? i * 3 + channel : channel * pixels + i];
            if (anime) v = (v + 1.0f) * 0.5f;
            out[i * 3 + channel] = static_cast<uint8_t>(
                std::clamp(v, 0.0f, 1.0f) * 255.0f);
        }
    }
}

#else
EnhancementProcessor::EnhancementProcessor() {}
EnhancementProcessor::~EnhancementProcessor() {}
bool EnhancementProcessor::load(const char*, EnhanceModelType, bool) { return false; }
bool EnhancementProcessor::process(const uint8_t*, int, int, uint8_t*, int, int) { return false; }
#endif

} // namespace lll
