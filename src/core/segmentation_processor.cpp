#include "segmentation_processor.hpp"
#include "../common/image-utils.hpp"
#include "../common/sight-log.h"
#include <util/bmem.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <thread>

namespace lll {

#ifdef USE_ONNX_MEDIAPIPE

SegmentationProcessor::SegmentationProcessor() 
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

SegmentationProcessor::~SegmentationProcessor() {
    for(char* p : input_names_ptr_) bfree(p);
    for(char* p : output_names_ptr_) bfree(p);
}

bool SegmentationProcessor::load(const char* model_path, SegModelType type, bool use_gpu) {
    ready_ = false;
    session_.reset();
    if (!model_path || !*model_path) return false;

    type_ = type;
    // Set parameters based on type
    if (type == SegModelType::U2Net) {
        input_size_ = 320; 
    } else if (type == SegModelType::RMBG) {
        input_size_ = 1024; // RMBG-1.4 usually 1024x1024 dynamic
    } else if (type == SegModelType::InSPyReNet) {
        input_size_ = 384; // SwinB standard size
    }

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SegProcessor");
        
        Ort::SessionOptions session_opts;
        // U2Net is convolution-heavy: using only one CPU thread measured
        // ~930ms per 320x320 frame on Ryzen 7 7700. Six workers measured
        // ~256ms while retaining physical cores for OBS video/graphics.
        // Keep the legacy processors' thread policy unchanged.
        const unsigned logical = std::thread::hardware_concurrency();
        const int cpu_threads = type_ == SegModelType::U2Net
            ? static_cast<int>(std::min(6u, std::max(1u, logical / 2u)))
            : 1;
        session_opts.SetIntraOpNumThreads(cpu_threads);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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

        // Reject a different ONNX model before interpreting its first output
        // as an alpha channel. The known U2Net export is float32 NCHW.
        if (type_ == SegModelType::U2Net) {
            if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1)
                throw std::runtime_error("U2Net requires one input and a mask output");
            auto in_type = session_->GetInputTypeInfo(0);
            auto out_type = session_->GetOutputTypeInfo(0);
            const auto in = in_type.GetTensorTypeAndShapeInfo();
            const auto out = out_type.GetTensorTypeAndShapeInfo();
            if (in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                in.GetShape() != std::vector<int64_t>{1, 3, 320, 320} ||
                out.GetShape() != std::vector<int64_t>{1, 1, 320, 320})
                throw std::runtime_error("Expected float32 U2Net 1x3x320x320 -> 1x1x320x320");
        }


        Ort::AllocatorWithDefaultOptions allocator;

        // Clear previous
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
        
        // Pre-allocate buffers
        input_tensor_values_.resize(input_size_ * input_size_ * 3);

        sight_logf(LOG_INFO, "[SegProcessor] Loaded %s (Size: %d)", model_path, input_size_);
        ready_ = true;
        return true;

    } catch (const std::exception& e) {
        sight_logf(LOG_ERROR, "[SegProcessor] Failed to load: %s", e.what());
        ready_ = false;
        return false;
    }
}

void SegmentationProcessor::preprocess(const uint8_t* src, int w, int h) {
    if (input_tensor_values_.size() != input_size_ * input_size_ * 3) {
        input_tensor_values_.resize(input_size_ * input_size_ * 3);
    }

    // Resize input to input_size_ x input_size_
    std::vector<uint8_t> resized(input_size_ * input_size_ * 3);
    
    // Bilinear resize RGB
    // We can use lll::resize_bilinear (but it expects 4 channels usually in our utilisation, or check implementation)
    // The implementation in image-utils is 4-channel. We need 3-channel resize.
    // Let's do a simple implementation here or use utility if available.
    // For now, implementing simple bilinear resize for 3-channel.
    
    float x_ratio = (float)(w - 1) / (input_size_ - 1);
    float y_ratio = (float)(h - 1) / (input_size_ - 1);
    
    for (int y = 0; y < input_size_; y++) {
        for (int x = 0; x < input_size_; x++) {
            int x_l = (int)(x_ratio * x);
            int y_l = (int)(y_ratio * y);
            int x_h = std::min(x_l + 1, w - 1);
            int y_h = std::min(y_l + 1, h - 1);
            float x_weight = (x_ratio * x) - x_l;
            float y_weight = (y_ratio * y) - y_l;
            
            int idx_ll = (y_l * w + x_l) * 3;
            int idx_lh = (y_l * w + x_h) * 3;
            int idx_hl = (y_h * w + x_l) * 3;
            int idx_hh = (y_h * w + x_h) * 3;
            
            int out_idx = (y * input_size_ + x) * 3;
            
            for (int c = 0; c < 3; c++) {
                 float val = 
                    src[idx_ll + c] * (1 - x_weight) * (1 - y_weight) +
                    src[idx_lh + c] * x_weight * (1 - y_weight) +
                    src[idx_hl + c] * (1 - x_weight) * y_weight +
                    src[idx_hh + c] * x_weight * y_weight;
                 resized[out_idx + c] = (uint8_t)val;
            }
        }
    }

    // Normalize
    // U2Net: (x - 0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) (ImageNet)
    // RMBG-1.4: (x - 0.5) / 0.5
    
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    
    if (type_ == SegModelType::RMBG) {
        mean[0] = 0.5f; mean[1] = 0.5f; mean[2] = 0.5f;
        std[0] = 0.5f;  std[1] = 0.5f;  std[2] = 0.5f;
    }
    // InSPyReNet (SwinB) uses ImageNet mean/std (default above)

    for (int i = 0; i < input_size_ * input_size_; i++) {
        float r = resized[i*3 + 0] / 255.0f;
        float g = resized[i*3 + 1] / 255.0f;
        float b = resized[i*3 + 2] / 255.0f;

        input_tensor_values_[i] = (r - mean[0]) / std[0];
        input_tensor_values_[input_size_ * input_size_ + i] = (g - mean[1]) / std[1];
        input_tensor_values_[2 * input_size_ * input_size_ + i] = (b - mean[2]) / std[2];
    }
}

bool SegmentationProcessor::process(const uint8_t* rgb, int width, int height, uint8_t* alpha_out) {
    if (!ready_ || !rgb || !alpha_out || width <= 0 || height <= 0)
        return false;

    preprocess(rgb, width, height);

    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};

    try {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values_.data(), input_tensor_values_.size(),
            input_shape.data(), input_shape.size());

        auto outputs = session_->Run(Ort::RunOptions{nullptr}, 
            input_names_.data(), &input_tensor, 1, 
            output_names_.data(), 1);

        // Output usually 1,1,H,W (logits or prob)
        if (outputs.size() != 1 || !outputs[0].IsTensor())
            throw std::runtime_error("U2Net produced no mask tensor");
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{1, 1, input_size_, input_size_})
            throw std::runtime_error("Unexpected segmentation mask type or shape");
        const float* out_data = outputs[0].GetTensorData<float>();
        
        // Resize mask back to original size
        // Bilinear resize required for smooth edges
        float x_ratio = width > 1 ? (float)(input_size_ - 1) / (width - 1) : 0;
        float y_ratio = height > 1 ? (float)(input_size_ - 1) / (height - 1) : 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float mx = x * x_ratio;
                float my = y * y_ratio;
                int x_l = (int)mx;
                int y_l = (int)my;
                int x_h = std::min(x_l + 1, input_size_ - 1);
                int y_h = std::min(y_l + 1, input_size_ - 1);
                float x_w = mx - x_l;
                float y_w = my - y_l;
                
                float val_ll = out_data[y_l * input_size_ + x_l];
                float val_lh = out_data[y_l * input_size_ + x_h];
                float val_hl = out_data[y_h * input_size_ + x_l];
                float val_hh = out_data[y_h * input_size_ + x_h];
                
                float val = 
                    val_ll * (1 - x_w) * (1 - y_w) +
                    val_lh * x_w * (1 - y_w) +
                    val_hl * (1 - x_w) * y_w +
                    val_hh * x_w * y_w;
                
                // Sigmoid if needed (most output logits)
                // U2Net / InSPyReNet output probability [0,1] usually?
                // U2Net outputs probability map [0,1].
                // If val is logits, Sigmoid: 1 / (1 + exp(-val))
                
                // Let's assume [0,1] range for now. If output is negative, apply sigmoid.
                // U2Net output is in range [0, 1] directly via softmax/sigmoid layer inside model usually.
                // But raw logits are common. 
                
                // Clamp and scale
                val = std::clamp(val, 0.0f, 1.0f);
                alpha_out[y * width + x] = (uint8_t)(val * 255.0f);
            }
        }
        return true;

    } catch (const std::exception& e) {
        sight_logf(LOG_ERROR, "[SegProcessor] Infer failed: %s", e.what());
        return false;
    }
}

#else

// Dummy Implementation
SegmentationProcessor::SegmentationProcessor() {}
SegmentationProcessor::~SegmentationProcessor() {}
bool SegmentationProcessor::load(const char*, SegModelType, bool) { return false; }
bool SegmentationProcessor::process(const uint8_t*, int, int, uint8_t*) { return false; }

#endif

} // namespace lll
