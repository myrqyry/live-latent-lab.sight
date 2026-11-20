/**
 * Depth Anything v2 - Implementation
 * 
 * Monocular depth estimation using LiteRT.
 * Based on: https://github.com/DepthAnything/Depth-Anything-V2
 */

#include "depth_estimator.hpp"
#include "litert_runtime.hpp"

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <exception>
#include <string>

namespace lll {

DepthEstimator::DepthEstimator() {
    // Pre-allocate buffers for default size
    input_buffer_.resize(3 * 518 * 518);
    output_buffer_.resize(518 * 518);
}

DepthEstimator::~DepthEstimator() = default;

bool DepthEstimator::load(const char* model_path, ModelType type, bool use_gpu) {
    if (!model_path || !*model_path) {
        return false;
    }
    
    model_type_ = type;
    if (model_type_ == ModelType::MiDaS_Small) {
        input_size_ = 256;
    } else {
        input_size_ = 518;
    }
    
    // Resize buffers
    input_buffer_.resize(3 * input_size_ * input_size_);
    output_buffer_.resize(input_size_ * input_size_);
    
    model_path_ = model_path;
    use_gpu_ = use_gpu;
    if (!std::filesystem::is_regular_file(model_path_)) {
        return false;
    }
    ready_ = lll::LiteRtRuntime::instance().initialize();
    return ready_;
}

void DepthEstimator::preprocess(const uint8_t* rgb, int src_w, int src_h) {
    // Resize and normalize to CHW format
    const float x_ratio = static_cast<float>(src_w) / input_size_;
    const float y_ratio = static_cast<float>(src_h) / input_size_;
    
    const int channel_size = input_size_ * input_size_;
    float* r_channel = input_buffer_.data();
    float* g_channel = r_channel + channel_size;
    float* b_channel = g_channel + channel_size;
    
    float mean[3], std[3];
    if (model_type_ == ModelType::MiDaS_Small) {
        // MiDaS: [0.5, 0.5, 0.5]
        mean[0] = mean[1] = mean[2] = 0.5f;
        std[0] = std[1] = std[2] = 0.5f;
    } else {
        // Depth Anything: ImageNet
        mean[0] = 0.485f; mean[1] = 0.456f; mean[2] = 0.406f;
        std[0] = 0.229f; std[1] = 0.224f; std[2] = 0.225f;
    }
    
    for (int y = 0; y < input_size_; ++y) {
        for (int x = 0; x < input_size_; ++x) {
            // Nearest neighbor resizing
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = std::min(src_x, src_w - 1);
            src_y = std::min(src_y, src_h - 1);
            
            const int src_idx = (src_y * src_w + src_x) * 3;
            const int dst_idx = y * input_size_ + x;
            
            r_channel[dst_idx] = (rgb[src_idx + 0] / 255.0f - mean[0]) / std[0];
            g_channel[dst_idx] = (rgb[src_idx + 1] / 255.0f - mean[1]) / std[1];
            b_channel[dst_idx] = (rgb[src_idx + 2] / 255.0f - mean[2]) / std[2];
        }
    }
}

void DepthEstimator::postprocess(int dst_w, int dst_h, float* depth_out) {
    // Resize output from input_size_ x input_size_ to dst_w x dst_h
    // Also normalize to [0, 1] range
    
    // Find min/max for normalization
    float min_val = output_buffer_[0];
    float max_val = output_buffer_[0];
    for (size_t i = 1; i < output_buffer_.size(); ++i) {
        min_val = std::min(min_val, output_buffer_[i]);
        max_val = std::max(max_val, output_buffer_[i]);
    }
    
    const float range = max_val - min_val;
    const float scale = (range > 1e-6f) ? (1.0f / range) : 1.0f;
    
    const float x_ratio = static_cast<float>(input_size_) / dst_w;
    const float y_ratio = static_cast<float>(input_size_) / dst_h;
    
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = std::min(src_x, input_size_ - 1);
            src_y = std::min(src_y, input_size_ - 1);
            
            float raw_depth = output_buffer_[src_y * input_size_ + src_x];
            
            // Normalize to [0, 1] - higher values = closer (inverse depth)
            // Note: Depth Anything output is already relative inverse depth.
            depth_out[y * dst_w + x] = (raw_depth - min_val) * scale;
        }
    }
}

bool DepthEstimator::estimate(const uint8_t* rgb, int width, int height, float* depth_out) {
    if (!ready_ || !rgb || !depth_out || width <= 0 || height <= 0) {
        return false;
    }
    
    try {
        // Preprocess input
        preprocess(rgb, width, height);
        
        if (!LiteRtRuntime::instance().run(
                model_path_, use_gpu_ ? LiteRtDevice::Gpu : LiteRtDevice::Cpu,
                input_buffer_, output_buffer_, {.allow_cpu_fallback = use_gpu_})) {
            return false;
        }
        if (output_buffer_.size() !=
            static_cast<size_t>(input_size_) * input_size_) {
            return false;
        }
        
        // Postprocess and resize to target dimensions
        postprocess(width, height, depth_out);
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

bool DepthEstimator::estimate_u8(const uint8_t* rgb, int width, int height, uint8_t* depth_out) {
    if (!rgb || !depth_out || width <= 0 || height <= 0) {
        return false;
    }

    std::vector<float> depth_float(width * height);
    
    if (!estimate(rgb, width, height, depth_float.data())) {
        return false;
    }
    
    // Convert float [0,1] to uint8 [0,255]
    for (int i = 0; i < width * height; ++i) {
        depth_out[i] = static_cast<uint8_t>(std::clamp(depth_float[i] * 255.0f, 0.0f, 255.0f));
    }
    
    return true;
}

} // namespace lll
