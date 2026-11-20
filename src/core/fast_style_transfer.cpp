#include "fast_style_transfer.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace lll {

struct FastStyleTransfer::Impl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    
    // Model resolution (some models are dynamic, others fixed)
    // Common Fast Style Transfer models (e.g. from ONNX Zoo) are often dynamic or 224x224.
    // We will inspect input shape or assume a safe default and resize if needed.
    // For "Fast Neural Style" models (mosaic-9.onnx etc), acceptable inputs are usually flex.
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_names_str; 
    std::vector<std::string> output_names_str;
};

FastStyleTransfer::FastStyleTransfer() : impl(std::make_unique<Impl>()) {
    try {
        impl->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FastStyle");
        impl->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[FastStyle] Init failed: %s", e.what());
    }
}

FastStyleTransfer::~FastStyleTransfer() = default;

bool FastStyleTransfer::is_ready() const {
    return impl->session != nullptr;
}

bool FastStyleTransfer::load(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); 
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

        impl->session = std::make_unique<Ort::Session>(*impl->env, model_path.c_str(), session_options);

        // Query metadata to support variety of models (CartoonGAN vs FastNeuralStyle)
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Clear previous
        impl->input_names.clear();
        impl->output_names.clear();
        impl->input_names_str.clear();
        impl->output_names_str.clear();

        size_t num_input_nodes = impl->session->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = impl->session->GetInputNameAllocated(i, allocator);
            impl->input_names_str.push_back(name.get());
            impl->input_names.push_back(impl->input_names_str.back().c_str());
        }
        
        size_t num_output_nodes = impl->session->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = impl->session->GetOutputNameAllocated(i, allocator);
            impl->output_names_str.push_back(name.get());
            impl->output_names.push_back(impl->output_names_str.back().c_str());
        }

        blog(LOG_INFO, "[FastStyle] Loaded model: %s (Inputs: %zu, Outputs: %zu)", 
             model_path.c_str(), num_input_nodes, num_output_nodes);
             
        return true;
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[FastStyle] Failed to load model: %s", e.what());
        impl->session.reset();
        return false;
    }
}

// Helper: Resize Bilinear (Standard implementation)
static void resize_bilinear_rgb(const uint8_t* src, int sys_w, int sys_h, 
                                float* dst_r, float* dst_g, float* dst_b, int dst_w, int dst_h) {
                                
    float x_ratio = (float)(sys_w > 1 ? sys_w - 1 : 0) / (dst_w > 1 ? dst_w - 1 : 1);
    float y_ratio = (float)(sys_h > 1 ? sys_h - 1 : 0) / (dst_h > 1 ? dst_h - 1 : 1);
    
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float gx = x * x_ratio;
            float gy = y * y_ratio;
            int gxi = (int)gx;
            int gyi = (int)gy;
            
            float tx = gx - gxi;
            float ty = gy - gyi;
            
            int idx00 = (gyi * sys_w + gxi) * 3;
            int idx10 = (gyi * sys_w + (gxi + 1)) * 3;
            int idx01 = ((gyi + 1) * sys_w + gxi) * 3;
            int idx11 = ((gyi + 1) * sys_w + (gxi + 1)) * 3;
            
            // Channel 0 (Red) - assuming standard packing, but wait
            // ONNX models often expect RGB or BGR.
            // FastNeuralStyle (mosaic-9) expects: Float32 [1, 3, H, W]
            // We'll write to planar destination directly
            
            int dst_idx = (y * dst_w + x);
            
            // Helper for 3 channels
            for(int c=0; c<3; c++) {
                 float c00 = src[idx00 + c];
                 float c10 = src[idx10 + c];
                 float c01 = src[idx01 + c];
                 float c11 = src[idx11 + c];
                 
                 float interpolated = (1 - tx) * (1 - ty) * c00 + 
                                       tx * (1 - ty) * c10 + 
                                       (1 - tx) * ty * c01 + 
                                       tx * ty * c11;
                 
                 if (c==0) dst_r[dst_idx] = interpolated;
                 if (c==1) dst_g[dst_idx] = interpolated;
                 if (c==2) dst_b[dst_idx] = interpolated;
            }
        }
    }
}

bool FastStyleTransfer::process(const uint8_t* input_rgb, int width, int height, 
                                uint8_t* output_rgb, int out_width, int out_height) {
    if (!is_ready() || !input_rgb || !output_rgb) return false;

    try {
        // Prepare Input
        // Most styles models take dynamic size, but let's stick to output size or input size
        int infer_w = out_width;
        int infer_h = out_height;
        
        std::vector<float> input_tensor_values(infer_w * infer_h * 3);
        
        // Pointers to planar channels in the single vector
        float* ptr_r = &input_tensor_values[0];
        float* ptr_g = &input_tensor_values[infer_w * infer_h];
        float* ptr_b = &input_tensor_values[infer_w * infer_h * 2];
        
        // Resize and Planarize (HWC -> CHW)
        // Also: Fast Neural Style usually expects BGR 0-255.
        // Some expect RGB. Mosaic-9 from ONNX Zoo expects RGB [0, 255].
        // Whitebox Cartoonization usually expects RGB [-1, 1] or [0, 1] or [0, 255].
        // We will assume [0, 255] RGB float for generic "style transfer".
        
        resize_bilinear_rgb(input_rgb, width, height, ptr_r, ptr_g, ptr_b, infer_w, infer_h);
        
        std::vector<int64_t> input_shape = {1, 3, infer_h, infer_w};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *impl->memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // Run Inference
        auto output_tensors = impl->session->Run(
            Ort::RunOptions{nullptr}, 
            impl->input_names.data(), &input_tensor, 1, 
            impl->output_names.data(), 1);

        // Process Output
        float* out_data = output_tensors.front().GetTensorMutableData<float>();
        
        // Output is usually [1, 3, H, W]
        // Convert back to [H, W, 3] uint8
        size_t pixel_count = infer_w * infer_h;
        float* out_r = out_data;
        float* out_g = out_data + pixel_count;
        float* out_b = out_data + pixel_count * 2;
        
        for (int i = 0; i < pixel_count; i++) {
             // Clamp 0-255
             float r = std::min(255.0f, std::max(0.0f, out_r[i]));
             float g = std::min(255.0f, std::max(0.0f, out_g[i]));
             float b = std::min(255.0f, std::max(0.0f, out_b[i]));
             
             output_rgb[i * 3 + 0] = (uint8_t)r;
             output_rgb[i * 3 + 1] = (uint8_t)g;
             output_rgb[i * 3 + 2] = (uint8_t)b;
        }

        return true;
        
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[FastStyle] Inference failed: %s", e.what());
        return false;
    }
}

} // namespace lll
