#include "nsfw_detector.hpp"

#ifdef USE_ONNX_MEDIAPIPE
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <iostream>

namespace lll {

struct NSFWDetector::Impl {
#ifdef USE_ONNX_MEDIAPIPE
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    
    // Model specific constants (OpenNSFW)
    const int input_width = 224;
    const int input_height = 224;
    const int channels = 3;
    
    // Input/Output names
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
#endif
};

NSFWDetector::NSFWDetector() : impl(std::make_unique<Impl>()) {
#ifdef USE_ONNX_MEDIAPIPE
    try {
        impl->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NSFWDetector");
        impl->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[NSFW] Init failed: %s", e.what());
    }
#else
    blog(LOG_WARNING, "[NSFW] ONNX Runtime not found during build. Safety filter disabled.");
#endif
}

NSFWDetector::~NSFWDetector() = default;

bool NSFWDetector::is_ready() const {
#ifdef USE_ONNX_MEDIAPIPE
    return impl->session != nullptr;
#else
    return false;
#endif
}

bool NSFWDetector::load(const std::string& model_path) {
#ifdef USE_ONNX_MEDIAPIPE
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

        impl->session = std::make_unique<Ort::Session>(*impl->env, model_path.c_str(), session_options);

        // Setup IO names (Simplified: assuming standard OpenNSFW signature)
        static char input_name[] = "input"; // Generic fallback
        static char output_name[] = "nsfw_score"; // Generic fallback
        
        impl->input_names = {input_name};
        impl->output_names = {output_name};

        blog(LOG_INFO, "[NSFW] Model loaded: %s", model_path.c_str());
        return true;
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[NSFW] Failed to load model: %s", e.what());
        impl->session.reset();
        return false;
    }
#else
    return false;
#endif
}

float NSFWDetector::predict(const uint8_t* rgb_data, int width, int height) {
    if (!is_ready() || !rgb_data) return 0.0f;

#ifdef USE_ONNX_MEDIAPIPE
    try {
        // Preprocess: Resize to 224x224 and Normalize
        std::vector<float> input_tensor_values(impl->input_width * impl->input_height * impl->channels);
        
        float x_ratio = (float)width / impl->input_width;
        float y_ratio = (float)height / impl->input_height;
        
        for (int y = 0; y < impl->input_height; y++) {
            for (int x = 0; x < impl->input_width; x++) {
                int src_x = (int)(x * x_ratio);
                int src_y = (int)(y * y_ratio);
                int src_idx = (src_y * width + src_x) * 3; // RGB
                
                // HWC -> BGR/RGB conversion if needed. OpenNSFW is usually BGR.
                // Mean Subtraction: 104, 117, 123
                float b = (float)rgb_data[src_idx + 2] - 104.0f;
                float g = (float)rgb_data[src_idx + 1] - 117.0f;
                float r = (float)rgb_data[src_idx + 0] - 123.0f;
                
                int dest_idx = (y * impl->input_width + x) * 3;
                input_tensor_values[dest_idx + 0] = b;
                input_tensor_values[dest_idx + 1] = g;
                input_tensor_values[dest_idx + 2] = r;
            }
        }
        
        // Create Tensor
        std::vector<int64_t> input_shape = {1, impl->input_width, impl->input_height, 3}; // NHWC
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *impl->memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // Run Inference
        auto output_tensors = impl->session->Run(
            Ort::RunOptions{nullptr}, 
            impl->input_names.data(), &input_tensor, 1, 
            impl->output_names.data(), 1);

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        
        // Assuming index 1 is NSFW score
        return floatarr[1];
        
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[NSFW] Inference failed: %s", e.what());
        return 0.0f; 
    }
#else
    return 0.0f;
#endif
}

} // namespace lll
