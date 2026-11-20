#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

#ifdef USE_ONNX_MEDIAPIPE
#include <onnxruntime_cxx_api.h>
#endif

namespace lll {

    enum class SegModelType {
        U2Net,           // 320x320, standard normalization
        RMBG,            // 1024x1024, (x-0.5)/0.5
        InSPyReNet,      // 384x384 (SwinB) or 1024x1024, ImageNet norm
        General          // Generic fallback
    };

    class SegmentationProcessor {
    public:
        SegmentationProcessor();
        ~SegmentationProcessor();

        // Prevent copying
        SegmentationProcessor(const SegmentationProcessor&) = delete;
        SegmentationProcessor& operator=(const SegmentationProcessor&) = delete;

        /**
         * Load segmentation model
         * @param model_path Path to ONNX model
         * @param type Model type (defines input size and normalization)
         * @param use_gpu Enable GPU acceleration
         */
        bool load(const char* model_path, SegModelType type = SegModelType::U2Net, bool use_gpu = true);
        
        bool is_ready() const { return ready_; }

        /**
         * Generate alpha mask from RGB image
         * @param rgb Input RGB data (packed)
         * @param width Image width
         * @param height Image height
         * @param alpha_out Output alpha channel (8-bit, w*h)
         */
        bool process(const uint8_t* rgb, int width, int height, uint8_t* alpha_out);

    private:
        bool ready_ = false;
        SegModelType type_ = SegModelType::U2Net;
        int input_size_ = 320;
        
#ifdef USE_ONNX_MEDIAPIPE
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        
        std::vector<float> input_tensor_values_;
        std::vector<float> output_tensor_values_;
        
        std::vector<char*> input_names_ptr_;
        std::vector<char*> output_names_ptr_;
        std::vector<const char*> input_names_;
        std::vector<const char*> output_names_;
#endif

        void preprocess(const uint8_t* src, int w, int h);
    };

} // namespace lll
