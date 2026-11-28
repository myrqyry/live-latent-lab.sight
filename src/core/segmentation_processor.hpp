#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "edge_tensor_io_stats.hpp"

#ifdef USE_ONNX_MEDIAPIPE
#include <onnxruntime_cxx_api.h>
#endif

namespace lll {
#if defined(USE_ONNX_MEDIAPIPE) && defined(SIGHT_WITH_CUDA_CUTOUT_IO)
class CutoutDeviceIO;
#endif

    enum class SegModelType {
        U2Net,           // 320x320, standard normalization
        RMBG,            // 1024x1024, (x-0.5)/0.5
        InSPyReNet,      // 384x384 (SwinB) or 1024x1024, ImageNet norm
        General,         // Generic fallback
        MODNet,          // RGB normalized to [-1,1], dynamic NCHW
        Longpipe         // RGB [0,1], fixed rectangular NCHW
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
         * @param use_gpu Prefer GPU acceleration
         * @param require_cuda Reject CPU fallback on the validated native path
         */
        bool load(const char* model_path, SegModelType type = SegModelType::U2Net, bool use_gpu = true, bool require_cuda = false);
        
        EdgeTensorIOStats tensor_io_stats() const;
        const char* configured_provider() const {return provider_;}
        const std::string& last_error() const {return last_error_;}
        bool is_ready() const { return ready_; }
        int input_edge() const {return input_width_>input_height_?input_width_:input_height_;}
        unsigned input_width() const {return input_width_;}
        unsigned input_height() const {return input_height_;}
#ifdef USE_ONNX_MEDIAPIPE
        // Longpipe flow requires the exact CPU canvas even in device mode.
        // Other device-mode families do not retain a host float canvas.
        const std::vector<float>& canvas_rgb() const {return input_tensor_values_;}
#else
        const std::vector<float>& canvas_rgb() const {static const std::vector<float> empty;return empty;}
#endif

        /**
         * Generate alpha mask from RGB image
         * @param rgb Input RGB data (packed)
         * @param width Image width
         * @param height Image height
         * @param alpha_out Output alpha channel (8-bit, w*h)
         */
        bool process(const uint8_t* rgb, int width, int height, uint8_t* alpha_out);

    private:
        const char* provider_=nullptr;
        std::string last_error_;
        bool ready_ = false;
        SegModelType type_ = SegModelType::U2Net;
        int input_width_ = 320, input_height_ = 320;
        
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
#ifdef SIGHT_WITH_CUDA_CUTOUT_IO
        std::unique_ptr<CutoutDeviceIO> device_io_;
#endif
#endif

        void preprocess(const uint8_t* src, int w, int h);
    };

} // namespace lll
