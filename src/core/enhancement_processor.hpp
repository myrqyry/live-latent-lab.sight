#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

#ifdef USE_ONNX_MEDIAPIPE
#include <onnxruntime_cxx_api.h>
#endif

namespace lll {

    enum class EnhanceModelType {
        RealESRGAN,      // 4x Upscale
        AnimeGANv3,      // Style Transfer
        ZeroDCE          // Low Light Enhancement
    };

    class EnhancementProcessor {
    public:
        EnhancementProcessor();
        ~EnhancementProcessor();

        // Prevent copying
        EnhancementProcessor(const EnhancementProcessor&) = delete;
        EnhancementProcessor& operator=(const EnhancementProcessor&) = delete;

        /**
         * Load enhancement model
         */
        bool load(const char* model_path, EnhanceModelType type, bool use_gpu = true);
        
        bool is_ready() const { return ready_; }
        EnhanceModelType get_type() const { return type_; }

        /**
         * Process image
         * RealESRGAN: output size must be 4x input size.
         * Others: output size = input size.
         */
        bool process(const uint8_t* rgb_in, int w, int h, uint8_t* rgb_out, int out_w, int out_h);

    private:
        bool ready_ = false;
        EnhanceModelType type_ = EnhanceModelType::ZeroDCE;
        
#ifdef USE_ONNX_MEDIAPIPE
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        
        std::vector<char*> input_names_ptr_;
        std::vector<char*> output_names_ptr_;
        std::vector<const char*> input_names_;
        std::vector<const char*> output_names_;
#endif

        void process_esrgan(const uint8_t* in, int w, int h, uint8_t* out);
        void process_generic(const uint8_t* in, int w, int h, uint8_t* out);
    };

} // namespace lll
