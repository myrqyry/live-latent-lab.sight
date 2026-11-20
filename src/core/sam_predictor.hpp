#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

#ifdef USE_ONNX_MEDIAPIPE
#include <onnxruntime_cxx_api.h>
#endif

namespace lll {

    struct SamPoint {
        float x;
        float y;
    };

    class SamPredictor {
    public:
        SamPredictor();
        ~SamPredictor();
        
        // Prevent copying
        SamPredictor(const SamPredictor&) = delete;
        SamPredictor& operator=(const SamPredictor&) = delete;

        /**
         * Load Encoder and Decoder models
         */
        bool load(const char* encoder_path, const char* decoder_path, bool use_gpu = true);
        
        bool is_ready() const { return ready_; }

        /**
         * Step 1: Encode image to embedding (Run once when image changes)
         * Input should be 1024x1024 usually for SAM, but MobileSAM might differ.
         * We will resize internally.
         */
        bool encode_image(const uint8_t* rgb, int width, int height);

        /**
         * Step 2: Decode mask from embedding + point prompt
         * @param point Normalized point (0.0-1.0)
         * @param mask_out Output Mask (width * height) - we can output resized to input resolution
         */
        bool predict_mask(SamPoint point, uint8_t* mask_out, int out_w, int out_h);

    private:
        bool ready_ = false;
        bool has_embedding_ = false;
        int encoder_input_size_ = 1024;
        
#ifdef USE_ONNX_MEDIAPIPE
        std::unique_ptr<Ort::Env> env_;
        
        // Encoder Session
        std::unique_ptr<Ort::Session> encoder_session_;
        std::vector<char*> enc_in_names_ptr_;
        std::vector<char*> enc_out_names_ptr_;
        std::vector<const char*> enc_in_names_;
        std::vector<const char*> enc_out_names_;
        
        // Decoder Session
        std::unique_ptr<Ort::Session> decoder_session_;
        std::vector<char*> dec_in_names_ptr_;
        std::vector<char*> dec_out_names_ptr_;
        std::vector<const char*> dec_in_names_;
        std::vector<const char*> dec_out_names_;
        
        Ort::MemoryInfo memory_info_;
        
        // State
        std::vector<float> image_embedding_; // Cached embedding
        std::vector<float> encoder_input_;   // Preprocessed image
#endif
        
        void preprocess_image(const uint8_t* src, int w, int h);
    };

} // namespace lll
