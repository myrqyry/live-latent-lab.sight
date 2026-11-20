#pragma once
/**
 * Depth Anything v2 - Monocular Depth Estimation
 * 
 * This module wraps a LiteRT depth model for real-time
 * depth estimation from a single RGB image.
 * 
 * Model: https://huggingface.co/litert-community/Depth-Anything-3-Small
 * Expected input: 518x518 RGB normalized with ImageNet statistics
 * Output: Relative depth map (inverse depth, higher = closer)
 * 
 * Usage:
 *   DepthEstimator estimator;
 *   if (estimator.load("depth_anything_3_small.tflite", true)) {
 *       std::vector<float> depth(w * h);
 *       estimator.estimate(rgb_data, w, h, depth.data());
 *   }
 */

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

namespace lll {

    enum class ModelType {
        DepthAnythingV2, // 518x518, ImageNet stats
        MiDaS_Small      // 256x256, [0.5] stats
    };

class DepthEstimator {
public:
    DepthEstimator();
    ~DepthEstimator();

    // Prevent copying
    DepthEstimator(const DepthEstimator&) = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;

    /**
     * Load the Depth model
     * @param model_path Path to the .tflite file
     * @param type Model type (default: DepthAnythingV2)
     * @param use_gpu Use GPU acceleration if available
     * @return true if model loaded successfully
     */
    bool load(const char* model_path, ModelType type = ModelType::DepthAnythingV2, bool use_gpu = true);
    
    /**
     * Check if model is ready for inference
     */
    bool is_ready() const { return ready_; }
    
    /**
     * Estimate depth from RGB image
     * @param rgb Input RGB image (3 bytes per pixel, row-major)
     * @param width Image width
     * @param height Image height
     * @param depth_out Output depth buffer (width * height floats)
     *                  Values in range [0, 1] where higher = closer
     * @return true if estimation succeeded
     */
    bool estimate(const uint8_t* rgb, int width, int height, float* depth_out);
    
    /**
     * Estimate depth and convert to 8-bit grayscale
     * @param rgb Input RGB image
     * @param width Image width
     * @param height Image height
     * @param depth_out Output grayscale buffer (width * height bytes)
     * @return true if estimation succeeded
     */
    bool estimate_u8(const uint8_t* rgb, int width, int height, uint8_t* depth_out);
    
    /**
     * Get the model's native input size
     */
    int get_input_size() const { return input_size_; }

private:
    ModelType model_type_ = ModelType::DepthAnythingV2;
    int input_size_ = 518;
    
    bool ready_ = false;
    
    std::filesystem::path model_path_;
    bool use_gpu_ = true;
    std::vector<float> input_buffer_;   // CHW normalized
    std::vector<float> output_buffer_;  // Raw depth
    
    // Preprocessing helpers
    void preprocess(const uint8_t* rgb, int src_w, int src_h);
    void postprocess(int dst_w, int dst_h, float* depth_out);
};

} // namespace lll
