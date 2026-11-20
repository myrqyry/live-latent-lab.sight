#pragma once

/**
 * Head Pose Estimator (SixDRepNet)
 * 
 * Estimates 6D head pose (Pitch, Yaw, Roll) from a cropped face image.
 * Model: SixDRepNet (LiteRT)
 * Input: 256x256 RGB face crop
 * Output: 3 float values (Pitch, Yaw, Roll in degrees)
 */

#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include <filesystem>

namespace lll {

struct HeadPose {
    float pitch = 0.0f; // Up/Down
    float yaw = 0.0f;   // Left/Right
    float roll = 0.0f;  // Tilt
};

class HeadPoseEstimator {
public:
    HeadPoseEstimator();
    ~HeadPoseEstimator();

    // Prevent copying
    HeadPoseEstimator(const HeadPoseEstimator&) = delete;
    HeadPoseEstimator& operator=(const HeadPoseEstimator&) = delete;

    /**
     * Load the SixDRepNet LiteRT model
     */
    bool load(const char* model_path, bool use_gpu = true);

    bool is_ready() const { return ready_; }

    /**
     * Estimate pose from a face crop
     * @param rgb Face crop image (RGB, row-major)
     * @param width Image width (should be close to 256 for best results)
     * @param height Image height
     * @return Estimated HeadPose (degrees)
     */
    HeadPose estimate(const uint8_t* rgb, int width, int height);

private:
    static constexpr int INPUT_SIZE = 256;
    
    // SixDRepNet Normalization (ImageNet)
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3] = {0.229f, 0.224f, 0.225f};

    bool ready_ = false;

    std::filesystem::path model_path_;
    bool use_gpu_ = true;
    std::vector<float> input_buffer_;

    void preprocess(const uint8_t* rgb, int w, int h);
};

} // namespace lll
