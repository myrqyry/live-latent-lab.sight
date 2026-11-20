/**
 * Head Pose Estimator - Implementation
 */

#include "head_pose_estimator.hpp"
#include "litert_runtime.hpp"

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <exception>

namespace lll {

HeadPoseEstimator::HeadPoseEstimator() {
    input_buffer_.resize(3 * INPUT_SIZE * INPUT_SIZE);
}

HeadPoseEstimator::~HeadPoseEstimator() = default;

bool HeadPoseEstimator::load(const char* model_path, bool use_gpu) {
    if (!model_path || !*model_path) return false;

    model_path_ = model_path;
    use_gpu_ = use_gpu;
    if (!std::filesystem::is_regular_file(model_path_)) {
        return false;
    }
    ready_ = lll::LiteRtRuntime::instance().initialize();
    return ready_;
}

void HeadPoseEstimator::preprocess(const uint8_t* rgb, int w, int h) {
    // Resize to 256x256 and Normalize
    const float x_ratio = static_cast<float>(w) / INPUT_SIZE;
    const float y_ratio = static_cast<float>(h) / INPUT_SIZE;
    
    const int area = INPUT_SIZE * INPUT_SIZE;
    float* r_ptr = input_buffer_.data();
    float* g_ptr = r_ptr + area;
    float* b_ptr = g_ptr + area;

    for (int y = 0; y < INPUT_SIZE; ++y) {
        for (int x = 0; x < INPUT_SIZE; ++x) {
            int sx = static_cast<int>(x * x_ratio);
            int sy = static_cast<int>(y * y_ratio);
            sx = std::min(sx, w - 1);
            sy = std::min(sy, h - 1);
            
            int idx = (sy * w + sx) * 3;
            
            // Normalize: (val/255 - mean) / std
            r_ptr[y * INPUT_SIZE + x] = (rgb[idx+0] / 255.0f - MEAN[0]) / STD[0];
            g_ptr[y * INPUT_SIZE + x] = (rgb[idx+1] / 255.0f - MEAN[1]) / STD[1];
            b_ptr[y * INPUT_SIZE + x] = (rgb[idx+2] / 255.0f - MEAN[2]) / STD[2];
        }
    }
}

HeadPose HeadPoseEstimator::estimate(const uint8_t* rgb, int w, int h) {
    HeadPose pose;
    if (!ready_ || !rgb || w <= 0 || h <= 0) return pose;

    try {
        preprocess(rgb, w, h);

        std::vector<float> output;
        if (!LiteRtRuntime::instance().run(
                model_path_, use_gpu_ ? LiteRtDevice::Gpu : LiteRtDevice::Cpu,
                input_buffer_, output, {.allow_cpu_fallback = use_gpu_}) ||
            output.size() < 3) {
            return pose;
        }
        pose.pitch = output[0];
        pose.yaw = output[1];
        pose.roll = output[2];
    } catch (const std::exception&) {
        return pose;
    }
    return pose;
}

} // namespace lll
