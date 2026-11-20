#pragma once

#include "litert_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lll {

enum class LiteRtKeypointModel { Pose17, Face98, Hand21, WholeBody133 };

struct LiteRtKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 0.0f;
};

class LiteRtKeypointEstimator {
public:
    bool load(const std::filesystem::path &model_path,
              LiteRtKeypointModel model);

    bool estimate(const uint8_t *rgb,
                  int width,
                  int height,
                  std::vector<LiteRtKeypoint> &keypoints,
                  LiteRtDevice device = LiteRtDevice::Gpu);

    static bool decode_simcc(const std::vector<float> &x_output,
                             const std::vector<float> &y_output,
                             int keypoint_count,
                             int x_bins,
                             int y_bins,
                             float crop_x,
                             float crop_y,
                             float crop_width,
                             float crop_height,
                             int input_width,
                             int input_height,
                             std::vector<LiteRtKeypoint> &keypoints);

private:
    std::filesystem::path model_path_;
    LiteRtKeypointModel model_ = LiteRtKeypointModel::Pose17;
    int input_width_ = 0;
    int input_height_ = 0;
    int keypoint_count_ = 0;
    int x_bins_ = 0;
    int y_bins_ = 0;
};

}  // namespace lll
