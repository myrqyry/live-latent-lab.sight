#include "litert_keypoint_estimator.hpp"

#include <algorithm>
#include <cmath>

namespace lll {
namespace {

struct ModelShape {
    int width;
    int height;
    int keypoints;
    int x_bins;
    int y_bins;
};

ModelShape shape_for(LiteRtKeypointModel model)
{
    switch (model) {
    case LiteRtKeypointModel::Face98:
        return {256, 256, 98, 512, 512};
    case LiteRtKeypointModel::Hand21:
        return {256, 256, 21, 512, 512};
    case LiteRtKeypointModel::WholeBody133:
        return {192, 256, 133, 384, 512};
    case LiteRtKeypointModel::Pose17:
        return {192, 256, 17, 384, 512};
    }
    return {0, 0, 0, 0, 0};
}

float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

bool LiteRtKeypointEstimator::load(const std::filesystem::path &model_path,
                                   LiteRtKeypointModel model)
{
    const ModelShape shape = shape_for(model);
    if (shape.width == 0 || !std::filesystem::is_regular_file(model_path)) {
        return false;
    }

    model_path_ = model_path;
    model_ = model;
    input_width_ = shape.width;
    input_height_ = shape.height;
    keypoint_count_ = shape.keypoints;
    x_bins_ = shape.x_bins;
    y_bins_ = shape.y_bins;
    return true;
}

bool LiteRtKeypointEstimator::estimate(const uint8_t *rgb,
                                       int width,
                                       int height,
                                       std::vector<LiteRtKeypoint> &keypoints,
                                       LiteRtDevice device)
{
    if (!rgb || width <= 0 || height <= 0 || model_path_.empty()) {
        return false;
    }

    const float target_aspect = static_cast<float>(input_width_) / input_height_;
    const float source_aspect = static_cast<float>(width) / height;
    int crop_x = 0;
    int crop_y = 0;
    int crop_width = width;
    int crop_height = height;
    if (source_aspect > target_aspect) {
        crop_width = static_cast<int>(height * target_aspect);
        crop_x = (width - crop_width) / 2;
    } else if (source_aspect < target_aspect) {
        crop_height = static_cast<int>(width / target_aspect);
        crop_y = (height - crop_height) / 2;
    }

    std::vector<float> input(static_cast<size_t>(input_width_) * input_height_ * 3);
    constexpr float mean[3] = {123.675f, 116.28f, 103.53f};
    constexpr float standard_deviation[3] = {58.395f, 57.12f, 57.375f};
    for (int y = 0; y < input_height_; ++y) {
        const float source_y = crop_y + (y + 0.5f) * crop_height / input_height_ - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, height - 1);
        const int y1 = std::clamp(y0 + 1, 0, height - 1);
        const float fy = source_y - std::floor(source_y);
        for (int x = 0; x < input_width_; ++x) {
            const float source_x = crop_x + (x + 0.5f) * crop_width / input_width_ - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, width - 1);
            const int x1 = std::clamp(x0 + 1, 0, width - 1);
            const float fx = source_x - std::floor(source_x);
            for (int channel = 0; channel < 3; ++channel) {
                const auto pixel = [=](int px, int py) {
                    return static_cast<float>(rgb[(py * width + px) * 3 + channel]);
                };
                const float top = pixel(x0, y0) + (pixel(x1, y0) - pixel(x0, y0)) * fx;
                const float bottom = pixel(x0, y1) + (pixel(x1, y1) - pixel(x0, y1)) * fx;
                input[(channel * input_height_ + y) * input_width_ + x] =
                    (top + (bottom - top) * fy - mean[channel]) / standard_deviation[channel];
            }
        }
    }

    std::vector<std::vector<float>> outputs;
    if (!LiteRtRuntime::instance().run_multi(model_path_, device, input, outputs)) {
        return false;
    }
    if (outputs.size() != 2) {
        return false;
    }
    return decode_simcc(outputs[0], outputs[1], keypoint_count_, x_bins_, y_bins_,
                        static_cast<float>(crop_x), static_cast<float>(crop_y),
                        static_cast<float>(crop_width), static_cast<float>(crop_height),
                        input_width_, input_height_, keypoints);
}

bool LiteRtKeypointEstimator::decode_simcc(const std::vector<float> &x_output,
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
                                           std::vector<LiteRtKeypoint> &keypoints)
{
    if (keypoint_count <= 0 || x_bins <= 0 || y_bins <= 0 || input_width <= 0 ||
        input_height <= 0 || crop_width <= 0.0f || crop_height <= 0.0f ||
        x_output.size() != static_cast<size_t>(keypoint_count) * x_bins ||
        y_output.size() != static_cast<size_t>(keypoint_count) * y_bins) {
        return false;
    }

    keypoints.resize(keypoint_count);
    for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
        const auto x_begin = x_output.begin() + keypoint * x_bins;
        const auto y_begin = y_output.begin() + keypoint * y_bins;
        const auto x_max = std::max_element(x_begin, x_begin + x_bins);
        const auto y_max = std::max_element(y_begin, y_begin + y_bins);
        const int x_index = static_cast<int>(x_max - x_begin);
        const int y_index = static_cast<int>(y_max - y_begin);
        keypoints[keypoint] = {
            crop_x + (x_index / 2.0f) * crop_width / input_width,
            crop_y + (y_index / 2.0f) * crop_height / input_height,
            std::min(sigmoid(*x_max), sigmoid(*y_max)),
        };
    }
    return true;
}

}  // namespace lll
