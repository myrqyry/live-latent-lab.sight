#include "preprocessors_processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace preprocessors {
namespace {

const std::vector<ModeSpec> kModes = {
    // GPU-free native algorithms. The OBS filter executes them on a worker.
    {"off", "Off (passthrough)", "General", Backend::Native},
    {"canny", "Canny Edge", "Lines", Backend::Native},
    {"binary", "Binary Lines", "Lines", Backend::Native},
    {"standard_lineart", "Lineart (luminance-adapted)", "Lines", Backend::Native},
    {"pyracanny", "PyraCanny (nine scales)", "Lines", Backend::Native},
    {"scribble_canny", "Scribble (Canny)", "Lines", Backend::Native},
    {"scribble_xdog", "Scribble XDoG", "Lines", Backend::Native},
    {"luminance", "Recolor: Luminance", "Color", Backend::Native},
    {"intensity", "Recolor: Intensity", "Color", Backend::Native},
    {"color_palette", "Color Palette", "Color", Backend::Native},
    {"tile_shuffle", "Tile Shuffle", "Color", Backend::Native},
    {"tile", "Tile (original pixels)", "Color", Backend::Native},

    // Exactly one filter type. Neural modes are in the inventory but not
    // advertised as functional without a tested backend and model.
    {"lineart_realistic", "Realistic Lineart (model)", "Lines", Backend::ModelRequired},
    {"lineart_coarse", "Coarse Lineart (model)", "Lines", Backend::ModelRequired},
    {"anime_lineart", "Anime Lineart (model)", "Lines", Backend::ModelRequired},
    {"manga_lineart", "Manga Lineart (model)", "Lines", Backend::ModelRequired},
    {"hed", "HED Soft Edge (model)", "Lines", Backend::ModelRequired},
    {"pidinet", "PiDiNet Soft Edge (model)", "Lines", Backend::ModelRequired},
    {"teed", "TEED Soft Edge (model)", "Lines", Backend::ModelRequired},
    {"anyline", "AnyLine (models)", "Lines", Backend::ModelRequired},
    {"mlsd", "M-LSD Lines (model)", "Lines", Backend::ModelRequired},
    {"scribble_hed", "Scribble HED (model)", "Lines", Backend::ModelRequired},
    {"scribble_pidinet", "Scribble PiDiNet (model)", "Lines", Backend::ModelRequired},
    {"diffusion_edge", "Diffusion Edge (model)", "Lines", Backend::ModelRequired},

    {"depth_midas", "MiDaS Depth (model)", "Depth and Normals", Backend::ModelRequired},
    {"depth_leres", "LeReS Depth (model)", "Depth and Normals", Backend::ModelRequired},
    {"depth_zoe", "Zoe Depth (model)", "Depth and Normals", Backend::ModelRequired},
    {"depth_anything", "Depth Anything (model)", "Depth and Normals", Backend::ModelRequired},
    {"depth_anything_v2", "Depth Anything V2 (model)", "Depth and Normals", Backend::ModelRequired},
    {"normal_midas", "MiDaS Normal (model)", "Depth and Normals", Backend::ModelRequired},
    {"normal_bae", "BAE Normal (model)", "Depth and Normals", Backend::ModelRequired},
    {"normal_dsine", "DSINE Normal (model)", "Depth and Normals", Backend::ModelRequired},
    {"metric3d_depth", "Metric3D Depth (model)", "Depth and Normals", Backend::ModelRequired},
    {"metric3d_normal", "Metric3D Normal (model)", "Depth and Normals", Backend::ModelRequired},
    {"hand_refiner", "Hand Refiner (model + hand input)", "Depth and Normals", Backend::ModelRequired},

    {"dwpose", "DWPose (model)", "Pose and Faces", Backend::ModelRequired},
    {"openpose", "OpenPose (model)", "Pose and Faces", Backend::ModelRequired},
    {"openpose_full", "OpenPose Full (model)", "Pose and Faces", Backend::ModelRequired},
    {"mediapipe_face", "MediaPipe Face Mesh (model)", "Pose and Faces", Backend::ModelRequired},
    {"animal_pose", "Animal Pose (model)", "Pose and Faces", Backend::ModelRequired},
    {"densepose", "DensePose (model)", "Pose and Faces", Backend::ModelRequired},

    {"oneformer_ade", "OneFormer ADE20K (model)", "Segmentation", Backend::ModelRequired},
    {"oneformer_coco", "OneFormer COCO (model)", "Segmentation", Backend::ModelRequired},
    {"uniformer", "UniFormer Segmentor (model)", "Segmentation", Backend::ModelRequired},
    {"anime_face_segment", "Anime Face Segment (model)", "Segmentation", Backend::ModelRequired},
    {"segment_anything", "Segment Anything (model + prompts)", "Segmentation", Backend::ModelRequired},

    {"unimatch_flow", "UniMatch Optical Flow (model + history)", "Motion", Backend::ModelRequired},
};

cv::Mat gray_from_rgba(const cv::Mat &rgba)
{
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}

void invert_if(cv::Mat &gray, bool invert)
{
    if (invert) cv::bitwise_not(gray, gray);
}

cv::Mat standard_lineart(const cv::Mat &rgba, const Settings &settings)
{
    cv::Mat gray, original, blur;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    gray.convertTo(original, CV_32FC1);
    const double sigma = std::clamp(settings.sigma, 0.25, 15.0);
    cv::GaussianBlur(original, blur, cv::Size(0, 0), sigma);
    // Luminance-adapted Standard Lineart. The reference uses
    // min(blur - original) across RGB, which collapses to a blank result
    // on common opposite-color boundaries in live camera footage.
    cv::Mat diff = blur - original;
    cv::Mat out(rgba.rows, rgba.cols, CV_32FC1);
    std::vector<float> nonzero;
    nonzero.reserve(out.total() / 8);
    const float threshold = static_cast<float>(settings.low_threshold) / 8.f;
    for (int y = 0; y < rgba.rows; ++y) {
        const auto *row = diff.ptr<float>(y);
        auto *dst = out.ptr<float>(y);
        for (int x = 0; x < rgba.cols; ++x) {
            const float value = std::clamp(row[x], 0.f, 255.f);
            dst[x] = value;
            if (value > threshold) nonzero.push_back(value);
        }
    }
    float median = 16.f;
    if (!nonzero.empty()) {
        const size_t mid = nonzero.size() / 2;
        std::nth_element(nonzero.begin(), nonzero.begin() + mid, nonzero.end());
        median = std::max(16.f, nonzero[mid]);
    }
    cv::Mat result;
    out.convertTo(result, CV_8U, 127.f / median);
    return result;
}

cv::Mat pyracanny(const cv::Mat &rgba, const Settings &settings)
{
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    cv::Mat accumulator;
    // Multi-scale, per-color-channel Canny, matching the broad PyraCanny
    // strategy without claiming bit-identical output to every Python version.
    for (int step = 2; step <= 10; ++step) {
        const double scale = step / 10.0;
        const int w = std::max(3, static_cast<int>(std::round(rgba.cols * scale)));
        const int h = std::max(3, static_cast<int>(std::round(rgba.rows * scale)));
        cv::Mat small;
        cv::resize(bgr, small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        std::vector<cv::Mat> channels;
        cv::split(small, channels);
        cv::Mat combined(h, w, CV_32FC1, cv::Scalar(0));
        for (const auto &channel : channels) {
            cv::Mat edge, e32;
            cv::Canny(channel, edge, settings.low_threshold, settings.high_threshold);
            edge.convertTo(e32, CV_32F, 1.f / 255.f);
            combined += e32;
        }
        if (accumulator.empty()) {
            accumulator = combined;
        } else {
            cv::Mat enlarged;
            cv::resize(accumulator, enlarged, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            accumulator = enlarged * 0.75f + combined * 0.25f;
        }
    }
    if (accumulator.size() != rgba.size())
        cv::resize(accumulator, accumulator, rgba.size(), 0, 0, cv::INTER_LINEAR);
    std::vector<float> values;
    values.assign(reinterpret_cast<const float *>(accumulator.datastart),
                  reinterpret_cast<const float *>(accumulator.dataend));
    if (values.empty()) return cv::Mat::zeros(rgba.size(), CV_8UC1);
    const auto at_quantile = [&](double ratio) {
        const auto index = static_cast<size_t>(ratio * (values.size() - 1));
        std::nth_element(values.begin(), values.begin() + index, values.end());
        return values[index];
    };
    const float low = at_quantile(0.01), high = at_quantile(0.99);
    cv::Mat normalized, output;
    normalized = (accumulator - low) * (255.f / std::max(1e-5f, high - low));
    normalized.convertTo(output, CV_8U);
    return output;
}

cv::Mat xdog(const cv::Mat &rgba, const Settings &settings)
{
    cv::Mat gray = gray_from_rgba(rgba);
    cv::Mat f, g1, g2;
    gray.convertTo(f, CV_32F, 1.f / 255.f);
    const double sigma = std::clamp(settings.sigma, 0.25, 10.0);
    cv::GaussianBlur(f, g1, cv::Size(0, 0), sigma);
    cv::GaussianBlur(f, g2, cv::Size(0, 0), sigma * 1.6);
    cv::Mat difference = g1 - 0.99 * g2;
    cv::Mat output(gray.size(), CV_8UC1);
    const float epsilon = static_cast<float>(settings.low_threshold) / 255.f;
    for (int y = 0; y < gray.rows; ++y) {
        auto *dst = output.ptr<uint8_t>(y);
        const auto *d = difference.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const float v = d[x] >= epsilon ? 1.f : 1.f + std::tanh(12.f * (d[x] - epsilon));
            dst[x] = static_cast<uint8_t>(std::clamp(v * 255.f, 0.f, 255.f));
        }
    }
    return output;
}

cv::Mat shuffle_tiles(const cv::Mat &rgba)
{
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    const int cells = 4;
    cv::Mat result = bgr.clone();
    for (int y = 0; y < cells; ++y)
        for (int x = 0; x < cells; ++x) {
            const int x0 = x * rgba.cols / cells;
            const int y0 = y * rgba.rows / cells;
            const int x1 = (x + 1) * rgba.cols / cells;
            const int y1 = (y + 1) * rgba.rows / cells;
            const int sx = (x * 3 + 1) % cells;
            const int sy = (y * 3 + 2) % cells;
            const cv::Rect source(sx * rgba.cols / cells, sy * rgba.rows / cells,
                                  (sx + 1) * rgba.cols / cells - sx * rgba.cols / cells,
                                  (sy + 1) * rgba.rows / cells - sy * rgba.rows / cells);
            cv::Mat patch = bgr(source);
            cv::resize(patch, result(cv::Rect(x0, y0, x1 - x0, y1 - y0)),
                       cv::Size(x1 - x0, y1 - y0), 0, 0, cv::INTER_LINEAR);
        }
    return result;
}

} // namespace

const std::vector<ModeSpec> &catalog() { return kModes; }

const ModeSpec *find_mode(std::string_view id)
{
    for (const auto &entry : kModes)
        if (id == entry.id) return &entry;
    return nullptr;
}

bool is_native(std::string_view id)
{
    const auto *mode = find_mode(id);
    return mode && mode->backend == Backend::Native;
}

bool process_rgba(const uint8_t *input, int width, int height, int stride,
                  std::string_view mode_id, const Settings &s,
                  std::vector<uint8_t> &result)
{
    if (!input || width < 8 || height < 8 || width > 8192 || height > 8192 ||
        stride < width * 4 || !is_native(mode_id))
        return false;
    try {
        cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t *>(input), stride);
        cv::Mat processed;
        const std::string mode(mode_id);
        if (mode == "off" || mode == "tile") {
            processed = rgba.clone();
        } else if (mode == "canny" || mode == "scribble_canny") {
            cv::Canny(gray_from_rgba(rgba), processed, s.low_threshold, s.high_threshold);
            if (mode == "scribble_canny") {
                cv::dilate(processed, processed, cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(3, 3)));
            }
            invert_if(processed, s.invert);
        } else if (mode == "binary") {
            cv::threshold(gray_from_rgba(rgba), processed, s.low_threshold,
                          255, cv::THRESH_BINARY);
            invert_if(processed, s.invert);
        } else if (mode == "standard_lineart") {
            processed = standard_lineart(rgba, s);
            invert_if(processed, s.invert);
        } else if (mode == "pyracanny") {
            processed = pyracanny(rgba, s);
            invert_if(processed, s.invert);
        } else if (mode == "scribble_xdog") {
            processed = xdog(rgba, s);
            invert_if(processed, s.invert);
        } else if (mode == "luminance") {
            processed = gray_from_rgba(rgba);
        } else if (mode == "intensity") {
            cv::Mat bgr;
            cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
            std::vector<cv::Mat> channels;
            cv::split(bgr, channels);
            cv::Mat accum;
            channels[0].convertTo(accum, CV_32F);
            for (int i = 1; i < 3; ++i) {
                cv::Mat temp; channels[i].convertTo(temp, CV_32F);
                accum += temp;
            }
            accum.convertTo(processed, CV_8U, 1.0 / 3.0);
        } else if (mode == "color_palette") {
            cv::Mat bgr;
            cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
            const int grid = std::clamp(s.palette_size, 4, 128);
            cv::Mat small;
            cv::resize(bgr, small, cv::Size(grid, grid), 0, 0, cv::INTER_AREA);
            cv::resize(small, processed, rgba.size(), 0, 0, cv::INTER_NEAREST);
            // Coarse palette quantization is intentionally visible, even when
            // source geometry happens to line up with the tile grid.
            processed = processed / 24 * 24;
        } else if (mode == "tile_shuffle") {
            processed = shuffle_tiles(rgba);
        } else {
            return false;
        }
        if (processed.size() != rgba.size()) return false;
        cv::Mat output;
        if (processed.channels() == 1)
            cv::cvtColor(processed, output, cv::COLOR_GRAY2RGBA);
        else if (processed.channels() == 3)
            cv::cvtColor(processed, output, cv::COLOR_BGR2RGBA);
        else
            output = processed.clone();

        // Never destroy source alpha/matting.
        std::vector<cv::Mat> in_channels, out_channels;
        cv::split(rgba, in_channels);
        cv::split(output, out_channels);
        out_channels[3] = in_channels[3];
        cv::merge(out_channels, output);
        if (!output.isContinuous()) output = output.clone();

        result.resize(output.total()*output.elemSize());
        std::copy(output.datastart,output.dataend,result.begin());
        return true;
    } catch (const cv::Exception &) {
        return false;
    }
}
} // namespace preprocessors
