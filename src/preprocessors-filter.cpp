#include "common/preferences.hpp"
#include <obs-module.h>
#include <callback/proc.h>
#include <graphics/graphics.h>
#include <util/bmem.h>

#include "core/preprocessors_processor.hpp"
#include "core/onnx_depth_processor.hpp"
#include "core/onnx_teed_processor.hpp"
#include "core/onnx_pidinet_processor.hpp"
#include "core/onnx_hed_processor.hpp"
#include "core/pidinet_scribble.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>


namespace {
constexpr const char *kMode = "mode_id";
constexpr uint64_t kMaxOutputAgeMs = 2500;

struct Filter {
    obs_source_t *source = nullptr;
    gs_texrender_t *capture = nullptr;
    gs_stagesurf_t *stage = nullptr;
    gs_texture_t *output_texture = nullptr;
    uint32_t stage_w = 0, stage_h = 0;

    std::mutex input_mutex;
    std::condition_variable input_ready;
    std::thread worker;
    bool stopping = false;
    bool frame_queued = false;
    bool capture_in_flight = false;
    uint64_t generation = 1;
    std::string mode_id = "off";
    std::string depth_model_path;
    std::string teed_model_path;
    std::string pidinet_model_path;
    std::string hed_model_path;
    int hed_resolution = 256;
    bool hed_safe = false;
    int pidinet_resolution = 256;
    bool pidinet_safe = false;
    int teed_resolution = 256;
    int teed_safe_steps = 2;
    preprocessors::Settings settings;
    uint32_t incoming_w = 0, incoming_h = 0;
    std::vector<uint8_t> incoming;
    std::atomic<uint64_t> captures{0}, completed{0}, skipped{0}, failed{0};
    std::atomic<uint64_t> depth_completed{0}, teed_completed{0}, pidinet_completed{0};
    std::atomic<uint64_t> hed_completed{0};
    std::atomic<bool> clear_gpu_output{false};

    std::mutex output_mutex;
    uint32_t output_w = 0, output_h = 0;
    uint64_t output_generation = 0;
    std::chrono::steady_clock::time_point output_when{};
    bool upload_due = false;
    std::vector<uint8_t> output_pixels;
};

uint64_t millis_since(std::chrono::steady_clock::time_point moment)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - moment).count());
}

// Reuse Sight's validated ONNX Depth Anything V2 backend. Only RGBA->RGB
// conversion, divisible-by-14 sizing and full-resolution grayscale output
// belong here; there is no synthetic image fallback or external service.
bool depth_from_rgba(lll::OnnxDepthProcessor &processor,
                     const std::vector<uint8_t> &input,
                     uint32_t w, uint32_t h, std::vector<uint8_t> &output)
{
    if (!processor.ready() || w < 14 || h < 14 ||
        input.size() != static_cast<size_t>(w) * h * 4)
        return false;
    constexpr int edge = 392;
    const float scale = float(edge) / float(std::max(w,h));
    const auto dim = [&](uint32_t value) {
        return std::clamp(
            static_cast<int>(std::round(float(value) * scale / 14.f)) * 14,
            14, edge);
    };
    const int input_w = dim(w), input_h = dim(h);
    cv::Mat src(static_cast<int>(h),static_cast<int>(w),CV_8UC4,
                const_cast<uint8_t *>(input.data()));
    cv::Mat resized, rgb;
    cv::resize(src, resized, cv::Size(input_w,input_h), 0, 0,
               scale < 1.f ? cv::INTER_AREA : cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_RGBA2RGB);
    std::vector<uint8_t> depth;
    if (!processor.estimate(rgb.data, input_w, input_h, depth) ||
        depth.size() != size_t(input_w)*input_h)
        return false;
    cv::Mat depth_map(input_h, input_w, CV_8UC1, depth.data());
    cv::Mat full;
    cv::resize(depth_map,full,cv::Size(int(w),int(h)),0,0,cv::INTER_LINEAR);
    std::vector<uint8_t> next(input.size());
    for (size_t i = 0; i < size_t(w)*h; ++i) {
        const uint8_t d = full.data[i];
        next[i*4]=next[i*4+1]=next[i*4+2]=d;
        next[i*4+3]=input[i*4+3];
    }
    output.swap(next);
    return true;
}

// The actual TEED ONNX detector uses raw RGB8 values (0..255), four
// float32 edge maps, sigmoid and safe-step quantization. Both model execution
// and resizing happen on the worker; the OBS render thread only sees pixels.
bool teed_from_rgba(lll::OnnxTeedProcessor &processor,
                    const std::vector<uint8_t> &input, uint32_t w, uint32_t h,
                    int resolution, int safe_steps, std::vector<uint8_t> &output)
{
    if (!processor.ready() || w < 8 || h < 8 ||
        input.size() != size_t(w) * h * 4)
        return false;
    const float scale = float(resolution) / float(std::max(w, h));
    const auto dim = [&](uint32_t v) {
        return std::clamp(
            int(std::round(float(v) * scale / 8.f)) * 8, 8, 1024);
    };
    const int in_w = dim(w), in_h = dim(h);
    cv::Mat source(int(h), int(w), CV_8UC4,
                   const_cast<uint8_t *>(input.data()));
    cv::Mat resized, rgb;
    cv::resize(source, resized, cv::Size(in_w, in_h), 0, 0,
               scale < 1.f ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::cvtColor(resized, rgb, cv::COLOR_RGBA2RGB);
    std::vector<uint8_t> gray;
    if (!processor.estimate(rgb.data, uint32_t(in_w), uint32_t(in_h),
                            uint32_t(safe_steps), gray) ||
        gray.size() != size_t(in_w) * in_h)
        return false;
    cv::Mat result(in_h, in_w, CV_8UC1, gray.data());
    cv::Mat full;
    cv::resize(result, full, cv::Size(int(w), int(h)), 0, 0, cv::INTER_LINEAR);
    std::vector<uint8_t> next(input.size());
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const auto value = full.data[i];
        next[4 * i] = next[4 * i + 1] = next[4 * i + 2] = value;
        next[4 * i + 3] = input[4 * i + 3];
    }
    output.swap(next);
    return true;
}

// PiDiNet's actual local model expects BGR float32 normalized by 255,
// not TEED's raw RGB. Preserve source alpha and return full-size edges.
bool pidinet_from_rgba(lll::OnnxPidiNetProcessor &processor,
                       const std::vector<uint8_t> &input, uint32_t w, uint32_t h,
                       int resolution, bool safe, bool scribble,
                       std::vector<uint8_t> &output)
{
    if (!processor.ready() || w < 8 || h < 8 ||
        input.size() != size_t(w) * h * 4)
        return false;
    const float scale = float(resolution) / float(std::max(w, h));
    const auto dim = [&](uint32_t v) {
        return std::clamp(
            int(std::round(float(v) * scale / 8.f)) * 8, 8, 1024);
    };
    const int in_w = dim(w), in_h = dim(h);
    cv::Mat source(int(h), int(w), CV_8UC4,
                   const_cast<uint8_t *>(input.data()));
    cv::Mat resized, bgr;
    cv::resize(source, resized, cv::Size(in_w, in_h), 0, 0,
               scale < 1.f ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::cvtColor(resized, bgr, cv::COLOR_RGBA2BGR);
    std::vector<uint8_t> gray;
    if (!processor.estimate(bgr.data, uint32_t(in_w), uint32_t(in_h),
                            safe, gray) || gray.size() != size_t(in_w) * in_h)
        return false;
    if (scribble && !lll::pidinet_scribble_edges(
            gray, uint32_t(in_w), uint32_t(in_h)))
        return false;
    cv::Mat result(in_h, in_w, CV_8UC1, gray.data());
    cv::Mat full;
    cv::resize(result, full, cv::Size(int(w), int(h)), 0, 0, cv::INTER_LINEAR);
    std::vector<uint8_t> next(input.size());
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const auto value = full.data[i];
        next[4 * i] = next[4 * i + 1] = next[4 * i + 2] = value;
        next[4 * i + 3] = input[4 * i + 3];
    }
    output.swap(next);
    return true;
}

// Actual HED has five multiscale neural edge projections. All resizing,
// ONNX execution, sigmoid and optional directional scribble NMS stay on the
// bounded worker. Native Sight graphics code only uploads finished RGBA.
bool hed_from_rgba(lll::OnnxHedProcessor &processor,
                   const std::vector<uint8_t> &input, uint32_t w, uint32_t h,
                   int resolution, bool safe, bool scribble,
                   std::vector<uint8_t> &output)
{
    if (!processor.ready() || w < 16 || h < 16 ||
        input.size() != size_t(w) * h * 4)
        return false;
    const float scale = float(resolution) / float(std::max(w, h));
    const auto dim = [&](uint32_t v) {
        return std::clamp(int(std::round(float(v) * scale / 16.f)) * 16, 16, 1024);
    };
    const int in_w = dim(w), in_h = dim(h);
    cv::Mat source(int(h), int(w), CV_8UC4,
                   const_cast<uint8_t *>(input.data()));
    cv::Mat resized, rgb;
    cv::resize(source, resized, cv::Size(in_w, in_h), 0, 0,
               scale < 1.f ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::cvtColor(resized, rgb, cv::COLOR_RGBA2RGB);
    std::vector<uint8_t> gray;
    if (!processor.estimate(rgb.data, uint32_t(in_w), uint32_t(in_h),
                            safe, gray) ||
        gray.size() != size_t(in_w) * in_h)
        return false;
    if (scribble && !lll::pidinet_scribble_edges(
            gray, uint32_t(in_w), uint32_t(in_h)))
        return false;
    cv::Mat result(in_h, in_w, CV_8UC1, gray.data());
    cv::Mat full;
    cv::resize(result, full, cv::Size(int(w), int(h)), 0, 0, cv::INTER_LINEAR);
    std::vector<uint8_t> next(input.size());
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const uint8_t value = full.data[i];
        next[4 * i] = next[4 * i + 1] = next[4 * i + 2] = value;
        next[4 * i + 3] = input[4 * i + 3];
    }
    output.swap(next);
    return true;
}

void worker_main(Filter *filter)
{
    lll::OnnxDepthProcessor depth_processor;
    std::string loaded_path;
    uint64_t last_attempt_generation = 0;
    lll::OnnxTeedProcessor teed_processor;
    std::string teed_loaded_path;
    uint64_t teed_last_attempt_generation = 0;
    lll::OnnxPidiNetProcessor pidinet_processor;
    std::string pidinet_loaded_path;
    uint64_t pidinet_last_attempt_generation = 0;
    lll::OnnxHedProcessor hed_processor;
    std::string hed_loaded_path;
    uint64_t hed_last_attempt_generation = 0;
    for (;;) {
        std::vector<uint8_t> input;
        std::string mode;
        preprocessors::Settings settings;
        std::string depth_model_path, teed_model_path, pidinet_model_path;
        std::string hed_model_path;
        int hed_resolution = 256;
        bool hed_safe = false;
        int teed_resolution = 256, teed_safe_steps = 2;
        int pidinet_resolution = 256;
        bool pidinet_safe = false;
        uint32_t width = 0, height = 0;
        uint64_t request_generation = 0;
        {
            std::unique_lock lock(filter->input_mutex);
            filter->input_ready.wait(lock, [&] {
                return filter->stopping || filter->frame_queued;
            });
            if (filter->stopping) return;
            input = std::move(filter->incoming);
            width = filter->incoming_w;
            height = filter->incoming_h;
            mode = filter->mode_id;
            settings = filter->settings;
            depth_model_path = filter->depth_model_path;
            teed_model_path = filter->teed_model_path;
            teed_resolution = filter->teed_resolution;
            teed_safe_steps = filter->teed_safe_steps;
            pidinet_model_path = filter->pidinet_model_path;
            pidinet_resolution = filter->pidinet_resolution;
            pidinet_safe = filter->pidinet_safe;
            hed_model_path = filter->hed_model_path;
            hed_resolution = filter->hed_resolution;
            hed_safe = filter->hed_safe;
            request_generation = filter->generation;
            filter->frame_queued = false;
        }
        std::vector<uint8_t> result;
        bool valid = false;
        if (mode == "depth_anything_v2") {
            // Loading and inference stay on the worker: never block OBS
            // render, UI, or network. Retry a failed path only after an
            // explicit setting change (new generation).
            if (!depth_model_path.empty() &&
                loaded_path != depth_model_path &&
                request_generation != last_attempt_generation) {
                last_attempt_generation = request_generation;
                if (depth_processor.load(depth_model_path)) {
                    loaded_path = depth_model_path;
                    blog(LOG_INFO, "[Sight/Preprocessors] Native Depth Anything V2 "
                         "model loaded: %s", depth_model_path.c_str());
                } else {
                    loaded_path.clear();
                    blog(LOG_ERROR, "[Sight/Preprocessors] Depth model load failed: %s",
                         depth_processor.last_error().c_str());
                }
            }
            if (depth_model_path.empty() &&
                request_generation != last_attempt_generation) {
                last_attempt_generation = request_generation;
                blog(LOG_WARNING, "[Sight/Preprocessors] Depth Anything V2 "
                     "requires a local ONNX model path; no inference performed");
            }
            if (loaded_path == depth_model_path && depth_processor.ready()) {
                valid = depth_from_rgba(depth_processor, input, width, height, result);
                if (!valid) {
                    blog(LOG_ERROR, "[Sight/Preprocessors] Native Depth Anything V2 "
                         "inference failed: %s",
                         depth_processor.last_error().c_str());
                }
            }
        } else if (mode == "teed") {
            if (!teed_model_path.empty() &&
                teed_loaded_path != teed_model_path &&
                request_generation != teed_last_attempt_generation) {
                teed_last_attempt_generation = request_generation;
                if (teed_processor.load(teed_model_path)) {
                    teed_loaded_path = teed_model_path;
                    blog(LOG_INFO, "[Sight/Preprocessors] Native TEED ONNX "
                         "model loaded: %s", teed_model_path.c_str());
                } else {
                    teed_loaded_path.clear();
                    blog(LOG_ERROR, "[Sight/Preprocessors] Native TEED "
                         "model load failed: %s",
                         teed_processor.last_error().c_str());
                }
            }
            if (teed_model_path.empty() &&
                request_generation != teed_last_attempt_generation) {
                teed_last_attempt_generation = request_generation;
                blog(LOG_WARNING, "[Sight/Preprocessors] TEED requires a "
                     "local exported ONNX model; passing through");
            }
            if (teed_loaded_path == teed_model_path && teed_processor.ready()) {
                valid = teed_from_rgba(teed_processor, input, width, height,
                                       teed_resolution, teed_safe_steps, result);
                if (!valid)
                    blog(LOG_ERROR, "[Sight/Preprocessors] Native TEED "
                         "inference failed: %s",
                         teed_processor.last_error().c_str());
            }
        } else if (mode == "hed" || mode == "scribble_hed") {
            if (!hed_model_path.empty() &&
                hed_loaded_path != hed_model_path &&
                request_generation != hed_last_attempt_generation) {
                hed_last_attempt_generation = request_generation;
                if (hed_processor.load(hed_model_path)) {
                    hed_loaded_path = hed_model_path;
                    blog(LOG_INFO, "[Sight/Preprocessors] Native HED ONNX "
                         "model loaded: %s", hed_model_path.c_str());
                } else {
                    hed_loaded_path.clear();
                    blog(LOG_ERROR, "[Sight/Preprocessors] Native HED load failed: %s",
                         hed_processor.last_error().c_str());
                }
            }
            if (hed_model_path.empty() &&
                request_generation != hed_last_attempt_generation) {
                hed_last_attempt_generation = request_generation;
                blog(LOG_WARNING, "[Sight/Preprocessors] HED requires a "
                     "local ONNX checkpoint, using passthrough");
            }
            if (hed_loaded_path == hed_model_path && hed_processor.ready()) {
                valid = hed_from_rgba(hed_processor, input, width, height,
                                      hed_resolution, hed_safe,
                                      mode == "scribble_hed", result);
                if (!valid)
                    blog(LOG_ERROR, "[Sight/Preprocessors] HED inference failed: %s",
                         hed_processor.last_error().c_str());
            }
        } else if (mode == "pidinet" || mode == "scribble_pidinet") {
            if (!pidinet_model_path.empty() &&
                pidinet_loaded_path != pidinet_model_path &&
                request_generation != pidinet_last_attempt_generation) {
                pidinet_last_attempt_generation = request_generation;
                if (pidinet_processor.load(pidinet_model_path)) {
                    pidinet_loaded_path = pidinet_model_path;
                    blog(LOG_INFO, "[Sight/Preprocessors] Native PiDiNet ONNX "
                         "model loaded: %s", pidinet_model_path.c_str());
                } else {
                    pidinet_loaded_path.clear();
                    blog(LOG_ERROR, "[Sight/Preprocessors] PiDiNet model "
                         "load failed: %s", pidinet_processor.last_error().c_str());
                }
            }
            if (pidinet_model_path.empty() &&
                request_generation != pidinet_last_attempt_generation) {
                pidinet_last_attempt_generation = request_generation;
                blog(LOG_WARNING, "[Sight/Preprocessors] PiDiNet requires a "
                     "local ONNX model; using passthrough");
            }
            if (pidinet_loaded_path == pidinet_model_path &&
                pidinet_processor.ready()) {
                valid = pidinet_from_rgba(pidinet_processor, input, width,
                                          height, pidinet_resolution, pidinet_safe,
                                          mode == "scribble_pidinet", result);
                if (!valid)
                    blog(LOG_ERROR, "[Sight/Preprocessors] Native PiDiNet "
                         "inference failed: %s",
                         pidinet_processor.last_error().c_str());
            }
        } else {
            valid = preprocessors::process_rgba(input.data(),
                static_cast<int>(width), static_cast<int>(height),
                static_cast<int>(width) * 4, mode, settings, result);
        }
        {
            std::lock_guard lock(filter->input_mutex);
            if (request_generation != filter->generation)
                continue; // Model or mode changed. Never upload stale pixels.
            if (valid) {
                std::lock_guard out_lock(filter->output_mutex);
                filter->output_pixels = std::move(result);
                filter->output_w = width;
                filter->output_h = height;
                filter->output_generation = request_generation;
                filter->output_when = std::chrono::steady_clock::now();
                filter->upload_due = true;
                ++filter->completed;
                if (mode == "depth_anything_v2")
                    ++filter->depth_completed;
                if (mode == "teed")
                    ++filter->teed_completed;
                if (mode == "pidinet" || mode == "scribble_pidinet")
                    ++filter->pidinet_completed;
                if (mode == "hed" || mode == "scribble_hed")
                    ++filter->hed_completed;
            } else {
                ++filter->failed;
            }
            filter->capture_in_flight = false;
        }
    }
}

void stats_proc(void *data, calldata_t *args)
{
    const auto *f = static_cast<Filter *>(data);
    calldata_set_int(args, "captures", static_cast<int64_t>(f->captures.load()));
    calldata_set_int(args, "completed", static_cast<int64_t>(f->completed.load()));
    calldata_set_int(args, "skipped", static_cast<int64_t>(f->skipped.load()));
    calldata_set_int(args, "failed", static_cast<int64_t>(f->failed.load()));
    calldata_set_int(args, "depth_completed",
                    static_cast<int64_t>(f->depth_completed.load()));
}

void teed_stats_proc(void *data, calldata_t *args)
{
    auto *f = static_cast<Filter *>(data);
    calldata_set_int(args, "teed_completed",
                     static_cast<int64_t>(f->teed_completed.load()));
}

void pidinet_stats_proc(void *data, calldata_t *args)
{
    auto *f = static_cast<Filter *>(data);
    calldata_set_int(args, "pidinet_completed",
                     static_cast<int64_t>(f->pidinet_completed.load()));
}

void hed_stats_proc(void *data, calldata_t *args)
{
    auto *f = static_cast<Filter *>(data);
    calldata_set_int(args, "hed_completed",
                     static_cast<int64_t>(f->hed_completed.load()));
}

void update(void *data, obs_data_t *settings)
{
    auto *f = static_cast<Filter *>(data);
    const std::string requested = obs_data_get_string(settings, kMode);
    const auto *mode = preprocessors::find_mode(requested);
    // Missing or invalid saved modes resolve to passthrough. No imitation
    // of model-based semantics under the native provider.
    const std::string normalized = mode &&
        (mode->backend == preprocessors::Backend::Native ||
         requested == "depth_anything_v2" ||
         requested == "teed" || requested == "pidinet" ||
         requested == "scribble_pidinet" || requested == "hed" ||
         requested == "scribble_hed") ? requested : "off";
    if (requested != "off" && normalized == "off") {
        blog(LOG_WARNING, "[LLL/Preprocessors] Unavailable mode '%s': "
             "neural backend not connected or invalid ID; using passthrough",
             requested.c_str());
    }
    preprocessors::Settings opts;
    opts.low_threshold = std::clamp(static_cast<int>(
        obs_data_get_int(settings, "low")), 0, 255);
    opts.high_threshold = std::clamp(static_cast<int>(
        obs_data_get_int(settings, "high")), 0, 255);
    if (opts.high_threshold < opts.low_threshold)
        std::swap(opts.low_threshold, opts.high_threshold);
    opts.sigma = std::clamp(obs_data_get_double(settings, "sigma"), 0.25, 15.0);
    opts.invert = obs_data_get_bool(settings, "invert");
    opts.palette_size = std::clamp(static_cast<int>(
        obs_data_get_int(settings, "palette_size")), 4, 128);
    {
        std::lock_guard lock(f->input_mutex);
        f->mode_id = normalized;
        f->depth_model_path = obs_data_get_string(settings, "depth_model_path");
        f->teed_model_path = obs_data_get_string(settings, "teed_model_path");
        f->teed_resolution = std::clamp(
            int(obs_data_get_int(settings, "teed_resolution")), 128, 768);
        f->teed_safe_steps = std::clamp(
            int(obs_data_get_int(settings, "teed_safe_steps")), 0, 10);
        f->pidinet_model_path = obs_data_get_string(settings, "pidinet_model_path");
        f->pidinet_resolution = std::clamp(
            int(obs_data_get_int(settings, "pidinet_resolution")), 128, 768);
        f->pidinet_safe = obs_data_get_bool(settings, "pidinet_safe");
        f->hed_model_path = obs_data_get_string(settings, "hed_model_path");
        f->hed_resolution = std::clamp(
            int(obs_data_get_int(settings, "hed_resolution")), 128, 768);
        f->hed_safe = obs_data_get_bool(settings, "hed_safe");
        f->settings = opts;
        ++f->generation;
        f->incoming.clear();
        f->frame_queued = false;
        f->capture_in_flight = false;
    }
    {
        std::lock_guard lock(f->output_mutex);
        f->output_pixels.clear();
        f->upload_due = false;
    }
    f->clear_gpu_output = true;
}

void *create(obs_data_t *settings, obs_source_t *source)
{
    std::unique_ptr<Filter> f(new Filter);
    f->source = source;
    update(f.get(), settings);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_preprocessor_stats_v1(out int captures, out int completed, out int skipped, out int failed, out int depth_completed)",
        stats_proc, f.get());
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_preprocessor_teed_stats_v1(out int teed_completed)",
        teed_stats_proc, f.get());
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_preprocessor_pidinet_stats_v1(out int pidinet_completed)",
        pidinet_stats_proc, f.get());
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_preprocessor_hed_stats_v1(out int hed_completed)",
        hed_stats_proc, f.get());
    f->worker = std::thread(worker_main, f.get());
    return f.release();
}

void destroy(void *data)
{
    auto *f = static_cast<Filter *>(data);
    {
        std::lock_guard lock(f->input_mutex);
        f->stopping = true;
        f->incoming.clear();
        f->frame_queued = false;
    }
    f->input_ready.notify_one();
    if (f->worker.joinable()) f->worker.join();
    obs_enter_graphics();
    if (f->capture) gs_texrender_destroy(f->capture);
    if (f->stage) gs_stagesurface_destroy(f->stage);
    if (f->output_texture) gs_texture_destroy(f->output_texture);
    obs_leave_graphics();
    delete f;
}

void defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, kMode, "standard_lineart");
    obs_data_set_default_int(settings, "low", 64);
    obs_data_set_default_int(settings, "high", 128);
    obs_data_set_default_double(settings, "sigma", 2.0);
    obs_data_set_default_bool(settings, "invert", false);
    obs_data_set_default_int(settings, "palette_size", 32);
    obs_data_set_default_string(settings, "depth_model_path", "");
    obs_data_set_default_string(settings, "teed_model_path", "");
    obs_data_set_default_int(settings, "teed_resolution", 256);
    obs_data_set_default_int(settings, "teed_safe_steps", 2);
    obs_data_set_default_string(settings, "pidinet_model_path", "");
    obs_data_set_default_int(settings, "pidinet_resolution", 256);
    obs_data_set_default_bool(settings, "pidinet_safe", false);
    obs_data_set_default_string(settings, "hed_model_path", "");
    obs_data_set_default_int(settings, "hed_resolution", 256);
    obs_data_set_default_bool(settings, "hed_safe", false);
    obs_data_set_default_string(settings, "backend_status",
        "Native OpenCV, Depth Anything V2, TEED, PiDiNet and HED ONNX. "
        "Other model-backed preprocessors remain unimplemented.");
}

void set_control_visibility(obs_properties_t *properties, const std::string &mode)
{
    const bool edge = mode == "canny" || mode == "pyracanny" ||
                      mode == "scribble_canny" || mode == "scribble_xdog" ||
                      mode == "binary" || mode == "standard_lineart";
    const bool blur = mode == "standard_lineart" || mode == "scribble_xdog";
    if (auto *p = obs_properties_get(properties, "low"))
        obs_property_set_visible(p, edge);
    if (auto *p = obs_properties_get(properties, "high"))
        obs_property_set_visible(p, mode == "canny" ||
            mode == "pyracanny" || mode == "scribble_canny");
    if (auto *p = obs_properties_get(properties, "sigma"))
        obs_property_set_visible(p, blur);
    if (auto *p = obs_properties_get(properties, "invert"))
        obs_property_set_visible(p, edge);
    if (auto *p = obs_properties_get(properties, "palette_size"))
        obs_property_set_visible(p, mode == "color_palette");
    if (auto *p = obs_properties_get(properties, "depth_model_path"))
        obs_property_set_visible(p, mode == "depth_anything_v2");
    for (const char *key : {"teed_model_path", "teed_resolution", "teed_safe_steps"})
        if (auto *p = obs_properties_get(properties, key))
            obs_property_set_visible(p, mode == "teed");
    for (const char *key : {"pidinet_model_path", "pidinet_resolution", "pidinet_safe"})
        if (auto *p = obs_properties_get(properties, key))
            obs_property_set_visible(p, mode == "pidinet" ||
                                            mode == "scribble_pidinet");
    for (const char *key : {"hed_model_path", "hed_resolution", "hed_safe"})
        if (auto *p = obs_properties_get(properties, key))
            obs_property_set_visible(p, mode == "hed" || mode == "scribble_hed");
}

bool modified_mode(obs_properties_t *properties, obs_property_t *,
                   obs_data_t *settings)
{
    set_control_visibility(properties, obs_data_get_string(settings, kMode));
    return true; // OBS refreshes widgets after this callback safely unwinds.
}

obs_properties_t *properties(void *data)
{
    auto *p = obs_properties_create();
    auto *combo = obs_properties_add_list(p, kMode,
        "Preprocessor", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    std::string category;
    for (const auto &mode : preprocessors::catalog()) {
        if (mode.backend != preprocessors::Backend::Native &&
            std::strcmp(mode.id, "depth_anything_v2") != 0 &&
            std::strcmp(mode.id, "teed") != 0 &&
            std::strcmp(mode.id, "pidinet") != 0 &&
            std::strcmp(mode.id, "scribble_pidinet") != 0 &&
            std::strcmp(mode.id, "hed") != 0 &&
            std::strcmp(mode.id, "scribble_hed") != 0) continue;
        // Tile Preprocessor is an identity map without a downstream
        // ControlNet consumer; don't clutter Sight with a duplicate of Off.
        if (std::strcmp(mode.id, "tile") == 0) continue;
        if (category != mode.category) {
            category = mode.category;
            const std::string divider = "── " + category + " ──";
            const auto i = obs_property_list_add_string(combo, divider.c_str(),
                                                         ("category/" + category).c_str());
            obs_property_list_item_disable(combo, i, true);
        }
        const auto i = obs_property_list_add_string(combo, mode.label, mode.id);

    }
    obs_property_set_long_description(combo,
        "Native OpenCV and model-backed Depth Anything V2, TEED, "
        "PiDiNet, HED and their native Scribble NMS variants. "
        "Neural backends run locally on an isolated worker. "
        "No ComfyUI process, HTTP API, or network inference.");
    obs_property_set_modified_callback(combo, modified_mode);
    obs_properties_add_int_slider(p, "low", "Low threshold", 0, 255, 1);
    obs_properties_add_int_slider(p, "high", "High threshold", 0, 255, 1);
    obs_properties_add_float_slider(p, "sigma", "Line softness", 0.25, 15.0, 0.25);
    obs_properties_add_bool(p, "invert", "Invert black/white");
    obs_properties_add_int_slider(p, "palette_size", "Palette grid", 4, 128, 1);
    obs_properties_add_path(p, "depth_model_path", "Depth Anything V2 ONNX",
                            OBS_PATH_FILE, "ONNX models (*.onnx)", sight_preferences::directory().c_str());
    obs_properties_add_path(p, "teed_model_path", "TEED ONNX model",
                            OBS_PATH_FILE, "ONNX models (*.onnx)", sight_preferences::directory().c_str());
    obs_properties_add_int_slider(p, "teed_resolution",
                                  "TEED detection resolution", 128, 768, 32);
    obs_properties_add_int_slider(p, "teed_safe_steps",
                                  "TEED Safe Steps (0 = smooth)", 0, 10, 1);
    obs_properties_add_path(p, "pidinet_model_path", "PiDiNet ONNX model",
                            OBS_PATH_FILE, "ONNX models (*.onnx)", sight_preferences::directory().c_str());
    obs_properties_add_int_slider(p, "pidinet_resolution",
                                  "PiDiNet detection resolution", 128, 768, 32);
    obs_properties_add_bool(p, "pidinet_safe",
                            "PiDiNet safe edges (3-level quantization)");
    obs_properties_add_path(p, "hed_model_path", "HED ONNX model",
                            OBS_PATH_FILE, "ONNX models (*.onnx)", sight_preferences::directory().c_str());
    obs_properties_add_int_slider(p, "hed_resolution",
                                  "HED detection resolution", 128, 768, 32);
    obs_properties_add_bool(p, "hed_safe",
                            "HED safe edges (3-level quantization)");
    obs_properties_add_text(p, "backend_status", "Available backends", OBS_TEXT_INFO);
    std::string selected = "standard_lineart";
    if (auto *f = static_cast<Filter *>(data)) {
        std::lock_guard lock(f->input_mutex);
        selected = f->mode_id;
    }
    set_control_visibility(p, selected);
    return p;
}

const char *name(void *)
{
    return "lll.sight Preprocessors";
}

void render(void *data, gs_effect_t *)
{
    auto *f = static_cast<Filter *>(data);
    auto *target = obs_filter_get_target(f->source);
    if (!target || !obs_filter_get_parent(f->source)) {
        obs_source_skip_video_filter(f->source);
        return;
    }
    const uint32_t w = obs_source_get_base_width(target);
    const uint32_t h = obs_source_get_base_height(target);
    if (w < 8 || h < 8 || w > 8192 || h > 8192) {
        obs_source_skip_video_filter(f->source);
        return;
    }

    if (f->clear_gpu_output.exchange(false) && f->output_texture) {
        gs_texture_destroy(f->output_texture);
        f->output_texture = nullptr;
    }
    bool capture_due = false;
    uint64_t request_generation = 0;
    {
        std::lock_guard lock(f->input_mutex);
        if (f->mode_id == "off") {
            obs_source_skip_video_filter(f->source);
            return;
        }
        if (!f->capture_in_flight) {
            f->capture_in_flight = true;
            request_generation = f->generation;
            capture_due = true;
        } else {
            ++f->skipped;
        }
    }
    if (capture_due) {
        bool staged = false;
        if (!f->capture) f->capture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        if (!f->stage || f->stage_w != w || f->stage_h != h) {
            if (f->stage) gs_stagesurface_destroy(f->stage);
            f->stage = gs_stagesurface_create(w, h, GS_RGBA);
            f->stage_w = w;
            f->stage_h = h;
        }
        if (f->capture && f->stage) {
            gs_texrender_reset(f->capture);
            if (gs_texrender_begin(f->capture, w, h)) {
                gs_projection_push();
                gs_matrix_push();
                gs_matrix_identity();
                gs_ortho(0, static_cast<float>(w), 0,
                         static_cast<float>(h), -100.f, 100.f);
                auto *base = obs_get_base_effect(OBS_EFFECT_DEFAULT);
                if (base)
                    while (gs_effect_loop(base, "Draw"))
                        obs_source_video_render(target);
                gs_matrix_pop();
                gs_projection_pop();
                gs_texrender_end(f->capture);
                auto *tex = gs_texrender_get_texture(f->capture);
                if (tex) {
                    gs_stage_texture(f->stage, tex);
                    uint8_t *mapped = nullptr;
                    uint32_t pitch = 0;
                    if (gs_stagesurface_map(f->stage, &mapped, &pitch)) {
                        if (pitch >= w * 4) {
                            std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 4);
                            for (uint32_t y = 0; y < h; ++y)
                                std::memcpy(frame.data() + static_cast<size_t>(y) * w * 4,
                                            mapped + static_cast<size_t>(y) * pitch, w * 4);
                            {
                                std::lock_guard lock(f->input_mutex);
                                if (f->generation == request_generation && !f->stopping) {
                                    f->incoming = std::move(frame);
                                    f->incoming_w = w;
                                    f->incoming_h = h;
                                    f->frame_queued = true;
                                    ++f->captures;
                                    staged = true;
                                    f->input_ready.notify_one();
                                }
                            }
                        }
                        gs_stagesurface_unmap(f->stage);
                    }
                }
            }
        }
        if (!staged) {
            std::lock_guard lock(f->input_mutex);
            if (request_generation == f->generation)
                f->capture_in_flight = false;
        }
    }

    bool output_active = false;
    {
        // Only graphics thread owns output_texture. Worker owns the CPU
        // packet; no GPU calls cross into the worker.
        std::lock_guard lock(f->output_mutex);
        const bool fresh = f->output_when.time_since_epoch().count() != 0 &&
                           millis_since(f->output_when) <= kMaxOutputAgeMs;
        if (f->upload_due && fresh && !f->output_pixels.empty()) {
            if (!f->output_texture ||
                gs_texture_get_width(f->output_texture) != f->output_w ||
                gs_texture_get_height(f->output_texture) != f->output_h) {
                if (f->output_texture) gs_texture_destroy(f->output_texture);
                f->output_texture = gs_texture_create(f->output_w, f->output_h,
                    GS_RGBA, 1, nullptr, GS_DYNAMIC);
            }
            if (f->output_texture) {
                gs_texture_set_image(f->output_texture, f->output_pixels.data(),
                    f->output_w * 4, false);
                f->upload_due = false;
            }
        }
        output_active = fresh && f->output_texture &&
                        gs_texture_get_width(f->output_texture) == w &&
                        gs_texture_get_height(f->output_texture) == h;
    }

    if (!output_active) {
        obs_source_skip_video_filter(f->source);
        return;
    }
    auto *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if (effect) {
        auto *image = gs_effect_get_param_by_name(effect, "image");
        if (image) {
            gs_effect_set_texture(image, f->output_texture);
            while (gs_effect_loop(effect, "Draw"))
                gs_draw_sprite(f->output_texture, 0, w, h);
        }
    }
}

} // namespace

struct obs_source_info sight_preprocessors_filter_info = {
    .id = "sight_preprocessors_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
    .get_name = name,
    .create = create,
    .destroy = destroy,
    .get_defaults = defaults,
    .get_properties = properties,
    .update = update,
    .video_render = render,
};
