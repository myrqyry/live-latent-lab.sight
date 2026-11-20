#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <memory>
#include <utility>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/onnx_depth_processor.hpp"
#include "core/depth_temporal_policy.hpp"
#include "core/analysis_capture.hpp"
#include "core/sight_depth_v1.h"

// Depth Anything v2: genuine ONNX depth inference, with independent
// capture, CPU worker, and graphics texture ownership.
namespace {
std::atomic<uint64_t> depth_sequence{0};
constexpr const char *MODEL_PATH = "model_path";
constexpr const char *INFERENCE_EDGE = "inference_edge";
constexpr const char *INFERENCE_CADENCE = "inference_cadence";
constexpr const char *TEMPORAL_BLEND = "temporal_blend";
constexpr const char *REJECT_STALE_DEPTH = "reject_stale_depth";
constexpr const char *MAX_DEPTH_AGE_SECONDS = "max_depth_age_seconds";
constexpr const char *VIEW_MODE = "view_mode";
constexpr const char *FOG_AMOUNT = "fog_amount";

uint32_t valid_edge(int64_t input)
{
    return input == 252 || input == 336 || input == 518
        ? static_cast<uint32_t>(input) : 252u;
}
unsigned valid_cadence(int64_t input)
{
    return input == 1 || input == 2 || input == 4 || input == 8
        ? static_cast<unsigned>(input) : 4u;
}
int valid_blend(int64_t input)
{
    return static_cast<int>(std::clamp<int64_t>(input, 0, 95));
}
unsigned valid_max_age_seconds(int64_t input)
{
    return input == 0 || input == 2 || input == 5 || input == 10 || input == 20
        ? static_cast<unsigned>(input) : 5u;
}

uint64_t monotonic_milliseconds()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::pair<uint32_t, uint32_t> capture_dimensions(
    uint32_t width, uint32_t height, uint32_t max_edge)
{
    if (width == 0 || height == 0)
        return {0, 0};
    const uint32_t longest = std::max(width, height);
    auto aligned = [&](uint32_t axis) {
        // DPT patch embedding requires spatial input sizes divisible by 14.
        const uint32_t raw = static_cast<uint32_t>(
            (static_cast<uint64_t>(axis) * max_edge + longest / 2) / longest);
        return std::max(14u, ((raw + 7u) / 14u) * 14u);
    };
    return {aligned(width), aligned(height)};
}

struct DepthFilter {
    obs_source_t *context = nullptr;
    std::mutex input_mutex;
    std::condition_variable frame_cv;
    std::thread worker;
    bool stopping = false;
    bool load_requested = false;
    bool frame_pending = false;
    std::string requested_path;
    std::string active_path;
    uint64_t model_generation = 0;
    uint32_t frame_width = 0, frame_height = 0;
    uint32_t frame_source_width=0,frame_source_height=0;
    uint64_t frame_capture_ms=0;
    std::vector<uint8_t> input_rgba;
    std::atomic<bool> model_ready{false};
    std::atomic<uint32_t> edge{252};
    std::atomic<unsigned> cadence{4};
    std::atomic<int> temporal_blend{65};
    std::atomic<bool> reject_stale_depth{false};
    // A disabled model should not leave a frozen map displayed indefinitely.
    std::atomic<unsigned> max_depth_age_seconds{5};
    std::atomic<uint64_t> last_successful_ms{0};
    // Graphics-thread state only, validated against the current source size.
    lll::DepthSourceGeometry source_geometry;
    bool announced_expired_depth = false;
    std::atomic<int> view_mode{0};
    std::atomic<int> fog_amount{40};
    // Graphics thread owns capture cadence; a worker can only request work.
    uint64_t render_number = 0;
    std::atomic<uint64_t> next_capture{0};
    // The render thread captures only after the worker requests a frame.
    // Cadence provides an additional user-selected limit.
    std::atomic<bool> wants_frame{false};

    std::mutex output_mutex;
    std::vector<uint8_t> output_mask;
    uint32_t output_width = 0, output_height = 0;
    uint32_t output_source_width=0,output_source_height=0;
    uint64_t output_capture_ms=0,output_sequence=0,output_generation=0;
    bool new_output = false;
    std::atomic<bool> reset_output_texture{false};
    bool announced_first_frame = false;

    // Graphics-thread resources; never touched by inference worker.
    gs_texrender_t *capture = nullptr;
    gs_stagesurf_t *stage = nullptr;
    gs_texture_t *output_texture = nullptr; // Small one-channel alpha mask only
    gs_texrender_t *full_resolution = nullptr;
    gs_effect_t *matte_effect = nullptr; // GPU depth visualization/fog
    uint32_t stage_width = 0, stage_height = 0;
};

void worker_main(DepthFilter *f)
{
    lll::OnnxDepthProcessor processor;
    std::vector<uint8_t> previous_depth;
    std::vector<uint8_t> previous_rgb;
    uint64_t previous_generation = 0;
    uint32_t previous_width = 0, previous_height = 0;
    using Clock = std::chrono::steady_clock;
    double accumulated_inference_ms = 0;
    uint32_t measured_frames = 0;
    bool announced_change_aware = false;
    for (;;) {
        std::vector<uint8_t> frame;
        std::string model_path;
        uint32_t w = 0, h = 0,sw=0,sh=0;
        uint64_t capture_ms=0;
        uint64_t generation = 0;
        bool do_load = false;
        {
            std::unique_lock lock(f->input_mutex);
            f->frame_cv.wait(lock, [&] {
                return f->stopping || f->load_requested || f->frame_pending;
            });
            if (f->stopping)
                return;
            if (f->load_requested) {
                do_load = true;
                model_path = f->requested_path;
                generation = f->model_generation;
                f->load_requested = false;
                f->frame_pending = false;
            } else {
                generation = f->model_generation;
                w = f->frame_width;
                h = f->frame_height;
                sw=f->frame_source_width;sh=f->frame_source_height;capture_ms=f->frame_capture_ms;
                frame = std::move(f->input_rgba);
                f->frame_pending = false;
            }
        }

        if (do_load) {
            const bool loaded = !model_path.empty() && processor.load(model_path);
            previous_depth.clear();
            previous_rgb.clear();
            previous_generation = 0;
            previous_width = previous_height = 0;
            accumulated_inference_ms = 0;
            measured_frames = 0;
            announced_change_aware = false;
            {
                std::lock_guard lock(f->input_mutex);
                if (generation != f->model_generation)
                    continue;
                f->active_path = loaded ? model_path : std::string();
                f->model_ready = loaded;
                f->wants_frame = loaded;
            }
            if (loaded)
                blog(LOG_INFO, "[Sight/Depth] Loaded real ONNX model: %s", model_path.c_str());
            else if (!model_path.empty())
                blog(LOG_ERROR, "[Sight/Depth] Failed to load %s: %s",
                     model_path.c_str(), processor.last_error().c_str());
            continue;
        }

        if (!f->model_ready || !w || !h ||
            frame.size() != static_cast<size_t>(w) * h * 4) {
            std::lock_guard lock(f->input_mutex);
            if (f->model_ready && generation == f->model_generation)
                f->wants_frame = true;
            continue;
        }

        std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i)
            std::memcpy(&rgb[i * 3], &frame[i * 4], 3);

        std::vector<uint8_t> mask;
        const auto inference_start = Clock::now();
        const bool succeeded = processor.estimate(rgb.data(), w, h, mask);
        const double inference_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - inference_start).count();
        if (!succeeded || mask.size() != static_cast<size_t>(w) * h) {
            blog(LOG_ERROR, "[Sight/Depth] Actual model inference failed: %ux%u %s",
                 w, h, processor.last_error().c_str());
            std::lock_guard lock(f->input_mutex);
            if (f->model_ready && generation == f->model_generation) {
                std::lock_guard out_lock(f->output_mutex);
                f->output_sequence=0;
                f->wants_frame = true;
            }
            continue;
        }

        // Drop stale results *before* applying temporal smoothing, so
        // discarded old-model/old-resolution results cannot poison history.
        {
            std::lock_guard lock(f->input_mutex);
            if (generation != f->model_generation || !f->model_ready)
                continue;
            // History from another model/input generation is never valid,
            // even if a quick setting change returns to identical dimensions.
            if (generation != previous_generation) {
                previous_depth.clear();
                previous_rgb.clear();
                previous_width = previous_height = 0;
            }
            // Smoothing runs only on genuine inference updates. RGB at the
            // same model-input resolution can reject stale history in pixels
            // with obvious appearance changes; it does not estimate flow.
            if (previous_width == w && previous_height == h) {
                const auto blend = static_cast<unsigned>(f->temporal_blend.load());
                if (f->reject_stale_depth.load()) {
                    lll::blend_depth_result_change_aware(
                        mask, previous_depth, rgb, previous_rgb, blend);
                    if (!announced_change_aware) {
                        blog(LOG_INFO, "[Sight/Depth] Change-aware history blending active");
                        announced_change_aware = true;
                    }
                } else {
                    lll::blend_depth_result(mask, previous_depth, blend);
                }
            }
            previous_rgb = std::move(rgb);
            previous_depth = mask;
            previous_width = w;
            previous_height = h;
            previous_generation = generation;
            std::lock_guard out_lock(f->output_mutex);
            f->output_mask = std::move(mask);
            f->output_width = w;
            f->output_height = h;
            f->output_source_width=sw;f->output_source_height=sh;
            f->output_capture_ms=capture_ms;f->output_generation=generation;
            f->output_sequence=++depth_sequence;
            f->new_output = true;
            f->last_successful_ms = monotonic_milliseconds();
            accumulated_inference_ms += inference_ms;
            ++measured_frames;
            if (measured_frames == 8 && accumulated_inference_ms > 0) {
                const double avg_ms = accumulated_inference_ms / measured_frames;
                blog(LOG_INFO,
                     "[Sight/Depth] Inference average: %.1f ms/frame (%.1f inference fps, first %u frames)",
                     avg_ms, 1000.0 / avg_ms, measured_frames);
            }
            if (!f->announced_first_frame) {
                blog(LOG_INFO, "[Sight/Depth] First real depth map: %ux%u", w, h);
                f->announced_first_frame = true;
            }
            f->wants_frame = true;
        }
    }
}

void read_depth(void* data,calldata_t* params)
{
    auto* f=static_cast<DepthFilter*>(data);
    calldata_set_bool(params,"available",false);
    void* buffer=nullptr;long long age=0,known=0;
    if(!calldata_get_ptr(params,"snapshot",&buffer)||!buffer||
       !calldata_get_int(params,"max_age_ms",&age)||age<1||age>5000||
       !calldata_get_int(params,"known_sequence",&known))return;
    auto* out=static_cast<sight_depth_v1*>(buffer);
    if(out->abi_version!=SIGHT_DEPTH_ABI_V1||out->struct_size!=sizeof(*out))return;
    out->sequence=0;
    if(!obs_source_enabled(f->context)||f->view_mode!=2)return;
    auto* target=obs_filter_get_target(f->context);
    if(!target||!obs_filter_get_parent(f->context))return;
    const uint32_t sw=obs_source_get_base_width(target),sh=obs_source_get_base_height(target);
    std::unique_lock input(f->input_mutex,std::try_to_lock);
    if(!input||!f->model_ready||f->load_requested)return;
    std::unique_lock output(f->output_mutex,std::try_to_lock);
    if(!output||!f->output_sequence||f->output_generation!=f->model_generation||
       !sw||!sh||sw!=f->output_source_width||sh!=f->output_source_height||
       !f->output_width||!f->output_height||f->output_width>SIGHT_DEPTH_MAX_EDGE||
       f->output_height>SIGHT_DEPTH_MAX_EDGE||
       !lll::depth_result_recent(monotonic_milliseconds(),f->output_capture_ms,uint64_t(age)))return;
    const size_t count=size_t(f->output_width)*f->output_height;
    if(f->output_mask.size()!=count)return;
    out->width=f->output_width;out->height=f->output_height;
    out->source_width=sw;out->source_height=sh;out->sequence=f->output_sequence;
    if(uint64_t(known)!=out->sequence)std::memcpy(out->pixels,f->output_mask.data(),count);
    calldata_set_bool(params,"available",true);
}

void *create(obs_data_t *settings, obs_source_t *context)
{
    auto *f = new DepthFilter;
    f->context = context;
    f->edge = valid_edge(obs_data_get_int(settings, INFERENCE_EDGE));
    f->cadence = valid_cadence(obs_data_get_int(settings, INFERENCE_CADENCE));
    f->temporal_blend = valid_blend(obs_data_get_int(settings, TEMPORAL_BLEND));
    f->reject_stale_depth = obs_data_get_bool(settings, REJECT_STALE_DEPTH);
    f->max_depth_age_seconds = valid_max_age_seconds(
        obs_data_get_int(settings, MAX_DEPTH_AGE_SECONDS));
    f->view_mode = static_cast<int>(std::clamp<int64_t>(
        obs_data_get_int(settings, VIEW_MODE), 0, 2));
    f->fog_amount = static_cast<int>(std::clamp<int64_t>(
        obs_data_get_int(settings, FOG_AMOUNT), 0, 100));
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    if (path && *path) {
        f->requested_path = path;
        f->load_requested = true;
        ++f->model_generation;
    }
    if(auto* procs=obs_source_get_proc_handler(context))
        proc_handler_add(procs,"void sight_depth_read_v1(in int max_age_ms, in int known_sequence, in ptr snapshot, out bool available)",read_depth,f);
    f->worker = std::thread(worker_main, f);
    if (f->load_requested)
        f->frame_cv.notify_one();
    return f;
}

void destroy(void *data)
{
    auto *f = static_cast<DepthFilter *>(data);
    if (!f)
        return;
    {
        std::lock_guard lock(f->input_mutex);
        f->stopping = true;
    }
    f->frame_cv.notify_one();
    if (f->worker.joinable())
        f->worker.join();

    obs_enter_graphics();
    if (f->output_texture)
        gs_texture_destroy(f->output_texture);
    if (f->full_resolution)
        gs_texrender_destroy(f->full_resolution);
    if (f->matte_effect)
        gs_effect_destroy(f->matte_effect);
    if (f->stage)
        gs_stagesurface_destroy(f->stage);
    if (f->capture)
        gs_texrender_destroy(f->capture);
    obs_leave_graphics();
    delete f;
}

void update(void *data, obs_data_t *settings)
{
    auto *f = static_cast<DepthFilter *>(data);
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    const std::string selected = path ? path : "";
    const uint32_t edge = valid_edge(obs_data_get_int(settings, INFERENCE_EDGE));
    f->cadence = valid_cadence(obs_data_get_int(settings, INFERENCE_CADENCE));
    f->temporal_blend = valid_blend(obs_data_get_int(settings, TEMPORAL_BLEND));
    f->reject_stale_depth = obs_data_get_bool(settings, REJECT_STALE_DEPTH);
    f->max_depth_age_seconds = valid_max_age_seconds(
        obs_data_get_int(settings, MAX_DEPTH_AGE_SECONDS));
    f->view_mode = static_cast<int>(std::clamp<int64_t>(
        obs_data_get_int(settings, VIEW_MODE), 0, 2));
    f->fog_amount = static_cast<int>(std::clamp<int64_t>(
        obs_data_get_int(settings, FOG_AMOUNT), 0, 100));
    {
        std::lock_guard lock(f->input_mutex);
        const bool reload = selected != f->requested_path;
        if (!reload && edge == f->edge)
            return;
        f->edge = edge;
        if (reload || !f->model_ready) {
            f->requested_path = selected;
            f->load_requested = true;
            f->model_ready = false;
            f->wants_frame = false;
        } else {
            // In-flight inference from the previous edge is stale. Rearm
            // capture or this filter could wait forever for a discarded map.
            f->wants_frame = true;
        }
        f->frame_pending = false;
        ++f->model_generation;
        f->next_capture = 0;
        f->announced_first_frame = false;
        f->reset_output_texture = true;
        f->last_successful_ms = 0;
    }
    {
        std::lock_guard lock(f->output_mutex);
        f->new_output = false;
        f->output_mask.clear();
        f->output_sequence=0;
    }
    f->frame_cv.notify_one();
}

void defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, MODEL_PATH, "");
    obs_data_set_default_int(settings, INFERENCE_EDGE, 252);
    obs_data_set_default_int(settings, INFERENCE_CADENCE, 4);
    obs_data_set_default_int(settings, TEMPORAL_BLEND, 65);
    obs_data_set_default_bool(settings, REJECT_STALE_DEPTH, false);
    obs_data_set_default_int(settings, MAX_DEPTH_AGE_SECONDS, 5);
    obs_data_set_default_int(settings, VIEW_MODE, 0);
    obs_data_set_default_int(settings, FOG_AMOUNT, 40);
}

// OBS uses the final obs_properties_add_path argument as the Browse dialog's
// initial directory. Prefer the selected model's folder when it still exists,
// otherwise use the persistent Sight model directory (not the source checkout).
std::string model_browser_directory(DepthFilter *filter)
{
    namespace fs = std::filesystem;
    std::error_code ec;

    if (filter) {
        std::string selected;
        {
            std::lock_guard lock(filter->input_mutex);
            selected = filter->requested_path;
        }
        if (!selected.empty() && fs::is_regular_file(selected, ec)) {
            const fs::path directory = fs::absolute(fs::path(selected).parent_path(), ec);
            if (!ec && fs::is_directory(directory))
                return directory.string();
        }
    }

    char *module_path = obs_module_config_path("models");
    fs::path directory;
    if (module_path) {
        directory = module_path;
        bfree(module_path);
    }
    if (directory.empty() || directory.is_relative()) {
        // Standalone module loaders can return a relative module config path.
        // Don't turn it into an arbitrary CWD-dependent directory.
        char *config_path = os_get_config_path_ptr(
            "obs-studio/plugin_config/live-latent-lab-sight/models");
        if (!config_path)
            return {};
        directory = config_path;
        bfree(config_path);
    }
    if (!directory.is_absolute())
        return {};
    fs::create_directories(directory, ec);
    if (ec && !fs::is_directory(directory))
        return {};
    return directory.string();
}

obs_properties_t *properties(void *data)
{
    obs_properties_t *p = obs_properties_create();
    const auto folder = model_browser_directory(static_cast<DepthFilter *>(data));
    auto *path = obs_properties_add_path(
        p, MODEL_PATH, "Depth Anything V2 ONNX model",
        OBS_PATH_FILE, "ONNX model (*.onnx)",
        folder.empty() ? nullptr : folder.c_str());
    obs_property_set_long_description(path,
        "Tested: depth_anything_v2_small_fp32.onnx (float32, NCHW). "
        "Depth is relative, not measured distance. Without a valid model "
        "the source passes through unchanged.");
    auto *size = obs_properties_add_list(p, INFERENCE_EDGE,
        "Inference detail (longest side)",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(size, "Fast — 252 px", 252);
    obs_property_list_add_int(size, "Balanced — 336 px", 336);
    obs_property_list_add_int(size, "Detail — 518 px (slow CPU)", 518);
    auto *cadence = obs_properties_add_list(p, INFERENCE_CADENCE,
        "Capture no more often than every N OBS frames",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    for (int n : {1, 2, 4, 8})
        obs_property_list_add_int(cadence, std::to_string(n).c_str(), n);
    obs_property_set_long_description(cadence,
        "Throttles new inference inputs; preserves the previous result. "
        "Slow models already compute less frequently than OBS renders.");
    obs_properties_add_int_slider(p, TEMPORAL_BLEND,
        "Temporal smoothing (reuse previous map, %)", 0, 95, 5);
    auto *reject = obs_properties_add_bool(p, REJECT_STALE_DEPTH,
        "Reject stale depth at changed pixels (experimental)");
    obs_property_set_long_description(reject,
        "When new inference arrives, reduce smoothing where the source RGB "
        "changed significantly. This is not optical-flow reprojection: "
        "it cannot fix lag between model updates or low-contrast motion.");
    auto *max_age = obs_properties_add_list(p, MAX_DEPTH_AGE_SECONDS,
        "Discard depth if no fresh result for",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(max_age, "2 seconds", 2);
    obs_property_list_add_int(max_age, "5 seconds (recommended)", 5);
    obs_property_list_add_int(max_age, "10 seconds", 10);
    obs_property_list_add_int(max_age, "20 seconds", 20);
    obs_property_list_add_int(max_age, "Never (keep last map)", 0);
    obs_property_set_long_description(max_age,
        "Pass through the original camera image if inference stops producing "
        "new depth maps. Limits indefinite frozen depth after model failure; "
        "does not eliminate latency while an inference is still running.");
    auto *mode = obs_properties_add_list(p, VIEW_MODE, "Visualization",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(mode, "Relative depth (grayscale)", 0);
    obs_property_list_add_int(mode, "Depth fog (original colors)", 1);
    obs_property_list_add_int(mode,"Original source (share depth)",2);
    obs_properties_add_int_slider(p, FOG_AMOUNT, "Depth fog strength", 0, 100, 5);
    return p;
}

const char *name(void *)
{
    return "LLL.sight Depth Camera (Depth Anything)";
}

void render(void *data, gs_effect_t *)
{
    auto *f = static_cast<DepthFilter *>(data);
    obs_source_t *target = obs_filter_get_target(f->context);
    if (!target || !obs_filter_get_parent(f->context)) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    const uint32_t width = obs_source_get_base_width(target);
    const uint32_t height = obs_source_get_base_height(target);
    if (!width || !height) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    // A new camera/source size invalidates the spatial correspondence of
    // every old depth pixel. Cancel stale queued/in-flight inference and
    // demand a fresh capture; do not restart a model still loading.
    if (f->source_geometry.observe(width, height)) {
        {
            std::lock_guard lock(f->input_mutex);
            if (f->model_ready && !f->load_requested) {
                ++f->model_generation;
                f->wants_frame = true;
            }
            f->frame_pending = false;
            f->input_rgba.clear();
            f->next_capture = 0;
        }
        {
            std::lock_guard lock(f->output_mutex);
            f->new_output = false;
            f->output_mask.clear();
            f->output_sequence=0;
            f->last_successful_ms = 0;
        }
        if (f->output_texture) {
            gs_texture_destroy(f->output_texture);
            f->output_texture = nullptr;
        }
        f->announced_expired_depth = false;
        blog(LOG_INFO, "[Sight/Depth] Source geometry changed to %ux%u; depth invalidated",
             width, height);
    }
    if (f->reset_output_texture.exchange(false) && f->output_texture) {
        gs_texture_destroy(f->output_texture);
        f->output_texture = nullptr;
    }
    if (!f->matte_effect) {
        char *path = obs_module_file("sight-depth-camera.effect");
        if (path) {
            char *errors = nullptr;
            f->matte_effect = gs_effect_create_from_file(path, &errors);
            if (errors) {
                blog(LOG_ERROR, "[Sight/Depth] Depth shader: %s", errors);
                bfree(errors);
            }
            bfree(path);
        }
        if (!f->matte_effect) {
            obs_source_skip_video_filter(f->context);
            return;
        }
    }
    const auto [pw, ph] = capture_dimensions(width, height, f->edge.load());
    ++f->render_number;

    if (f->model_ready && f->wants_frame &&
        lll::depth_capture_due(f->render_number, f->next_capture) &&
        f->wants_frame.exchange(false)) {
        bool submitted = false;
        if (!f->capture)
            f->capture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        if (!f->stage || f->stage_width != pw || f->stage_height != ph) {
            if (f->stage)
                gs_stagesurface_destroy(f->stage);
            f->stage = gs_stagesurface_create(pw, ph, GS_RGBA);
            f->stage_width = pw;
            f->stage_height = ph;
        }
        if (f->capture && f->stage) {
            gs_texrender_reset(f->capture);
            if (gs_texrender_begin(f->capture, pw, ph)) {
                gs_projection_push();
                gs_matrix_push();
                gs_matrix_identity();
                gs_ortho(0, static_cast<float>(width), 0,
                    static_cast<float>(height), -100, 100);
                const uint64_t capture_ms=monotonic_milliseconds();
                struct vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);
                gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
                sight_draw_analysis_source(target);
                gs_blend_state_pop();
                gs_matrix_pop();
                gs_projection_pop();
                gs_texrender_end(f->capture);

                gs_texture_t *tex = gs_texrender_get_texture(f->capture);
                if (tex) {
                    gs_stage_texture(f->stage, tex);
                    uint8_t *pixels = nullptr;
                    uint32_t pitch = 0;
                    if (gs_stagesurface_map(f->stage, &pixels, &pitch)) {
                        if (pitch >= pw * 4) {
                            std::vector<uint8_t> frame(static_cast<size_t>(pw) * ph * 4);
                            for (uint32_t y = 0; y < ph; ++y)
                                std::memcpy(frame.data() + static_cast<size_t>(y) * pw * 4,
                                    pixels + static_cast<size_t>(y) * pitch, pw * 4);
                            std::lock_guard lock(f->input_mutex);
                            if (f->model_ready && !f->load_requested) {
                                f->input_rgba = std::move(frame);
                                f->frame_width = pw;
                                f->frame_height = ph;
                                f->frame_source_width=width;f->frame_source_height=height;
                                f->frame_capture_ms=capture_ms;
                                f->frame_pending = true;
                                submitted = true;
                                f->next_capture = lll::next_depth_capture(
                                    f->render_number, f->cadence.load());
                                f->frame_cv.notify_one();
                            }
                        }
                        gs_stagesurface_unmap(f->stage);
                    }
                }
            }
        }
        // If capture failed, retry on the next OBS render. A successful
        // submission can only be rearmed after inference finishes.
        if (!submitted && f->model_ready)
            f->wants_frame = true;
    }

    if(f->view_mode==2){obs_source_skip_video_filter(f->context);return;}
    {
        std::lock_guard lock(f->output_mutex);
        if (f->new_output) {
            if (!f->output_texture ||
                gs_texture_get_width(f->output_texture) != f->output_width ||
                gs_texture_get_height(f->output_texture) != f->output_height) {
                if (f->output_texture)
                    gs_texture_destroy(f->output_texture);
                f->output_texture = gs_texture_create(
                    f->output_width, f->output_height, GS_R8, 1, nullptr, GS_DYNAMIC);
            }
            if (f->output_texture)
                gs_texture_set_image(f->output_texture, f->output_mask.data(),
                    f->output_width, false);
            f->new_output = false;
        }
    }
    if (!f->model_ready || !f->output_texture) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    const uint64_t last_success = f->last_successful_ms.load();
    const uint64_t max_age_ms = uint64_t(f->max_depth_age_seconds.load()) * 1000;
    if (!lll::depth_result_recent(monotonic_milliseconds(), last_success,
                                  max_age_ms)) {
        if (last_success && !f->announced_expired_depth) {
            blog(LOG_WARNING,
                 "[Sight/Depth] Last depth result expired; passing through source until inference recovers");
            f->announced_expired_depth = true;
        }
        obs_source_skip_video_filter(f->context);
        return;
    }
    f->announced_expired_depth = false;

    // The map was inferred at <= 518px. The original colors must remain
    // full-resolution in the output; this capture stays on the GPU.
    gs_texture_t *native_texture = f->output_texture;
    if (f->view_mode == 1) {
        // Fog needs the original RGB, but a grayscale depth map does not.
        // Avoid an unnecessary full-resolution FBO pass in the default mode.
        if (!f->full_resolution)
            f->full_resolution = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        if (!f->full_resolution) {
            obs_source_skip_video_filter(f->context);
            return;
        }
        gs_texrender_reset(f->full_resolution);
        if (!gs_texrender_begin(f->full_resolution, width, height)) {
            obs_source_skip_video_filter(f->context);
            return;
        }
        gs_projection_push();
        gs_matrix_push();
        gs_matrix_identity();
        gs_ortho(0, static_cast<float>(width), 0,
                 static_cast<float>(height), -100, 100);
        struct vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);
        sight_draw_analysis_source(target);
        gs_matrix_pop();
        gs_projection_pop();
        gs_texrender_end(f->full_resolution);
        native_texture = gs_texrender_get_texture(f->full_resolution);
    }
    gs_eparam_t *source_param = gs_effect_get_param_by_name(f->matte_effect, "original_frame");
    gs_eparam_t *mask_param = gs_effect_get_param_by_name(f->matte_effect, "depth_map");
    if (!native_texture || !source_param || !mask_param) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_effect_set_texture(source_param, native_texture);
    gs_effect_set_texture(mask_param, f->output_texture);
    auto *view_param = gs_effect_get_param_by_name(f->matte_effect, "view_mode");
    auto *fog_param = gs_effect_get_param_by_name(f->matte_effect, "fog_strength");
    if (!view_param || !fog_param) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_effect_set_int(view_param, f->view_mode);
    gs_effect_set_float(fog_param, f->fog_amount.load() / 100.0f);
    while (gs_effect_loop(f->matte_effect, "Draw"))
        gs_draw_sprite(native_texture, 0, width, height);
}
} // namespace

// A distinct persistent OBS filter ID. Does not overwrite Portrait Sketch.
struct obs_source_info sight_depth_filter_info = {
    .id = "sight_depth_filter",
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
