#include "common/backend-status.hpp"
#include "common/model-diagnostics.hpp"
#include "common/preferences.hpp"
#include <obs-module.h>
#include <callback/proc.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
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

#include "core/enhancement_processor.hpp"

// This is the one enabled Sight effect. The other processors stay in the tree
// but neither compile nor register until they have their own verified path.
namespace {
constexpr const char *MODEL_PATH = "model_path";
constexpr const char *DETAIL_SIZE = "detail_size";
constexpr uint32_t DEFAULT_DETAIL_SIZE = 512;
uint32_t validated_detail_size(int64_t size)
{
    return size == 256 || size == 512 || size == 768 || size == 1024
        ? static_cast<uint32_t>(size) : DEFAULT_DETAIL_SIZE;
}
// The convolution/downsample/upsample branches in this AnimeGANv3 model
// require aligned spatial dimensions. Pad inference rather than stretching
// the captured frame: the output is cropped before upload to OBS.
constexpr uint32_t MODEL_ALIGNMENT = 32;

constexpr uint32_t aligned_size(uint32_t size)
{
    return ((size + MODEL_ALIGNMENT - 1) / MODEL_ALIGNMENT) * MODEL_ALIGNMENT;
}

std::pair<uint32_t, uint32_t> capture_dimensions(
    uint32_t width, uint32_t height, uint32_t limit)
{
    if (width == 0 || height == 0)
        return {0, 0};
    const uint32_t longest = std::max(width, height);
    if (longest <= limit)
        return {width, height};
    return {
        std::max(1u, static_cast<uint32_t>(
            static_cast<uint64_t>(width) * limit / longest)),
        std::max(1u, static_cast<uint32_t>(
            static_cast<uint64_t>(height) * limit / longest))
    };
}

struct AnimeGanFilter {
    obs_source_t *context = nullptr;
    std::mutex input_mutex;
    std::condition_variable frame_cv;
    std::thread worker;
    bool stopping = false;
    bool load_requested = false;
    bool frame_pending = false;
    // One capture per worker inference. Protected by input_mutex.
    bool capture_in_flight = false;
    std::atomic<int64_t> captures{0}, completed{0}, skipped{0};
    std::atomic<int64_t> buffer_growths{0}, tensor_growths{0};
    std::string requested_path;
    std::string active_path;
    sight_backend::Status backend_status;
    uint64_t model_generation = 0;
    uint32_t frame_width = 0, frame_height = 0;
    std::vector<uint8_t> input_rgba;
    std::vector<uint8_t> capture_scratch; // graphics-thread owned
    std::atomic<bool> model_ready{false};
    std::atomic<uint32_t> detail_size{DEFAULT_DETAIL_SIZE};

    std::mutex output_mutex;
    std::vector<uint8_t> output_rgba;
    uint32_t output_width = 0, output_height = 0;
    bool new_output = false;
    std::atomic<bool> reset_output_texture{false};
    bool announced_first_frame = false;

    // Graphics-thread resources; never touched by inference worker.
    gs_texrender_t *capture = nullptr;
    gs_stagesurf_t *stage = nullptr;
    gs_texture_t *output_texture = nullptr;
    uint32_t stage_width = 0, stage_height = 0;
};

// Keep capacity across frames, counting actual capacity-growth requests.
void resize_pixels(AnimeGanFilter *f, std::vector<uint8_t> &pixels, size_t size)
{
    if (pixels.capacity() < size) ++f->buffer_growths;
    pixels.resize(size);
}

void worker_main(AnimeGanFilter *f)
{
    lll::EnhancementProcessor processor;
    std::vector<uint8_t> frame, rgb, result, rgba;
    for (;;) {
        std::string model_path;
        uint32_t w = 0, h = 0;
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
                frame.swap(f->input_rgba);
                f->frame_pending = false;
            }
        }

        if (do_load) {
            const bool loaded = !model_path.empty() &&
                processor.load(model_path.c_str(), lll::EnhanceModelType::AnimeGANv3, true);
            {
                std::lock_guard lock(f->input_mutex);
                if (generation != f->model_generation)
                    continue;
                f->capture_in_flight = false;
                f->active_path = loaded ? model_path : std::string();
                f->backend_status.complete(model_path,loaded,processor.configured_provider(),processor.last_error());
                f->model_ready = loaded;
            }
            if (loaded)
                blog(LOG_INFO, "[Sight/AnimeGAN] Loaded real ONNX model: %s", model_path.c_str());
            else if (!model_path.empty())
                blog(LOG_ERROR, "[Sight/AnimeGAN] Failed to load model: %s", model_path.c_str());
            continue;
        }

        if (!f->model_ready || !w || !h ||
            frame.size() != static_cast<size_t>(w) * h * 4) {
            std::lock_guard lock(f->input_mutex);
            if (generation == f->model_generation) {
                frame.swap(f->input_rgba);
                f->capture_in_flight = false;
            }
            continue;
        }

        const uint32_t infer_w = aligned_size(w);
        const uint32_t infer_h = aligned_size(h);
        // Replicate edge pixels into the padded region. Unlike resizing the
        // source to aligned dimensions, this preserves the original aspect.
        resize_pixels(f, rgb, static_cast<size_t>(infer_w) * infer_h * 3);
        for (uint32_t y = 0; y < infer_h; ++y) {
            const uint32_t src_y = std::min(y, h - 1);
            for (uint32_t x = 0; x < infer_w; ++x) {
                const uint32_t src_x = std::min(x, w - 1);
                const size_t src = (static_cast<size_t>(src_y) * w + src_x) * 4;
                const size_t dst = (static_cast<size_t>(y) * infer_w + x) * 3;
                std::memcpy(&rgb[dst], &frame[src], 3);
            }
        }
        resize_pixels(f, result, rgb.size());
        const auto inference_start=sight_backend::InferenceHealth::Clock::now();
        const bool succeeded=processor.process(rgb.data(),infer_w,infer_h,result.data(),infer_w,infer_h);
        f->tensor_growths = static_cast<int64_t>(processor.input_buffer_growths());
        const double inference_ms=std::chrono::duration<double,std::milli>(
            sight_backend::InferenceHealth::Clock::now()-inference_start).count();
        if (!succeeded) {
            blog(LOG_ERROR, "[Sight/AnimeGAN] Actual ONNX frame inference failed: %ux%u (captured %ux%u)",
                 infer_w, infer_h, w, h);
            std::lock_guard lock(f->input_mutex);
            if (generation == f->model_generation) {
                f->backend_status.health.record(false,inference_ms,sight_backend::InferenceHealth::Clock::now());
                frame.swap(f->input_rgba);
                f->capture_in_flight = false;
            }
            continue;
        }
        // Crop padding from the inference result before scaling in OBS.
        resize_pixels(f, rgba, static_cast<size_t>(w) * h * 4);
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                const size_t src = (static_cast<size_t>(y) * infer_w + x) * 3;
                const size_t dst = (static_cast<size_t>(y) * w + x) * 4;
                std::memcpy(&rgba[dst], &result[src], 3);
                rgba[dst + 3] = 255;
            }
        }
        // Drop stale results if the user selected a different model mid-run.
        {
            std::lock_guard lock(f->input_mutex);
            if (generation != f->model_generation || !f->model_ready)
                continue;
            std::lock_guard out_lock(f->output_mutex);
            f->output_rgba.swap(rgba);
            frame.swap(f->input_rgba);
            f->output_width = w;
            f->output_height = h;
            f->new_output = true;
            f->backend_status.health.record(true,inference_ms,sight_backend::InferenceHealth::Clock::now());
            ++f->completed;
            f->capture_in_flight = false;
            if (!f->announced_first_frame) {
                blog(LOG_INFO, "[Sight/AnimeGAN] SIGHT_SKETCH_REUSE_20261127 First real frame processed: %ux%u (inference %ux%u)",
                     w, h, infer_w, infer_h);
                f->announced_first_frame = true;
            }
        }
    }
}

void capture_stats_proc(void *data, calldata_t *args)
{
    auto *f = static_cast<AnimeGanFilter *>(data);
    calldata_set_int(args, "captures", f->captures.load());
    calldata_set_int(args, "completed", f->completed.load());
    calldata_set_int(args, "skipped", f->skipped.load());
}

void buffer_stats_proc(void *data, calldata_t *args)
{
    auto *f = static_cast<AnimeGanFilter *>(data);
    calldata_set_int(args, "completed", f->completed.load());
    calldata_set_int(args, "growths", f->buffer_growths.load() + f->tensor_growths.load());
}

void *create(obs_data_t *settings, obs_source_t *context)
{
    auto *f = new AnimeGanFilter;
    f->context = context;
    proc_handler_add(obs_source_get_proc_handler(context),
        "void sight_sketch_capture_stats_v1(out int captures, out int completed, out int skipped)",
        capture_stats_proc, f);
    proc_handler_add(obs_source_get_proc_handler(context),
        "void sight_sketch_buffer_stats_v1(out int completed, out int growths)",
        buffer_stats_proc, f);
    f->detail_size = validated_detail_size(obs_data_get_int(settings, DETAIL_SIZE));
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    if (path && *path) {
        f->requested_path = path;
        f->backend_status.request(path);
        f->load_requested = true;
        ++f->model_generation;
    }
    f->worker = std::thread(worker_main, f);
    if (f->load_requested)
        f->frame_cv.notify_one();
    return f;
}

void destroy(void *data)
{
    auto *f = static_cast<AnimeGanFilter *>(data);
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
    if (f->stage)
        gs_stagesurface_destroy(f->stage);
    if (f->capture)
        gs_texrender_destroy(f->capture);
    obs_leave_graphics();
    delete f;
}

void update(void *data, obs_data_t *settings)
{
    auto *f = static_cast<AnimeGanFilter *>(data);
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    const std::string selected = path ? path : "";
    const uint32_t detail = validated_detail_size(obs_data_get_int(settings, DETAIL_SIZE));
    {
        std::lock_guard lock(f->input_mutex);
        const bool path_changed = selected != f->requested_path;
        if (!path_changed && detail == f->detail_size)
            return;
        f->detail_size = detail;
        if (path_changed || (!f->model_ready && !selected.empty())) {
            // If detail changes while model loading is underway, the old
            // generation will be discarded. Request a new load, rather than
            // leaving the filter with no ready model forever.
            f->requested_path = selected;
            f->backend_status.request(selected);
            f->load_requested = true;
            f->model_ready = false;
        }
        f->frame_pending = false;
        f->capture_in_flight = false;
        f->backend_status.health={};
        ++f->model_generation;
        f->announced_first_frame = false;
        f->reset_output_texture = true;
    }
    {
        std::lock_guard lock(f->output_mutex);
        f->new_output = false;
        f->output_rgba.clear();
    }
    f->frame_cv.notify_one();
}

void defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, MODEL_PATH, "");
    obs_data_set_default_int(settings, DETAIL_SIZE, DEFAULT_DETAIL_SIZE);
}

// OBS uses the final obs_properties_add_path argument as the Browse dialog's
// initial directory. Prefer the selected model's folder when it still exists,
// otherwise use the persistent Sight model directory (not the source checkout).
std::string model_browser_directory(AnimeGanFilter *filter)
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

    if(auto configured=sight_preferences::directory();!configured.empty())return configured;
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
    const auto folder = model_browser_directory(static_cast<AnimeGanFilter *>(data));
    obs_property_t *path = obs_properties_add_path(
        p, MODEL_PATH, "Portrait Sketch ONNX model",
        OBS_PATH_FILE, "ONNX model (*.onnx)",
        folder.empty() ? nullptr : folder.c_str());
    sight_diagnostics::add<sight_diagnostics::Format::Onnx>(p,path);
    sight_backend::add(p,static_cast<AnimeGanFilter *>(data));
    obs_property_set_long_description(path,
        "PortraitSketch emphasizes linework and simplifies colors. It can "
        "serve as sketch guidance or preprocessing for compatible ControlNet "
        "workflows; it does not itself run ControlNet. Select the supplied "
        "float32 NHWC PortraitSketch ONNX model. No model = passthrough.");
    obs_property_t *resolution = obs_properties_add_list(
        p, DETAIL_SIZE, "Sketch detail (longest inference edge)",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(resolution, "Quick — 256 px", 256);
    obs_property_list_add_int(resolution, "Balanced — 512 px", 512);
    obs_property_list_add_int(resolution, "Detail — 768 px (slower)", 768);
    obs_property_list_add_int(resolution, "Fine — 1024 px (much slower on CPU)", 1024);
    obs_property_set_long_description(resolution,
        "Higher settings preserve more linework, but greatly increase ONNX "
        "inference time and memory usage. The CPU runner uses a bounded number "
        "of threads so OBS remains responsive. This sketch model intentionally "
        "simplifies textures; higher detail does not change its style.");
    return p;
}

const char *name(void *)
{
    return "lll.sight Portrait Sketch (ControlNet Prep)";
}

void render(void *data, gs_effect_t *)
{
    auto *f = static_cast<AnimeGanFilter *>(data);
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
    if (f->reset_output_texture.exchange(false) && f->output_texture) {
        gs_texture_destroy(f->output_texture);
        f->output_texture = nullptr;
    }
    const auto [pw, ph] = capture_dimensions(width, height, f->detail_size.load());
    bool capture_due = false;
    uint64_t capture_generation = 0;
    {
        std::lock_guard lock(f->input_mutex);
        if (f->model_ready && !f->load_requested) {
            if (f->capture_in_flight) {
                ++f->skipped;
            } else {
                f->capture_in_flight = true;
                capture_generation = f->model_generation;
                capture_due = true;
            }
        }
    }
    if (capture_due) {
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
                gs_effect_t *capture_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
                if (capture_effect) {
                    while (gs_effect_loop(capture_effect, "Draw"))
                        obs_source_video_render(target);
                }
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
                            auto &frame = f->capture_scratch;
                            resize_pixels(f, frame, static_cast<size_t>(pw) * ph * 4);
                            for (uint32_t y = 0; y < ph; ++y)
                                std::memcpy(frame.data() + static_cast<size_t>(y) * pw * 4,
                                    pixels + static_cast<size_t>(y) * pitch, pw * 4);
                            std::lock_guard lock(f->input_mutex);
                            if (capture_generation == f->model_generation &&
                                f->model_ready && !f->load_requested) {
                                f->input_rgba.swap(frame);
                                f->frame_width = pw;
                                f->frame_height = ph;
                                f->frame_pending = true;
                                ++f->captures;
                                f->frame_cv.notify_one();
                                capture_due = false;
                            }
                        }
                        gs_stagesurface_unmap(f->stage);
                    }
                }
            }
        }
        // No frame was queued (graphics failure or a concurrent model change).
        // Do not accidentally release a newer generation's capture claim.
        if (capture_due) {
            std::lock_guard lock(f->input_mutex);
            if (capture_generation == f->model_generation)
                f->capture_in_flight = false;
        }
    }

    {
        std::lock_guard lock(f->output_mutex);
        if (f->new_output) {
            if (!f->output_texture ||
                gs_texture_get_width(f->output_texture) != f->output_width ||
                gs_texture_get_height(f->output_texture) != f->output_height) {
                if (f->output_texture)
                    gs_texture_destroy(f->output_texture);
                f->output_texture = gs_texture_create(
                    f->output_width, f->output_height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
            }
            if (f->output_texture)
                gs_texture_set_image(f->output_texture, f->output_rgba.data(),
                    f->output_width * 4, false);
            f->new_output = false;
        }
    }
    if (!f->model_ready || !f->output_texture) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    // This filter performs its own multi-pass capture and draw. Declare it
    // CUSTOM_DRAW and explicitly own the effect around the final draw.
    gs_effect_t *output_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if (output_effect) {
        gs_eparam_t *image = gs_effect_get_param_by_name(output_effect, "image");
        if (image) {
            gs_effect_set_texture(image, f->output_texture);
            while (gs_effect_loop(output_effect, "Draw"))
                gs_draw_sprite(f->output_texture, 0, width, height);
        }
    }
}
} // namespace

// C++20 designated source info intentionally retains the original id so
// existing OBS scenes can keep their filter settings in the validated build.
struct obs_source_info neural_enhance_filter_info = {
    .id = "neural_enhance_filter",
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
