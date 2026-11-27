#include "common/backend-status.hpp"
#include "common/model-diagnostics.hpp"
#include "common/preferences.hpp"
#include <obs-module.h>
#include "core/analysis_capture.hpp"
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

#include "core/segmentation_processor.hpp"
#include "core/subject_matte_history.hpp"
#include "core/sight_matte_v1.h"
#include "core/sight_motion_v1.h"
#include "core/subject_motion.hpp"
#include "core/longpipe-flow.hpp"

// U2Net subject matte. This is a distinct, independently-verified effect.
// Frame capture, ONNX inference, and output upload have separate thread ownership.
namespace {
constexpr const char *MODEL_PATH = "model_path";
constexpr const char *MODEL_FAMILY = "cutout_model_family";
constexpr const char *DEVICE = "cutout_device";
constexpr const char *FLOW_PATH = "cutout_flow_model";
enum class Device {Auto=0, CPU=1, CUDA=2};
Device requested_device(obs_data_t* settings){
    switch(obs_data_get_int(settings,DEVICE)){
        case 1:return Device::CPU;
        case 2:return Device::CUDA;
        default:return Device::Auto;
    }
}
lll::SegModelType model_family(obs_data_t* settings){
    switch(obs_data_get_int(settings,MODEL_FAMILY)){
        case 1:return lll::SegModelType::MODNet;
        case 2:return lll::SegModelType::Longpipe;
        default:return lll::SegModelType::U2Net;
    }
}
constexpr const char *TREATMENT_MODE = "subject_treatment";
constexpr const char *TREATMENT_STRENGTH = "subject_strength";
constexpr const char *REFINE_EDGES = "refine_edges";
constexpr int MAX_TREATMENT = 3;

std::pair<uint32_t, uint32_t> capture_dimensions(uint32_t width, uint32_t height,uint32_t edge)
{
    if (width == 0 || height == 0)
        return {0, 0};
    const uint32_t longest = std::max(width, height);
    if (longest <= edge)
        return {width, height};
    return {
        std::max(1u, static_cast<uint32_t>(
            static_cast<uint64_t>(width) * edge / longest)),
        std::max(1u, static_cast<uint32_t>(
            static_cast<uint64_t>(height) * edge / longest))
    };
}

// Keep the public v1 matte bounded independently of render quality.
std::vector<uint8_t> shared_matte(const std::vector<uint8_t>&mask,
                                  uint32_t w,uint32_t h,uint32_t sw,uint32_t sh)
{
    if(w==sw&&h==sh)return {};
    std::vector<uint8_t> result(size_t(sw)*sh);
    for(uint32_t y=0;y<sh;++y){
        const float fy=sh>1?float(y)*float(h-1)/float(sh-1):float(h-1)*.5f;
        const auto y0=std::min(uint32_t(fy),h-1),y1=std::min(y0+1,h-1);
        const float dy=fy-y0;
        for(uint32_t x=0;x<sw;++x){
            const float fx=sw>1?float(x)*float(w-1)/float(sw-1):float(w-1)*.5f;
            const auto x0=std::min(uint32_t(fx),w-1),x1=std::min(x0+1,w-1);
            const float dx=fx-x0;
            const float top=mask[size_t(y0)*w+x0]*(1-dx)+mask[size_t(y0)*w+x1]*dx;
            const float bot=mask[size_t(y1)*w+x0]*(1-dx)+mask[size_t(y1)*w+x1]*dx;
            result[size_t(y)*sw+x]=uint8_t(std::clamp(std::lround(top*(1-dy)+bot*dy),0L,255L));
        }
    }
    return result;
}

std::atomic<uint64_t> matte_sequence{0};

struct U2NetFilter {
    obs_source_t *context = nullptr;
    std::mutex input_mutex;
    std::condition_variable frame_cv;
    std::thread worker;
    bool stopping = false;
    bool load_requested = false;
    bool frame_pending = false;
    std::string requested_path,requested_flow;
    lll::SegModelType requested_family=lll::SegModelType::U2Net;
    Device device=Device::Auto;
    std::atomic<uint32_t> capture_edge{320};
    std::string active_path;
    sight_backend::Status backend_status;
    uint64_t model_generation = 0;
    uint32_t frame_width = 0, frame_height = 0;
    uint32_t frame_source_width = 0, frame_source_height = 0;
    std::chrono::steady_clock::time_point frame_time{};
    std::vector<uint8_t> input_rgba;
    std::atomic<bool> model_ready{false};
    // OBS property writes can arrive off the graphics thread.
    std::atomic<int> treatment_mode{0};
    std::atomic<float> treatment_strength{0.7f};
    std::atomic<bool> refine_edges{false};
    std::atomic<float> matte_gamma{1.f}, matte_low{0.f}, matte_high{1.f}, retention{0.f};
    std::atomic<int> matte_view{0};
    std::atomic<bool> reset_history{false};
    // The render thread only captures after the worker requests a frame.
    // Avoid copying pixels off the GPU 60 times/sec for ~4fps CPU inference.
    std::atomic<bool> wants_frame{false};

    std::mutex output_mutex;
    std::vector<uint8_t> output_mask, raw_mask, shared_mask;
    uint32_t shared_width=0,shared_height=0;
    uint32_t output_width = 0, output_height = 0;
    uint32_t output_source_width = 0, output_source_height = 0;
    std::chrono::steady_clock::time_point output_time{};
    uint64_t output_sequence = 0;
    sight::MotionSample output_motion{};
    bool new_output = false;
    std::atomic<bool> reset_output_texture{false};
    bool announced_first_frame = false;

    // Graphics-thread resources; never touched by inference worker.
    gs_texrender_t *capture = nullptr;
    gs_stagesurf_t *stage = nullptr;
    gs_texture_t *output_texture = nullptr; // Small one-channel alpha mask only
    gs_texrender_t *full_resolution = nullptr; // Original full-resolution color
    gs_effect_t *matte_effect = nullptr;
    bool uploaded_raw = false;
    bool announced_calibration = false;
    bool announced_refinement = false; // graphics thread only
    uint32_t stage_width = 0, stage_height = 0;
};

void worker_main(U2NetFilter *f)
{
    lll::SegmentationProcessor processor;
    sight::SubjectMatteHistory history;
    sight::LongpipeFlow flow;
    sight::SubjectMotion motion;
    using Clock = std::chrono::steady_clock;
    double accumulated_inference_ms = 0;
    uint32_t measured_frames = 0;
    bool announced_history = false;
    Clock::time_point previous_result{};
    for (;;) {
        std::vector<uint8_t> frame;
        std::string model_path,flow_path;
        auto family=lll::SegModelType::U2Net;
        auto device=Device::Auto;
        uint32_t w = 0, h = 0, source_w = 0, source_h = 0;
        Clock::time_point captured{};
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
                flow_path=f->requested_flow;
                family=f->requested_family;
                device=f->device;
                generation = f->model_generation;
                f->load_requested = false;
                f->frame_pending = false;
            } else {
                generation = f->model_generation;
                w = f->frame_width;
                h = f->frame_height;
                source_w = f->frame_source_width;
                source_h = f->frame_source_height;
                captured = f->frame_time;
                frame = std::move(f->input_rgba);
                f->frame_pending = false;
            }
        }

        if (do_load) {
            history.reset();
            motion.reset();
            const bool loaded = !model_path.empty() &&
                processor.load(model_path.c_str(), family, device!=Device::CPU, device==Device::CUDA);
            flow.load(loaded&&family==lll::SegModelType::Longpipe?flow_path:std::string(),
                processor.input_width(),processor.input_height(),device!=Device::CPU,device==Device::CUDA);
            accumulated_inference_ms = 0;
            measured_frames = 0;
            {
                std::lock_guard lock(f->input_mutex);
                if (generation != f->model_generation)
                    continue;
                f->active_path = loaded ? model_path : std::string();
                f->backend_status.complete(model_path,loaded,processor.configured_provider(),processor.last_error());
                f->backend_status.detail=flow.status();
                f->capture_edge=family==lll::SegModelType::U2Net?320:
                    family==lll::SegModelType::Longpipe&&loaded?std::max(512,processor.input_edge()):512;
                f->model_ready = loaded;
                f->wants_frame = loaded;
            }
            if (loaded) {
                blog(LOG_INFO, "[Sight/U2Net] Loaded real ONNX model: %s", model_path.c_str());
                blog(LOG_INFO, "[Sight/U2Net] SIGHT_CUTOUT_FAMILY_20261126_K427 family=%d device=%d", int(family),int(device));
                blog(LOG_INFO, "[Sight/U2Net] SIGHT_MATTE_ABI_BOUND_20261126_Q582 capture_edge=%u shared_edge=%u", f->capture_edge.load(),SIGHT_MATTE_MAX_EDGE);
            }
            else if (!model_path.empty())
                blog(LOG_ERROR, "[Sight/U2Net] Failed to load model: %s", model_path.c_str());
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

        std::vector<uint8_t> mask(static_cast<size_t>(w) * h);
        const auto inference_start = Clock::now();
        const bool succeeded = processor.process(rgb.data(), w, h, mask.data());
        double inference_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - inference_start).count();
        if (!succeeded) {
            history.reset();flow.reset();
            motion.reset();
            blog(LOG_ERROR, "[Sight/U2Net] Actual subject mask inference failed: %ux%u", w, h);
            std::lock_guard lock(f->input_mutex);
            if (f->model_ready && generation == f->model_generation) {
                f->backend_status.health.record(false,inference_ms,Clock::now());
                f->wants_frame = true;
            }
            continue;
        }

        if (f->reset_history.exchange(false)){
            history.reset();flow.reset();
        }
        std::vector<uint8_t> raw;
        const float retention = f->retention.load();
        if (retention > 0.f)
            raw = mask;
        const auto now = Clock::now();
        if (now - previous_result > std::chrono::seconds(2)){
            history.reset();flow.reset();
        }
        previous_result = now;
        size_t blended=0;
        if(flow.ready()){
            history.reset();
            const auto flow_start=Clock::now();
            auto prior_runs=flow.runs();
            blended=flow.apply(mask,rgb,processor.canvas_rgb(),w,h,source_w,source_h,captured,retention);
            inference_ms+=std::chrono::duration<double,std::milli>(Clock::now()-flow_start).count();
            if((prior_runs==0&&flow.runs()>0)||!flow.ready())
                blog(LOG_INFO,"[Sight/Cutout] SIGHT_LONGPIPE_FLOW_ADAPTIVE_20261126_N417 %s",flow.status().c_str());
        }else blended=history.apply(mask, rgb, w, h, retention);
        if (blended && !announced_history) {
            blog(LOG_INFO, "[Sight/U2Net] SIGHT_MATTE_HISTORY_20261117: %zu stable pixels blended", blended);
            announced_history = true;
        }

        const auto motion_sample = motion.update(rgb,w,h,source_w,source_h,captured);
        const auto [sw,sh]=capture_dimensions(w,h,SIGHT_MATTE_MAX_EDGE);
        auto compact=shared_matte(mask,w,h,sw,sh);

        // Inference only returns a small matte. Rendering later applies it to
        // the live original-resolution source entirely on the GPU, so no
        // low-resolution RGB frame is ever stretched into the final output.
        // Drop stale results if the user selected a different model mid-run.
        {
            std::lock_guard lock(f->input_mutex);
            if (generation != f->model_generation || !f->model_ready)
                continue;
            std::lock_guard out_lock(f->output_mutex);
            f->output_mask = std::move(mask);
            f->shared_mask=std::move(compact);
            f->shared_width=sw;f->shared_height=sh;
            f->raw_mask = std::move(raw);
            f->output_width = w;
            f->output_height = h;
            f->output_source_width = source_w;
            f->output_source_height = source_h;
            f->output_time = captured;
            f->output_sequence = ++matte_sequence;
            f->output_motion = motion_sample;
            f->new_output = true;
            f->backend_status.detail=flow.status();
            if(flow.ready())f->backend_status.detail+="; flow runs="+std::to_string(flow.runs());
            f->backend_status.health.record(true,inference_ms,Clock::now());
            accumulated_inference_ms += inference_ms;
            ++measured_frames;
            if (measured_frames == 8 && accumulated_inference_ms > 0) {
                const double avg_ms = accumulated_inference_ms / measured_frames;
                blog(LOG_INFO,
                     "[Sight/U2Net] Inference average: %.1f ms/frame (%.1f inference fps, first %u frames)",
                     avg_ms, 1000.0 / avg_ms, measured_frames);
            }
            if (!f->announced_first_frame) {
                blog(LOG_INFO, "[Sight/U2Net] First real subject matte: %ux%u", w, h);
                f->announced_first_frame = true;
            }
            f->wants_frame = true;
        }
    }
}

void read_matte(void *data, calldata_t *params)
{
    auto *f = static_cast<U2NetFilter *>(data);
    calldata_set_bool(params, "available", false);
    void *buffer = nullptr;
    long long max_age = 0, known_sequence = 0;
    if (!calldata_get_ptr(params,"snapshot",&buffer) || !buffer ||
        !calldata_get_int(params,"max_age_ms",&max_age) || max_age < 1 || max_age > 5000)
        return;
    calldata_get_int(params,"known_sequence",&known_sequence);
    auto *out = static_cast<sight_matte_v1 *>(buffer);
    if (out->abi_version != SIGHT_MATTE_ABI_V1 || out->struct_size != sizeof(*out))
        return;
    out->sequence = 0;
    // Never wait for a producer upload or a model change.
    std::unique_lock lock(f->output_mutex, std::try_to_lock);
    if (!lock || !f->model_ready || !obs_source_enabled(f->context) ||
        !f->output_sequence || f->output_mask.empty() ||
        std::chrono::steady_clock::now() - f->output_time > std::chrono::milliseconds(max_age))
        return;
    const auto&pixels=f->shared_mask.empty()?f->output_mask:f->shared_mask;
    const auto count=size_t(f->shared_width)*f->shared_height;
    if (!count || f->shared_width>SIGHT_MATTE_MAX_EDGE || f->shared_height>SIGHT_MATTE_MAX_EDGE ||
        count != pixels.size() || count > SIGHT_MATTE_MAX_PIXELS)
        return;
    out->width = f->shared_width;
    out->height = f->shared_height;
    out->source_width = f->output_source_width;
    out->source_height = f->output_source_height;
    out->sequence = f->output_sequence;
    if (uint64_t(known_sequence) != out->sequence)
        std::memcpy(out->pixels,pixels.data(),count);
    calldata_set_bool(params,"available",true);
}

void read_motion(void *data, calldata_t *params)
{
    auto *f = static_cast<U2NetFilter *>(data);
    calldata_set_bool(params,"available",false);
    void *buffer=nullptr;
    long long max_age=0;
    if (!calldata_get_ptr(params,"snapshot",&buffer) || !buffer ||
        !calldata_get_int(params,"max_age_ms",&max_age) || max_age<1 || max_age>5000)
        return;
    auto *out=static_cast<sight_motion_v1 *>(buffer);
    if (out->abi_version!=SIGHT_MOTION_ABI_V1 || out->struct_size!=sizeof(*out)) return;
    out->sequence=0; out->energy=0.f;
    std::unique_lock lock(f->output_mutex,std::try_to_lock);
    if (!lock || !f->model_ready || !obs_source_enabled(f->context) ||
        !f->output_sequence || !f->output_motion.valid ||
        std::chrono::steady_clock::now()-f->output_time>std::chrono::milliseconds(max_age))
        return;
    out->source_width=f->output_source_width;
    out->source_height=f->output_source_height;
    out->sequence=f->output_sequence;
    out->energy=f->output_motion.energy;
    out->reserved=0;
    calldata_set_bool(params,"available",true);
}

void *create(obs_data_t *settings, obs_source_t *context)
{
    auto *f = new U2NetFilter;
    f->context = context;
    f->treatment_mode = std::clamp(
        static_cast<int>(obs_data_get_int(settings, TREATMENT_MODE)), 0, MAX_TREATMENT);
    f->treatment_strength = std::clamp(
        static_cast<float>(obs_data_get_double(settings, TREATMENT_STRENGTH)), 0.0f, 1.0f);
    f->refine_edges = obs_data_get_bool(settings, REFINE_EDGES);
    f->matte_gamma = std::clamp(float(obs_data_get_double(settings, "matte_gamma")), .1f, 4.f);
    f->matte_low = std::clamp(float(obs_data_get_double(settings, "matte_low")), 0.f, .49f);
    f->matte_high = std::clamp(float(obs_data_get_double(settings, "matte_high")), .51f, 1.f);
    const float retention = std::clamp(float(obs_data_get_double(settings, "matte_retention")), 0.f, .9f);
    if (f->retention.exchange(retention) != retention)
        f->reset_history = true;
    f->matte_view = std::clamp(int(obs_data_get_int(settings, "matte_view")), 0, 4);
    f->requested_family=model_family(settings);
    f->requested_flow=obs_data_get_string(settings,FLOW_PATH);
    f->device=requested_device(settings);
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    if (path && *path) {
        f->requested_path = path;
        f->backend_status.request(path);
        f->load_requested = true;
        ++f->model_generation;
    }
    if (auto *procs = obs_source_get_proc_handler(context)) {
        proc_handler_add(procs,
            "void sight_matte_read_v1(in int max_age_ms, in int known_sequence, in ptr snapshot, out bool available)",
            read_matte, f);
        proc_handler_add(procs,
            "void sight_motion_read_v1(in int max_age_ms, in ptr snapshot, out bool available)",
            read_motion,f);
    }
    f->worker = std::thread(worker_main, f);
    if (f->load_requested)
        f->frame_cv.notify_one();
    return f;
}

void destroy(void *data)
{
    auto *f = static_cast<U2NetFilter *>(data);
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
    auto *f = static_cast<U2NetFilter *>(data);
    const char *path = obs_data_get_string(settings, MODEL_PATH);
    const std::string selected = path ? path : "";
    const auto family=model_family(settings);
    const auto device=requested_device(settings);
    const std::string flow_path=obs_data_get_string(settings,FLOW_PATH);
    f->treatment_mode = std::clamp(
        static_cast<int>(obs_data_get_int(settings, TREATMENT_MODE)), 0, MAX_TREATMENT);
    f->treatment_strength = std::clamp(
        static_cast<float>(obs_data_get_double(settings, TREATMENT_STRENGTH)), 0.0f, 1.0f);
    f->refine_edges = obs_data_get_bool(settings, REFINE_EDGES);
    f->matte_gamma = std::clamp(float(obs_data_get_double(settings, "matte_gamma")), .1f, 4.f);
    f->matte_low = std::clamp(float(obs_data_get_double(settings, "matte_low")), 0.f, .49f);
    f->matte_high = std::clamp(float(obs_data_get_double(settings, "matte_high")), .51f, 1.f);
    const float retention = std::clamp(float(obs_data_get_double(settings, "matte_retention")), 0.f, .9f);
    if (f->retention.exchange(retention) != retention)
        f->reset_history = true;
    f->matte_view = std::clamp(int(obs_data_get_int(settings, "matte_view")), 0, 4);
    {
        std::lock_guard lock(f->input_mutex);
        if (selected == f->requested_path && family==f->requested_family && device==f->device && flow_path==f->requested_flow)
            return;
        f->requested_path = selected;
        f->requested_family=family;
        f->requested_flow=flow_path;
        f->device=device;
        f->backend_status.request(selected);
        f->load_requested = true;
        f->model_ready = false;
        f->wants_frame = false;
        f->frame_pending = false;
        f->backend_status.health={};
        ++f->model_generation;
        f->announced_first_frame = false;
        f->reset_output_texture = true;
    }
    {
        std::lock_guard lock(f->output_mutex);
        f->new_output = false;
        f->output_mask.clear();
        f->shared_mask.clear();f->shared_width=f->shared_height=0;
        f->output_sequence = 0;
        f->output_motion = {};
        f->raw_mask.clear();
    }
    f->frame_cv.notify_one();
}

void defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, MODEL_PATH, "");
    obs_data_set_default_int(settings,MODEL_FAMILY,0);
    obs_data_set_default_int(settings,DEVICE,0);
    obs_data_set_default_string(settings,FLOW_PATH,"");
    obs_data_set_default_int(settings, TREATMENT_MODE, 0);
    obs_data_set_default_double(settings, TREATMENT_STRENGTH, 0.7);
    obs_data_set_default_bool(settings, REFINE_EDGES, false);
    obs_data_set_default_double(settings, "matte_gamma", 1.);
    obs_data_set_default_double(settings, "matte_low", 0.);
    obs_data_set_default_double(settings, "matte_high", 1.);
    obs_data_set_default_double(settings, "matte_retention", 0.);
    obs_data_set_default_int(settings, "matte_view", 0);
}

// OBS uses the final obs_properties_add_path argument as the Browse dialog's
// initial directory. Prefer the selected model's folder when it still exists,
// otherwise use the persistent Sight model directory (not the source checkout).
std::string model_browser_directory(U2NetFilter *filter)
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
    const auto folder = model_browser_directory(static_cast<U2NetFilter *>(data));
    auto*family=obs_properties_add_list(p,MODEL_FAMILY,"Model family",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(family,"U²-Net / U²-NetP (general subjects)",0);
    obs_property_list_add_int(family,"MODNet (portrait matting)",1);
    obs_property_list_add_int(family,"Longpipe (portrait matting)",2);
    obs_property_t *path = obs_properties_add_path(
        p, MODEL_PATH, "Foreground / alpha model",
        OBS_PATH_FILE, "ONNX model (*.onnx)",
        folder.empty() ? nullptr : folder.c_str());
    sight_diagnostics::add<sight_diagnostics::Format::Onnx>(p,path);
    auto*device=obs_properties_add_list(p,DEVICE,"Inference device",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(device,"Auto (CUDA when available)",0);
    obs_property_list_add_int(device,"CPU",1);
    obs_property_list_add_int(device,"CUDA (required)",2);
    obs_property_set_long_description(device,
        "Choose per filter. CPU is useful for lightweight FP32 models. "
        "CUDA requires an initialized CUDA session; individual operators may still use CPU. "
        "The process-wide SIGHT_ONNX_DEVICE=cpu override disables CUDA. "
        "Changing device reloads the selected model. Failed loading passes through video.");
    auto*flow_picker=obs_properties_add_path(p,FLOW_PATH,"Longpipe optical flow (optional)",
        OBS_PATH_FILE,"ONNX model (*.onnx)",folder.empty()?nullptr:folder.c_str());
    obs_property_set_long_description(flow_picker,
        "Select the matching tier/orientation flow graph. Used with nonzero Matte stabilization. "
        "Fresh matting still runs each cycle; mismatched or failed flow falls back to ordinary stabilization.");
    obs_property_set_modified_callback(family,[](obs_properties_t*props,obs_property_t*,obs_data_t*settings){
        auto*field=obs_properties_get(props,FLOW_PATH);
        const bool visible=model_family(settings)==lll::SegModelType::Longpipe;
        const bool changed=obs_property_visible(field)!=visible;
        obs_property_set_visible(field,visible);return changed;
    });
    obs_property_set_visible(flow_picker,false);
    sight_backend::add(p,static_cast<U2NetFilter *>(data));
    obs_property_set_long_description(path,
        "Select the matching model family and local ONNX file. The default Cutout mode preserves "
        "the original behavior. No model or mask = passthrough.");
    obs_property_t *mode = obs_properties_add_list(p, TREATMENT_MODE,
        "Mask treatment", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(mode, "Cutout (transparent background)", 0);
    obs_property_list_add_int(mode, "Spotlight (desaturate background)", 1);
    obs_property_list_add_int(mode, "Warm subject (color treatment)", 2);
    obs_property_list_add_int(mode, "Soft subject edge glow", 3);
    obs_property_set_long_description(mode,
        "Use the selected model’s foreground mask to alter only the subject or surroundings. "
        "This detects foreground, not arbitrary prompt-selected objects or SAM3 tracking.");
    obs_properties_add_float_slider(p, TREATMENT_STRENGTH,
        "Treatment strength", 0.0, 1.0, 0.01);
    auto *refine = obs_properties_add_bool(p, REFINE_EDGES, "Edge-aware matte refinement");
    obs_property_set_long_description(refine,
        "Use source colors to guide mask upsampling. Optional GPU work; "
        "does not change the model or guarantee accurate hair detail.");
    obs_properties_add_float_slider(p, "matte_gamma", "Matte gamma", .1, 4., .05);
    obs_properties_add_float_slider(p, "matte_low", "Background cutoff", 0., .49, .01);
    obs_properties_add_float_slider(p, "matte_high", "Foreground cutoff", .51, 1., .01);
    auto *retention = obs_properties_add_float_slider(p, "matte_retention", "Matte stabilization", 0., .9, .05);
    obs_property_set_long_description(retention,
        "Blend recent inference masks only where source color and alpha remain similar. "
        "Zero disables history. Higher values may lag during subtle motion.");
    auto *view = obs_properties_add_list(p, "matte_view", "Diagnostic view",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(view, "Normal treatment", 0);
    obs_property_list_add_int(view, "Final matte", 1);
    obs_property_list_add_int(view, "Raw inference matte", 2);
    obs_property_list_add_int(view, "Stabilized matte", 3);
    obs_property_list_add_int(view, "Original source", 4);
    obs_property_set_long_description(view,
        "Changes this filter's actual video output. Select Normal treatment before streaming.");
    return p;
}

const char *name(void *)
{
    return "LLL.sight Subject Cutout (U2Net)";
}

void render(void *data, gs_effect_t *)
{
    auto *f = static_cast<U2NetFilter *>(data);
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
    if (!f->matte_effect) {
        char *path = obs_module_file("sight-subject-matte.effect");
        if (path) {
            char *errors = nullptr;
            f->matte_effect = gs_effect_create_from_file(path, &errors);
            if (errors) {
                blog(LOG_ERROR, "[Sight/U2Net] Matte shader: %s", errors);
                bfree(errors);
            }
            bfree(path);
        }
        if (!f->matte_effect) {
            obs_source_skip_video_filter(f->context);
            return;
        }
    }
    const auto edge=f->capture_edge.load();
    const auto [pw, ph] = capture_dimensions(width, height,edge);

    if (f->model_ready && f->wants_frame.exchange(false)) {
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
                            if (f->model_ready && !f->load_requested && edge==f->capture_edge.load()) {
                                f->input_rgba = std::move(frame);
                                f->frame_width = pw;
                                f->frame_height = ph;
                                f->frame_source_width = width;
                                f->frame_source_height = height;
                                f->frame_time = std::chrono::steady_clock::now();
                                f->frame_pending = true;
                                submitted = true;
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

    const int matte_view = f->matte_view.load();
    // Analysis-only mode publishes worker results without uploading/rendering
    // the treatment, and is safe to place among other analysis-only filters.
    if(matte_view==4){obs_source_skip_video_filter(f->context);return;}
    const bool want_raw = matte_view == 2;
    {
        std::lock_guard lock(f->output_mutex);
        if (!f->output_mask.empty() && (f->new_output || !f->output_texture || f->uploaded_raw != want_raw)) {
            if (!f->output_texture ||
                gs_texture_get_width(f->output_texture) != f->output_width ||
                gs_texture_get_height(f->output_texture) != f->output_height) {
                if (f->output_texture)
                    gs_texture_destroy(f->output_texture);
                f->output_texture = gs_texture_create(
                    f->output_width, f->output_height, GS_R8, 1, nullptr, GS_DYNAMIC);
            }
            if (f->output_texture) {
                const auto &pixels = want_raw && !f->raw_mask.empty() ? f->raw_mask : f->output_mask;
                gs_texture_set_image(f->output_texture, pixels.data(), f->output_width, false);
                f->uploaded_raw = want_raw;
                f->new_output = false;
            }
        }
    }
    if (!f->model_ready || !f->output_texture) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    // The matte has its own inference resolution. The original colors must remain
    // full-resolution in the output; this capture stays on the GPU.
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

    gs_texture_t *native_texture = gs_texrender_get_texture(f->full_resolution);
    gs_eparam_t *source_param = gs_effect_get_param_by_name(f->matte_effect, "original_frame");
    gs_eparam_t *mask_param = gs_effect_get_param_by_name(f->matte_effect, "segmentation_mask");
    gs_eparam_t *mode_param = gs_effect_get_param_by_name(f->matte_effect, "treatment_mode");
    gs_eparam_t *strength_param = gs_effect_get_param_by_name(f->matte_effect, "treatment_strength");
    gs_eparam_t *texel_param = gs_effect_get_param_by_name(f->matte_effect, "mask_texel");
    gs_eparam_t *refine_param = gs_effect_get_param_by_name(f->matte_effect, "refine_edges");
    auto *gamma_param = gs_effect_get_param_by_name(f->matte_effect, "matte_gamma");
    auto *low_param = gs_effect_get_param_by_name(f->matte_effect, "matte_low");
    auto *high_param = gs_effect_get_param_by_name(f->matte_effect, "matte_high");
    auto *view_param = gs_effect_get_param_by_name(f->matte_effect, "matte_view");
    if (!gamma_param || !low_param || !high_param || !view_param ||
        !native_texture || !source_param || !mask_param || !mode_param ||
        !strength_param || !texel_param || !refine_param) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_effect_set_texture(source_param, native_texture);
    gs_effect_set_texture(mask_param, f->output_texture);
    gs_effect_set_int(mode_param, f->treatment_mode.load());
    gs_effect_set_float(strength_param, f->treatment_strength.load());
    struct vec2 texel = {1.0f / gs_texture_get_width(f->output_texture), 1.0f / gs_texture_get_height(f->output_texture)};
    gs_effect_set_vec2(texel_param, &texel);
    const bool refine = f->refine_edges.load();
    gs_effect_set_bool(refine_param, refine);
    const float gamma = f->matte_gamma.load(), low = f->matte_low.load(), high = f->matte_high.load();
    gs_effect_set_float(gamma_param, gamma);
    gs_effect_set_float(low_param, low);
    gs_effect_set_float(high_param, high);
    gs_effect_set_int(view_param, matte_view);
    if (!f->announced_calibration && (gamma != 1.f || low != 0.f || high != 1.f || matte_view != 0)) {
        f->announced_calibration = true;
        blog(LOG_INFO, "[Sight/U2Net] SIGHT_MATTE_CALIBRATION_20261117: native controls active");
    }
    if (refine && !f->announced_refinement) {
        blog(LOG_INFO, "[Sight/U2Net] SIGHT_EDGE_MATTE_REFINEMENT_20261117: active native shader");
        f->announced_refinement = true;
    }
    while (gs_effect_loop(f->matte_effect, "Draw"))
        gs_draw_sprite(native_texture, 0, width, height);
}
} // namespace

// A distinct persistent OBS filter ID. Does not overwrite Portrait Sketch.
struct obs_source_info sight_u2net_filter_info = {
    .id = "sight_u2net_filter",
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
