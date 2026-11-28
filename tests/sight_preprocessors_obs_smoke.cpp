#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <callback/proc.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <utility>

namespace {
constexpr uint32_t W = 320, H = 192;

std::mutex ui_mutex;
std::vector<std::pair<obs_task_t, void *>> ui_tasks;

void queue_ui(obs_task_t task, void *param, bool wait)
{
    if (wait) {
        std::cerr << "FAIL: synchronous UI task in test\n";
        return;
    }
    std::lock_guard lock(ui_mutex);
    ui_tasks.emplace_back(task, param);
}

void drain_ui()
{
    std::vector<std::pair<obs_task_t, void *>> ready;
    {
        std::lock_guard lock(ui_mutex);
        ready.swap(ui_tasks);
    }
    for (const auto &item : ready)
        item.first(item.second);
}

struct Pattern { gs_texture_t *tex = nullptr; };
const char *pattern_name(void *) { return "Preprocessors test pattern"; }
uint32_t pattern_width(void *) { return W; }
uint32_t pattern_height(void *) { return H; }

void *pattern_create(obs_data_t *, obs_source_t *)
{
    auto *p = new Pattern;
    std::vector<uint8_t> rgba(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const auto i = (y * W + x) * 4;
            const bool block = x > 60 && x < 260 && y > 45 && y < 152;
            rgba[i] = block ? 245 : static_cast<uint8_t>(20 + x / 6);
            rgba[i + 1] = block ? 30 : static_cast<uint8_t>(90 + y / 3);
            rgba[i + 2] = block ? 60 : static_cast<uint8_t>(30 + y / 7);
            rgba[i + 3] = 255;
        }
    const uint8_t *bytes = rgba.data();
    obs_enter_graphics();
    p->tex = gs_texture_create(W, H, GS_RGBA, 1, &bytes, 0);
    obs_leave_graphics();
    return p;
}
void pattern_destroy(void *data)
{
    auto *p = static_cast<Pattern *>(data);
    obs_enter_graphics();
    if (p->tex) gs_texture_destroy(p->tex);
    obs_leave_graphics();
    delete p;
}
void pattern_render(void *data, gs_effect_t *)
{
    auto *p = static_cast<Pattern *>(data);
    if (p->tex) obs_source_draw(p->tex, 0, 0, W, H, false);
}

obs_source_info pattern_info = {
    .id = "preprocessors_smoke_pattern",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = pattern_name,
    .create = pattern_create,
    .destroy = pattern_destroy,
    .get_width = pattern_width,
    .get_height = pattern_height,
    .video_render = pattern_render,
};

bool render(obs_source_t *source, std::vector<uint8_t> &pixels)
{
    obs_enter_graphics();
    auto *surface = gs_stagesurface_create(W, H, GS_RGBA);
    auto *target = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    bool success = false;
    if (surface && target && gs_texrender_begin(target, W, H)) {
        gs_projection_push();
        gs_matrix_push();
        gs_matrix_identity();
        gs_ortho(0.f, float(W), 0.f, float(H), -100.f, 100.f);
        obs_source_video_render(source);
        gs_matrix_pop();
        gs_projection_pop();
        gs_texrender_end(target);
        auto *tex = gs_texrender_get_texture(target);
        if (tex) {
            gs_stage_texture(surface, tex);
            uint8_t *bytes = nullptr;
            uint32_t stride = 0;
            if (gs_stagesurface_map(surface, &bytes, &stride)) {
                if (stride >= W * 4) {
                    pixels.resize(size_t(W) * H * 4);
                    for (uint32_t y = 0; y < H; ++y)
                        std::memcpy(pixels.data() + size_t(y) * W * 4,
                                    bytes + size_t(y) * stride, W * 4);
                    success = true;
                }
                gs_stagesurface_unmap(surface);
            }
        }
    }
    if (surface) gs_stagesurface_destroy(surface);
    if (target) gs_texrender_destroy(target);
    obs_leave_graphics();
    drain_ui();
    return success;
}

size_t changed(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
    if (a.size() != b.size()) return 0;
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); i += 4)
        if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2])
            ++diff;
    return diff;
}
int64_t stat(obs_source_t *filter, const char *key)
{
    calldata_t args;
    calldata_init(&args);
    const bool called = proc_handler_call(
        obs_source_get_proc_handler(filter),
        std::strcmp(key, "teed_completed") == 0
            ? "sight_preprocessor_teed_stats_v1"
            : std::strcmp(key, "pidinet_completed") == 0
            ? "sight_preprocessor_pidinet_stats_v1"
            : std::strcmp(key, "hed_completed") == 0
            ? "sight_preprocessor_hed_stats_v1"
            : "sight_preprocessor_stats_v1",
        &args);
    const int64_t count = called ? calldata_int(&args, key) : -1;
    calldata_free(&args);
    return count;
}
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: sight_preprocessors_obs_smoke /path/to/module.so /path/to/data\n";
        return 2;
    }
    if (!obs_startup("en-US", nullptr, nullptr)) return 1;
    obs_set_ui_task_handler(queue_ui);
    obs_video_info video{};
    video.graphics_module = "libobs-opengl";
    video.fps_num = 30;
    video.fps_den = 1;
    video.base_width = W;
    video.base_height = H;
    video.output_width = W;
    video.output_height = H;
    video.output_format = VIDEO_FORMAT_RGBA;
    video.colorspace = VIDEO_CS_DEFAULT;
    video.range = VIDEO_RANGE_DEFAULT;
    video.scale_type = OBS_SCALE_DISABLE;
    if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) {
        std::cerr << "FAIL: OBS OpenGL video initialization\n";
        obs_shutdown();
        return 1;
    }
    obs_module_t *module = nullptr;
    if (obs_open_module(&module, argv[1], argv[2]) != MODULE_SUCCESS ||
        !obs_init_module(module)) {
        std::cerr << "FAIL: native plugin did not load\n";
        obs_shutdown();
        return 1;
    }
    size_t index = 0, count = 0;
    const char *type = nullptr;
    bool found_sight = false, found_unified = false, found_preprocessors = false;
    while (obs_enum_filter_types(index++, &type)) {
        ++count;
        if (std::strcmp(type, "sight_preprocessors_filter") == 0)
            found_preprocessors = true;
        if (std::strcmp(type, "sight_unified_tracking_filter") == 0)
            found_unified = true;
        if (std::strcmp(type, "neural_enhance_filter") == 0)
            found_sight = true;
    }
    const char *display = obs_source_get_display_name("sight_preprocessors_filter");
    if (!found_preprocessors || !found_unified || !found_sight || !display ||
        std::strcmp(display, "lll.sight Preprocessors") != 0 ||
        (obs_get_source_output_flags("sight_preprocessors_filter") &
             OBS_SOURCE_CAP_OBSOLETE) != 0) {
        std::cerr << "FAIL: unified preprocessors not exposed from Sight\n";
        obs_shutdown(); return 1;
    }
    auto *properties = obs_get_source_properties("sight_preprocessors_filter");
    auto *modes = properties ? obs_properties_get(properties, "mode_id") : nullptr;
    bool native_found = false, neural_exposed = false, teed_listed = false;
    bool pidinet_listed = false, scribble_pidinet_listed = false;
    bool hed_listed = false, scribble_hed_listed = false;
    if (modes) for (size_t i = 0; i < obs_property_list_item_count(modes); ++i) {
        const std::string id = obs_property_list_item_string(modes, i);
        if (id == "standard_lineart" && !obs_property_list_item_disabled(modes, i))
            native_found = true;
        if (id == "anime_lineart")
            neural_exposed = true;
        if (id == "teed" && !obs_property_list_item_disabled(modes, i))
            teed_listed = true;
        if (id == "pidinet" && !obs_property_list_item_disabled(modes, i))
            pidinet_listed = true;
        if (id == "scribble_pidinet" &&
            !obs_property_list_item_disabled(modes, i))
            scribble_pidinet_listed = true;
        if (id == "hed" && !obs_property_list_item_disabled(modes, i))
            hed_listed = true;
        if (id == "scribble_hed" && !obs_property_list_item_disabled(modes, i))
            scribble_hed_listed = true;
    }
    if (properties) obs_properties_destroy(properties);
    if (!native_found || neural_exposed || !teed_listed ||
        !pidinet_listed || !scribble_pidinet_listed ||
        !hed_listed || !scribble_hed_listed) {
        std::cerr << "FAIL: native mode unavailable or incomplete neural mode exposed\n";
        obs_shutdown(); return 1;
    }

    obs_register_source(&pattern_info);
    obs_source_t *source = obs_source_create("preprocessors_smoke_pattern",
                                              "Native input", nullptr, nullptr);
    std::vector<uint8_t> baseline, output, previous;
    if (!source || !render(source, baseline)) {
        std::cerr << "FAIL: source baseline readback\n";
        obs_shutdown(); return 1;
    }

    auto *settings = obs_data_create();
    obs_data_set_string(settings, "mode_id", "canny");
    obs_source_t *filter = obs_source_create("sight_preprocessors_filter",
                                               "Native Preprocessors", settings, nullptr);
    obs_data_release(settings);
    if (!filter) {
        std::cerr << "FAIL: filter create\n";
        obs_source_release(source);obs_shutdown(); return 1;
    }
    obs_source_filter_add(source, filter);
    auto reuse=[&](const char* label){
        if(!std::getenv("SIGHT_PREPROCESSORS_REUSE"))return true;
        int64_t growths=0,uploads=0;
        auto stats=[&]{
            calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_preprocessor_buffer_stats_v1",&d);
            growths=calldata_int(&d,"growths");uploads=calldata_int(&d,"uploads");calldata_free(&d);return ok;
        };
        auto advance=[&](int64_t target){
            auto end=std::chrono::steady_clock::now()+std::chrono::seconds(15);
            while(std::chrono::steady_clock::now()<end){
                if(!stats())return false;
                if(uploads>=target)return true;
                if(!render(source,output))return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            return false;
        };
        bool warm=stats()&&advance(uploads+4);auto initial=growths;
        const auto reference=output;
        bool continued=warm&&advance(uploads+4);
        uint64_t error=0;unsigned max_error=0;
        if(output.size()==reference.size())for(size_t i=0;i<output.size();++i){
            const unsigned e=unsigned(std::abs(int(output[i])-int(reference[i])));
            error+=e;max_error=std::max(max_error,e);
        }
        const char* device=std::getenv("SIGHT_ONNX_DEVICE");
        const bool cuda_depth=device&&std::string(device)=="cuda"&&std::string(label)=="depth_anything_v2";
        const bool same=output==reference||(cuda_depth&&output.size()==reference.size()&&
            !output.empty()&&max_error<=1&&error<=output.size()/100);
        bool ok=continued&&growths==initial&&initial>0&&same;
        std::cout<<"SIGHT_PREPROCESSOR_REUSE mode="<<label<<" growths="<<initial<<" final="<<growths
            <<" uploads="<<uploads<<" error="<<error<<" max_error="<<max_error<<" pass="<<ok<<"\n";
        return ok;
    };
    for (const char *mode : {"canny", "standard_lineart", "pyracanny", "luminance"}) {
        auto *changed_settings = obs_data_create();
        obs_data_set_string(changed_settings, "mode_id", mode);
        obs_source_update(filter, changed_settings);
        obs_data_release(changed_settings);
        const int64_t start_completed = stat(filter, "completed");
        bool transformed = false;
        size_t diff = 0;
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (!render(source, output)) break;
            diff = changed(output, baseline);
            const bool newer = stat(filter, "completed") > start_completed;
            uint8_t minimum = 255, maximum = 0;
            for (size_t px = 0; px < output.size(); px += 4) {
                for (int channel = 0; channel < 3; ++channel) {
                    minimum = std::min(minimum, output[px + channel]);
                    maximum = std::max(maximum, output[px + channel]);
                }
            }
            const bool has_real_contrast = maximum > minimum + 24;
            if (newer && has_real_contrast && diff > W * H / 8 &&
                (previous.empty() || changed(output, previous) > W * H / 24)) {
                transformed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        std::cout << "mode=" << mode << " transformed=" << transformed
                  << " pixels_different=" << diff
                  << " captures=" << stat(filter, "captures")
                  << " completed=" << stat(filter, "completed")
                  << " skipped=" << stat(filter, "skipped") << "\n";
        if (!transformed) {
            std::cerr << "FAIL: source did not transform under " << mode << "\n";
            obs_source_filter_remove(source, filter);
            obs_source_release(filter);
            obs_source_release(source);
            drain_ui();
            obs_shutdown();
            return 1;
        }
        if(!reuse(mode))return 1;
        previous = output;
    }

    if (const char *model = std::getenv("SIGHT_PREPROCESSORS_DEPTH_MODEL")) {
        auto *depth_settings = obs_data_create();
        obs_data_set_string(depth_settings, "mode_id", "depth_anything_v2");
        obs_data_set_string(depth_settings, "depth_model_path", model);
        obs_source_update(filter, depth_settings);
        obs_data_release(depth_settings);
        const auto start_depth = stat(filter, "depth_completed");
        bool got_depth = false;
        for (int attempt = 0; attempt < 220; ++attempt) {
            if (!render(source, output)) break;
            uint8_t lowest = 255, highest = 0;
            for (size_t px = 0; px < output.size(); px += 4) {
                lowest = std::min(lowest, output[px]);
                highest = std::max(highest, output[px]);
            }
            const bool real_inference = stat(filter, "depth_completed") > start_depth;
            if (real_inference && highest > lowest + 8 &&
                changed(output, baseline) > W * H / 10) {
                got_depth = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
        }
        std::cout << "mode=depth_anything_v2 transformed=" << got_depth
                  << " completed=" << stat(filter, "completed")
                  << " failures=" << stat(filter, "failed") << "\n";
        if (!got_depth || !reuse("depth_anything_v2")) {
            std::cerr << "FAIL: real native Depth Anything V2 output missing\n";
            obs_source_filter_remove(source, filter);
            obs_source_release(filter);
            obs_source_release(source);
            drain_ui();
            obs_shutdown();
            return 1;
        }
    }

    if (const char *model = std::getenv("SIGHT_PREPROCESSORS_TEED_MODEL")) {
        auto *teed_settings = obs_data_create();
        obs_data_set_string(teed_settings, "mode_id", "teed");
        obs_data_set_string(teed_settings, "teed_model_path", model);
        obs_data_set_int(teed_settings, "teed_resolution", 256);
        obs_data_set_int(teed_settings, "teed_safe_steps", 2);
        obs_source_update(filter, teed_settings);
        obs_data_release(teed_settings);
        const auto start = stat(filter, "teed_completed");
        bool got_edges = false;
        for (int attempt = 0; attempt < 230; ++attempt) {
            if (!render(source, output)) break;
            uint8_t lowest = 255, highest = 0;
            for (size_t px = 0; px < output.size(); px += 4) {
                lowest = std::min(lowest, output[px]);
                highest = std::max(highest, output[px]);
            }
            if (stat(filter, "teed_completed") > start &&
                highest > lowest + 16 &&
                changed(output, baseline) > W * H / 10) {
                got_edges = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(45));
        }
        std::cout << "mode=teed real_edge_inference=" << got_edges
                  << " teed_completed=" << stat(filter, "teed_completed")
                  << " failures=" << stat(filter, "failed") << "\n";
        if (!got_edges || !reuse("teed")) {
            std::cerr << "FAIL: TEED native model did not render actual edges\n";
            return 1;
        }
    }

    if (std::getenv("SIGHT_PREPROCESSORS_TEED_BAD_MODEL_CASE")) {
        auto *bad = obs_data_create();
        obs_data_set_string(bad, "mode_id", "teed");
        obs_data_set_string(bad, "teed_model_path",
                            "/nonexistent/sight-teed-should-not-exist.onnx");
        obs_source_update(filter, bad);
        obs_data_release(bad);
        const auto before_failed = stat(filter, "failed");
        const auto before_teed = stat(filter, "teed_completed");
        bool rejected = false, passed_through = false;
        for (int i = 0; i < 100; ++i) {
            if (!render(source, output)) break;
            passed_through = passed_through || changed(output, baseline) == 0;
            rejected = stat(filter, "failed") > before_failed;
            if (passed_through && rejected) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        if (!rejected || !passed_through ||
            stat(filter, "teed_completed") != before_teed) {
            std::cerr << "FAIL: invalid TEED path did not return passthrough "
                         "without claiming neural inference\n";
            return 1;
        }
    }

    if (const char *model = std::getenv("SIGHT_PREPROCESSORS_PIDINET_MODEL")) {
        auto *setting = obs_data_create();
        obs_data_set_string(setting, "mode_id", "pidinet");
        obs_data_set_string(setting, "pidinet_model_path", model);
        obs_data_set_int(setting, "pidinet_resolution", 256);
        obs_data_set_bool(setting, "pidinet_safe", false);
        obs_source_update(filter, setting);
        obs_data_release(setting);
        const auto before = stat(filter, "pidinet_completed");
        bool detected = false;
        for (int i = 0; i < 230; ++i) {
            if (!render(source, output)) break;
            uint8_t minimum = 255, maximum = 0;
            for (size_t idx = 0; idx < output.size(); idx += 4) {
                minimum = std::min(minimum, output[idx]);
                maximum = std::max(maximum, output[idx]);
            }
            if (stat(filter, "pidinet_completed") > before &&
                maximum > minimum + 16 &&
                changed(output, baseline) > W * H / 10) {
                detected = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(45));
        }
        std::cout << "mode=pidinet real_inference=" << detected
                  << " pidinet_completed=" << stat(filter, "pidinet_completed")
                  << " failures=" << stat(filter, "failed") << "\n";
        if (!detected || !reuse("pidinet")) {
            std::cerr << "FAIL: real native PiDiNet OBS output missing\n";
            return 1;
        }
        const auto natural_output = output;
        auto *safe_settings = obs_data_create();
        obs_data_set_string(safe_settings, "mode_id", "pidinet");
        obs_data_set_string(safe_settings, "pidinet_model_path", model);
        obs_data_set_int(safe_settings, "pidinet_resolution", 256);
        obs_data_set_bool(safe_settings, "pidinet_safe", true);
        obs_source_update(filter, safe_settings);
        obs_data_release(safe_settings);
        const auto before_safe = stat(filter, "pidinet_completed");
        bool safe_changed = false;
        for (int i = 0; i < 150; ++i) {
            if (!render(source, output)) break;
            if (stat(filter, "pidinet_completed") > before_safe &&
                changed(natural_output, output) > W * H / 100) {
                safe_changed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        std::cout << "mode=pidinet safe_changed=" << safe_changed << "\n";
        if (!safe_changed) {
            std::cerr << "FAIL: PiDiNet safe mode did not change real OBS pixels\n";
            return 1;
        }
        auto *scribble_settings = obs_data_create();
        obs_data_set_string(scribble_settings, "mode_id", "scribble_pidinet");
        obs_data_set_string(scribble_settings, "pidinet_model_path", model);
        obs_data_set_int(scribble_settings, "pidinet_resolution", 256);
        obs_data_set_bool(scribble_settings, "pidinet_safe", false);
        obs_source_update(filter, scribble_settings);
        obs_data_release(scribble_settings);
        const auto before_scribble = stat(filter, "pidinet_completed");
        bool scribble_changed = false;
        for (int i = 0; i < 160; ++i) {
            if (!render(source, output)) break;
            uint8_t minimum = 255, maximum = 0;
            for (size_t idx = 0; idx < output.size(); idx += 4) {
                minimum = std::min(minimum, output[idx]);
                maximum = std::max(maximum, output[idx]);
            }
            if (stat(filter, "pidinet_completed") > before_scribble &&
                maximum > minimum + 100 &&
                changed(output, natural_output) > W * H / 100) {
                scribble_changed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        std::cout << "mode=scribble_pidinet real_scribble="
                  << scribble_changed << "\n";
        if (!scribble_changed) {
            std::cerr << "FAIL: real PiDiNet NMS scribble did not render\n";
            return 1;
        }
    }

    if (std::getenv("SIGHT_PREPROCESSORS_PIDINET_BAD_MODEL_CASE")) {
        auto *setting = obs_data_create();
        obs_data_set_string(setting, "mode_id", "pidinet");
        obs_data_set_string(setting, "pidinet_model_path",
                            "/nonexistent/sight-pidinet-should-not-exist.onnx");
        obs_source_update(filter, setting);
        obs_data_release(setting);
        const auto before_failed = stat(filter, "failed");
        const auto before_pidi = stat(filter, "pidinet_completed");
        bool rejected = false, passed_through = false;
        for (int i = 0; i < 100; ++i) {
            if (!render(source, output)) break;
            passed_through = passed_through || changed(output, baseline) == 0;
            rejected = stat(filter, "failed") > before_failed;
            if (passed_through && rejected) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        if (!rejected || !passed_through ||
            stat(filter, "pidinet_completed") != before_pidi) {
            std::cerr << "FAIL: invalid PiDiNet ONNX path did not fail "
                         "with passthrough and no inference completion\n";
            return 1;
        }
    }

    if (const char *model = std::getenv("SIGHT_PREPROCESSORS_HED_MODEL")) {
        auto configure = [&](const char *mode, bool safe) {
            auto *data = obs_data_create();
            obs_data_set_string(data, "mode_id", mode);
            obs_data_set_string(data, "hed_model_path", model);
            obs_data_set_int(data, "hed_resolution", 256);
            obs_data_set_bool(data, "hed_safe", safe);
            obs_source_update(filter, data);
            obs_data_release(data);
        };
        configure("hed", false);
        auto before = stat(filter, "hed_completed");
        auto await_hed = [&](const std::vector<uint8_t> *different,
                             bool require_contrast = true) {
            for (int attempt = 0; attempt < 200; ++attempt) {
                if (!render(source, output)) return false;
                uint8_t lo = 255, hi = 0;
                for (size_t p = 0; p < output.size(); p += 4) {
                    lo = std::min(lo, output[p]);
                    hi = std::max(hi, output[p]);
                }
                if (stat(filter, "hed_completed") > before &&
                    (!require_contrast || hi > lo + 16) &&
                    changed(output, baseline) > W * H / 10 &&
                    (!different || changed(output, *different) > W * H / 100))
                    return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(45));
            }
            return false;
        };
        if (!await_hed(nullptr)||!reuse("hed")) {
            std::cerr << "FAIL: native HED ONNX did not produce real edges\n";
            return 1;
        }
        const auto unsafed = output;
        before = stat(filter, "hed_completed");
        configure("hed", true);
        if (!await_hed(&unsafed)) {
            std::cerr << "FAIL: native HED safe-step setting did not change pixels\n";
            return 1;
        }
        before = stat(filter, "hed_completed");
        configure("scribble_hed", false);
        // The real reference returns ALL-BLACK scribble for this broad
        // synthetic OBS pattern at 256px; nonflat is an invalid assertion.
        // Demand fresh HED inference and changed actual rendered pixels.
        if (!await_hed(&unsafed, false)) {
            std::cerr << "FAIL: native Scribble HED did not render neural NMS edges\n";
            return 1;
        }
        std::cout << "mode=hed native_and_scribble=1 completions="
                  << stat(filter, "hed_completed") << "\n";
    }

    if (std::getenv("SIGHT_PREPROCESSORS_HED_BAD_MODEL_CASE")) {
        const auto previous = stat(filter, "hed_completed");
        const auto failures = stat(filter, "failed");
        auto *data = obs_data_create();
        obs_data_set_string(data, "mode_id", "hed");
        obs_data_set_string(data, "hed_model_path",
                            "/nonexistent/sight-hed-should-not-exist.onnx");
        obs_source_update(filter, data);
        obs_data_release(data);
        bool pass_through = false, rejected = false;
        for (int attempt = 0; attempt < 110; ++attempt) {
            if (!render(source, output)) break;
            pass_through = pass_through || changed(output, baseline) == 0;
            rejected = stat(filter, "failed") > failures;
            if (pass_through && rejected) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        if (!pass_through || !rejected ||
            stat(filter, "hed_completed") != previous) {
            std::cerr << "FAIL: invalid HED path must fail and pass through\n";
            return 1;
        }
    }

    if (std::getenv("SIGHT_PREPROCESSORS_BAD_MODEL_CASE")) {
        auto *bad = obs_data_create();
        obs_data_set_string(bad, "mode_id", "depth_anything_v2");
        obs_data_set_string(bad, "depth_model_path",
                            "/nonexistent/sight-depth-should-not-exist.onnx");
        obs_source_update(filter, bad);
        obs_data_release(bad);
        const auto before_failed = stat(filter, "failed");
        const auto before_depth = stat(filter, "depth_completed");
        bool rejected = false, passed_through = false;
        for (int i = 0; i < 100; ++i) {
            if (!render(source, output)) break;
            passed_through = passed_through || changed(output, baseline) == 0;
            rejected = stat(filter, "failed") > before_failed;
            if (passed_through && rejected) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        if (!rejected || !passed_through ||
            stat(filter, "depth_completed") != before_depth) {
            std::cerr << "FAIL: bad native ONNX path was not rejected "
                         "with real passthrough and zero depth inference\n";
            return 1;
        }
    }

    auto *off = obs_data_create();
    obs_data_set_string(off, "mode_id", "off");
    obs_source_update(filter, off);
    obs_data_release(off);
    bool passthrough = false;
    for (int i = 0; i < 50; ++i) {
        if (render(source, output) && changed(baseline, output) == 0) {
            passthrough = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::cout << "off_passthrough=" << passthrough << "\n";
    obs_source_filter_remove(source, filter);
    obs_source_release(filter);
    obs_source_release(source);
    drain_ui();
    obs_shutdown();
    if (!passthrough) {
        std::cerr << "FAIL: disabled mode did not restore original pixels\n";
        return 1;
    }
    std::cout << "PASS: one OBS filter, mode switching, actual image results, passthrough\n";
    return 0;
}
