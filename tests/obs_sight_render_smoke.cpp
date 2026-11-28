#include "../src/core/sight_depth_v1.h"
#include <obs.h>
#include <obs-module.h>
#include "../src/core/sight_snapshot_v1.h"
#include "../src/core/sight_matte_v1.h"
#include "../src/core/sight_motion_v1.h"
#include "../src/core/sight_text_v1.h"
#include "../src/core/sight_flow_v1.h"
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <array>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#ifdef SIGHT_HAVE_OCR_EDITOR
#include <QApplication>
#include <QDialog>
#include <QLabel>
#include <QComboBox>
#include <QSpinBox>
#include <QTimer>
#include <QMouseEvent>
#include <QDialogButtonBox>
#include <QPushButton>
#include <memory>
#endif

namespace {
// Isolated OBS smoke has no Qt frontend. Supply a real queued UI task
// dispatcher so deferred libobs property refreshes run on this test's
// owning thread and release their retained source references.
std::mutex smoke_ui_mutex;
std::vector<std::pair<obs_task_t, void*>> smoke_ui_tasks;
void smoke_queue_ui(obs_task_t task, void* param, bool wait)
{
    if (!task) return;
    if (wait) {
        // OBS's test calls do not request synchronous UI dispatch.
        std::fprintf(stderr, "FAIL: unexpected synchronous UI dispatch in smoke\n");
        return;
    }
    std::lock_guard<std::mutex> lock(smoke_ui_mutex);
    smoke_ui_tasks.emplace_back(task, param);
}
void smoke_drain_ui()
{
    std::vector<std::pair<obs_task_t, void*>> tasks;
    {
        std::lock_guard<std::mutex> lock(smoke_ui_mutex);
        tasks.swap(smoke_ui_tasks);
    }
    for (const auto& t : tasks)
        t.first(t.second);
}

// Test fixtures express tracker settings independently of the selected mode.
obs_data_t* tracking_settings(obs_data_t* raw,const std::string& mode) {
    auto* result=obs_data_create();
    obs_data_set_string(result,"tracking_mode",mode.c_str());
    for(auto* item=obs_data_first(raw);item;obs_data_item_next(&item)){
        const auto key=mode+"_"+obs_data_item_get_name(item);
        switch(obs_data_item_gettype(item)){
        case OBS_DATA_STRING:obs_data_set_string(result,key.c_str(),obs_data_item_get_string(item));break;
        case OBS_DATA_BOOLEAN:obs_data_set_bool(result,key.c_str(),obs_data_item_get_bool(item));break;
        case OBS_DATA_NUMBER:
            if(obs_data_item_numtype(item)==OBS_DATA_NUM_INT)obs_data_set_int(result,key.c_str(),obs_data_item_get_int(item));
            else obs_data_set_double(result,key.c_str(),obs_data_item_get_double(item));
            break;
        default:break;
        }
    }
    return result;
}
obs_source_t* create_tracker(const char* name,obs_data_t* raw,const std::string& mode){
    auto* settings=tracking_settings(raw,mode);
    auto* result=obs_source_create("sight_unified_tracking_filter",name,settings,nullptr);
    obs_data_release(settings);return result;
}
// Let CTest exercise square, landscape, portrait and misaligned aspect ratios.
uint32_t W = 128, H = 128;
struct Pattern { gs_texture_t *texture = nullptr; };
const char *pattern_name(void *) { return "Sight smoke pattern"; }
uint32_t source_w(void *) {return W;}
uint32_t source_h(void *) {return H;}
void *pattern_create(obs_data_t *, obs_source_t *)
{
    auto *p = new Pattern;
    std::vector<uint8_t> rgba(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x) {
        const size_t i = (y * W + x) * 4;
        rgba[i] = x * 2;
        rgba[i + 1] = y * 2;
        rgba[i + 2] = (x + y) & 255;
        rgba[i + 3] = 255;
    }
    const uint8_t *pixels = rgba.data();
    obs_enter_graphics();
    p->texture = gs_texture_create(W, H, GS_RGBA, 1, &pixels, 0);
    obs_leave_graphics();
    return p;
}
void pattern_destroy(void *v)
{
    auto *p = static_cast<Pattern *>(v);
    obs_enter_graphics();
    if (p->texture) gs_texture_destroy(p->texture);
    obs_leave_graphics();
    delete p;
}
void pattern_render(void *v, gs_effect_t *)
{
    auto *p = static_cast<Pattern *>(v);
    if (p->texture)
        obs_source_draw(p->texture, 0, 0, W, H, false);
}
// Generate a disposable BMP so this regression can exercise the real OBS
// image-source module without requiring a private JPEG or checked-in fixture.
struct TemporaryImage {
    std::string path;
    ~TemporaryImage() {
        if (!path.empty())
            std::remove(path.c_str());
    }
    bool make(uint32_t width, uint32_t height, const std::vector<uint8_t>* rgba=nullptr, int dx=0, int dy=0) {
        char pattern[] = "/tmp/sight-obs-image-XXXXXX.bmp";
        const int fd = mkstemps(pattern, 4);
        if (fd < 0)
            return false;
        path = pattern;
        FILE *file = fdopen(fd, "wb");
        if (!file) {
            close(fd);
            return false;
        }
        const uint32_t rowbytes = (width * 3 + 3) & ~3u;
        const uint32_t payload = rowbytes * height;
        std::array<uint8_t, 54> header{};
        auto set32 = [&](size_t i, uint32_t n) {
            for (size_t k = 0; k < 4; ++k)
                header[i + k] = static_cast<uint8_t>(n >> (8 * k));
        };
        header[0] = 'B'; header[1] = 'M';
        set32(2, 54 + payload);
        set32(10, 54);
        set32(14, 40);
        set32(18, width);
        set32(22, height);
        header[26] = 1;
        header[28] = 24;
        set32(34, payload);
        bool ok = fwrite(header.data(), 1, header.size(), file) == header.size();
        std::vector<uint8_t> row(rowbytes, 0);
        for (uint32_t y = 0; ok && y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                if(rgba&&rgba->size()==size_t(width)*height*4){
                    const auto sx=std::clamp(int(x)-dx,0,int(width)-1);
                    const auto sy=std::clamp(int(height-1-y)-dy,0,int(height)-1);
                    const auto i=(size_t(sy)*width+sx)*4;
                    row[x*3]=(*rgba)[i+2];row[x*3+1]=(*rgba)[i+1];row[x*3+2]=(*rgba)[i];
                }else{
                    row[x * 3] = static_cast<uint8_t>((x + y) & 255);
                    row[x * 3 + 1] = static_cast<uint8_t>((y * 2) & 255);
                    row[x * 3 + 2] = static_cast<uint8_t>((x * 3) & 255);
                }
            }
            ok = fwrite(row.data(), 1, row.size(), file) == row.size();
        }
        ok = fclose(file) == 0 && ok;
        return ok;
    }
};

std::atomic<double> routed_gain{-777.0};
std::atomic<double> gain_min{100000.0}, gain_max{-100000.0};
std::atomic<unsigned> gain_changes{0};
std::atomic<unsigned> gate_intermediate{0};
struct Controlled {obs_source_t* context=nullptr;};
const char* controlled_name(void*) {return "Sight numeric router verification target";}
void* controlled_create(obs_data_t*,obs_source_t* source) {
    return new Controlled{source};
}
void controlled_destroy(void* data) {delete static_cast<Controlled*>(data);}
void controlled_update(void*,obs_data_t* settings) {
    const double value=obs_data_get_double(settings,"gain");
    routed_gain=value;
    gain_changes.fetch_add(1);
    if(std::abs(value-10.)>.001 && std::abs(value-90.)>.001)gate_intermediate.fetch_add(1);
    gain_min.store(std::min(gain_min.load(),value));
    gain_max.store(std::max(gain_max.load(),value));
}
void controlled_defaults(obs_data_t* settings) {
    obs_data_set_default_double(settings,"gain",-1.0);
}
obs_properties_t* controlled_properties(void*) {
    auto* props=obs_properties_create();
    auto* group=obs_properties_create();
    obs_properties_add_float_slider(group,"gain","Controlled gain",0,100,.001);
    obs_properties_add_bool(group,"not_numeric","Not numeric");
    obs_properties_add_group(props,"controls","Controls",OBS_GROUP_NORMAL,group);
    return props;
}
void controlled_render(void* data,gs_effect_t*) {
    auto* controlled=static_cast<Controlled*>(data);
    obs_source_skip_video_filter(controlled->context);
}
obs_source_info controlled_info={
    .id="sight_smoke_controlled_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO,
    .get_name=controlled_name,
    .create=controlled_create,
    .destroy=controlled_destroy,
    .get_defaults=controlled_defaults,
    .get_properties=controlled_properties,
    .update=controlled_update,
    .video_render=controlled_render,
};

obs_source_info pattern_info = {
    .id="sight_smoke_test_source",
    .type=OBS_SOURCE_TYPE_INPUT,
    .output_flags=OBS_SOURCE_VIDEO,
    .get_name=pattern_name,
    .create=pattern_create,
    .destroy=pattern_destroy,
    .get_width=source_w,
    .get_height=source_h,
    .video_render=pattern_render,
};
bool render_image(obs_source_t *source, std::vector<uint8_t> &frame)
{
    bool success = false;
    obs_enter_graphics();
    auto *texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    auto *stage = gs_stagesurface_create(W, H, GS_RGBA);
    if (texrender && stage && gs_texrender_begin(texrender, W, H)) {
        gs_projection_push();
        gs_matrix_push();
        gs_matrix_identity();
        gs_ortho(0, static_cast<float>(W), 0, static_cast<float>(H), -100, 100);
        obs_source_video_render(source);
        gs_matrix_pop();
        gs_projection_pop();
        gs_texrender_end(texrender);
        auto *tex = gs_texrender_get_texture(texrender);
        if (tex) {
            gs_stage_texture(stage, tex);
            uint8_t *data = nullptr;
            uint32_t pitch = 0;
            if (gs_stagesurface_map(stage, &data, &pitch)) {
                if (pitch >= W * 4) {
                    frame.resize(W * H * 4);
                    for (uint32_t y = 0; y < H; ++y)
                        std::memcpy(frame.data() + y * W * 4, data + y * pitch, W * 4);
                    success = true;
                }
                gs_stagesurface_unmap(stage);
            }
        }
    }
    if (stage) gs_stagesurface_destroy(stage);
    if (texrender) gs_texrender_destroy(texrender);
    obs_leave_graphics();
    smoke_drain_ui();  // thread-safe, main-thread-only; never inside graphics lock
    return success;
}
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: sight_obs_smoke /path/to/plugin.so /path/to/animegan.onnx /path/to/data\n";
        return 2;
    }
#ifdef SIGHT_HAVE_OCR_EDITOR
    int qt_argc=1;char* qt_argv[]={argv[0],nullptr};
    std::unique_ptr<QApplication> qt_app;
    if(std::getenv("SIGHT_SMOKE_OCR_EDITOR"))qt_app=std::make_unique<QApplication>(qt_argc,qt_argv);
#endif
    if (const char *env = std::getenv("SIGHT_SMOKE_WIDTH"))
        W = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
    if (const char *env = std::getenv("SIGHT_SMOKE_HEIGHT"))
        H = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
    if (W < 64 || H < 64 || W > 1280 || H > 1280) {
        std::cerr << "invalid smoke dimensions\n";
        return 2;
    }
    std::cout << "source_dimensions=" << W << "x" << H << "\n";
    if (!obs_startup("en-US", std::getenv("SIGHT_SMOKE_MODULE_CONFIG_ROOT"), nullptr)) {
        std::cerr << "obs_startup failed\n";
        return 1;
    }
    obs_set_ui_task_handler(smoke_queue_ui);
    smoke_drain_ui();
    obs_video_info info = {};
    info.graphics_module = "libobs-opengl";
    info.fps_num = 30; info.fps_den = 1;
    info.base_width = W; info.base_height = H;
    info.output_width = W; info.output_height = H;
    info.output_format = VIDEO_FORMAT_RGBA;
    info.colorspace = VIDEO_CS_DEFAULT;
    info.range = VIDEO_RANGE_DEFAULT;
    info.scale_type = OBS_SCALE_DISABLE;
    const auto video_result = obs_reset_video(&info);
    if (video_result != OBS_VIDEO_SUCCESS) {
        std::cerr << "obs_reset_video failed: " << video_result << "\n";
        obs_shutdown();
        return 1;
    }
    obs_module_t *module = nullptr;
    const int opened = obs_open_module(&module, argv[1], argv[3]);
    if (opened != MODULE_SUCCESS || !obs_init_module(module)) {
        std::cerr << "Sight module did not load and initialize: " << opened << "\n";
        obs_shutdown();
        return 1;
    }
    size_t count = 0;
    const char *name = nullptr;
    bool verified_only = true;
    bool found_native_sd_live=false;
    bool found_neural_rendering=false;
#ifdef SIGHT_SMOKE_EXPECT_NEURAL_RENDERING
    constexpr bool expect_neural_rendering=true;
#else
    constexpr bool expect_neural_rendering=false;
#endif
#ifdef SIGHT_SMOKE_EXPECT_NATIVE_SD_LIVE
    constexpr bool expect_native_sd_live=true;
#else
    constexpr bool expect_native_sd_live=false;
#endif
    bool found_sketch = false, found_u2net = false, found_depth = false;
    bool found_receiver = false;
    bool found_router = false;
    bool found_privacy = false;
    bool found_ocr = false, found_flow = false, found_unified = false;
    bool found_preprocessors=false;
#ifdef SIGHT_SMOKE_EXPECT_PREPROCESSORS
    constexpr bool expect_preprocessors=true;
#else
    constexpr bool expect_preprocessors=false;
#endif
    const char *selected_filter = std::getenv("SIGHT_SMOKE_EFFECT");
    if (!selected_filter || !*selected_filter)
        selected_filter = "neural_enhance_filter";
    while (obs_enum_filter_types(count++, &name)) {
        std::cout << "registered_filter=" << name << "\n";
        const auto* label = obs_source_get_display_name(name);
        if (!label || std::strncmp(label, "lll.sight ", 10) != 0) {
            std::cerr << "FAIL: filter display namespace: " << name << "\n";
            obs_shutdown();return 1;
        }
        if (std::strcmp(name, "neural_enhance_filter") == 0)
            found_sketch = true;
        else if (std::strcmp(name, "sight_u2net_filter") == 0)
            found_u2net = true;
        else if (std::strcmp(name, "sight_depth_filter") == 0)
            found_depth = true;
        else if(std::strcmp(name,"sight_unified_tracking_filter")==0)
            found_unified=true;
        else if (std::strcmp(name,"sight_privacy_v2_filter")==0)
            found_privacy=true;
        else if (std::strcmp(name, "sight_tracking_receiver_filter") == 0)
            found_receiver = true;
        else if (std::strcmp(name,"sight_value_router_filter")==0)
            found_router = true;
        else if (std::strcmp(name,"sight_text_obfuscation_filter")==0)
            found_ocr=true;
        else if(std::strcmp(name,"sight_preprocessors_filter")==0)
            found_preprocessors=true;
        else if(std::strcmp(name,"sight_spatial_motion_filter")==0)
            found_flow=true;
        else if(std::strcmp(name,"sight_native_sd_live_filter")==0)
            found_native_sd_live=true;
        else if(std::strcmp(name,"sight_neural_rendering_filter")==0)
            found_neural_rendering=true;
        else
            verified_only = false;
    }
    const char *display_name = obs_source_get_display_name("neural_enhance_filter");
    if (!display_name || std::string(display_name) !=
            "lll.sight Portrait Sketch (ControlNet Prep)") {
        std::cerr << "FAIL: registered source is mislabelled as anime color style\n";
        obs_shutdown();
        return 1;
    }
    const char *cutout_name = obs_source_get_display_name("sight_u2net_filter");
    if (!cutout_name || std::string(cutout_name) != "lll.sight Subject Cutout (U2Net)") {
        std::cerr << "FAIL: U2Net filter missing or misnamed\n";
        obs_shutdown();
        return 1;
    }
    const char *depth_name = obs_source_get_display_name("sight_depth_filter");
    if (!depth_name || std::string(depth_name) !=
            "lll.sight Depth Camera (Depth Anything)") {
        std::cerr << "FAIL: Depth Camera missing or misnamed\n";
        obs_shutdown();
        return 1;
    }
    const auto* tracking_name=obs_source_get_display_name("sight_unified_tracking_filter");
    if(!tracking_name||std::strcmp(tracking_name,"lll.sight Tracking")){
        std::cerr<<"FAIL: consolidated tracker missing\n";obs_shutdown();return 1;
    }
    const char* receiver_name=obs_source_get_display_name("sight_tracking_receiver_filter");
    if (!receiver_name || std::strcmp(receiver_name,"lll.sight Tracking Signal Receiver")) {
        std::cerr << "FAIL: tracking signal receiver missing or misnamed\n";
        obs_shutdown();
        return 1;
    }
    const char* router_name=obs_source_get_display_name("sight_value_router_filter");
    if(!router_name || std::strcmp(router_name,"lll.sight Value Router (Signals + Waves)")) {
        std::cerr << "FAIL: Value Router is not registered correctly\n";
        obs_shutdown();
        return 1;
    }
    const auto *privacy_name=obs_source_get_display_name("sight_privacy_v2_filter");
    if(!privacy_name||std::strcmp(privacy_name,
        "lll.sight Privacy V2 — Conceal All Faces (Experimental)")) {
        std::cerr<<"FAIL: Privacy V2 filter not registered with correct name\n";
        obs_shutdown();return 1;
    }
    {
        const uint32_t privacy_flags =
            obs_get_source_output_flags("sight_privacy_v2_filter");
        if ((privacy_flags & OBS_SOURCE_CUSTOM_DRAW) == 0 ||
            (privacy_flags & OBS_SOURCE_VIDEO) == 0) {
            std::cerr<<"FAIL: Privacy V2 does not own its declared draw path (CUSTOM_DRAW)\n";
            obs_shutdown();return 1;
        }
    }
    if (count != 8+size_t(expect_preprocessors)+size_t(found_ocr)+size_t(found_flow)+size_t(expect_native_sd_live)+size_t(expect_neural_rendering) || found_preprocessors!=expect_preprocessors || found_neural_rendering!=expect_neural_rendering || found_native_sd_live!=expect_native_sd_live || !verified_only || !found_unified ||
        !found_privacy || !found_sketch || !found_u2net || !found_depth ||
        !found_receiver || !found_router) {
        std::cerr << "FAIL: expected exactly the verified Sight filter set\n";
        obs_shutdown();return 1;
    }
    const bool tracking_direct=std::strcmp(selected_filter,"sight_unified_tracking_filter")==0;
    const std::string tracking_mode=std::getenv("SIGHT_SMOKE_TRACK_MODE")?std::getenv("SIGHT_SMOKE_TRACK_MODE"):"face";
    const auto tracking_key=[&](const char* key){return tracking_direct?tracking_mode+"_"+key:std::string(key);};
    const bool holistic_direct=tracking_direct&&tracking_mode=="holistic";
    const bool object_direct=tracking_direct&&tracking_mode=="object";
    const bool signal_receiver =
        std::strcmp(selected_filter, "sight_tracking_receiver_filter") == 0;
    const bool value_router =
        std::strcmp(selected_filter, "sight_value_router_filter") == 0;
    const bool router_gate=value_router && std::getenv("SIGHT_SMOKE_ROUTER_GATE");
    const bool tracked_gate=router_gate && std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL");
    bool gate_released_on_loss=false;
    const bool privacy_v2 =
        std::strcmp(selected_filter,"sight_privacy_v2_filter")==0;
    const bool text_ocr=std::strcmp(selected_filter,"sight_text_obfuscation_filter")==0;
    const bool spatial_flow=std::strcmp(selected_filter,"sight_spatial_motion_filter")==0;
    if(spatial_flow&&!found_flow){std::cerr<<"FAIL: spatial flow unavailable\n";obs_shutdown();return 1;}
    if(text_ocr&&!found_ocr) {
        std::cerr<<"FAIL: native OCR filter not compiled\n";obs_shutdown();return 1;
    }
    const bool expect_passthrough=std::getenv("SIGHT_SMOKE_EXPECT_PASSTHROUGH");
    // Check the real OBS property metadata for the model browser, except for
    // the receiver which intentionally has no model (it never infers).
    auto *unconfigured = obs_get_source_properties(
        (signal_receiver||value_router||privacy_v2||text_ocr||spatial_flow) ?
            "sight_unified_tracking_filter" : selected_filter);
    const auto *picker = unconfigured ? obs_properties_get(unconfigured, (signal_receiver||value_router||privacy_v2||text_ocr||spatial_flow)?"face_model_path":tracking_key("model_path").c_str()) : nullptr;
    const char *starting_folder = picker ? obs_property_path_default_path(
        const_cast<obs_property_t *>(picker)) : nullptr;
    if (!starting_folder || !std::filesystem::is_directory(starting_folder) ||
        !std::filesystem::path(starting_folder).is_absolute() ||
        std::filesystem::path(starting_folder).filename() != "models" ||
        std::filesystem::path(starting_folder).parent_path().filename() != "live-latent-lab-sight" ||
        std::filesystem::path(starting_folder).parent_path().parent_path().filename() != "plugin_config") {
        std::cerr << "FAIL: new model browser does not start in Sight models/\n";
        if (unconfigured) obs_properties_destroy(unconfigured);
        obs_shutdown();
        return 1;
    }
    std::cout << "default_model_folder=" << starting_folder << "\n";
    if (std::strcmp(selected_filter, "neural_enhance_filter") == 0) {
        if(std::getenv("SIGHT_SMOKE_CUDA_TEXTURE") && !obs_properties_get(unconfigured,"cuda_texture")){
            std::cerr<<"FAIL: missing CUDA texture option\n";obs_properties_destroy(unconfigured);obs_shutdown();return 1;
        }
        const char *requested = std::getenv("SIGHT_SMOKE_SKETCH_SIZE");
        if (requested && std::strtoul(requested, nullptr, 10) == 1024) {
            obs_property_t *detail = obs_properties_get(unconfigured, "detail_size");
            bool found = false;
            if (detail) {
                for (size_t index = 0; index < obs_property_list_item_count(detail); ++index) {
                    if (obs_property_list_item_int(detail, index) == 1024)
                        found = true;
                }
            }
            if (!found) {
                std::cerr << "FAIL: Portrait Sketch does not expose 1024 px inference detail\n";
                obs_properties_destroy(unconfigured);
                obs_shutdown();
                return 1;
            }
        }
    }
    obs_properties_destroy(unconfigured);

    obs_register_source(&pattern_info);
    obs_register_source(&controlled_info);
    obs_source_t *source = nullptr;
    TemporaryImage fixture;
    if (const char *image_plugin = std::getenv("SIGHT_SMOKE_IMAGE_PLUGIN")) {
        const char *image = std::getenv("SIGHT_SMOKE_IMAGE_FILE");
        if (!image) {
            if (!fixture.make(W, H)) {
                std::cerr << "FAIL: couldn't create real image-source fixture\n";
                obs_shutdown();
                return 2;
            }
            image = fixture.path.c_str();
        }
        if (!image_plugin || !std::filesystem::is_regular_file(image_plugin)) {
            std::cerr << "FAIL: real image-source test requires SIGHT_SMOKE_IMAGE_PLUGIN\n";
            obs_shutdown();
            return 2;
        }
        obs_module_t *image_module = nullptr;
        const int module_result = obs_open_module(
            &image_module, image_plugin, "/usr/share/obs/obs-plugins/image-source");
        if (module_result != MODULE_SUCCESS || !obs_init_module(image_module)) {
            std::cerr << "FAIL: real OBS image-source module unavailable: "
                      << module_result << "\n";
            obs_shutdown();
            return 2;
        }
        auto *image_settings = obs_data_create();
        obs_data_set_string(image_settings, "file", image);
        source = obs_source_create("image_source", "OBS image input", image_settings, nullptr);
        obs_data_release(image_settings);
        if (!source) {
            std::cerr << "FAIL: real OBS image source creation failed\n";
            obs_shutdown();
            return 2;
        }
        if (obs_source_get_base_width(source) != W ||
            obs_source_get_base_height(source) != H) {
            std::cerr << "FAIL: expected a " << W << "x" << H << " image, got "
                      << obs_source_get_base_width(source) << "x"
                      << obs_source_get_base_height(source) << "\n";
            obs_source_release(source);
            obs_shutdown();
            return 2;
        }
        std::cout << "real_obs_image_source=" << image << "\n";
    } else {
        source = obs_source_create("sight_smoke_test_source", "gradient", nullptr, nullptr);
    }
    obs_data_t *settings = obs_data_create();
    obs_data_set_string(settings, "model_path", argv[2]);
    if(std::getenv("SIGHT_SMOKE_CUTOUT_REUSE_DRAW"))obs_data_set_bool(settings,"reuse_source_draw",true);
    if(const char*family=std::getenv("SIGHT_SMOKE_CUTOUT_MODEL"))
        obs_data_set_int(settings,"cutout_model_family",std::atoi(family));
    if(const char*device=std::getenv("SIGHT_SMOKE_CUTOUT_DEVICE"))
        obs_data_set_int(settings,"cutout_device",std::atoi(device));
    if(const char*flow=std::getenv("SIGHT_SMOKE_CUTOUT_FLOW")){
        obs_data_set_string(settings,"cutout_flow_model",flow);
        obs_data_set_double(settings,"matte_retention",.6);
    }
    if (std::getenv("SIGHT_SMOKE_REFINE_EDGES"))
        obs_data_set_bool(settings, "refine_edges", true);
    if (std::getenv("SIGHT_SMOKE_MATTE_CONTROLS")) {
        obs_data_set_double(settings, "matte_gamma", 1.5);
        obs_data_set_double(settings, "matte_low", .1);
        obs_data_set_double(settings, "matte_high", .9);
        obs_data_set_double(settings, "matte_retention", .4);
    }
    if(text_ocr) {
        const char* model_dir=std::getenv("SIGHT_SMOKE_OCR_DATA");
        if(!model_dir){obs_data_release(settings);obs_source_release(source);obs_shutdown();return 2;}
        if(!std::getenv("SIGHT_SMOKE_OCR_USE_DEFAULT"))obs_data_set_string(settings,"tessdata_path",model_dir);
        obs_data_set_int(settings,"ocr_layout",6);
        obs_data_set_int(settings,"ocr_interval",100);
    }
    if(privacy_v2) {
        obs_data_set_int(settings,"privacy_cadence",1);
        if(const char* model=std::getenv("SIGHT_SMOKE_SEMANTIC_MODEL")) {
            obs_data_set_bool(settings,"semantic_enabled",true);
            obs_data_set_string(settings,"semantic_model",model);
            obs_data_set_int(settings,"semantic_class_count",0);
            obs_data_set_string(settings,"semantic_classes","1");
            obs_data_set_double(settings,"semantic_confidence",.99);
            obs_data_set_string(settings,"semantic_thresholds","1:0.20");
            obs_data_set_int(settings,"semantic_hold_ms",180);
            if(std::getenv("SIGHT_SMOKE_SEMANTIC_BUFFERED"))
                obs_data_set_bool(settings,"semantic_buffered",true);
            obs_data_set_int(settings,"privacy_edge",512);
        }
        if(const char* zones=std::getenv("SIGHT_SMOKE_PRIVACY_TEXT_ZONES"))
            obs_data_set_string(settings,"privacy_text_zones",zones);
    }
    if(value_router) {
        obs_data_set_int(settings,"input_mode",std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL")?1:0);
        const char* hand_signal=std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL");
        const bool is_hand=hand_signal&&std::strcmp(hand_signal,"hand")==0;
        const bool is_rtmw=hand_signal&&std::strcmp(hand_signal,"rtmw")==0;
        obs_data_set_string(settings,"signal_channel",is_rtmw?"wholebody":is_hand?"hand":"face");
        obs_data_set_int(settings,"signal_feature",is_rtmw?9:is_hand?7:0);
        obs_data_set_int(settings,"signal_index",is_rtmw?0:4);
        obs_data_set_string(settings,"target_filter","numeric target");
        obs_data_set_string(settings,"target_setting","gain");
        obs_data_set_double(settings,"output_min",10);
        obs_data_set_double(settings,"output_max",90);
        obs_data_set_double(settings,"wave_frequency",1.);
        if(router_gate) {
            obs_data_set_int(settings,"response_mode",1);
            obs_data_set_double(settings,"gate_activate",tracked_gate?.1:.65);
            obs_data_set_double(settings,"gate_release",tracked_gate?.05:.45);
        }
    }
    if(signal_receiver)
        obs_data_set_string(settings,"signal_channel",
            std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL") ?
                std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL") : "face");
    if (const char *detail = std::getenv("SIGHT_SMOKE_SKETCH_SIZE"))
        obs_data_set_int(settings, "detail_size", std::atoi(detail));
    if(std::getenv("SIGHT_SMOKE_CUDA_TEXTURE"))obs_data_set_bool(settings,"cuda_texture",true);
    if (const char *edge = std::getenv("SIGHT_SMOKE_DEPTH_EDGE"))
        obs_data_set_int(settings, "inference_edge", std::atoi(edge));
    if (const char *cadence = std::getenv("SIGHT_SMOKE_DEPTH_CADENCE"))
        obs_data_set_int(settings, "inference_cadence", std::atoi(cadence));
    if (const char *smooth = std::getenv("SIGHT_SMOKE_DEPTH_SMOOTH"))
        obs_data_set_int(settings, "temporal_blend", std::atoi(smooth));
    if (std::getenv("SIGHT_SMOKE_DEPTH_REJECT_STALE"))
        obs_data_set_bool(settings, "reject_stale_depth", true);
    if (const char *runtime = std::getenv("SIGHT_SMOKE_TASK_RUNTIME"))
        obs_data_set_string(settings, "mediapipe_runtime", runtime);
    if (const char *edge = std::getenv("SIGHT_SMOKE_TRACK_EDGE"))
        obs_data_set_int(settings, "inference_edge", std::atoi(edge));
    if (const char *rate = std::getenv("SIGHT_SMOKE_TRACK_CADENCE"))
        obs_data_set_int(settings, "inference_cadence", std::atoi(rate));
    if (const char *smooth = std::getenv("SIGHT_SMOKE_TRACK_SMOOTH"))
        obs_data_set_int(settings, "temporal_blend", std::atoi(smooth));
    if(object_direct && std::getenv("SIGHT_SMOKE_OBJECT_OVERLAY_OFF"))
        obs_data_set_bool(settings,"draw_overlay",false);
    if (holistic_direct) {
        if(std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY"))
            obs_data_set_bool(settings,"draw_overlay",true);
        if(const char* mode=std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE"))
            obs_data_set_int(settings,"holistic_overlay_mode",std::atoi(mode));
        if(const char* mode=std::getenv("SIGHT_SMOKE_STABILITY_MODE"))
            obs_data_set_int(settings,"holistic_stability_mode",std::atoi(mode));
        if(const char* conf=std::getenv("SIGHT_SMOKE_CONFIDENCE"))
            obs_data_set_int(settings,"holistic_min_confidence",std::atoi(conf));
        if(std::getenv("SIGHT_SMOKE_DISABLE_FADE"))
            obs_data_set_bool(settings,"holistic_fade_lost",false);
        if(std::getenv("SIGHT_SMOKE_HOLISTIC_PERSON_MASK"))
            obs_data_set_bool(settings,"holistic_person_mask",true);
    }
    if (const char *age = std::getenv("SIGHT_SMOKE_DEPTH_MAX_AGE"))
        obs_data_set_int(settings, "max_depth_age_seconds", std::atoi(age));
    if (const char *fog = std::getenv("SIGHT_SMOKE_DEPTH_FOG")) {
        obs_data_set_int(settings, "view_mode", 1);
        obs_data_set_int(settings, "fog_amount", std::atoi(fog));
    }
    obs_source_t *filter = tracking_direct?create_tracker("sight smoke filter",settings,tracking_mode):obs_source_create(selected_filter, "sight smoke filter", settings, nullptr);
    obs_data_release(settings);
    if (!source || !filter) {
        std::cerr << "Could not instantiate gradient source or AnimeGAN filter\n";
        if (filter) obs_source_release(filter);
        if (source) obs_source_release(source);
        obs_shutdown();
        return 1;
    }
    auto *configured = obs_source_properties(filter);
    auto *configured_picker = configured ? obs_properties_get(configured,tracking_key("model_path").c_str()) : nullptr;
    if(std::getenv("SIGHT_SMOKE_CUTOUT_MODEL")){
        auto*family=configured?obs_properties_get(configured,"cutout_model_family"):nullptr;
        if(!family||obs_property_list_item_count(family)!=3){
            std::cerr<<"FAIL: Subject Cutout model family selector missing\n";
            if(configured)obs_properties_destroy(configured);
            obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;
        }
    }
    const char *configured_folder = configured_picker
        ? obs_property_path_default_path(configured_picker) : nullptr;
    const auto expected = std::filesystem::absolute(std::filesystem::path(argv[2]).parent_path());
    if (!(signal_receiver||value_router||privacy_v2||text_ocr||spatial_flow) &&
        (!configured_folder || std::filesystem::path(configured_folder) != expected)) {
        std::cerr << "FAIL: picker did not remember the selected model folder\n";
        if (configured) obs_properties_destroy(configured);
        obs_source_release(filter);
        obs_source_release(source);
        obs_shutdown();
        return 1;
    }
    if (std::strcmp(selected_filter, "sight_u2net_filter") == 0) {
        auto *toggle = configured ? obs_properties_get(configured, "refine_edges") : nullptr;
        auto *current_settings = obs_source_get_settings(filter);
        const bool enabled = obs_data_get_bool(current_settings, "refine_edges");
        const bool controls = std::getenv("SIGHT_SMOKE_MATTE_CONTROLS") != nullptr;
        bool valid_matte_controls = configured != nullptr;
        for (const char *key : {"matte_gamma","matte_low","matte_high","matte_retention"}) {
            auto *property = configured ? obs_properties_get(configured,key) : nullptr;
            valid_matte_controls = valid_matte_controls && property &&
                obs_property_get_type(property)==OBS_PROPERTY_FLOAT;
        }
        auto *view = configured ? obs_properties_get(configured,"matte_view") : nullptr;
        valid_matte_controls = valid_matte_controls && view &&
            obs_property_get_type(view)==OBS_PROPERTY_LIST && obs_property_list_item_count(view)==5
            && obs_data_get_int(current_settings,"matte_view")==0
            && std::abs(obs_data_get_double(current_settings,"matte_gamma")-(controls?1.5:1.))<.001
            && std::abs(obs_data_get_double(current_settings,"matte_low")-(controls?.1:0.))<.001
            && std::abs(obs_data_get_double(current_settings,"matte_high")-(controls?.9:1.))<.001
            && std::abs(obs_data_get_double(current_settings,"matte_retention")-(controls?.4:std::getenv("SIGHT_SMOKE_CUTOUT_FLOW")?.6:0.))<.001;
        if(std::getenv("SIGHT_SMOKE_CUTOUT_FLOW")){
            auto*flow=obs_properties_get(configured,"cutout_flow_model");
            valid_matte_controls=valid_matte_controls&&flow&&obs_property_visible(flow)&&
                obs_property_get_type(flow)==OBS_PROPERTY_PATH;
        }
        obs_data_release(current_settings);
        if (!valid_matte_controls || !toggle || obs_property_get_type(toggle) != OBS_PROPERTY_BOOL ||
            enabled != (std::getenv("SIGHT_SMOKE_REFINE_EDGES") != nullptr)) {
            std::cerr << "FAIL: Subject Cutout refinement control/default invalid\n";
            if (configured) obs_properties_destroy(configured);
            obs_source_release(filter); obs_source_release(source); obs_shutdown();
            return 1;
        }
    }
    if(object_direct) {
        const auto* picker=configured ?
            obs_properties_get(configured,tracking_key("model_path").c_str()):nullptr;
        const auto* minimum=configured ?
            obs_properties_get(configured,tracking_key("object_min_score").c_str()):nullptr;
        const auto* overlay=configured ?
            obs_properties_get(configured,tracking_key("draw_overlay").c_str()):nullptr;
        auto* current=obs_source_get_settings(filter);
        const bool enabled=obs_data_get_bool(current,tracking_key("draw_overlay").c_str());
        const int confidence=obs_data_get_int(current,tracking_key("object_min_score").c_str());
        obs_data_release(current);
        if(!picker || !minimum || !overlay ||
           confidence!=35 ||
           enabled==(std::getenv("SIGHT_SMOKE_OBJECT_OVERLAY_OFF")!=nullptr)) {
            std::cerr<<"FAIL: Object Tracking control defaults invalid\n";
            if(configured)obs_properties_destroy(configured);
            obs_source_release(filter);obs_source_release(source);obs_shutdown();
            return 1;
        }
        std::cout<<"SIGHT_OBJECT_CONTROLS_PASS=1\n";
    }
    if(holistic_direct) {
        const auto* toggle=configured ?
            obs_properties_get(configured,tracking_key("draw_overlay").c_str()) : nullptr;
        const auto* mode=configured ?
            obs_properties_get(configured,tracking_key("holistic_overlay_mode").c_str()) : nullptr;
        const auto* stability=configured ?
            obs_properties_get(configured,tracking_key("holistic_stability_mode").c_str()) : nullptr;
        const auto* minimum=configured ?
            obs_properties_get(configured,tracking_key("holistic_min_confidence").c_str()) : nullptr;
        const auto* fade=configured ?
            obs_properties_get(configured,tracking_key("holistic_fade_lost").c_str()) : nullptr;
        const auto* person_mask=configured ?
            obs_properties_get(configured,tracking_key("holistic_person_mask").c_str()) : nullptr;
        if(!toggle || !mode || !stability || !minimum || !fade || !person_mask ||
           obs_property_get_type(const_cast<obs_property_t*>(toggle))!=OBS_PROPERTY_BOOL ||
           obs_property_get_type(const_cast<obs_property_t*>(mode))!=OBS_PROPERTY_LIST ||
           obs_property_list_item_count(const_cast<obs_property_t*>(mode))!=4 ||
           obs_property_get_type(const_cast<obs_property_t*>(stability))!=OBS_PROPERTY_LIST ||
           obs_property_list_item_count(const_cast<obs_property_t*>(stability))!=4 ||
           obs_property_get_type(const_cast<obs_property_t*>(minimum))!=OBS_PROPERTY_INT ||
           obs_property_get_type(const_cast<obs_property_t*>(fade))!=OBS_PROPERTY_BOOL ||
           obs_property_get_type(const_cast<obs_property_t*>(person_mask))!=OBS_PROPERTY_BOOL) {
            std::cerr<<"FAIL: Holistic overlay controls missing\n";
            if(configured)obs_properties_destroy(configured);
            obs_source_release(filter);obs_source_release(source);obs_shutdown();
            return 1;
        }
        auto* current_settings=obs_source_get_settings(filter);
        const bool active=obs_data_get_bool(current_settings,tracking_key("draw_overlay").c_str());
        const int overlay_mode=obs_data_get_int(current_settings,tracking_key("holistic_overlay_mode").c_str());
        const int stability_mode=obs_data_get_int(current_settings,tracking_key("holistic_stability_mode").c_str());
        const int minimum_confidence=obs_data_get_int(current_settings,tracking_key("holistic_min_confidence").c_str());
        const bool fade_lost=obs_data_get_bool(current_settings,tracking_key("holistic_fade_lost").c_str());
        const bool mask_enabled=obs_data_get_bool(current_settings,tracking_key("holistic_person_mask").c_str());
        obs_data_release(current_settings);
        if(active != (std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY")!=nullptr) ||
           overlay_mode!=(std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE")
              ? std::atoi(std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY_MODE")):0) ||
           stability_mode!=(std::getenv("SIGHT_SMOKE_STABILITY_MODE")
              ? std::atoi(std::getenv("SIGHT_SMOKE_STABILITY_MODE")):2) ||
           minimum_confidence!=(std::getenv("SIGHT_SMOKE_CONFIDENCE")
              ? std::atoi(std::getenv("SIGHT_SMOKE_CONFIDENCE")):55) ||
           fade_lost != (std::getenv("SIGHT_SMOKE_DISABLE_FADE")==nullptr) ||
           mask_enabled != (std::getenv("SIGHT_SMOKE_HOLISTIC_PERSON_MASK")!=nullptr)) {
            std::cerr<<"FAIL: Holistic overlay default or mode incorrect\n";
            if(configured)obs_properties_destroy(configured);
            obs_source_release(filter);obs_source_release(source);obs_shutdown();
            return 1;
        }
        std::cout<<"HOLISTIC_OVERLAY_CONTROLS_PASS=1 enabled="<<active
                 <<" mode="<<overlay_mode<<"\n";
    }
    if(configured_folder)
        std::cout << "selected_model_folder=" << configured_folder << "\n";
    if(value_router) {
        for(const char* key:{"input_mode","wave_type","wave_frequency",
                             "signal_channel","signal_feature","signal_index",
                             "target_source","target_filter","target_setting",
                             "output_min","output_max","max_updates_per_second",
                             "response_mode","gate_activate","gate_release"}) {
            if(!obs_properties_get(configured,key)) {
                std::cerr<<"FAIL: value router control missing "<<key<<"\n";
                obs_properties_destroy(configured);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
    }
    if(privacy_v2) {
        for(const char* key:{"privacy_edge","privacy_cadence","privacy_block_ratio","privacy_text_zones"}) {
            if(!obs_properties_get(configured,key)) {
                std::cerr<<"FAIL: privacy control missing "<<key<<"\n";
                obs_properties_destroy(configured);obs_source_release(filter);
                obs_source_release(source);obs_shutdown();return 1;
            }
        }
    }
    if(signal_receiver) {
        for(const char* key:{"signal_channel","signal_max_age_ms","overlay_resolution"}) {
            if(!obs_properties_get(configured,key)) {
                std::cerr<<"FAIL: receiver control missing "<<key<<"\n";
                obs_properties_destroy(configured);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
    }
    if (std::strcmp(selected_filter, "neural_enhance_filter") == 0) {
        auto *detail_property = obs_properties_get(configured, "detail_size");
        if (!detail_property || obs_property_get_type(detail_property) != OBS_PROPERTY_LIST) {
            std::cerr << "FAIL: sketch detail control missing\n";
            obs_properties_destroy(configured);
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
    }
    if (std::strcmp(selected_filter, "sight_depth_filter") == 0) {
        for (const char *key : {"inference_edge", "inference_cadence",
                                "temporal_blend", "reject_stale_depth",
                                "max_depth_age_seconds", "view_mode", "fog_amount"}) {
            if (!obs_properties_get(configured, key)) {
                std::cerr << "FAIL: depth control missing " << key << "\n";
                obs_properties_destroy(configured);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
    }
    if (tracking_direct&&!holistic_direct&&!object_direct) {
        for (const char *key : {"model_path", "mediapipe_runtime", "inference_edge",
                                "inference_cadence", "temporal_blend"}) {
            if (!obs_properties_get(configured, tracking_key(key).c_str())) {
                std::cerr << "FAIL: tracking control missing " << key << "\n";
                obs_properties_destroy(configured);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
    }
    if(text_ocr) {
        for(const char* key:{"tessdata_path","ocr_language","ocr_layout","ocr_view","ocr_edge",
                             "ocr_padding","ocr_confidence","ocr_interval","ocr_age","ocr_block"})
            if(!configured||!obs_properties_get(configured,key)){
                std::cerr<<"FAIL: missing OCR property "<<key<<"\n";
                if(configured)obs_properties_destroy(configured);
                obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;
            }
    }
    obs_properties_destroy(configured);
    if(text_ocr&&std::getenv("SIGHT_SMOKE_OCR_USE_DEFAULT")){
        auto* actual=obs_source_get_settings(filter);
        bool correct=std::string(obs_data_get_string(actual,"tessdata_path"))==
            std::getenv("SIGHT_SMOKE_OCR_DATA")&&!obs_data_has_user_value(actual,"tessdata_path");
        obs_data_release(actual);
        std::cout<<"SIGHT_OCR_LOCAL_DEFAULT="<<correct<<std::endl;
        if(!correct){obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
    }

    std::vector<uint8_t> baseline, current;
    if (!render_image(source, baseline)) {
        std::cerr << "Baseline rendering failed\n";
        obs_source_release(filter);
        obs_source_release(source);
        obs_shutdown();
        return 1;
    }
    if(std::strcmp(selected_filter,"sight_depth_filter")==0&&std::getenv("SIGHT_SMOKE_DEPTH_BRIDGE")){
        auto* settings=obs_data_create();obs_data_set_int(settings,"view_mode",2);
        obs_data_set_int(settings,"temporal_blend",0);
        obs_source_update(filter,settings);obs_data_release(settings);
        obs_source_filter_add(source,filter);
        sight_depth_v1 snap{};
        auto read=[&](int age,uint64_t known=0,bool abi=true){
            snap.abi_version=abi?SIGHT_DEPTH_ABI_V1:99;snap.struct_size=sizeof(snap);
            calldata_t args;calldata_init(&args);calldata_set_ptr(&args,"snapshot",&snap);
            calldata_set_int(&args,"max_age_ms",age);calldata_set_int(&args,"known_sequence",known);
            bool available=false;
            bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_depth_read_v1",&args);
            bool returned=calldata_get_bool(&args,"available",&available);
            calldata_free(&args);return called&&returned&&available;
        };
        auto wait=[&](auto predicate){
            for(int i=0;i<100;++i){
                if(!render_image(source,current)||current.size()!=baseline.size())return false;
                if(predicate())return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(35));
            }
            return false;
        };
        auto same=[&](){
            for(size_t i=0;i<current.size();++i)if(std::abs(int(current[i])-int(baseline[i]))>3)return false;
            return true;
        };
        bool ready=wait([&](){return read(1500)&&same();});
        bool range=false,metadata=false,known=false,abi=false,disabled=false,expired=false,recovered=false;
        if(ready){
            metadata=snap.width==252&&snap.height==252&&snap.source_width==W&&snap.source_height==H&&snap.sequence;
            auto bounds=std::minmax_element(snap.pixels,snap.pixels+snap.width*snap.height);
            range=*bounds.first<5&&*bounds.second>245;
            for(int retry=0;retry<20&&!known;++retry){
                if(read(1500)){
                    const uint64_t seq=snap.sequence;
                    std::fill_n(snap.pixels,SIGHT_DEPTH_MAX_PIXELS,17);
                    known=read(1500,seq)&&snap.sequence==seq&&snap.pixels[0]==17&&
                        snap.pixels[snap.width*snap.height-1]==17;
                }
                if(!known)std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            abi=!read(0)&&!read(1500,0,false)&&!read(5001);
            obs_source_set_enabled(filter,false);disabled=!read(1500);
            obs_source_set_enabled(filter,true);
            // Still attached but not rendering: capture-age expiry must
            // reject the map independently of the filter attachment check.
            std::this_thread::sleep_for(std::chrono::milliseconds(1600));
            expired=!read(1500);
            obs_source_filter_remove(source,filter);
            expired=expired&&!read(5000);
            obs_source_filter_add(source,filter);
            recovered=wait([&](){return read(1500)&&same();});
        }
        bool triple_ok=true;
        if(recovered&&std::getenv("SIGHT_SMOKE_DEPTH_COMPOSE")){
            obs_module_t* mod=nullptr;
            if(obs_open_module(&mod,std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE"),
                std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA"))!=MODULE_SUCCESS||!obs_init_module(mod))return 1;
            settings=obs_data_create();obs_data_set_string(settings,"model_path",std::getenv("SIGHT_SMOKE_DEPTH_COMPOSE"));
            obs_data_set_int(settings,"matte_view",4);
            auto* matte=obs_source_create("sight_u2net_filter","Depth composition matte",settings,nullptr);
            obs_data_release(settings);
            settings=obs_data_create();obs_data_set_int(settings,"flow_view",0);
            obs_data_set_int(settings,"flow_interval",100);
            auto* flow=obs_source_create("sight_spatial_motion_filter","Depth composition flow",settings,nullptr);
            obs_data_release(settings);
            settings=obs_data_create();
            obs_data_set_string(settings,"shader_file",std::getenv("SIGHT_SMOKE_DEPTH_PROBE"));
            obs_data_set_bool(settings,"sight_use_tracking",true);
            auto* hyper=obs_source_create("hypershader_filter","Three signal probe",settings,nullptr);
            obs_data_release(settings);
            if(!matte||!flow||!hyper)return 1;
            obs_source_filter_add(source,matte);obs_source_filter_add(source,flow);
            obs_source_filter_add(source,hyper);
            obs_source_inc_showing(source);obs_set_output_source(0,source);
            auto presence=[&](bool depth,bool subject,bool motion){
                return wait([&](){
                    if(current[0]!=(depth?255:0)||current[1]!=(subject?255:0)||current[2]!=(motion?255:0))return false;
                    size_t lo=0,hi=0,ml=0,mh=0,confidence=0;
                    int min_depth=255,max_depth=0;
                    for(size_t i=size_t(W)*H/10*4;i<current.size();i+=4){
                        min_depth=std::min(min_depth,int(current[i]));max_depth=std::max(max_depth,int(current[i]));
                        lo+=current[i]<80;hi+=current[i]>160;ml+=current[i+1]<24;mh+=current[i+1]>230;
                        confidence+=current[i+2]>64;
                    }
                    return (!depth||(max_depth-min_depth>100&&lo>100&&hi>100))&&
                        (!subject||(ml>100&&mh>100))&&(!motion||confidence>W*H/12);
                });
            };
            const bool all=presence(true,true,true);
            // Match independent CPU snapshot to point-sampled GPU texels.
            bool values=false;
            if(all&&read(1500)){
                size_t agree=0,n=0;
                for(unsigned y=H/8;y<H;y+=17)for(unsigned x=0;x<W;x+=17){
                    const auto sx=std::min(snap.width-1,unsigned((x+.5)*snap.width/W));
                    const auto sy=std::min(snap.height-1,unsigned((y+.5)*snap.height/H));
                    agree+=std::abs(int(current[(size_t(y)*W+x)*4])-int(snap.pixels[sy*snap.width+sx]))<=3;++n;
                }
                values=agree>n*.95;
                std::cout<<"DEPTH_TEXTURE_COMPARE agree="<<agree<<" samples="<<n<<std::endl;
            }
            obs_source_filter_set_order(source,filter,OBS_ORDER_MOVE_BOTTOM);
            // Put consumer back at the end, leaving depth immediately upstream.
            obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_BOTTOM);
            const bool reordered=all&&presence(true,true,true);
            obs_source_set_enabled(filter,false);
            const bool missing=reordered&&presence(false,true,true);
            obs_source_set_enabled(filter,true);
            const bool back=missing&&presence(true,true,true);
            settings=obs_data_create();obs_data_set_int(settings,"view_mode",0);
            obs_source_update(filter,settings);obs_data_release(settings);
            const bool barrier=back&&presence(false,false,false);
            settings=obs_data_create();obs_data_set_int(settings,"view_mode",2);
            obs_source_update(filter,settings);obs_data_release(settings);
            const bool restored=barrier&&presence(true,true,true);
            obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
            smoke_drain_ui();obs_source_filter_remove(source,hyper);obs_source_release(hyper);
            bool glass_ok=true;
            if(restored&&std::getenv("SIGHT_SMOKE_DEPTH_GLASS")){
                settings=obs_data_create();
                obs_data_set_string(settings,"shader_file",std::getenv("SIGHT_SMOKE_DEPTH_GLASS"));
                obs_data_set_bool(settings,"sight_use_tracking",true);
                obs_data_set_double(settings,"param_glass_strength",24.);
                obs_data_set_double(settings,"param_wave_speed",0.);
                obs_data_set_double(settings,"param_wave_length",64.);
                obs_data_set_double(settings,"param_depth_center",.2);
                obs_data_set_double(settings,"param_depth_width",.35);
                obs_data_set_double(settings,"param_edge_tolerance",.18);
                obs_data_set_double(settings,"param_effect_mix",1.);
                auto* glass=obs_source_create("hypershader_filter","Depth Glass consumer",settings,nullptr);
                obs_data_release(settings);
                if(!glass)return 1;
                obs_source_filter_add(source,glass);
                obs_source_inc_showing(source);obs_set_output_source(0,source);
                size_t changed=0,near_changed=0,far_changed=0,near_count=0,far_count=0;
                const bool warped=wait([&](){
                    if(!read(1500))return false;
                    changed=near_changed=far_changed=near_count=far_count=0;
                    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
                        size_t i=(size_t(y)*W+x)*4;
                        bool altered=false;
                        for(unsigned c=0;c<3;++c)altered|=std::abs(int(current[i+c])-int(baseline[i+c]))>3;
                        if(current[i+3]!=baseline[i+3])return false;
                        changed+=altered;
                        // Match the shader's linear sampling (not nearest
                        // texels) at pixel centers. Sharp depth transitions
                        // can straddle four inference-grid texels.
                        const double u=(x+.5)*snap.width/W-.5,v=(y+.5)*snap.height/H-.5;
                        const int ix=int(std::floor(u)),iy=int(std::floor(v));
                        auto at=[&](int px,int py){
                            return double(snap.pixels[size_t(std::clamp(py,0,int(snap.height)-1))*snap.width+
                                std::clamp(px,0,int(snap.width)-1)]);
                        };
                        const double fx=u-ix,fy=v-iy;
                        const double d=(1-fy)*((1-fx)*at(ix,iy)+fx*at(ix+1,iy))+
                            fy*((1-fx)*at(ix,iy+1)+fx*at(ix+1,iy+1));
                        if(d>190){++near_count;near_changed+=altered;}
                        if(d<80){++far_count;far_changed+=altered;}
                    }
                    return far_count>100&&near_count>100&&far_changed>200&&near_changed==0;
                });
                const auto measured=changed,far_pixels=far_changed,near_pixels=near_changed;
                settings=obs_data_create();obs_data_set_bool(settings,"param_debug_band",true);
                obs_source_update(glass,settings);obs_data_release(settings);
                const bool debug=wait([&](){
                    size_t white=0,black=0;
                    for(size_t i=0;i<current.size();i+=4){
                        if(current[i]!=current[i+1]||current[i]!=current[i+2])return false;
                        white+=current[i]>230;black+=current[i]<10;
                    }
                    return white>100&&black>100;
                });
                settings=obs_data_create();obs_data_set_bool(settings,"param_debug_band",false);
                obs_data_set_double(settings,"param_effect_mix",0.);
                obs_source_update(glass,settings);obs_data_release(settings);
                const bool zero=wait(same);
                settings=obs_data_create();obs_data_set_double(settings,"param_effect_mix",1.);
                obs_data_set_bool(settings,"param_protect_subject",true);
                obs_source_update(glass,settings);obs_data_release(settings);
                const bool protected_active=wait([&](){return !same();});
                obs_source_set_enabled(matte,false);
                const bool matte_missing=wait(same);
                obs_source_set_enabled(matte,true);
                const bool matte_back=wait([&](){return !same();});
                obs_source_set_enabled(filter,false);
                const bool missing_depth=wait(same);
                obs_source_set_enabled(filter,true);
                const bool depth_back=wait([&](){return read(1500)&&!same();});
                obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
                smoke_drain_ui();obs_source_filter_remove(source,glass);obs_source_release(glass);smoke_drain_ui();
                glass_ok=warped&&debug&&zero&&protected_active&&matte_missing&&matte_back&&missing_depth&&depth_back;
                std::cout<<"SIGHT_DEPTH_GLASS_BRIDGE warped="<<warped<<" changed="<<measured
                         <<" far="<<far_pixels<<" near="<<near_pixels<<" debug="<<debug<<" zero="<<zero
                         <<" protected="<<protected_active<<" matte_missing="<<matte_missing<<" matte_back="<<matte_back
                         <<" depth_missing="<<missing_depth<<" depth_back="<<depth_back<<std::endl;
            }
            bool layer_ok=true;
            if(restored&&std::getenv("SIGHT_SMOKE_DEPTH_SPARKS")){
                TemporaryImage shifted;
                if(!shifted.make(W,H,&baseline,14,-8))return 1;
                settings=obs_data_create();obs_data_set_string(settings,"file",shifted.path.c_str());
                auto* ref=obs_source_create("image_source","Depth sparks shifted reference",settings,nullptr);
                obs_data_release(settings);
                std::vector<uint8_t> shifted_pixels;
                bool reference=ref&&render_image(ref,shifted_pixels)&&shifted_pixels.size()==baseline.size();
                if(ref)obs_source_release(ref);
                settings=obs_data_create();
                obs_data_set_string(settings,"shader_file",std::getenv("SIGHT_SMOKE_DEPTH_SPARKS"));
                obs_data_set_bool(settings,"sight_use_tracking",true);
                obs_data_set_bool(settings,"param_depth_occlusion",true);
                obs_data_set_double(settings,"param_spark_depth",.75);
                obs_data_set_double(settings,"param_depth_softness",.05);
                obs_data_set_int(settings,"param_spark_region",2);
                obs_data_set_double(settings,"param_emission",1.);
                obs_data_set_double(settings,"param_effect_mix",1.);
                obs_data_set_double(settings,"param_retention",3.);
                obs_data_set_double(settings,"param_spacing",10.);
                obs_data_set_double(settings,"param_speed_threshold",1.);
                obs_data_set_int(settings,"param_spark_color",0xff0000ff);
                auto* sparks=obs_source_create("hypershader_filter","Depth layer Flow Sparks",settings,nullptr);
                obs_data_release(settings);
                if(!sparks)return 1;
                obs_source_filter_add(source,sparks);
                obs_source_inc_showing(source);obs_set_output_source(0,source);
                const bool quiet=wait([&](){return same();});
                for(int i=0;i<12;++i){render_image(source,current);std::this_thread::sleep_for(std::chrono::milliseconds(50));}
                settings=obs_data_create();obs_data_set_string(settings,"file",shifted.path.c_str());
                obs_source_update(source,settings);obs_data_release(settings);
                auto same_shift=[&](){
                    if(!reference||current.size()!=shifted_pixels.size())return false;
                    for(size_t j=0;j<current.size();++j)if(std::abs(int(current[j])-int(shifted_pixels[j]))>3)return false;
                    return true;
                };
                size_t lit=0;
                const bool emitted=reference&&wait([&](){
                    lit=0;
                    for(size_t j=0;j<current.size();j+=4)
                        lit+=int(current[j])-int(shifted_pixels[j])>15&&
                            std::abs(int(current[j+1])-int(shifted_pixels[j+1]))<=3&&
                            std::abs(int(current[j+2])-int(shifted_pixels[j+2]))<=3;
                    return lit>80;
                });
                obs_source_set_enabled(filter,false);
                const bool lost=wait(same_shift);
                obs_source_set_enabled(filter,true);
                // The previous density must not pop back into view upon
                // recovering a valid depth snapshot on stationary input.
                const bool clean=wait([&](){return read(1500)&&same_shift();});
                obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
                smoke_drain_ui();obs_source_filter_remove(source,sparks);obs_source_release(sparks);smoke_drain_ui();
                layer_ok=quiet&&emitted&&lost&&clean;
                std::cout<<"SIGHT_DEPTH_SPARKS quiet="<<quiet<<" emitted="<<emitted<<" pixels="<<lit
                         <<" lost="<<lost<<" clean="<<clean<<std::endl;
            }
            obs_source_filter_remove(source,flow);obs_source_release(flow);
            obs_source_filter_remove(source,matte);obs_source_release(matte);smoke_drain_ui();
            triple_ok=all&&values&&reordered&&missing&&back&&barrier&&restored&&layer_ok&&glass_ok;
            std::cout<<"SIGHT_DEPTH_COMPOSED all="<<all<<" values="<<values<<" reordered="<<reordered
                     <<" missing="<<missing<<" back="<<back<<" barrier="<<barrier<<" restored="<<restored<<std::endl;
        }
        std::cout<<"SIGHT_DEPTH_SNAPSHOT ready="<<ready<<" range="<<range<<" metadata="<<metadata
                 <<" known="<<known<<" abi="<<abi<<" disabled="<<disabled<<" expired="<<expired<<" recovered="<<recovered<<std::endl;
        obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);
        smoke_drain_ui();obs_shutdown();
        return !(ready&&range&&metadata&&known&&abi&&disabled&&expired&&recovered&&triple_ok);
    }
    if (std::strcmp(selected_filter,"sight_u2net_filter")==0 &&
        std::getenv("SIGHT_SMOKE_COMPOSED_EFFECT")) {
        obs_module_t* module=nullptr;
        if(obs_open_module(&module,std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE"),
           std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA"))!=MODULE_SUCCESS||!obs_init_module(module))return 1;
        auto* settings=obs_data_create();
        obs_data_set_int(settings,"flow_view",0);
        obs_data_set_int(settings,"flow_interval",100);
        auto* flow=obs_source_create("sight_spatial_motion_filter","Composed spatial motion",settings,nullptr);
        obs_data_release(settings);
        settings=obs_data_create();
        obs_data_set_string(settings,"shader_file",std::getenv("SIGHT_SMOKE_COMPOSED_EFFECT"));
        obs_data_set_bool(settings,"sight_use_tracking",true);
        auto* hyper=obs_source_create("hypershader_filter","Composed signals",settings,nullptr);
        obs_data_release(settings);
        if(!flow||!hyper)return 1;
        auto* orig=obs_data_create();obs_data_set_int(orig,"matte_view",4);
        obs_source_update(filter,orig);obs_data_release(orig);
        obs_source_filter_add(source,filter);
        obs_source_filter_add(source,flow);
        obs_source_filter_add(source,hyper);
        obs_source_inc_showing(source);obs_set_output_source(0,source);
        std::vector<uint8_t> preview;
        auto presence=[&](bool matte,bool motion){
            for(int i=0;i<85;++i){
                if(!render_image(source,preview)||preview.size()!=baseline.size())return false;
                size_t low=0,high=0,conf=0;
                for(size_t j=size_t(W)*H/10*4;j<preview.size();j+=4){
                    low+=preview[j]<24;high+=preview[j]>230;conf+=preview[j+1]>64;
                }
                if(preview[0]==(matte?255:0)&&preview[1]==(motion?255:0)&&
                   (!matte||(low>100&&high>100))&&(!motion||conf>W*H/12))return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }
            std::cout<<"COMPOSED_DIAG expected="<<matte<<motion<<" actual="<<int(preview[0])<<","<<int(preview[1])<<","<<int(preview[2])<<std::endl;
            return false;
        };
        const bool first=presence(true,true);
        std::cout<<"COMPOSED_INITIAL="<<first<<std::endl;
        obs_source_filter_set_order(source,flow,OBS_ORDER_MOVE_TOP);
        const bool reverse=first&&presence(true,true);
        obs_source_set_enabled(filter,false);
        const bool disabled=reverse&&presence(false,true);
        obs_source_set_enabled(filter,true);
        const bool recovered=disabled&&presence(true,true);
        // A nontransparent upstream output invalidates deeper data. The
        // immediate producer also cannot share pre-warp coordinates.
        settings=obs_data_create();obs_data_set_int(settings,"matte_view",1);
        obs_source_update(filter,settings);obs_data_release(settings);
        const bool blocked=recovered&&presence(false,false);
        settings=obs_data_create();obs_data_set_int(settings,"matte_view",4);
        obs_source_update(filter,settings);obs_data_release(settings);
        const bool restored=blocked&&presence(true,true);
        obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
        smoke_drain_ui();obs_source_filter_remove(source,hyper);obs_source_release(hyper);
        bool sparks_ok=true;
        if(restored&&std::getenv("SIGHT_SMOKE_COMPOSED_SPARKS")){
            TemporaryImage shifted;
            if(!shifted.make(W,H,&baseline,12,-8))return 1;
            settings=obs_data_create();
            obs_data_set_string(settings,"file",shifted.path.c_str());
            auto* ref=obs_source_create("image_source","Shifted portrait reference",settings,nullptr);
            obs_data_release(settings);
            std::vector<uint8_t> shifted_pixels;
            const bool reference=ref&&render_image(ref,shifted_pixels)&&shifted_pixels.size()==baseline.size();
            if(ref)obs_source_release(ref);
            settings=obs_data_create();
            obs_data_set_string(settings,"shader_file",std::getenv("SIGHT_SMOKE_COMPOSED_SPARKS"));
            obs_data_set_bool(settings,"sight_use_tracking",true);
            obs_data_set_int(settings,"param_spark_region",1);
            obs_data_set_double(settings,"param_emission",1.0);
            obs_data_set_double(settings,"param_effect_mix",1.0);
            obs_data_set_double(settings,"param_retention",3.0);
            obs_data_set_double(settings,"param_spacing",10.0);
            obs_data_set_double(settings,"param_speed_threshold",1.0);
            obs_data_set_int(settings,"param_spark_color",0xff0000ff);
            auto* sparks=obs_source_create("hypershader_filter","Masked Flow Sparks",settings,nullptr);
            obs_data_release(settings);
            if(!sparks)return 1;
            obs_source_filter_add(source,sparks);
            obs_source_inc_showing(source);obs_set_output_source(0,source);
            auto wait=[&](auto predicate){
                for(int i=0;i<100;++i){
                    if(!render_image(source,preview)||preview.size()!=baseline.size())return false;
                    if(predicate())return true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(35));
                }return false;
            };
            auto same=[&](const std::vector<uint8_t>& reference){
                if(preview.size()!=reference.size())return false;
                for(size_t j=0;j<preview.size();++j)if(std::abs(int(preview[j])-int(reference[j]))>3)return false;
                return true;
            };
            const bool quiet=wait([&](){return same(baseline);});
            // Let the graph seed its sequence before the first movement.
            for(int i=0;i<10;++i){render_image(source,preview);std::this_thread::sleep_for(std::chrono::milliseconds(45));}
            settings=obs_data_create();obs_data_set_string(settings,"file",shifted.path.c_str());
            obs_source_update(source,settings);obs_data_release(settings);
            size_t lit=0;
            const bool subject=reference&&wait([&](){
                lit=0;
                for(size_t j=0;j<preview.size();j+=4)
                    lit+=int(preview[j])-int(shifted_pixels[j])>15&&
                        std::abs(int(preview[j+1])-int(shifted_pixels[j+1]))<=3&&
                        std::abs(int(preview[j+2])-int(shifted_pixels[j+2]))<=3;
                return lit>100;
            });
            obs_source_set_enabled(filter,false);
            const bool missing=reference&&wait([&](){return same(shifted_pixels);});
            obs_source_set_enabled(filter,true);
            const bool clean=reference&&wait([&](){return same(shifted_pixels);});
            obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
            smoke_drain_ui();obs_source_filter_remove(source,sparks);obs_source_release(sparks);smoke_drain_ui();
            sparks_ok=quiet&&subject&&missing&&clean;
            std::cout<<"SIGHT_MASKED_SPARKS quiet="<<quiet<<" subject="<<subject<<" pixels="<<lit
                     <<" missing="<<missing<<" clean="<<clean<<std::endl;
        }
        obs_source_filter_remove(source,flow);obs_source_release(flow);smoke_drain_ui();
        std::cout<<"SIGHT_COMPOSED_SIGNALS first="<<first<<" reverse="<<reverse<<" disabled="<<disabled
                 <<" recovered="<<recovered<<" blocked="<<blocked<<" restored="<<restored<<std::endl;
        obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);
        smoke_drain_ui();obs_shutdown();
        return !(first&&reverse&&disabled&&recovered&&blocked&&restored&&sparks_ok);
    }
    if(spatial_flow) {
        auto* opts=obs_data_create();obs_data_set_int(opts,"flow_interval",200);
        obs_source_update(filter,opts);obs_data_release(opts);
        obs_source_filter_add(source,filter);
        sight_flow_v1 snap{};
        auto read=[&](int age,bool abi=true){
            snap.abi_version=abi?SIGHT_FLOW_ABI_V1:99;snap.struct_size=sizeof(snap);
            calldata_t args;calldata_init(&args);calldata_set_ptr(&args,"snapshot",&snap);
            calldata_set_int(&args,"max_age_ms",age);
            bool available=false;
            bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_flow_read_v1",&args);
            bool returned=calldata_get_bool(&args,"available",&available);
            calldata_free(&args);
            return called&&returned&&available;
        };
        auto wait=[&](auto predicate) {
            for(int i=0;i<180;++i){
                if(!render_image(source,current)||current.size()!=baseline.size())return false;
                if(predicate())return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return false;
        };
        auto view=[&](int value){auto* s=obs_data_create();obs_data_set_int(s,"flow_view",value);
            obs_source_update(filter,s);obs_data_release(s);};
        auto image=[&](const char* file){auto* s=obs_data_create();obs_data_set_string(s,"file",file);
            obs_source_update(source,s);obs_data_release(s);};
        auto still=[&](){
            if(!read(500)||snap.coverage<.5f)return false;
            for(auto c:snap.cells)if(std::abs(c.dx)>.001||std::abs(c.dy)>.001)return false;
            return true;
        };
        const bool original=wait([&](){return still()&&current==baseline;});
        const bool abi=!read(500,false)&&!read(0)&&!read(1001);
        view(1);
        image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
        double error=0;unsigned n=0;
        bool moved=wait([&](){
            if(!read(500)||snap.source_width!=W||snap.source_height!=H||snap.interval_seconds<=0)return false;
            error=0;n=0;
            for(unsigned y=2;y<16;++y)for(unsigned x=2;x<30;++x){
                auto c=snap.cells[y*32+x];
                if(c.confidence>.5){error+=std::hypot(c.dx*W+5,c.dy*H-3);++n;}
            }
            return n>250&&error/n<.8;
        });
        bool gpu=false;
        if(moved) {
            // Compare observed rendered pixel with current snapshot direction color.
            const unsigned x=16,y=9;const auto c=snap.cells[y*32+x];
            size_t j=(size_t((2*y+1)*H/36)*W+(2*x+1)*W/64)*4;
            float er=std::clamp(.5f-12*c.dx*c.confidence,0.f,1.f);
            float eg=std::clamp(.5f-12*c.dy*c.confidence,0.f,1.f);
            gpu=c.confidence>.5&&std::abs(current[j]/255.f-er)<.06f&&
                std::abs(current[j+1]/255.f-eg)<.06f&&std::abs(current[j+2]/255.f-c.confidence)<.06f;
        }
        const bool settled=wait(still);
        view(2);
        const bool confidence=wait([&](){if(!read(500))return false;
            const auto c=snap.cells[9*32+16];size_t j=(size_t(95)*W+165)*4;
            return c.confidence>.5&&std::abs(current[j]/255.f-c.confidence)<.06f&&
                   current[j]==current[j+1]&&current[j]==current[j+2];});
        obs_source_set_enabled(filter,false);
        const bool disabled=!read(500);
        obs_source_set_enabled(filter,true);
        image(std::getenv("SIGHT_SMOKE_FLOW_FLAT"));
        const bool flat=wait([&](){if(read(500))return false;
            for(size_t j=0;j<current.size();j+=4)if(current[j]!=128||current[j+1]!=128||current[j+2]!=128)return false;
            return true;});
        image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
        view(0);
        const bool recovered=wait([&](){return still()&&current==baseline;});
        auto stabilize=[&](int ms,int crop){
            auto* s=obs_data_create();obs_data_set_int(s,"flow_view",3);
            obs_data_set_int(s,"flow_stabilize_ms",ms);obs_data_set_int(s,"flow_crop_percent",crop);
            obs_source_update(filter,s);obs_data_release(s);
        };
        auto difference=[&](const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){
            if(a.size()!=b.size()||a.size()!=size_t(W)*H*4)return 255.0;
            double sum=0;size_t n=0;
            for(unsigned y=H/4;y<H*3/4;++y)for(unsigned x=W/4;x<W*3/4;++x){
                size_t j=(size_t(y)*W+x)*4;sum+=std::abs(int(a[j])-int(b[j]));++n;
            }
            return sum/n;
        };
        // Build actual centered-crop references with no trajectory history.
        stabilize(1500,6);
        const bool crop_ready=wait([&](){return still()&&current!=baseline;});
        const auto cropped_original=current;
        view(0);image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
        const bool shift_ready=wait(still);
        stabilize(1500,6);
        const bool shift_crop_ready=wait([&](){return still()&&difference(current,cropped_original)>4;});
        const auto cropped_shift=current;
        const double uncorrected=difference(cropped_shift,cropped_original);
        // Restore the first image and allow history to reset before the actual challenge.
        view(0);image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
        const bool first_ready=wait([&](){return still()&&current==baseline;});
        stabilize(1500,6);
        const bool center_ready=wait([&](){return still()&&difference(current,cropped_original)<.5;});
        image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
        double corrected=255;
        const bool stabilized=wait([&](){
            if(!read(500)||snap.cells[9*32+16].dx>-.01f||snap.cells[9*32+16].dy<.01f)return false;
            corrected=difference(current,cropped_original);
            return uncorrected>4&&corrected<uncorrected*.65;
        });
        // A new frame with no texture clears the correction, retaining the
        // fixed crop. No black/uninitialized border pixels may appear.
        image(std::getenv("SIGHT_SMOKE_FLOW_FLAT"));
        const bool reset_crop=wait([&](){
            if(read(500))return false;
            for(size_t j=0;j<current.size();j+=4)
                if(current[j]!=128||current[j+1]!=128||current[j+2]!=128||current[j+3]!=255)return false;
            return true;
        });
        image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
        stabilize(1500,0);
        const bool zero_crop=wait([&](){return still()&&current==baseline;});
        view(0);
        const bool original_back=wait([&](){return still()&&current==baseline;});
        const bool camera_ok=crop_ready&&shift_ready&&shift_crop_ready&&first_ready&&center_ready&&stabilized&&reset_crop&&zero_crop&&original_back;
        std::cout<<"SIGHT_CAMERA_STABILIZER_GPU crop="<<crop_ready<<" shifted_ref="<<shift_crop_ready
                 <<" center="<<center_ready<<" corrected="<<stabilized<<" reset="<<reset_crop
                 <<" zero="<<zero_crop<<" original="<<original_back
                 <<" raw_mae="<<uncorrected<<" corrected_mae="<<corrected<<std::endl;
        bool bridge=true;
        if(const char* module=std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE")) {
            const char* data=std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA");
            const char* effect=std::getenv("SIGHT_SMOKE_FLOW_EFFECT");
            obs_module_t* mod=nullptr;
            if(!data||!effect||obs_open_module(&mod,module,data)!=MODULE_SUCCESS||!obs_init_module(mod))return 1;
            auto* options=obs_data_create();
            obs_data_set_string(options,"shader_file",effect);
            obs_data_set_bool(options,"sight_use_tracking",true);
            obs_data_set_bool(options,"param_debug_flow",true);
            auto* hyper=obs_source_create("hypershader_filter","Spatial flow consumer",options,nullptr);
            obs_data_release(options);
            if(!hyper)return 1;
            obs_source_filter_add(source,hyper);
            obs_source_inc_showing(source);obs_set_output_source(0,source);
            auto same=[&](const std::vector<uint8_t>& pixels){
                if(current.size()!=pixels.size())return false;
                for(size_t j=0;j<current.size();++j)if(std::abs(int(current[j])-int(pixels[j]))>3)return false;
                return true;
            };
            auto pixel=[&](){
                auto c=snap.cells[9*32+16];size_t j=(size_t(95)*W+165)*4;
                return c.confidence>.5&&
                    std::abs(current[j]/255.f-std::clamp(.5f-12*c.dx*c.confidence,0.f,1.f))<.06f&&
                    std::abs(current[j+1]/255.f-std::clamp(.5f-12*c.dy*c.confidence,0.f,1.f))<.06f&&
                    std::abs(current[j+2]/255.f-c.confidence)<.06f;
            };
            auto* ref_options=obs_data_create();
            obs_data_set_string(ref_options,"file",std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
            auto* ref_source=obs_source_create("image_source","Spatial flow independent shifted reference",ref_options,nullptr);
            obs_data_release(ref_options);
            std::vector<uint8_t> shifted;
            const bool reference_ok=ref_source&&render_image(ref_source,shifted)&&shifted.size()==baseline.size();
            if(ref_source)obs_source_release(ref_source);
            const bool ready=wait([&](){return still()&&pixel();});
            image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
            const bool mapped=wait([&](){return read(500)&&snap.cells[9*32+16].dx<-.01f&&snap.cells[9*32+16].dy>.01f&&pixel();});
            bool shift_still=wait([&](){return still()&&pixel();});
            obs_source_set_enabled(filter,false);
            const bool absent=reference_ok&&wait([&](){return !read(500)&&same(shifted);});
            obs_source_set_enabled(filter,true);
            const bool restored=wait([&](){return still()&&pixel();});
            obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_TOP);
            const bool ordered=wait([&](){return same(shifted);});
            obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_BOTTOM);
            const bool reordered=wait([&](){return still()&&pixel();});
            auto* set=obs_data_create();obs_data_set_bool(set,"param_debug_flow",false);
            obs_source_update(hyper,set);obs_data_release(set);
            const bool idle=wait([&](){return still()&&same(shifted);});
            image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
            size_t changed=0;
            const bool warped=wait([&](){
                if(!read(500)||snap.cells[9*32+16].dx<.01f)return false;
                changed=0;
                for(size_t j=0;j<current.size();j+=4)changed+=std::abs(int(current[j])-int(baseline[j]))>6;
                return changed>10000;
            });
            const bool settle=wait([&](){return still()&&same(baseline);});
            obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
            smoke_drain_ui();obs_source_filter_remove(source,hyper);obs_source_release(hyper);smoke_drain_ui();
            bool echo_ok=true;
            if(const char* echo_path=std::getenv("SIGHT_SMOKE_FLOW_ECHO_EFFECT")) {
                auto* opts=obs_data_create();
                obs_data_set_string(opts,"shader_file",echo_path);
                obs_data_set_bool(opts,"sight_use_tracking",true);
                obs_data_set_double(opts,"param_retention",5.0);
                obs_data_set_double(opts,"param_tracking",1.0);
                obs_data_set_double(opts,"param_effect_mix",1.0);
                auto* echo=obs_source_create("hypershader_filter","Spatial flow echo consumer",opts,nullptr);
                obs_data_release(opts);
                if(!echo){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
                obs_source_filter_add(source,echo);
                obs_source_inc_showing(source);obs_set_output_source(0,source);
                const bool seeded=wait([&](){return still()&&same(baseline);});
                image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
                size_t retained=0;
                // The bright history is carried across actual source frames.
                const bool trail=wait([&](){
                    if(!read(500))return false;
                    retained=0;
                    for(size_t j=0;j<current.size();j+=4)retained+=int(current[j])-int(shifted[j])>8;
                    return retained>1000;
                });
                const auto retained_count=retained;
                auto* clear=obs_data_create();obs_data_set_bool(clear,"param_clear_history",true);
                obs_source_update(echo,clear);obs_data_release(clear);
                const bool cleared=wait([&](){return still()&&same(shifted);});
                clear=obs_data_create();obs_data_set_bool(clear,"param_clear_history",false);
                obs_source_update(echo,clear);obs_data_release(clear);
                image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
                const bool again=wait([&](){
                    if(!read(500))return false;
                    size_t n=0;
                    for(size_t j=0;j<current.size();j+=4)n+=int(current[j])-int(baseline[j])>8;
                    return n>1000;
                });
                obs_source_set_enabled(filter,false);
                const bool missing=wait([&](){return !read(500)&&same(baseline);});
                obs_source_set_enabled(filter,true);
                const bool back=wait([&](){return still()&&same(baseline);});
                image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
                const bool before_mix=wait([&](){
                    if(!read(500))return false;
                    size_t n=0;
                    for(size_t j=0;j<current.size();j+=4)n+=int(current[j])-int(shifted[j])>8;
                    return n>1000;
                });
                auto* mix=obs_data_create();obs_data_set_double(mix,"param_effect_mix",0.0);
                obs_source_update(echo,mix);obs_data_release(mix);
                // The processor and the graph consume asynchronously. Keep
                // bypass active across two stationary capture sequences;
                // returning dry pixels before the tick catches up is not
                // sufficient to prove the history is clean.
                uint64_t dry_sequence=0;
                const bool dry=wait([&](){
                    if(!still()||!same(shifted))return false;
                    if(!dry_sequence){dry_sequence=snap.sequence;return false;}
                    return snap.sequence!=dry_sequence;
                });
                mix=obs_data_create();obs_data_set_double(mix,"param_effect_mix",1.0);
                obs_source_update(echo,mix);obs_data_release(mix);
                unsigned remain=0,max_delta=0;float max_displacement=0;bool valid_still=false;
                uint64_t clean_sequence=0;
                const bool no_old=wait([&](){
                    valid_still=still();
                    remain=0;max_delta=0;max_displacement=0;
                    for(size_t j=0;j<current.size();j+=4){
                        unsigned d=std::abs(int(current[j])-int(shifted[j]));remain+=d>3;max_delta=std::max(max_delta,d);
                    }
                    for(auto v:snap.cells)max_displacement=std::max(max_displacement,float(std::hypot(v.dx*W,v.dy*H)));
                    if(!valid_still||!same(shifted)){clean_sequence=0;return false;}
                    if(!clean_sequence){clean_sequence=snap.sequence;return false;}
                    return snap.sequence!=clean_sequence;
                });
                std::cout<<"FLOW_ECHO_BYPASS_DIAGNOSTIC remain="<<remain<<" max_delta="<<max_delta
                    <<" still="<<valid_still<<" displacement="<<max_displacement<<std::endl;
                obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
                smoke_drain_ui();obs_source_filter_remove(source,echo);obs_source_release(echo);smoke_drain_ui();
                echo_ok=seeded&&trail&&cleared&&again&&missing&&back&&before_mix&&dry&&no_old;
                std::cout<<"SIGHT_FLOW_ECHO_BRIDGE seeded="<<seeded<<" trail="<<trail<<" retained="<<retained_count
                    <<" clear="<<cleared<<" again="<<again<<" missing="<<missing<<" back="<<back
                    <<" before_mix="<<before_mix<<" dry="<<dry<<" no_old="<<no_old<<std::endl;
            }
            bool sparks_ok=true;
            if(const char* sparks_path=std::getenv("SIGHT_SMOKE_FLOW_SPARKS_EFFECT")){
                auto* opts=obs_data_create();
                obs_data_set_string(opts,"shader_file",sparks_path);
                obs_data_set_bool(opts,"sight_use_tracking",true);
                obs_data_set_double(opts,"param_retention",.65);
                obs_data_set_double(opts,"param_tracking",1.0);
                obs_data_set_double(opts,"param_effect_mix",1.0);
                obs_data_set_double(opts,"param_emission",1.0);
                obs_data_set_double(opts,"param_spacing",12.0);
                obs_data_set_double(opts,"param_speed_threshold",1.0);
                auto* sparks=obs_source_create("hypershader_filter","Spatial flow sparks consumer",opts,nullptr);
                obs_data_release(opts);
                if(!sparks){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
                obs_source_filter_add(source,sparks);
                obs_source_inc_showing(source);obs_set_output_source(0,source);
                // Start from the shifted reference following the echo test.
                const bool quiet=wait([&](){return still()&&same(shifted);});
                size_t lit=0;
                auto count_sparks=[&](){
                    size_t count=0;
                    for(size_t j=0;j<current.size();j+=4)
                        count+=int(current[j+1])-int(current[j])>20&&
                               int(current[j+2])-int(current[j])>20&&current[j+3]==255;
                    return count;
                };
                image(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
                const bool emitted=wait([&](){lit=count_sparks();return read(500)&&lit>200&&lit<W*H/2;});
                const size_t emitted_count=lit;
                const bool lingering=wait([&](){return still()&&count_sparks()>50;});
                auto* clear=obs_data_create();obs_data_set_bool(clear,"param_clear_history",true);
                obs_source_update(sparks,clear);obs_data_release(clear);
                const bool cleared=wait([&](){return same(baseline);});
                clear=obs_data_create();obs_data_set_bool(clear,"param_clear_history",false);
                obs_data_set_int(clear,"param_spark_color",0xff0000ff);
                obs_source_update(sparks,clear);obs_data_release(clear);
                const bool no_birth=wait([&](){return still()&&same(baseline);});
                image(std::getenv("SIGHT_SMOKE_FLOW_SHIFT"));
                const bool reemitted=wait([&](){
                    size_t red=0;
                    for(size_t j=0;j<current.size();j+=4)
                        red+=int(current[j])-int(current[j+1])>20&&
                             std::abs(int(current[j+1])-int(current[j+2]))<=3&&current[j+3]==255;
                    return read(500)&&red>200;
                });
                obs_source_set_enabled(filter,false);
                const bool lost=wait([&](){return !read(500)&&same(shifted);});
                obs_source_set_enabled(filter,true);
                const bool clean=wait([&](){return still()&&same(shifted);});
                obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
                smoke_drain_ui();obs_source_filter_remove(source,sparks);obs_source_release(sparks);smoke_drain_ui();
                sparks_ok=quiet&&emitted&&lingering&&cleared&&no_birth&&reemitted&&lost&&clean;
                std::cout<<"SIGHT_FLOW_SPARKS_BRIDGE quiet="<<quiet<<" emitted="<<emitted
                    <<" pixels="<<emitted_count<<" lingering="<<lingering<<" clear="<<cleared
                    <<" no_birth="<<no_birth<<" reemitted="<<reemitted<<" lost="<<lost<<" clean="<<clean<<std::endl;
            }
            bridge=reference_ok&&ready&&mapped&&shift_still&&absent&&restored&&ordered&&reordered&&idle&&warped&&settle&&echo_ok&&sparks_ok;
            std::cout<<"SIGHT_SPATIAL_FLOW_BRIDGE ready="<<ready<<" mapped="<<mapped<<" still="<<shift_still
               <<" absent="<<absent<<" restored="<<restored<<" order="<<ordered<<" reordered="<<reordered
               <<" idle="<<idle<<" warp="<<warped<<" settled="<<settle<<" changed="<<changed<<std::endl;
        }
        obs_source_filter_remove(source,filter);
        std::this_thread::sleep_for(std::chrono::milliseconds(550));
        const bool expired=!read(500);
        std::cout<<"SIGHT_SPATIAL_FLOW_OBS original="<<original<<" moved="<<moved<<" gpu="<<gpu
                 <<" settled="<<settled<<" confidence="<<confidence<<" disabled="<<disabled
                 <<" flat="<<flat<<" recovered="<<recovered<<" expired="<<expired<<" ABI="<<abi
                 <<" n="<<n<<" endpoint_error="<<error/std::max(1u,n)<<std::endl;
        obs_source_release(filter);obs_source_release(source);smoke_drain_ui();obs_shutdown();
        return !(original&&moved&&gpu&&settled&&confidence&&disabled&&flat&&recovered&&expired&&abi&&bridge&&camera_ok);
    }
    if(text_ocr) {
        obs_source_filter_add(source,filter);
        auto* orig=std::getenv("SIGHT_SMOKE_IMAGE_FILE");
        auto* blank=std::getenv("SIGHT_SMOKE_OCR_BLANK");
        auto update_mode=[&](int view) {
            auto* set=obs_data_create();obs_data_set_int(set,"ocr_view",view);
            obs_source_update(filter,set);obs_data_release(set);
        };
        auto source_file=[&](const char* path) {
            auto* set=obs_data_create();obs_data_set_string(set,"file",path);
            obs_source_update(source,set);obs_data_release(set);
        };
        auto wait=[&](auto predicate) {
            for(int i=0;i<100;++i) {
                if(!render_image(source,current)||current.size()!=baseline.size())return false;
                if(predicate())return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(45));
            }
            return false;
        };
        size_t ink=0,covered=0,blacked=0,outside=0;
        auto measure=[&]() {
            ink=covered=blacked=outside=0;
            for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x) {
                size_t i=(size_t(y)*W+x)*4;
                bool black=current[i]==0&&current[i+1]==0&&current[i+2]==0&&current[i+3]==255;
                if(baseline[i]<230){++ink;covered+=black;}
                if(baseline[i]>250&&black)++blacked;
                bool allowed=x>=45&&x<=660&&((y>=25&&y<=170)||(y>=195&&y<=355));
                if(!allowed && std::memcmp(&baseline[i],&current[i],4)!=0)++outside;
            }
            return ink>1000&&covered>=ink*99/100&&blacked>1000&&outside==0;
        };
        const bool redacted=wait(measure);
        const auto ink_n=ink,covered_n=covered,black_n=blacked,out_n=outside;
        sight_text_v1 snapshot{};
        auto read_text=[&](int age,bool abi=true) {
            snapshot.abi_version=abi?SIGHT_TEXT_ABI_V1:99;
            snapshot.struct_size=sizeof(snapshot);
            calldata_t args;calldata_init(&args);
            calldata_set_int(&args,"max_age_ms",age);
            calldata_set_ptr(&args,"snapshot",&snapshot);
            bool available=false;
            bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_text_read_v1",&args);
            bool returned=calldata_get_bool(&args,"available",&available);
            calldata_free(&args);
            return called&&returned&&available;
        };
        bool geometry=wait([&](){
            if(!read_text(2000)||snapshot.count!=2||
               snapshot.source_width!=W||snapshot.source_height!=H)return false;
            const auto& a=snapshot.lines[0];const auto& b=snapshot.lines[1];
            return a.x0>.05f&&a.x1<.85f&&a.y0<.4f&&a.y1<.45f&&
                   b.x0>.05f&&b.x1<.85f&&b.y0>.45f&&b.y1<.9f&&
                   a.confidence>=0&&a.confidence<=1&&b.confidence>=0&&b.confidence<=1;
        });
        const bool abi=!read_text(2000,false)&&!read_text(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const bool expired=!read_text(1);
        update_mode(2);
        bool mask=wait([&](){
            size_t white=0,black=0;
            for(size_t i=0;i<current.size();i+=4){
                white+=current[i]==255&&current[i+1]==255&&current[i+2]==255;
                black+=current[i]==0&&current[i+1]==0&&current[i+2]==0;
            }
            return white>1000&&black>W*H/2&&white+black==W*H;
        });
        update_mode(1);
        bool mosaic=wait([&](){
            size_t changed=0,white=0,outer=0;
            for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
                size_t i=(size_t(y)*W+x)*4;
                if(std::memcmp(&baseline[i],&current[i],4)!=0) {
                    ++changed;white+=current[i]>0;
                    if(x<45||x>660||!((y>=25&&y<=170)||(y>=195&&y<=355)))++outer;
                }
            }
            return changed>500&&white>200&&outer==0;
        });
        update_mode(3);
        bool original=wait([&](){return current==baseline;});
        update_mode(0);
        bool cleared=false,recovered=false;
        if(orig&&blank) {
            source_file(blank);
            update_mode(2);
            const bool blank_mask=wait([&](){
                if(!read_text(2000)||snapshot.count!=0)return false;
                for(size_t i=0;i<current.size();i+=4)
                    if(current[i]!=0||current[i+1]!=0||current[i+2]!=0||current[i+3]!=255)return false;
                return true;
            });
            update_mode(0);
            cleared=blank_mask&&wait([&](){
                for(size_t i=0;i<current.size();i+=4)
                    if(current[i]!=255||current[i+1]!=255||current[i+2]!=255||current[i+3]!=255)return false;
                return true;
            });
            source_file(orig);recovered=wait(measure);
        }
        std::string output,region;
        auto read_output=[&](int age){
            calldata_t args;calldata_init(&args);calldata_set_int(&args,"max_age_ms",age);
            bool available=false;
            bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_ocr_output_v1",&args);
            calldata_get_bool(&args,"available",&available);
            const char* text=calldata_string(&args,"text"),*name=calldata_string(&args,"region");
            output=text?text:"";region=name?name:"";calldata_free(&args);
            return called&&available;
        };
        bool private_default=!read_output(2000)&&output.empty();
        obs_source_t* destination=nullptr;
        if(const char* module=std::getenv("SIGHT_SMOKE_OCR_TEXT_MODULE")){
            const char* data=std::getenv("SIGHT_SMOKE_OCR_TEXT_DATA");
            obs_module_t* mod=nullptr;
            if(!data||obs_open_module(&mod,module,data)!=MODULE_SUCCESS||!obs_init_module(mod))return 1;
            auto* settings=obs_data_create();obs_data_set_string(settings,"text","Initial");
            destination=obs_source_create("text_ft2_source","OCR routing fixture",settings,nullptr);obs_data_release(settings);
            if(!destination)return 1;
        }
        auto destination_text=[&](){
            if(!destination)return std::string{};
            auto* settings=obs_source_get_settings(destination);
            std::string text=obs_data_get_string(settings,"text");obs_data_release(settings);return text;
        };
        auto configure_output=[&](int view,bool enabled,int y,int h){
            auto* cfg=obs_data_create();
            obs_data_set_int(cfg,"ocr_view",view);obs_data_set_bool(cfg,"ocr_export",enabled);
            obs_data_set_int(cfg,"ocr_x",0);obs_data_set_int(cfg,"ocr_y",y);
            obs_data_set_int(cfg,"ocr_w",100);obs_data_set_int(cfg,"ocr_h",h);
            obs_data_set_int(cfg,"ocr_stable",2);obs_data_set_bool(cfg,"ocr_change_only",true);
            obs_data_set_string(cfg,"ocr_region_name","Score");
            obs_data_set_string(cfg,"ocr_format","{region}: {text}");
            if(destination)obs_data_set_string(cfg,"ocr_target_uuid",obs_source_get_uuid(destination));
            obs_source_update(filter,cfg);obs_data_release(cfg);
        };
        configure_output(3,true,50,50);
        bool not_immediate=!read_output(2000);
        bool cropped=wait([&](){
            return current==baseline&&read_output(2000)&&output=="CODE 12345"&&region=="Score"&&
                read_text(2000)&&snapshot.count==1&&snapshot.lines[0].y0>.45f&&
                (!destination||destination_text()=="Score: CODE 12345");
        });
        auto stats=[&](){
            calldata_t args;calldata_init(&args);
            long long runs=0,hits=0;
            proc_handler_call(obs_source_get_proc_handler(filter),"sight_ocr_stats_v1",&args);
            calldata_get_int(&args,"recognitions",&runs);calldata_get_int(&args,"cache_hits",&hits);
            calldata_free(&args);return std::pair{runs,hits};
        };
        auto before=stats();
        bool reused=wait([&](){auto now=stats();return now.second>=before.second+2&&now.first==before.first&&read_output(2000)&&output=="CODE 12345";});
        // Real router target receives normalized OCR numeric input.
        auto* ns=obs_data_create();obs_data_set_double(ns,"gain",10);
        auto* numeric=obs_source_create("sight_smoke_controlled_filter","OCR numeric target",ns,nullptr);obs_data_release(ns);
        if(!numeric)return 1;
        obs_source_filter_add(source,numeric);
        auto* rs=obs_data_create();obs_data_set_int(rs,"input_mode",2);
        obs_data_set_string(rs,"ocr_input_filter",obs_source_get_name(filter));
        obs_data_set_double(rs,"ocr_input_min",0);obs_data_set_double(rs,"ocr_input_max",20000);
        obs_data_set_string(rs,"target_filter","OCR numeric target");obs_data_set_string(rs,"target_setting","gain");
        obs_data_set_double(rs,"output_min",0);obs_data_set_double(rs,"output_max",100);
        auto* router=obs_source_create("sight_value_router_filter","OCR value router",rs,nullptr);obs_data_release(rs);
        if(!router)return 1;
        obs_source_filter_add(source,router);
        obs_source_inc_showing(source);obs_set_output_source(0,source);
        auto gain=[&](){auto* cfg=obs_source_get_settings(numeric);double v=obs_data_get_double(cfg,"gain");obs_data_release(cfg);return v;};
        bool numeric_routed=wait([&](){return std::abs(gain()-61.725)<.01;});
        auto read_slot=[&](int slot,std::string& value,std::string& name,long long& seq){
            calldata_t args;calldata_init(&args);
            calldata_set_int(&args,"max_age_ms",2000);calldata_set_int(&args,"slot",slot);
            bool available=false;
            bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_ocr_region_v1",&args);
            if(called){
                calldata_get_bool(&args,"available",&available);
                auto* text=calldata_string(&args,"text");auto* region=calldata_string(&args,"region");
                value=text?text:"";name=region?region:"";calldata_get_int(&args,"sequence",&seq);
            }
            calldata_free(&args);return called&&available;
        };
        obs_source_t* second_destination=nullptr;
        if(destination){
            auto* cfg=obs_data_create();obs_data_set_string(cfg,"text","Initial second");
            second_destination=obs_source_create("text_ft2_source","OCR second routing fixture",cfg,nullptr);obs_data_release(cfg);
            if(!second_destination)return 1;
        }
        auto second_text=[&](){
            if(!second_destination)return std::string{};
            auto* cfg=obs_source_get_settings(second_destination);
            std::string text=obs_data_get_string(cfg,"text");obs_data_release(cfg);return text;
        };
        auto* extra=obs_data_create();
        if(second_destination)obs_data_set_string(extra,"ocr_r2_target_uuid",obs_source_get_uuid(second_destination));
        obs_data_set_string(extra,"ocr_r2_format","{region}: {text}");
        obs_data_set_bool(extra,"ocr_r2_enabled",true);
        obs_data_set_string(extra,"ocr_r2_name","Title");
        obs_data_set_int(extra,"ocr_r2_x",0);obs_data_set_int(extra,"ocr_r2_y",0);
        obs_data_set_int(extra,"ocr_r2_w",100);obs_data_set_int(extra,"ocr_r2_h",50);
        obs_data_set_int(extra,"ocr_r2_layout",6);
        obs_data_set_int(extra,"ocr_r2_stable",2);
        obs_data_set_bool(extra,"ocr_r2_export",true);
        obs_data_set_bool(extra,"ocr_r2_change_only",true);
        obs_source_update(filter,extra);obs_data_release(extra);
        std::string v1,v2,n1,n2;long long seq1=0,seq2=0;
        bool multi=wait([&](){
            return current==baseline&&read_slot(1,v1,n1,seq1)&&read_slot(2,v2,n2,seq2)&&
                v1=="CODE 12345"&&v2=="SIGHT TEST"&&n1=="Score"&&n2=="Title"&&
                seq1==seq2&&read_text(2000)&&snapshot.count==2&&
                (!second_destination||(second_text()=="Title: SIGHT TEST"&&destination_text()=="Score: CODE 12345"));
        });
        // Both regions must retain independent caches over identical frames.
        auto multi_stats=stats();
        bool multi_reuse=wait([&](){
            auto now=stats();return now.second>=multi_stats.second+4&&now.first==multi_stats.first;
        });
        bool duplicate_safe=true;
        if(second_destination){
            extra=obs_data_create();
            obs_data_set_string(extra,"ocr_r2_target_uuid",obs_source_get_uuid(destination));
            obs_source_update(filter,extra);obs_data_release(extra);
            auto* manual=obs_data_create();obs_data_set_string(manual,"text","manual");
            obs_source_update(destination,manual);obs_data_release(manual);
            bool active=wait([&](){return read_slot(1,v1,n1,seq1)&&read_slot(2,v2,n2,seq2)&&destination_text()=="manual";});
            for(int i=0;i<12;++i){
                duplicate_safe=duplicate_safe&&render_image(source,current)&&destination_text()=="manual";
                std::this_thread::sleep_for(std::chrono::milliseconds(45));
            }
            duplicate_safe=duplicate_safe&&active;
            extra=obs_data_create();obs_data_set_string(extra,"ocr_r2_target_uuid",obs_source_get_uuid(second_destination));
            obs_source_update(filter,extra);obs_data_release(extra);
            duplicate_safe=duplicate_safe&&wait([&](){return destination_text()=="Score: CODE 12345"&&second_text()=="Title: SIGHT TEST";});
        }
        bool invalid_slot=!read_slot(0,v1,n1,seq1)&&!read_slot(5,v1,n1,seq1)&&!read_slot(3,v1,n1,seq1);
        auto* slot=obs_data_create();obs_data_set_int(slot,"ocr_input_slot",2);
        obs_data_set_int(slot,"ocr_input_kind",2);obs_data_set_string(slot,"ocr_input_query","SIGHT TEST");
        obs_source_update(router,slot);obs_data_release(slot);
        bool slot_routed=wait([&](){return std::abs(gain()-100)<.01;});
        extra=obs_data_create();obs_data_set_bool(extra,"ocr_r2_export",false);
        obs_source_update(filter,extra);obs_data_release(extra);
        bool slot_private=wait([&](){return !read_slot(2,v2,n2,seq2)&&read_slot(1,v1,n1,seq1)&&
            v1=="CODE 12345"&&std::abs(gain()-10)<.01;});
        configure_output(0,false,50,50);
        bool multi_mask=wait([&](){return measure()&&read_text(2000)&&snapshot.count==2&&!read_slot(2,v2,n2,seq2);});
        configure_output(3,true,50,50);
        extra=obs_data_create();obs_data_set_bool(extra,"ocr_r2_enabled",false);
        obs_source_update(filter,extra);obs_data_release(extra);
        bool removed_slot=wait([&](){return !read_slot(2,v2,n2,seq2)&&read_text(2000)&&snapshot.count==1;});
        slot=obs_data_create();obs_data_set_int(slot,"ocr_input_slot",1);
        obs_source_update(router,slot);obs_data_release(slot);
        if(second_destination){obs_source_release(second_destination);smoke_drain_ui();}
        bool multi_ok=multi&&multi_reuse&&duplicate_safe&&multi_mask&&invalid_slot&&slot_routed&&slot_private&&removed_slot;
        std::cout<<"SIGHT_OCR_MULTI_REAL multi="<<multi<<" invalid="<<invalid_slot
            <<" routed="<<slot_routed<<" private="<<slot_private<<" removed="<<removed_slot
            <<" reuse="<<multi_reuse<<" duplicate_safe="<<duplicate_safe<<" mask="<<multi_mask
            <<" PASS="<<multi_ok<<std::endl;

        auto match=[&](int kind,const char* query){
            auto* cfg=obs_data_create();obs_data_set_int(cfg,"ocr_input_kind",kind);
            obs_data_set_string(cfg,"ocr_input_query",query);obs_source_update(router,cfg);obs_data_release(cfg);
        };
        match(1,"CODE");bool match_routed=wait([&](){return std::abs(gain()-100)<.01;});
        match(2,"CODE");bool no_match=wait([&](){return std::abs(gain())<.01;});
        configure_output(0,true,50,50);
        bool concealment_private=wait([&](){return !read_output(2000)&&output.empty()&&current!=baseline;});
        bool router_released=wait([&](){return std::abs(gain()-10)<.01;});
        obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
        smoke_drain_ui();obs_source_filter_remove(source,router);obs_source_release(router);
        smoke_drain_ui();obs_source_filter_remove(source,numeric);obs_source_release(numeric);
        configure_output(3,true,0,50);
        bool upper=wait([&](){return read_output(2000)&&output=="SIGHT TEST"&&
            (!destination||destination_text()=="Score: SIGHT TEST");});
        source_file(blank);
        bool empty_output=wait([&](){return read_output(2000)&&output.empty()&&
            (!destination||destination_text().empty());});
        source_file(orig);
        bool returned=wait([&](){return read_output(2000)&&output=="SIGHT TEST";});
        obs_source_set_enabled(filter,false);
        bool disabled_output=!read_output(2000)&&output.empty();
        obs_source_set_enabled(filter,true);
        configure_output(3,false,0,100);
        bool opt_out=wait([&](){return !read_output(2000)&&output.empty()&&current==baseline;});
        configure_output(4,false,0,100);
        auto* prep=obs_data_create();obs_data_set_int(prep,"ocr_preprocessing",1);
        obs_source_update(filter,prep);obs_data_release(prep);
        bool preview=wait([&](){
            size_t correct=0;
            for(size_t i=0;i<current.size();i+=4)
                correct+=std::abs(int(current[i])-(255-int(baseline[i])))<=1&&
                    current[i]==current[i+1]&&current[i]==current[i+2];
            return correct>W*H*99/100&&!read_output(2000);
        });
        prep=obs_data_create();obs_data_set_int(prep,"ocr_preprocessing",0);
        obs_source_update(filter,prep);obs_data_release(prep);
        configure_output(0,false,0,100);
        bool original_mask=wait(measure);
        bool editor_ok=true;
#ifdef SIGHT_HAVE_OCR_EDITOR
        if(qt_app){
            auto* props=obs_source_properties(filter);
            auto* button=obs_properties_get(props,"ocr_visual_editor");
            bool button_ok=button!=nullptr;
            bool snapshot_seen=false,pixels_seen=false,drag_ok=false,second_ok=false,cancel_ok=false,applied=false;
            auto interact=[&](bool accept){
                QTimer timer;timer.setInterval(75);
                int attempts=0;bool interacted=false;
                QObject::connect(&timer,&QTimer::timeout,[&](){
                    smoke_drain_ui();
                    auto* dialog=qobject_cast<QDialog*>(QApplication::activeModalWidget());
                    if(!dialog||dialog->objectName()!="sightOcrRegionEditor")return;
                    auto* status=dialog->findChild<QLabel*>("sightOcrStatus");
                    if(!status||!status->text().contains("Frozen source snapshot")){
                        if(++attempts>100)dialog->reject();
                        else render_image(source,current);
                        return;
                    }
                    snapshot_seen=true;
                    auto* canvas=dialog->findChild<QWidget*>("sightOcrCanvas");
                    auto* slots_box=dialog->findChild<QComboBox*>("sightOcrSlot");
                    if(!canvas||!slots_box){dialog->reject();return;}
                    const double scale=std::min(double(canvas->width())/W,double(canvas->height())/H);
                    const double fw=W*scale,fh=H*scale,l=(canvas->width()-fw)/2,t=(canvas->height()-fh)/2;
                    // Inspect rendered canvas pixels away from labels and
                    // borders. Match known source fixture dark/bright areas.
                    auto image=canvas->grab().toImage();
                    size_t dark=0,bright=0,dark_match=0,bright_match=0;
                    const double dpr=image.devicePixelRatio();
                    for(unsigned sy=H/8;sy<H*7/8;sy+=2)for(unsigned sx=W/8;sx<W*7/8;sx+=2){
                        auto index=(size_t(sy)*W+sx)*4;
                        auto color=image.pixelColor(int((l+fw*(sx+.5)/W)*dpr),int((t+fh*(sy+.5)/H)*dpr));
                        int gray=qGray(color.rgb());
                        if(baseline[index]<16){++dark;dark_match+=gray<110;}
                        if(baseline[index]>245){++bright;bright_match+=gray>150;}
                    }
                    pixels_seen=dark>50&&bright>50&&dark_match*100>dark*60&&bright_match*100>bright*85;
                    auto drag=[&](int slot,double x,double y,double w,double h){
                        slots_box->setCurrentIndex(slot);
                        QPointF p(l+x*fw,t+y*fh),q(l+(x+w)*fw,t+(y+h)*fh);
                        QMouseEvent press(QEvent::MouseButtonPress,p,canvas->mapToGlobal(p.toPoint()),Qt::LeftButton,Qt::LeftButton,Qt::NoModifier);
                        QApplication::sendEvent(canvas,&press);
                        QMouseEvent release(QEvent::MouseButtonRelease,q,canvas->mapToGlobal(q.toPoint()),Qt::LeftButton,Qt::NoButton,Qt::NoModifier);
                        QApplication::sendEvent(canvas,&release);
                        const int expected[]={int(x*100+.5),int(y*100+.5),int(w*100+.5),int(h*100+.5)};
                        for(int i=0;i<4;++i){
                            auto* spin=dialog->findChild<QSpinBox*>(QString("sightOcrCoord%1").arg(i));
                            if(!spin||spin->value()!=expected[i])return false;
                        }
                        return true;
                    };
                    drag_ok=drag(0,0,.5,1,.5);second_ok=drag(1,0,0,1,.5);
                    interacted=true;timer.stop();
                    auto* buttons=dialog->findChild<QDialogButtonBox*>();
                    if(buttons&&accept)buttons->button(QDialogButtonBox::Ok)->click();
                    else dialog->reject();
                });
                timer.start();
                const bool changed=button&&obs_property_button_clicked(button,filter);
                timer.stop();return interacted&&changed==accept;
            };
            // Run against the real source capture in the modal loop.
            if(button_ok){
                cancel_ok=interact(false);
                auto* cfg=obs_source_get_settings(filter);
                cancel_ok=cancel_ok&&obs_data_get_int(cfg,"ocr_y")==0&&obs_data_get_int(cfg,"ocr_h")==100&&
                    !obs_data_get_bool(cfg,"ocr_r2_enabled");obs_data_release(cfg);
                applied=interact(true);
                cfg=obs_source_get_settings(filter);
                applied=applied&&obs_data_get_int(cfg,"ocr_y")==50&&obs_data_get_int(cfg,"ocr_h")==50&&
                    obs_data_get_bool(cfg,"ocr_r2_enabled")&&obs_data_get_int(cfg,"ocr_r2_y")==0&&
                    obs_data_get_int(cfg,"ocr_r2_h")==50&&!obs_data_get_bool(cfg,"ocr_r2_export");
                obs_data_release(cfg);
            }
            obs_properties_destroy(props);
            bool mask_after=wait([&](){return measure()&&read_text(2000)&&snapshot.count==2;});
            auto* restore=obs_data_create();obs_data_set_bool(restore,"ocr_r2_enabled",false);
            obs_data_set_int(restore,"ocr_y",0);obs_data_set_int(restore,"ocr_h",100);
            obs_source_update(filter,restore);obs_data_release(restore);
            bool restored_editor=wait(measure);
            editor_ok=button_ok&&snapshot_seen&&pixels_seen&&drag_ok&&second_ok&&cancel_ok&&applied&&mask_after&&restored_editor;
            std::cout<<"SIGHT_OCR_EDITOR_REAL button="<<button_ok<<" snapshot="<<snapshot_seen<<" pixels="<<pixels_seen<<" drag="<<drag_ok
                <<" second="<<second_ok<<" cancel="<<cancel_ok<<" applied="<<applied
                <<" mask="<<mask_after<<" restored="<<restored_editor<<" PASS="<<editor_ok<<std::endl;
        }
#endif
        bool processing_ok=editor_ok&&multi_ok&&reused&&numeric_routed&&match_routed&&no_match&&router_released&&preview;
        std::cout<<"SIGHT_OCR_PROCESSING_REAL reused="<<reused<<" numeric="<<numeric_routed
            <<" match="<<match_routed<<" no_match="<<no_match<<" released="<<router_released
            <<" preview="<<preview<<" PASS="<<processing_ok<<std::endl;
        if(destination){obs_source_release(destination);smoke_drain_ui();}
        bool output_ok=private_default&&not_immediate&&cropped&&concealment_private&&upper&&empty_output&&returned&&disabled_output&&opt_out&&original_mask&&processing_ok;
        std::cout<<"SIGHT_OCR_OUTPUT_REAL private="<<private_default<<" invalidated="<<not_immediate
            <<" crop="<<cropped<<" concealment_private="<<concealment_private<<" upper="<<upper
            <<" empty="<<empty_output<<" returned="<<returned<<" disabled="<<disabled_output
            <<" opt_out="<<opt_out<<" mask_restored="<<original_mask<<" routed="<<(destination!=nullptr)
            <<" PASS="<<output_ok<<std::endl;
        bool bridge_ok=true;
        if(const char* module=std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE")) {
            const char* data=std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA");
            const char* shader=std::getenv("SIGHT_SMOKE_HYPERSHADER_EFFECT");
            const char* preset=std::getenv("SIGHT_SMOKE_TEXT_PRESET");
            obs_module_t* mod=nullptr;
            if(!data||!shader||!preset||obs_open_module(&mod,module,data)!=MODULE_SUCCESS||!obs_init_module(mod))
                return 1;
            update_mode(3);
            auto make_hyper=[&](const char* path) {
                auto* opts=obs_data_create();
                obs_data_set_string(opts,"shader_file",path);
                obs_data_set_bool(opts,"sight_use_tracking",true);
                auto* hs=obs_source_create("hypershader_filter","Shared OCR probe",opts,nullptr);
                obs_data_release(opts);
                if(hs)obs_source_filter_add(source,hs);
                return hs;
            };
            auto* hyper=make_hyper(shader);
            if(!hyper)return 1;
            obs_source_inc_showing(source);obs_set_output_source(0,source);
            auto probe=[&](int count) {
                return wait([&](){
                    size_t marked=0,white=0,black=0;
                    for(size_t i=0;i<current.size();i+=4) {
                        if(current[i+1]==count&&current[i+2]==255&&current[i+3]==255) {
                            ++marked;white+=current[i]==255;black+=current[i]==0;
                        }
                    }
                    return marked>W*H*9/10&&black>W*H/2&&
                        (count==0?white==0:white>1000);
                });
            };
            bool shared=probe(2);
            bool gpu_bounds=false;
            if(shared&&read_text(2000)&&snapshot.count==2) {
                gpu_bounds=true;
                for(unsigned i=0;i<8;++i) {
                    const auto& b=snapshot.lines[i/4];
                    float expected[]={b.x0,b.y0,b.x1,b.y1};
                    size_t j=(size_t(H/50)*W+(2*i+1)*W/16)*4;
                    gpu_bounds&=std::abs(current[j]/255.f-expected[i%4])<.015f&&
                        current[j+1]==255&&current[j+2]==0;
                }
            }
            obs_source_set_enabled(filter,false);
            bool disabled=wait([&](){return current==baseline;});
            obs_source_set_enabled(filter,true);
            bool enabled=probe(2);
            obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_TOP);
            bool wrong_order=wait([&](){return current==baseline;});
            obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_BOTTOM);
            bool restored=probe(2);
            source_file(blank);
            bool empty=probe(0);
            source_file(orig);
            bool reappeared=probe(2);
            obs_source_filter_remove(source,filter);
            bool detached=wait([&](){return current==baseline;});
            obs_source_filter_add(source,filter);
            obs_source_filter_set_order(source,filter,OBS_ORDER_MOVE_TOP);
            bool reattached=probe(2);
            obs_source_filter_remove(source,hyper);obs_source_release(hyper);smoke_drain_ui();
            auto* fx=make_hyper(preset);
            if(!fx)return 1;
            bool noise=wait([&](){
                size_t changed=0,outer=0,colored=0;
                for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x) {
                    size_t i=(size_t(y)*W+x)*4;
                    if(std::memcmp(&baseline[i],&current[i],4)==0)continue;
                    ++changed;
                    colored+=current[i+2]>current[i]+5 && current[i+1]>current[i]+5;
                    if(x<45||x>660||!((y>=25&&y<=170)||(y>=195&&y<=355)))++outer;
                }
                return changed>20000&&colored>1000&&outer==0;
            });
            const auto noise_frame=current;
            bool animated=wait([&](){
                size_t changed=0;
                for(size_t i=0;i<current.size();i+=4)
                    changed+=std::memcmp(&noise_frame[i],&current[i],4)!=0;
                return changed>1000&&current!=baseline;
            });
            auto* black=obs_data_create();obs_data_set_int(black,"param_treatment",1);
            obs_source_update(fx,black);obs_data_release(black);
            bool blackout=wait(measure);
            auto* preview=obs_data_create();obs_data_set_int(preview,"param_treatment",2);
            obs_source_update(fx,preview);obs_data_release(preview);
            const bool preview_ok=wait([&](){
                size_t white=0,black=0;
                for(size_t i=0;i<current.size();i+=4){
                    white+=current[i]==255&&current[i+1]==255&&current[i+2]==255;
                    black+=current[i]==0&&current[i+1]==0&&current[i+2]==0;
                }
                return white>1000&&black>W*H/2&&white+black==W*H;
            });
            auto* black_again=obs_data_create();obs_data_set_int(black_again,"param_treatment",1);
            obs_source_update(fx,black_again);obs_data_release(black_again);
            blackout=wait(measure)&&blackout;
            obs_source_set_enabled(filter,false);
            bool absent=wait([&](){return current==baseline;});
            obs_source_set_enabled(filter,true);
            bool back=wait(measure);
            auto* broken=obs_data_create();
            obs_data_set_string(broken,"tessdata_path","/nonexistent/sight-bridge-tessdata");
            obs_source_update(filter,broken);obs_data_release(broken);
            bool model_lost=wait([&](){return current==baseline&&!read_text(2000);});
            for(int i=0;i<20;++i) {
                model_lost=render_image(source,current)&&current==baseline&&!read_text(2000)&&model_lost;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            auto* restored_model=obs_data_create();
            obs_data_set_string(restored_model,"tessdata_path",std::getenv("SIGHT_SMOKE_OCR_DATA"));
            obs_source_update(filter,restored_model);obs_data_release(restored_model);
            bool model_back=wait(measure);
            obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
            smoke_drain_ui();obs_source_filter_remove(source,fx);obs_source_release(fx);smoke_drain_ui();
            update_mode(0);
            const bool standalone_back=wait(measure);
            bridge_ok=preview_ok&&model_lost&&model_back&&standalone_back&&shared&&gpu_bounds&&disabled&&enabled&&wrong_order&&restored&&empty&&reappeared&&
                detached&&reattached&&noise&&animated&&blackout&&absent&&back;
            std::cout<<"SIGHT_TEXT_BRIDGE shared="<<shared<<" bounds="<<gpu_bounds
                <<" disabled="<<disabled<<" enabled="<<enabled<<" order="<<wrong_order
                <<" restored="<<restored<<" empty="<<empty<<" reappeared="<<reappeared
                <<" detached="<<detached<<" reattached="<<reattached
                <<" noise="<<noise<<" animated="<<animated<<" blackout="<<blackout
                <<" absent="<<absent<<" recovered="<<back
                <<" preview="<<preview_ok<<" model_lost="<<model_lost<<" model_back="<<model_back
                <<" standalone="<<standalone_back<<std::endl;
        }
        auto* bad=obs_data_create();obs_data_set_string(bad,"tessdata_path","/nonexistent/sight-test-tessdata");
        obs_source_update(filter,bad);obs_data_release(bad);
        bool unavailable=wait([&](){return current==baseline;});
        // Continue rendering so the invalid model is actually loaded and
        // the asynchronous failure cannot leave an earlier mask active.
        for(int i=0;i<60;++i) {
            const bool rendered=render_image(source,current);
            const bool same=current==baseline;
            const bool readable=read_text(2000);
            if(!rendered||!same||readable)
                std::cout<<"OCR_MISSING_TRANSITION "<<i<<" rendered="<<rendered<<" baseline="<<same<<" readable="<<readable<<std::endl;
            unavailable=rendered&&same&&!readable&&unavailable;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout<<"SIGHT_OCR_OBS redacted="<<redacted<<" mask="<<mask<<" mosaic="<<mosaic
            <<" geometry="<<geometry<<" ABI="<<abi<<" expired="<<expired
            <<" original="<<original<<" cleared="<<cleared<<" recovered="<<recovered
            <<" missing_model="<<unavailable<<" ink="<<ink_n<<" covered="<<covered_n
            <<" extra_black="<<black_n<<" outside="<<out_n<<"\n";
        obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_source_release(source);smoke_drain_ui();obs_shutdown();
        return !(redacted&&mask&&mosaic&&geometry&&abi&&expired&&original&&cleared&&recovered&&unavailable&&bridge_ok&&output_ok);
    }
    obs_source_t* producer=nullptr;
    obs_source_t* numeric=nullptr;
    if(value_router) {
        auto* numeric_settings=obs_data_create();
        obs_data_set_double(numeric_settings,"gain",10);
        numeric=obs_source_create("sight_smoke_controlled_filter",
                                  "numeric target",numeric_settings,nullptr);
        obs_data_release(numeric_settings);
        if(!numeric) {
            std::cerr<<"FAIL: numeric target missing\n";return 1;
        }
        obs_source_filter_add(source,numeric);
    }
    if(signal_receiver ||
       (value_router&&std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL"))) {
        const char* kind=std::getenv("SIGHT_SMOKE_SIGNAL_PRODUCER");
        if(!kind && value_router)kind=std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL");
        const bool pose=kind && std::strcmp(kind,"body")==0;
        const bool hand=kind && std::strcmp(kind,"hand")==0;
        const bool holistic=kind && std::strcmp(kind,"holistic")==0;
        const bool rtmw=kind && std::strcmp(kind,"rtmw")==0;
        auto* producer_settings=obs_data_create();
        obs_data_set_string(producer_settings,"model_path",argv[2]);
        if(const auto* path=std::getenv("SIGHT_SMOKE_TASK_RUNTIME"))
            obs_data_set_string(producer_settings,"mediapipe_runtime",path);
        obs_data_set_string(producer_settings,"signal_channel",
                            rtmw?"wholebody":hand?"hand":pose?"body":"face");
        if(rtmw) {
            const char* detector=std::getenv("SIGHT_SMOKE_RTMW_DETECTOR");
            if(!detector)return 1;
            obs_data_set_string(producer_settings,"rtmw_person_detector",detector);
        }
        if(holistic) {
            obs_data_set_string(producer_settings,"signal_channel_face","holistic/face");
            obs_data_set_string(producer_settings,"signal_channel_body","holistic/body");
            obs_data_set_string(producer_settings,"signal_channel_hand","holistic/hand");
        }
        obs_data_set_bool(producer_settings,"draw_overlay",false);
        obs_data_set_int(producer_settings,"inference_edge",512);
        if(std::getenv("SIGHT_SMOKE_STATIC_TRACKING"))
            obs_data_set_int(producer_settings,"inference_cadence",1);
        producer=create_tracker("Invisible Sight tracking publisher",producer_settings,
            rtmw?"rtmw":holistic?"holistic":hand?"hand":pose?"body":"face");
        obs_data_release(producer_settings);
        if(!producer) {
            std::cerr<<"FAIL: could not create independent signal publisher\n";
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
        if(holistic) {
            auto* options=obs_source_properties(producer);
            bool complete=options;
            for(const char* key:{"model_path","mediapipe_runtime",
                                 "signal_channel_face","signal_channel_body",
                                 "signal_channel_hand","inference_edge",
                                 "inference_cadence"}) {
                complete &= options && obs_properties_get(options,(std::string("holistic_")+key).c_str())!=nullptr;
            }
            // A Holistic producer exposes an optional local overlay, but
            // this independent signal producer deliberately keeps it off.
            const auto* toggle=options?
                obs_properties_get(options,"holistic_draw_overlay"):nullptr;
            const auto* selector=options?
                obs_properties_get(options,"holistic_holistic_overlay_mode"):nullptr;
            complete &= toggle && selector &&
                obs_property_get_type(const_cast<obs_property_t*>(toggle))==OBS_PROPERTY_BOOL &&
                obs_property_get_type(const_cast<obs_property_t*>(selector))==OBS_PROPERTY_LIST &&
                obs_property_list_item_count(const_cast<obs_property_t*>(selector))==4;
            auto* producer_settings_now=obs_source_get_settings(producer);
            complete &= producer_settings_now &&
                !obs_data_get_bool(producer_settings_now,"holistic_draw_overlay");
            if(producer_settings_now)obs_data_release(producer_settings_now);
            if(options)obs_properties_destroy(options);
            if(!complete) {
                std::cerr<<"FAIL: Holistic producer properties incomplete or misleading\n";
                obs_source_release(producer);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
        obs_source_filter_add(source,producer);
    }
    if(value_router) {
        gain_min=100000.0;
        gain_max=-100000.0;
        gain_changes=0;
        gate_intermediate=0;
    }
    obs_source_filter_add(source, filter);
    if(holistic_direct && std::getenv("SIGHT_SMOKE_HOLISTIC_MASK_TOGGLE")) {
        auto* changed=obs_data_create();
        obs_data_set_bool(changed,"holistic_holistic_person_mask",true);
        obs_source_update(filter,changed);
        obs_data_release(changed);
    }
    if(holistic_direct && std::getenv("SIGHT_SMOKE_HOLISTIC_LIVE_TOGGLE")) {
        // Toggle and change the modality without reloading MediaPipe.
        auto* changed=obs_data_create();
        obs_data_set_bool(changed,"holistic_draw_overlay",true);
        obs_data_set_int(changed,"holistic_holistic_overlay_mode",3);
        obs_source_update(filter,changed);
        obs_data_release(changed);
    }
    if(value_router)
        obs_source_inc_showing(source);
    size_t diff = 0;
    const bool tracking=tracking_direct;
    const size_t required_diff =
        (tracking || signal_receiver || value_router || privacy_v2) ?
        250 : baseline.size() / 12;
    for (int i = 0; i < 70; ++i) {
        if (!render_image(source, current)) {
            std::cerr << "Filtered frame readback failed\n";
            break;
        }
        diff = 0;
        for (size_t j = 0; j < baseline.size(); ++j)
            diff += baseline[j] != current[j];
        if((value_router &&
            ((std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL") &&
              (tracked_gate?routed_gain==90:(routed_gain>15 && routed_gain<90))) ||
             (!std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL") &&
              gain_changes.load()>=5 &&
              gain_max.load()-gain_min.load()>20))) ||
           (!value_router && ((expect_passthrough && i==3) ||
                             (!expect_passthrough && diff > required_diff))))
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(70));
    }
    if (std::getenv("SIGHT_SMOKE_CAPTURE_GATE")) {
        calldata_t stats{};
        const bool called = proc_handler_call(obs_source_get_proc_handler(filter),
                                              "sight_sketch_capture_stats_v1", &stats);
        const int64_t captures = calldata_int(&stats, "captures");
        const int64_t completed = calldata_int(&stats, "completed");
        const int64_t skipped = calldata_int(&stats, "skipped");
        calldata_free(&stats);
        std::cout << "SKETCH_CAPTURE_GATE captures=" << captures
                  << " completed=" << completed << " skipped=" << skipped << "\n";
        if (!called || captures < 1 || completed < 1 || skipped < 1 ||
            captures > completed + 1) {
            std::cerr << "FAIL: Portrait Sketch redundantly captured frames while inference was busy\n";
            obs_source_filter_remove(source,filter);
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
    }
    if(std::getenv("SIGHT_SMOKE_CUDA_TEXTURE")){
        int64_t captures=0,completed=0,uploads=0,registrations=0,failures=0,readbacks=0,max_handoff=0,loads=0;
        auto stats=[&]{
            calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_sketch_gpu_stats_v1",&d);
            captures=calldata_int(&d,"captures");completed=calldata_int(&d,"completed");uploads=calldata_int(&d,"uploads");
            registrations=calldata_int(&d,"registrations");failures=calldata_int(&d,"failures");readbacks=calldata_int(&d,"readbacks");
            max_handoff=calldata_int(&d,"max_handoff_us");loads=calldata_int(&d,"loads");calldata_free(&d);return ok;
        };
        auto advance=[&](int64_t n){
            auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
            while(std::chrono::steady_clock::now()<end){
                if(!stats()||failures)return false;
                if(uploads>=n)return true;
                if(!render_image(source,current))return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            return false;
        };
        bool warm=advance(3);auto regs=registrations;
        bool stable=warm&&advance(6)&&regs==registrations&&registrations==2&&readbacks==0&&captures<=completed+1;
        std::cout<<"SIGHT_SKETCH_CUDA_RESIDENT warm="<<warm<<" uploads="<<uploads<<" registrations="<<registrations
            <<" readbacks="<<readbacks<<" failures="<<failures<<" max_handoff_us="<<max_handoff<<" pass="<<stable<<"\n";
        if(!stable){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        auto* change=obs_source_get_settings(filter);obs_data_set_int(change,"detail_size",512);obs_source_update(filter,change);obs_data_release(change);
        auto target=uploads+3;
        bool resized=advance(target)&&readbacks==0&&registrations==4&&loads==1;
        std::cout<<"SIGHT_SKETCH_CUDA_RESIZE uploads="<<uploads<<" registrations="<<registrations<<" loads="<<loads<<" pass="<<resized<<"\n";
        if(!resized){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        if(std::getenv("SIGHT_SMOKE_CUDA_TEXTURE_SWITCH")){
            auto all_completed=[&]{
                calldata_t d{};proc_handler_call(obs_source_get_proc_handler(filter),"sight_sketch_capture_stats_v1",&d);
                auto n=calldata_int(&d,"completed");calldata_free(&d);return n;
            };
            auto toggle=[&](bool enabled){
                auto* d=obs_source_get_settings(filter);obs_data_set_bool(d,"cuda_texture",enabled);obs_source_update(filter,d);obs_data_release(d);
            };
            if(!render_image(source,current))return 1;
            const auto gpu_frame=current;const auto prior=all_completed();
            toggle(false);
            bool staged=false;
            auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
            while(std::chrono::steady_clock::now()<end){
                if(!render_image(source,current)||!stats())break;
                if(all_completed()>=prior+2&&readbacks>0){
                    staged=render_image(source,current)&&current.size()==gpu_frame.size();break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            double mae=0;unsigned max_error=0;
            if(staged){
                for(size_t i=0;i<current.size();++i)if(i%4!=3){
                    auto e=unsigned(std::abs(int(current[i])-int(gpu_frame[i])));
                    mae+=e;max_error=std::max(max_error,e);
                }
                mae/=current.size()*0.75;
            }
            toggle(true);
            const auto next=uploads+3;
            // One staged result may have been in flight while switching; no
            // new staged reads are permitted after GPU publication resumes.
            bool returned=advance(next);const auto staged_reads=readbacks;
            returned=returned&&advance(next+2)&&readbacks==staged_reads&&loads==2&&registrations==6;
            bool ok=staged&&mae<1.0&&max_error<=5&&returned;
            std::cout<<"SIGHT_SKETCH_CUDA_SWITCH staged="<<staged<<" mae="<<mae<<" max_error="<<max_error
                <<" returned="<<returned<<" loads="<<loads<<" readbacks="<<readbacks<<" pass="<<ok<<"\n";
            if(!ok){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        }
    }
    if (std::getenv("SIGHT_SMOKE_SKETCH_REUSE")) {
        auto stats=[&](int64_t& completed,int64_t& growths){
            calldata_t d{};
            bool ok=proc_handler_call(obs_source_get_proc_handler(filter),
                                      "sight_sketch_buffer_stats_v1",&d);
            completed=calldata_int(&d,"completed");
            growths=calldata_int(&d,"growths");
            calldata_free(&d);
            return ok;
        };
        int64_t completed=0,growths=0;
        auto advance=[&](int64_t target){
            const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(15);
            while(std::chrono::steady_clock::now()<end){
                if(!stats(completed,growths))return false;
                if(completed>=target)return true;
                if(!render_image(source,current))return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            return false;
        };
        const bool warm=advance(3);
        const auto initial_growths=growths,initial_completed=completed;
        const bool continued=warm&&advance(initial_completed+3);
        const bool stable=continued&&growths==initial_growths&&growths>0&&growths<=8;
        std::cout<<"SKETCH_BUFFER_REUSE warm="<<warm<<" continued="<<continued
                 <<" growths="<<initial_growths<<" final="<<growths<<" completed="<<completed
                 <<" pass="<<stable<<"\n";
        if(!stable){
            obs_source_filter_remove(source,filter);
            obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(signal_receiver&&std::getenv("SIGHT_SMOKE_CONSUMER_DISCOVERY")){
        auto contains=[](obs_property_t* p,const char* key){
            if(!p||obs_property_get_type(p)!=OBS_PROPERTY_LIST||
                obs_property_list_type(p)!=OBS_COMBO_TYPE_LIST)return false;
            for(size_t i=0;i<obs_property_list_item_count(p);++i)
                if(std::string(obs_property_list_item_string(p,i))==key)return true;
            return false;
        };
        auto* props=obs_source_properties(filter);
        const bool receiver=contains(obs_properties_get(props,"signal_channel"),"face");
        obs_properties_destroy(props);
        auto* hs=obs_data_create();
        obs_data_set_string(hs,"model_path","");
        obs_data_set_string(hs,"signal_channel_hand","camera/hands");
        auto* holistic=create_tracker("Discovery holistic",hs,"holistic");
        obs_data_release(hs);
        if(!holistic)return 1;
        obs_source_set_enabled(holistic,false);
        obs_source_filter_add(source,holistic);
        props=obs_source_properties(filter);
        const bool hands=contains(obs_properties_get(props,"signal_channel"),"camera/hands");
        obs_properties_destroy(props);
        auto* inventory=obs_data_create();
        calldata_t call;calldata_init(&call);
        calldata_set_ptr(&call,"inventory",inventory);
        const bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_tracking_list_v1",&call);
        auto* rows=obs_data_get_array(inventory,"channels");
        bool face=false,hand=false;
        for(size_t i=0;rows&&i<obs_data_array_count(rows);++i){
            auto* row=obs_data_array_item(rows,i);
            const std::string name=obs_data_get_string(row,"channel");
            face|=name=="face"&&obs_data_get_int(row,"kind")==1;
            hand|=name=="camera/hands"&&obs_data_get_int(row,"kind")==3&&!obs_data_get_bool(row,"enabled");
            obs_data_release(row);
        }
        if(rows)obs_data_array_release(rows);
        calldata_free(&call);obs_data_release(inventory);
        obs_source_filter_remove(source,holistic);obs_source_release(holistic);
        props=obs_source_properties(filter);
        const bool removed=!contains(obs_properties_get(props,"signal_channel"),"camera/hands");
        obs_properties_destroy(props);
        std::cout<<"SIGHT_CONSUMER_DISCOVERY receiver="<<receiver<<" hands="<<hands
                 <<" bridge="<<(called&&face&&hand)<<" removed="<<removed<<std::endl;
        if(!(receiver&&hands&&called&&face&&hand&&removed))return 1;
    }
    if(value_router&&std::getenv("SIGHT_SMOKE_ROUTER_DISCOVERY")){
        auto has_string=[](obs_property_t* p,const char* value){
            if(!p||obs_property_get_type(p)!=OBS_PROPERTY_LIST)return false;
            for(size_t i=0;i<obs_property_list_item_count(p);++i){
                const char* item=obs_property_list_item_string(p,i);
                if(item&&std::string(item)==value)return true;
            }
            return false;
        };
        auto has_int=[](obs_property_t* p,int value){
            if(!p||obs_property_get_type(p)!=OBS_PROPERTY_LIST)return false;
            for(size_t i=0;i<obs_property_list_item_count(p);++i)
                if(obs_property_list_item_int(p,i)==value)return true;
            return false;
        };
        auto* props=obs_source_properties(filter);
        auto* settings=obs_source_get_settings(filter);
        auto* channel=obs_properties_get(props,"signal_channel");
        auto* feature=obs_properties_get(props,"signal_feature");
        const bool id_backed=obs_property_list_type(channel)==OBS_COMBO_TYPE_LIST&&
            obs_property_list_type(obs_properties_get(props,"target_source"))==OBS_COMBO_TYPE_LIST&&
            obs_property_list_type(obs_properties_get(props,"target_filter"))==OBS_COMBO_TYPE_LIST&&
            obs_property_list_type(obs_properties_get(props,"target_setting"))==OBS_COMBO_TYPE_LIST;
        const bool live=has_string(channel,"face")&&has_int(feature,2)&&!has_int(feature,9);
        obs_data_set_int(settings,"signal_feature",2);
        obs_property_modified(feature,settings);
        auto* index=obs_properties_get(props,"signal_index");
        bool names=false;
        if(index&&obs_property_get_type(index)==OBS_PROPERTY_LIST)
            for(size_t i=0;i<obs_property_list_item_count(index);++i){
                const char* name=obs_property_list_item_name(index,i);
                if(name&&std::string(name).find("mouthSmileLeft")!=std::string::npos)names=true;
            }
        obs_data_set_int(settings,"signal_feature",0);
        obs_property_modified(feature,settings);
        const bool outputs=has_string(obs_properties_get(props,"target_source"),obs_source_get_name(source))&&
            has_string(obs_properties_get(props,"target_filter"),"numeric target")&&
            has_string(obs_properties_get(props,"target_setting"),"gain");
        obs_data_release(settings);obs_properties_destroy(props);
        auto* idle_settings=obs_data_create();
        obs_data_set_string(idle_settings,"model_path","");
        obs_data_set_string(idle_settings,"signal_channel_hand","camera/holistic-hand");
        auto* idle_producer=create_tracker("Idle holistic",idle_settings,"holistic");
        obs_data_release(idle_settings);
        if(!idle_producer)return 1;
        obs_source_filter_add(source,idle_producer);
        props=obs_source_properties(filter);
        settings=obs_source_get_settings(filter);
        const auto before=std::string(obs_data_get_json(settings));
        auto* refresh=obs_properties_get(props,"refresh_io");
        const bool refreshed=refresh&&obs_property_button_clicked(refresh,filter)&&
            before==obs_data_get_json(settings)&&
            has_string(obs_properties_get(props,"signal_channel"),"camera/holistic-hand");
        // Simulate a deliberate channel selection in the property callback.
        auto* choice=obs_data_create_from_json(before.c_str());
        obs_data_set_string(choice,"signal_channel","camera/holistic-hand");
        obs_data_set_int(choice,"signal_feature",5);
        obs_property_modified(obs_properties_get(props,"signal_channel"),choice);
        const auto* landmark_names=obs_properties_get(props,"signal_index");
        const auto* index_tip=landmark_names?obs_property_list_item_name(const_cast<obs_property_t*>(landmark_names),8):nullptr;
        const bool hand_caps=has_int(obs_properties_get(props,"signal_feature"),7)&&
            !has_int(obs_properties_get(props,"signal_feature"),8)&&
            !has_int(obs_properties_get(props,"signal_feature"),0)&&
            index_tip&&std::string(index_tip).find("Index tip")!=std::string::npos;
        obs_data_release(choice);obs_data_release(settings);obs_properties_destroy(props);
        obs_source_filter_remove(source,idle_producer);obs_source_release(idle_producer);
        // Rename and disable the producer. Discovery should expose the configured
        // new channel before any frames are rendered, not the old live name.
        auto* changed=obs_data_create();
        obs_data_set_string(changed,"face_signal_channel","camera/renamed");
        obs_source_update(producer,changed);obs_data_release(changed);
        obs_source_set_enabled(producer,false);
        props=obs_source_properties(filter);
        channel=obs_properties_get(props,"signal_channel");
        const bool idle=has_string(channel,"camera/renamed")&&has_string(channel,"face");
        obs_properties_destroy(props);
        obs_source_filter_remove(source,producer);obs_source_release(producer);producer=nullptr;
        props=obs_source_properties(filter);
        const bool removed=!has_string(obs_properties_get(props,"signal_channel"),"camera/renamed");
        obs_properties_destroy(props);
        std::cout<<"SIGHT_ROUTER_DISCOVERY live="<<live<<" names="<<names<<" outputs="<<outputs
                 <<" idle_saved="<<idle<<" removed="<<removed<<" refresh="<<refreshed<<" hand_caps="<<hand_caps<<" id_backed="<<id_backed<<std::endl;
        obs_source_dec_showing(source);obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_source_filter_remove(source,numeric);obs_source_release(numeric);
        obs_source_release(source);smoke_drain_ui();obs_shutdown();
        return !(live&&names&&outputs&&idle&&removed&&refreshed&&hand_caps&&id_backed);
    }
    if(value_router&&std::getenv("SIGHT_SMOKE_ROUTER_RESTORE")){
        auto read=[&](obs_source_t* target){
            auto* data=obs_source_get_settings(target);
            double v=obs_data_get_double(data,"gain");obs_data_release(data);return v;
        };
        auto set=[&](obs_source_t* target,double v){
            auto* data=obs_data_create();obs_data_set_double(data,"gain",v);obs_source_update(target,data);obs_data_release(data);
        };
        auto wait=[&](auto pred){
            for(int i=0;i<60;++i){if(!render_image(source,current))return false;if(pred())return true;std::this_thread::sleep_for(std::chrono::milliseconds(35));}return false;
        };
        auto close=[](double a,double b){return std::abs(a-b)<.001;};
        auto constant=[&](const char* target){
            auto* data=obs_data_create();
            obs_data_set_string(data,"target_filter",target);
            obs_data_set_double(data,"wave_amplitude",0.);
            obs_data_set_double(data,"output_min",20.);
            obs_data_set_double(data,"output_max",80.);
            obs_source_update(filter,data);obs_data_release(data);
        };
        auto* props=obs_source_properties(filter);
        auto* settings=obs_source_get_settings(filter);
        auto* picker=obs_properties_get(props,"target_setting");
        bool picker_ok=false;
        for(size_t i=0;picker&&i<obs_property_list_item_count(picker);++i)
            picker_ok|=std::string(obs_property_list_item_string(picker,i))=="gain";
        auto* range=obs_properties_get(props,"use_target_range");
        bool range_ok=range&&obs_property_button_clicked(range,filter);
        range_ok=range_ok&&obs_data_get_double(settings,"output_min")==0&&obs_data_get_double(settings,"output_max")==100;
        obs_data_release(settings);obs_properties_destroy(props);
        std::cout<<"SIGHT_ROUTER_EASY picker="<<picker_ok<<" range="<<range_ok<<std::endl;
        constant("numeric target");
        bool active=wait([&]{return close(read(numeric),50);});
        obs_source_set_enabled(filter,false);
        bool disabled=wait([&]{return close(read(numeric),10);});
        set(numeric,27.);
        obs_source_set_enabled(filter,true);
        bool resumed=wait([&]{return close(read(numeric),50);});
        set(numeric,43.);
        bool manual=wait([&]{return close(read(numeric),50);});
        obs_source_set_enabled(filter,false);
        bool edited=wait([&]{return close(read(numeric),43);});
        auto* ns=obs_data_create();obs_data_set_double(ns,"gain",31.);
        auto* target2=obs_source_create("sight_smoke_controlled_filter","other numeric target",ns,nullptr);
        obs_data_release(ns);
        obs_source_filter_add(source,target2);
        obs_source_set_enabled(filter,true);wait([&]{return close(read(numeric),50);});
        constant("other numeric target");
        bool retarget=wait([&]{return close(read(numeric),43)&&close(read(target2),50);});
        // Serialize the actual source and parse the resulting JSON before
        // any queued modulation task may resume.
        auto saved_gain=[&](obs_source_t* root,const char* name){
            auto* saved=obs_save_source(root);
            auto* json=obs_data_create_from_json(obs_data_get_json(saved));
            auto* array=obs_data_get_array(json,"filters");
            double gain=-999;
            for(size_t i=0;i<obs_data_array_count(array);++i){
                auto* item=obs_data_array_item(array,i);
                if(std::string(obs_data_get_string(item,"name"))==name){
                    auto* data=obs_data_get_obj(item,"settings");gain=obs_data_get_double(data,"gain");obs_data_release(data);
                }
                obs_data_release(item);
            }
            obs_data_array_release(array);obs_data_release(json);obs_data_release(saved);return gain;
        };
        bool saved=close(saved_gain(source,"other numeric target"),31);
        bool save_resume=wait([&]{return close(read(target2),50);});
        obs_source_filter_set_order(source,filter,OBS_ORDER_MOVE_TOP);
        bool save_reorder=close(saved_gain(source,"other numeric target"),31);
        save_resume=save_resume&&wait([&]{return close(read(target2),50);});
        // Another router on the same property may not fight the owner.
        auto* other_settings=obs_data_create();
        obs_data_set_string(other_settings,"target_filter","other numeric target");
        obs_data_set_string(other_settings,"target_setting","gain");
        obs_data_set_double(other_settings,"wave_amplitude",0.);
        obs_data_set_double(other_settings,"output_min",60.);
        obs_data_set_double(other_settings,"output_max",80.);
        auto* other=obs_source_create("sight_value_router_filter","Contending router",other_settings,nullptr);
        obs_data_release(other_settings);
        obs_source_filter_add(source,other);
        bool exclusive=true;
        for(int i=0;i<20;++i){
            exclusive=render_image(source,current)&&close(read(target2),50)&&exclusive;
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        obs_source_filter_remove(source,filter);
        bool handoff=wait([&]{return close(read(target2),70);});
        obs_source_filter_remove(source,other);obs_source_release(other);
        bool removed=wait([&]{return close(read(target2),31);});
        std::cout<<"SIGHT_ROUTER_SAVE saved="<<saved<<" reorder="<<save_reorder
                 <<" resumed="<<save_resume<<" exclusive="<<exclusive<<" handoff="<<handoff<<std::endl;
        std::cout<<"SIGHT_ROUTER_RESTORE active="<<active<<" disabled="<<disabled<<" resumed="<<resumed
                 <<" manual="<<manual<<" edited="<<edited<<" retarget="<<retarget<<" removed="<<removed<<std::endl;
        obs_source_dec_showing(source);obs_source_release(filter);
        obs_source_filter_remove(source,target2);obs_source_release(target2);
        obs_source_filter_remove(source,numeric);obs_source_release(numeric);
        obs_source_release(source);smoke_drain_ui();obs_shutdown();
        return !(picker_ok&&range_ok&&active&&disabled&&resumed&&manual&&edited&&retarget&&removed&&saved&&save_reorder&&save_resume&&exclusive&&handoff);
    }
    if(tracked_gate) {
        const bool acquired=routed_gain==90 && producer;
        if(acquired) {
            // Destroy the actual model producer, not a fabricated snapshot.
            obs_source_filter_remove(source,producer);
            obs_source_release(producer);
            producer=nullptr;
            for(int i=0;i<40 && routed_gain!=10;++i) {
                if(!render_image(source,current))break;
                std::this_thread::sleep_for(std::chrono::milliseconds(70));
            }
            gate_released_on_loss=routed_gain==10;
        }
        std::cout<<"SIGHT_GATE_TRACKING_LOSS acquired="<<acquired
                 <<" released="<<gate_released_on_loss<<"\n";
    }
    // Keep rendering after first success when requested, so performance tests
    // can verify multiple real inferences without changing the production path.
    if (const char *hold_env = std::getenv("SIGHT_SMOKE_HOLD_MS")) {
        const int hold_ms = std::atoi(hold_env);
        if (hold_ms > 0 && hold_ms <= 15000) {
            const auto until = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(hold_ms);
            while (std::chrono::steady_clock::now() < until) {
                if (!render_image(source, current)) {
                    std::cerr << "FAIL: repeated OBS render failed\n";
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
            }
        }
    }
    // Observe actual published sequences, not render counts, during an
    // unchanged camera fixture. Require a fresh heartbeat throughout.
    if(signal_receiver && std::getenv("SIGHT_SMOKE_STATIC_TRACKING")) {
        const char* channel=std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL");
        if(!channel||!*channel)channel="face";
        const std::string name(channel);
        const unsigned expected=name=="body"?33:name=="hand"?21:478;
        const unsigned stride=name.starts_with("holistic/")?3:1;
        auto read=[&](uint64_t& sequence) {
            sight_snapshot_v1 snapshot{};
            snapshot.abi_version=SIGHT_SNAPSHOT_ABI_V1;
            snapshot.struct_size=sizeof(snapshot);
            calldata_t args;calldata_init(&args);
            calldata_set_string(&args,"channel",channel);
            calldata_set_int(&args,"max_age_ms",750);
            calldata_set_ptr(&args,"snapshot",&snapshot);
            const bool called=proc_handler_call(obs_source_get_proc_handler(filter),
                "sight_tracking_read_v1",&args);
            bool available=false;
            calldata_get_bool(&args,"available",&available);
            calldata_free(&args);
            sequence=snapshot.sequence;
            return called&&available&&snapshot.point_count==expected;
        };
        uint64_t first=0,last=0;
        if(!read(first))return 1;
        const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(2400);
        unsigned samples=0;
        while(std::chrono::steady_clock::now()<until) {
            if(!render_image(source,current)||!read(last))return 1;
            ++samples;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto publications=(last-first)/stride;
        std::cout<<"SIGHT_STATIC_TRACKING channel="<<channel<<" renders="<<samples<<" publications="
                 <<publications<<" fresh=1"<<std::endl;
        if(publications<5||publications>13) {
            std::cerr<<"FAIL: static tracking must publish a bounded heartbeat, not every render\n";
            return 1;
        }
    }
    /* Live producer -> immutable bus -> exported snapshot through the actual
     * OBS source procedure. No fake publisher or second model invocation. */
    if (signal_receiver && !expect_passthrough) {
        const char* channel = std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL");
        if (!channel || !channel[0]) channel = "face";
        sight_snapshot_v1 snapshot{};
        snapshot.abi_version = SIGHT_SNAPSHOT_ABI_V1;
        snapshot.struct_size = sizeof(snapshot);
        calldata_t call;
        calldata_init(&call);
        calldata_set_string(&call, "channel", channel);
        calldata_set_int(&call, "max_age_ms", 5000);
        calldata_set_ptr(&call, "snapshot", &snapshot);
        const bool called = proc_handler_call(obs_source_get_proc_handler(filter),
                                               "sight_tracking_read_v1", &call);
        bool available = false;
        const bool returned = calldata_get_bool(&call, "available", &available);
        calldata_free(&call);
        const uint32_t kind = std::strcmp(channel,"wholebody")==0 ? SIGHT_SNAPSHOT_WHOLEBODY :
            (std::strcmp(channel,"hand")==0 ||
            std::strcmp(channel,"holistic/hand")==0) ? SIGHT_SNAPSHOT_HAND :
            (std::strcmp(channel,"body")==0 ||
             std::strcmp(channel,"holistic/body")==0) ? SIGHT_SNAPSHOT_BODY : SIGHT_SNAPSHOT_FACE;
        const uint32_t count = kind == SIGHT_SNAPSHOT_WHOLEBODY ? 133 :
                               kind == SIGHT_SNAPSHOT_FACE ? 478 :
                               kind == SIGHT_SNAPSHOT_BODY ? 33 : 21;
        const bool pass = called && returned && available &&
            snapshot.abi_version == SIGHT_SNAPSHOT_ABI_V1 &&
            snapshot.struct_size == sizeof(snapshot) &&
            snapshot.kind == kind && snapshot.point_count == count &&
            snapshot.sequence > 0 && snapshot.source_width == W &&
            snapshot.source_height == H &&
            (kind != SIGHT_SNAPSHOT_HAND || (snapshot.hands_count >= 1 &&
                                             snapshot.hands_count <= 2));
        std::cout << "SIGHT_TRACKING_PROC_REAL kind=" << snapshot.kind
                  << " count=" << snapshot.point_count
                  << " hands=" << snapshot.hands_count
                  << " available=" << available
                  << " PASS=" << pass << "\n";
        if (!pass) {
            std::cerr << "FAIL: real Sight producer not exposed by snapshot procedure\n";
            obs_source_filter_remove(source,filter);
            obs_source_release(filter);
            if (producer) {
                obs_source_filter_remove(source,producer);
                obs_source_release(producer);
            }
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
    }
    if(signal_receiver && std::getenv("SIGHT_SMOKE_HOLISTIC_TRIPLE")) {
        // The three read-only OBS API snapshots must originate from the same
        // published Holistic frame (contiguous sequence, common producer).
        bool synchronized=false;
        for(int attempt=0;attempt<18 && !synchronized;++attempt) {
            sight_snapshot_v1 results[3]{};
            bool available[3]{};
            const char* channels[3]={"holistic/face","holistic/body","holistic/hand"};
            const uint32_t kinds[3]={SIGHT_SNAPSHOT_FACE,SIGHT_SNAPSHOT_BODY,SIGHT_SNAPSHOT_HAND};
            const uint32_t counts[3]={478,33,21};
            bool valid=true;
            for(size_t index=0;index<3;++index) {
                auto& snapshot=results[index];
                snapshot.abi_version=SIGHT_SNAPSHOT_ABI_V1;
                snapshot.struct_size=sizeof(snapshot);
                calldata_t data;calldata_init(&data);
                calldata_set_string(&data,"channel",channels[index]);
                calldata_set_int(&data,"max_age_ms",5000);
                calldata_set_ptr(&data,"snapshot",&snapshot);
                const bool called=proc_handler_call(
                    obs_source_get_proc_handler(filter),"sight_tracking_read_v1",&data);
                const bool got=calldata_get_bool(&data,"available",&available[index]);
                calldata_free(&data);
                valid &= called && got && available[index] &&
                    snapshot.kind==kinds[index] &&
                    snapshot.point_count==counts[index] &&
                    snapshot.source_width==W && snapshot.source_height==H;
            }
            synchronized=valid &&
                results[1].sequence==results[0].sequence+1 &&
                results[2].sequence==results[1].sequence+1 &&
                results[2].hands_count>=1 && results[2].hands_count<=2 &&
                results[2].gesture[0][0]=='\0' &&
                results[0].has_head_transform==0;
            if(!synchronized) {
                std::this_thread::sleep_for(std::chrono::milliseconds(35));
                if(!render_image(source,current))break;
            }
        }
        std::cout<<"SIGHT_HOLISTIC_THREE_SNAPSHOT_PASS="<<(synchronized?1:0)<<"\n";
        if(!synchronized) {
            std::cerr<<"FAIL: atomic real Holistic Face/Body/Hand snapshots unavailable\n";
            obs_source_filter_remove(source,filter);
            obs_source_release(filter);
            if(producer){obs_source_filter_remove(source,producer);obs_source_release(producer);}
            obs_source_release(source);obs_shutdown();
            return 1;
        }
    }
    if(object_direct) {
        // Actual model -> worker -> persistent tracks -> readonly libobs
        // source procedure, independent of whether the overlay is shown.
        bool verified=false;
        int64_t detected=0;
        std::string json;
        for(int attempt=0;attempt<70 && !verified;++attempt) {
            calldata_t query;
            calldata_init(&query);
            const bool called=proc_handler_call(
                obs_source_get_proc_handler(filter),"sight_object_read_v1",&query);
            bool available=false;
            const bool has_status=calldata_get_bool(&query,"available",&available);
            detected=calldata_int(&query,"count");
            const auto width=calldata_int(&query,"source_width");
            const auto height=calldata_int(&query,"source_height");
            const char* payload=calldata_string(&query,"objects_json");
            json=payload?payload:"";
            verified=called&&has_status&&available&&detected>=2&&
                width==W&&height==H&&
                json.find("cat")!=std::string::npos&&
                json.find("dog")!=std::string::npos&&
                json.find("id")!=std::string::npos;
            calldata_free(&query);
            if(!verified) {
                std::this_thread::sleep_for(std::chrono::milliseconds(70));
                if(!render_image(source,current))break;
            }
        }
        std::cout<<"SIGHT_OBJECT_REAL_OBS_PROC_PASS="<<verified
                 <<" detections="<<detected<<" json="<<json<<"\n";
        if(!verified) {
            std::cerr<<"FAIL: Real native object boxes/IDs unavailable in OBS\n";
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
        if(expect_passthrough && diff>250) {
            std::cerr<<"FAIL: invisible object tracking changed source pixels\n";
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(holistic_direct && expect_passthrough && diff>250) {
        std::cerr<<"FAIL: Holistic overlay-off changed camera pixels\n";
        obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_source_release(source);obs_shutdown();return 1;
    }
    if(holistic_direct)
        std::cout<<"HOLISTIC_DIRECT_OVERLAY_DIFF="<<diff
                 <<" enabled="<<(std::getenv("SIGHT_SMOKE_HOLISTIC_OVERLAY")?1:0)
                 <<"\n";
    std::cout << "obs_render_diff=" << diff << "/" << baseline.size() << "\n";
    if(!expect_passthrough && (std::strcmp(selected_filter,"sight_depth_filter")==0 ||
        std::strcmp(selected_filter,"sight_u2net_filter")==0 ||
        std::strcmp(selected_filter,"neural_enhance_filter")==0)){
        auto* health=obs_source_properties(filter);
        auto* label=health?obs_properties_get(health,"sight_inference_health"):nullptr;
        std::string status=label?obs_property_description(label):"missing";
        std::cout<<"SIGHT_HEALTH_REAL_OBS "<<status<<"\n";
        bool ok=status.find("Recent inference")!=std::string::npos &&
                status.find("Last inference:")!=std::string::npos &&
                status.find("Successful:")!=std::string::npos;
        if(health)obs_properties_destroy(health);
        if(!ok){
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if (std::strcmp(selected_filter, "sight_depth_filter") == 0) {
        uint8_t mn = 255, mx = 0;
        uint64_t channel_difference = 0;
        for (size_t p = 0; p + 3 < current.size(); p += 4) {
            mn = std::min(mn, current[p]);
            mx = std::max(mx, current[p]);
            channel_difference += static_cast<unsigned>(
                std::abs(int(current[p]) - int(current[p+1])));
        }
        std::cout << "depth_render_min=" << +mn
                  << " max=" << +mx
                  << " channel_diff=" << channel_difference << "\n";
        if (mx < mn + 24 ||
            (!std::getenv("SIGHT_SMOKE_DEPTH_FOG") && channel_difference > W*H/4)) {
            std::cerr << "FAIL: Depth output is constant or not grayscale\n";
            obs_source_filter_remove(source, filter);
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
    }
    if (std::strcmp(selected_filter, "sight_u2net_filter") == 0 &&
        std::getenv("SIGHT_SMOKE_MATTE_CONTROLS")) {
        bool valid = true;
        std::vector<uint8_t> preview, raw, final;
        for (int mode : {2,3,1,4,0}) {
            auto *settings = obs_data_create();
            obs_data_set_int(settings,"matte_view",mode);
            obs_source_update(filter,settings);
            obs_data_release(settings);
            for (int i=0;i<4;++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(70));
                valid = render_image(source,preview) && valid;
            }
            if (!valid || preview.size()!=baseline.size()) { valid=false; break; }
            if (mode>=1 && mode<=3) {
                for (size_t i=0;i<preview.size();i+=4)
                    if (preview[i]!=preview[i+1] || preview[i]!=preview[i+2] || preview[i+3]!=255) {
                        std::cerr<<"preview mode="<<mode<<" rgba="<<int(preview[i])<<","<<int(preview[i+1])<<","<<int(preview[i+2])<<","<<int(preview[i+3])<<"\n";
                        valid=false; break;
                    }
                if (mode==2) raw=preview;
                if (mode==1) final=preview;
            } else if (mode==4) {
                size_t errors=0;
                for (size_t i=0;i<preview.size();++i)
                    errors+=std::abs(int(preview[i])-int(baseline[i]))>3;
                std::cout<<"original view errors="<<errors<<"\n";
                valid=valid && errors<preview.size()/100;
            } else {
                current=preview;
                diff=0;
                for (size_t i=0;i<current.size();++i) diff+=current[i]!=baseline[i];
            }
        }
        std::cout<<"raw bytes="<<raw.size()<<" final bytes="<<final.size()<<" different="<<(raw!=final)<<"\n";
        // A synthetic gradient may have an entirely empty foreground.
        // Require changed calibration on the actual portrait fixture, not
        // on a valid all-zero matte.
        valid=valid && raw.size()==baseline.size() && final.size()==baseline.size()
            && diff>required_diff
            && (!std::getenv("SIGHT_SMOKE_EXPECT_FOREGROUND") || raw!=final);
        std::cout<<"SIGHT_MATTE_LIVE_VIEWS="<<valid<<"\n";
        if (!valid) {
            obs_source_filter_remove(source,filter); obs_source_release(filter);
            obs_source_release(source); obs_shutdown(); return 1;
        }
    }
    if (std::strcmp(selected_filter, "sight_u2net_filter") == 0) {
        if(std::getenv("SIGHT_SMOKE_CUTOUT_REUSE")){
            int64_t captures=0,completed=0,growths=0,uploads=0,draws=0,reused=0;
            auto stats=[&]{
                calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_cutout_buffer_stats_v1",&d);
                captures=calldata_int(&d,"captures");completed=calldata_int(&d,"completed");growths=calldata_int(&d,"growths");
                uploads=calldata_int(&d,"uploads");draws=calldata_int(&d,"source_draws");reused=calldata_int(&d,"reused_draws");
                calldata_free(&d);return ok;
            };
            auto advance=[&](int64_t n){
                auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
                while(std::chrono::steady_clock::now()<end){
                    if(!stats())return false;
                    if(completed>=n)return render_image(source,current)&&stats();
                    if(!render_image(source,current))return false;
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                return false;
            };
            bool warm=advance(4);auto initial=growths,prior=completed;
            bool stable=warm&&advance(prior+4)&&growths==initial&&initial>0&&captures<=completed+1&&uploads>0&&(std::getenv("SIGHT_SMOKE_CUTOUT_REUSE_DRAW")?reused>0:reused==0);
            std::cout<<"SIGHT_CUTOUT_REUSE warm="<<warm<<" growths="<<initial<<" final="<<growths<<" completed="<<completed
                <<" captures="<<captures<<" uploads="<<uploads<<" source_draws="<<draws<<" reused_draws="<<reused<<" pass="<<stable<<"\n";
            if(!stable){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        }
        if(const char*expected_flow=std::getenv("SIGHT_SMOKE_CUTOUT_FLOW_EXPECT")){
            auto*props=obs_source_properties(filter);
            auto*status=props?obs_properties_get(props,"sight_backend_status"):nullptr;
            std::string desc=status?obs_property_description(status):"missing";
            bool ok=desc.find("Loaded — ONNX Runtime")!=std::string::npos&&desc.find(expected_flow)!=std::string::npos;
            std::cout<<"SIGHT_CUTOUT_FLOW_STATUS="<<ok<<" "<<desc<<"\n";
            if(props)obs_properties_destroy(props);
            if(!ok){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        }
        if(std::getenv("SIGHT_SMOKE_CUTOUT_SHARED")){
            auto snap=std::make_unique<sight_matte_v1>();
            snap->abi_version=SIGHT_MATTE_ABI_V1;snap->struct_size=sizeof(*snap);
            auto read=[&](uint64_t known){
                calldata_t args;calldata_init(&args);
                calldata_set_ptr(&args,"snapshot",snap.get());
                calldata_set_int(&args,"max_age_ms",5000);
                calldata_set_int(&args,"known_sequence",known);
                bool available=false;
                bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_matte_read_v1",&args);
                bool returned=calldata_get_bool(&args,"available",&available);
                calldata_free(&args);return called&&returned&&available;
            };
            bool available=false;
            for(int i=0;i<40&&!available;++i){
                available=read(0);
                if(!available)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            const unsigned expected_w=W>=H?320:std::max(1u,unsigned(uint64_t(W)*320/H));
            const unsigned expected_h=H>=W?320:std::max(1u,unsigned(uint64_t(H)*320/W));
            bool ok=available&&snap->sequence&&snap->width==expected_w&&snap->height==expected_h&&
                snap->source_width==W&&snap->source_height==H;
            if(ok){
                const size_t n=size_t(snap->width)*snap->height;
                auto [lo,hi]=std::minmax_element(snap->pixels,snap->pixels+n);
                ok=*hi>*lo+20;
            }
            if(ok&&std::getenv("SIGHT_SMOKE_CUTOUT_SNAPSHOT")){
                FILE* output=std::fopen(std::getenv("SIGHT_SMOKE_CUTOUT_SNAPSHOT"),"wb");
                const size_t bytes=size_t(snap->width)*snap->height;
                if(!output)ok=false;
                else{ok=std::fwrite(snap->pixels,1,bytes,output)==bytes&&ok;std::fclose(output);}
            }
            std::cout<<"SIGHT_CUTOUT_SHARED_ABI="<<ok<<" available="<<available<<" mask="<<snap->width<<"x"<<snap->height<<" source="<<snap->source_width<<"x"<<snap->source_height<<"\n";
            if(!ok){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
            if(const char*moved=std::getenv("SIGHT_SMOKE_CUTOUT_FLOW_MOVE")){
                auto center=[&](){
                    double total=0,weighted=0;
                    for(unsigned y=0;y<snap->height;++y)for(unsigned x=0;x<snap->width;++x){
                        double a=snap->pixels[size_t(y)*snap->width+x];
                        total+=a;weighted+=a*(x+.5)/snap->width;
                    }
                    return total>0?weighted/total:-1.;
                };
                auto flow_runs=[&](){
                    auto*props=obs_source_properties(filter);
                    auto*label=props?obs_properties_get(props,"sight_backend_status"):nullptr;
                    std::string desc=label?obs_property_description(label):"";
                    if(props)obs_properties_destroy(props);
                    auto pos=desc.find("flow runs=");
                    return pos==std::string::npos?0ull:std::strtoull(desc.c_str()+pos+10,nullptr,10);
                };
                const auto initial_runs=flow_runs();
                const auto first=center();
                const auto seq=snap->sequence;
                auto change=[&](const char*path){
                    auto*d=obs_data_create();obs_data_set_string(d,"file",path);
                    obs_source_update(source,d);obs_data_release(d);
                };
                auto pump=[&](int count){
                    for(int i=0;i<count;++i){
                        if(!render_image(source,current))return false;
                        std::this_thread::sleep_for(std::chrono::milliseconds(70));
                    }
                    return true;
                };
                auto flow_status=[&](const char*expected){
                    auto*props=obs_source_properties(filter);
                    auto*label=props?obs_properties_get(props,"sight_backend_status"):nullptr;
                    const bool valid=label&&std::string(obs_property_description(label)).find(expected)!=std::string::npos;
                    if(props)obs_properties_destroy(props);
                    return valid;
                };
                change(moved);
                bool motion_ok=pump(20)&&read(0)&&snap->sequence>seq&&center()>first+.02&&flow_runs()>initial_runs;
                const double moved_center=center();
                auto retention=[&](double val){
                    auto*d=obs_data_create();obs_data_set_double(d,"matte_retention",val);
                    obs_source_update(filter,d);obs_data_release(d);
                };
                retention(0);const auto before_off=flow_runs();
                const bool off=pump(8)&&flow_status("Flow ready")&&flow_runs()==before_off;
                retention(.6);const bool on=pump(12)&&flow_status("stationary stabilization")&&flow_runs()==before_off;
                change(std::getenv("SIGHT_SMOKE_IMAGE_FILE"));
                const bool restored=pump(20)&&read(0)&&std::abs(center()-first)<.015&&flow_runs()>before_off;
                std::cout<<"SIGHT_FLOW_MOTION="<<motion_ok<<" OFF="<<off<<" ON="<<on<<" RESTORED="<<restored
                    <<" center_before="<<first<<" center_moved="<<moved_center<<" center_restored="<<center()<<" flow_runs="<<flow_runs()<<"\n";
                if(!(motion_ok&&off&&on&&restored)){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
            }
        }
        uint8_t lo = 255, hi = 0;
        for (size_t i = 3; i < current.size(); i += 4) {
            lo = std::min(lo, current[i]);
            hi = std::max(hi, current[i]);
        }
        std::cout << "obs_output_alpha_min=" << +lo << " alpha_max=" << +hi << "\n";
        // A mask may be small, but it must NOT downsample the input RGB.
        // Check actual native-resolution colors inside near-opaque regions.
        size_t opaque = 0;
        uint64_t color_error = 0;
        for (size_t i = 0; i + 3 < current.size(); i += 4) {
            const int a = current[i + 3];
            if (a >= 248 && baseline[i + 3] == 255) {
                ++opaque;
                for (size_t c = 0; c < 3; ++c)
                    color_error += std::abs(static_cast<int>(current[i + c]) -
                                            static_cast<int>(baseline[i + c]));
            }
        }
        if (opaque == 0 && std::getenv("SIGHT_SMOKE_EXPECT_FOREGROUND")) {
            std::cerr << "FAIL: no opaque native-resolution foreground pixels\n";
            obs_source_filter_remove(source, filter);
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
        if (opaque > 0) {
            const double mean_error = static_cast<double>(color_error) / (3 * opaque);
            std::cout << "native_color_preservation_pixels=" << opaque
                      << " mean_abs_rgb_error=" << mean_error << "\n";
            if (std::getenv("SIGHT_SMOKE_EXPECT_FOREGROUND") &&
                (opaque < 1000 || mean_error > 8)) {
                std::cerr << "FAIL: foreground RGB no longer matches original native image\n";
                obs_source_filter_remove(source, filter);
                obs_source_release(filter);
                obs_source_release(source);
                obs_shutdown();
                return 1;
            }
        }
        if (std::getenv("SIGHT_SMOKE_EXPECT_FOREGROUND") && hi <= lo + 20) {
            std::cerr << "FAIL: U2Net OBS output has no detectable alpha variation\n";
            obs_source_filter_remove(source, filter);
            obs_source_release(filter);
            obs_source_release(source);
            obs_shutdown();
            return 1;
        }
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_SEMANTIC_EXPECT_SLATE")) {
        size_t slate=0;
        for(size_t i=0;i+3<current.size();i+=4)
            slate+=current[i]==24&&current[i+1]==24&&current[i+2]==24&&current[i+3]==255;
        std::cout<<"semantic_buffered_slate_pixels="<<slate<<"\n";
        if(slate!=size_t(W)*H){std::cerr<<"FAIL: unapproved frame escaped buffered mode\n";return 1;}
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_SEMANTIC_MODEL") &&
       std::getenv("SIGHT_SMOKE_PRIVACY_EXPECT_MASKED")) {
        calldata_t args;calldata_init(&args);
        bool available=false;
        const bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_semantic_read_v1",&args);
        const bool returned=calldata_get_bool(&args,"available",&available);
        const char* json=calldata_string(&args,"regions_json");
        const bool ok=called&&returned&&available&&calldata_int(&args,"count")>0&&
            calldata_int(&args,"source_width")==W&&calldata_int(&args,"source_height")==H&&
            json&&std::strstr(json,"\"category\":1")&&std::strstr(json,"\"id\":");
        std::cout<<"semantic_snapshot_valid="<<ok<<" regions="<<calldata_int(&args,"count")<<" json="<<(json?json:"")<<"\n";
        calldata_free(&args);
        if(!ok){std::cerr<<"FAIL: semantic snapshot absent or invalid\n";return 1;}
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_PRIVACY_EXPECT_MASKED")) {
        size_t visible=0,transformed=0,unchanged=0;
        if(current.size()==baseline.size())
            for(size_t pixel=0;pixel+3<current.size();pixel+=4) {
                bool same=true;
                for(size_t k=0;k<3;++k)
                    same &= std::abs(int(current[pixel+k])-int(baseline[pixel+k]))<=2;
                visible += current[pixel]>8 || current[pixel+1]>8 || current[pixel+2]>8;
                transformed += !same;
                unchanged += same;
            }
        std::cout<<"privacy_visible_pixels="<<visible<<" transformed_pixels="
                 <<transformed<<" unchanged_pixels="<<unchanged<<"\n";
        if(std::getenv("SIGHT_SMOKE_PRIVACY_EXPECT_TWO_MASKS")) {
            size_t left=0,right=0;
            for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x) {
                const size_t pixel=(size_t(y)*W+x)*4;
                bool changed=false;
                for(size_t k=0;k<3;++k)
                    changed|=std::abs(int(current[pixel+k])-int(baseline[pixel+k]))>2;
                if(x<W/2)left+=changed;else right+=changed;
            }
            std::cout<<"privacy_left_changed="<<left
                     <<" privacy_right_changed="<<right<<"\n";
            if(left<1500||right<1500) {
                std::cerr<<"FAIL: both detected bystanders were not concealed\n";
                obs_source_filter_remove(source,filter);obs_source_release(filter);
                obs_source_release(source);obs_shutdown();return 1;
            }
        }
        if(visible<10000||transformed<1000||unchanged<10000) {
            std::cerr<<"FAIL: Privacy V2 did not render a selectively concealed real image\n";
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_PRIVACY_EXPECT_MANUAL")) {
        // Real OBS filter: synthetic ROI should alter only the middle square.
        size_t inside_changed=0,outside_changed=0,inside_opaque=0;
        for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x) {
            const size_t i=(size_t(y)*W+x)*4;
            bool changed=false;
            for(unsigned k=0;k<3;++k)
                changed|=std::abs(int(current[i+k])-int(baseline[i+k]))>2;
            const bool inside=x>=W/4&&x<3*W/4&&y>=H/4&&y<3*H/4;
            (inside?inside_changed:outside_changed)+=changed;
            if(inside)
                inside_opaque+=current[i]==0&&current[i+1]==0&&current[i+2]==0;
        }
        std::cout<<"privacy_manual_inside_changed="<<inside_changed
                 <<" opaque_pixels="<<inside_opaque
                 <<" outside_changed="<<outside_changed<<"\n";
        if(inside_changed<200||inside_opaque<2000||outside_changed>500) {
            std::cerr<<"FAIL: manual text ROI not selectively concealed in OBS\n";
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_PRIVACY_REQUIRE_BLACK")) {
        bool fully_black=!current.empty();
        for(size_t pixel=0;pixel+3<current.size();pixel+=4)
            fully_black &= (current[pixel]==0 && current[pixel+1]==0 &&
                            current[pixel+2]==0 && current[pixel+3]==255);
        std::cout<<"privacy_black_fallback="<<fully_black<<"\n";
        if(!fully_black){
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_EXPECT_PASSTHROUGH")) {
        size_t preserved=0;
        for(size_t i=0;i+3<current.size()&&i+3<baseline.size();i+=4) {
            bool same=true;
            for(unsigned c=0;c<3;++c)
                same &= std::abs(int(current[i+c])-int(baseline[i+c]))<=2;
            preserved+=same;
        }
        const bool pass=current.size()==baseline.size()&&
            preserved>=size_t(W)*H*98/100;
        std::cout<<"privacy_zero_face_passthrough="<<pass
                 <<" preserved_pixels="<<preserved<<"\n";
        if(!pass) {
            std::cerr<<"FAIL: empty-camera privacy processing changed normal video\n";
            obs_source_filter_remove(source,filter);obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    if(privacy_v2 && std::getenv("SIGHT_SMOKE_PRIVACY_CHECK_FREEZE")) {
        // A new source image must reach the filter output without waiting
        // for the face CNN to approve the replacement frame.
        TemporaryImage changed_input;
        if(!changed_input.make(W,H)) {
            std::cerr<<"FAIL: could not generate replacement camera frame\n";return 1;
        }
        auto* ref_settings=obs_data_create();
        obs_data_set_string(ref_settings,"file",changed_input.path.c_str());
        auto* ref_source=obs_source_create("image_source","privacy replacement reference",
                                           ref_settings,nullptr);
        obs_data_release(ref_settings);
        std::vector<uint8_t> replacement_baseline;
        const bool ref_ok=ref_source&&render_image(ref_source,replacement_baseline);
        if(ref_source)obs_source_release(ref_source);
        if(!ref_ok||replacement_baseline.size()!=baseline.size()) {
            std::cerr<<"FAIL: replacement image reference failed\n";return 1;
        }
        auto* settings=obs_data_create();
        obs_data_set_string(settings,"file",changed_input.path.c_str());
        obs_source_update(source,settings);
        obs_data_release(settings);
        bool live_changed=false,ever_black=false;
        size_t best_match=0;
        for(int attempt=0;attempt<18;++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if(!render_image(source,current)) {
                std::cerr<<"FAIL: live privacy frame readback failed\n";return 1;
            }
            bool black=!current.empty();
            for(size_t i=0;i+3<current.size();i+=4)
                black &= current[i]==0&&current[i+1]==0&&
                         current[i+2]==0&&current[i+3]==255;
            ever_black|=black;
            size_t match=0;
            for(uint32_t y=0;y<64;++y)for(uint32_t x=0;x<64;++x) {
                const size_t i=(size_t(y)*W+x)*4;
                bool same=true;
                for(unsigned c=0;c<3;++c)
                    same&=std::abs(int(current[i+c])-int(replacement_baseline[i+c]))<=2;
                match+=same;
            }
            best_match=std::max(best_match,match);
            if(match>=3900)live_changed=true;
        }
        std::cout<<"privacy_live_source_updates="<<live_changed
                 <<" privacy_never_black="<<!ever_black
                 <<" matched_background="<<best_match<<"\n";
        if(!live_changed||ever_black) {
            std::cerr<<"FAIL: privacy filter froze live source or emitted black\n";
            return 1;
        }
    }
    if(std::strcmp(selected_filter,"sight_depth_filter")==0&&std::getenv("SIGHT_SMOKE_DEPTH_REUSE")){
        int64_t captures=0,completed=0,growths=0,uploads=0,loads=0;
        auto stats=[&]{
            calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_depth_buffer_stats_v1",&d);
            captures=calldata_int(&d,"captures");completed=calldata_int(&d,"completed");growths=calldata_int(&d,"growths");
            uploads=calldata_int(&d,"uploads");loads=calldata_int(&d,"loads");calldata_free(&d);return ok;
        };
        auto advance=[&](int64_t n){
            auto end=std::chrono::steady_clock::now()+std::chrono::seconds(25);
            while(std::chrono::steady_clock::now()<end){
                if(!stats())return false;
                if(completed>=n)return render_image(source,current)&&stats();
                if(!render_image(source,current))return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            return false;
        };
        bool warm=advance(4);auto initial=growths,prior=completed;
        bool stable=warm&&advance(prior+4)&&growths==initial&&initial>0&&uploads>0&&loads==1;
        std::cout<<"SIGHT_DEPTH_REUSE warm="<<warm<<" growths="<<initial<<" final="<<growths<<" completed="<<completed
            <<" captures="<<captures<<" uploads="<<uploads<<" loads="<<loads<<" pass="<<stable<<"\n";
        if(!stable){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        if(std::getenv("SIGHT_SMOKE_DEPTH_REUSE_EDGE")){
            auto* edit=obs_source_get_settings(filter);obs_data_set_int(edit,"inference_edge",336);obs_source_update(filter,edit);obs_data_release(edit);
            bool changed=advance(completed+4)&&loads==1;
            auto grown=growths;
            changed=changed&&advance(completed+4)&&growths==grown&&loads==1;
            std::cout<<"SIGHT_DEPTH_REUSE_EDGE loads="<<loads<<" growths="<<grown<<" final="<<growths<<" pass="<<changed<<"\n";
            if(!changed){obs_source_filter_remove(source,filter);obs_source_release(filter);obs_source_release(source);obs_shutdown();return 1;}
        }
    }
    if (std::strcmp(selected_filter, "sight_depth_filter") == 0 &&
        std::getenv("SIGHT_SMOKE_DEPTH_VERIFY_EXPIRY")) {
        // After real model inference, pause all OBS rendering so capture
        // cannot refresh the map; on the next render, it must fail open.
        const int sleep_ms = std::atoi(std::getenv("SIGHT_SMOKE_DEPTH_VERIFY_EXPIRY"));
        if (sleep_ms < 2000 || sleep_ms > 15000) {
            std::cerr << "FAIL: invalid expiry test hold\n";
            return 1;
        }
        std::unique_ptr<obs_properties_t,decltype(&obs_properties_destroy)> health_owner(obs_source_properties(filter),obs_properties_destroy);
        auto* health_props=health_owner.get();
        auto* health_label=health_props?obs_properties_get(health_props,"sight_inference_health"):nullptr;
        auto* refresh=health_props?obs_properties_get(health_props,"sight_backend_refresh"):nullptr;
        auto health_text=[&](){
            if(!health_label||!refresh||!obs_property_button_clicked(refresh,filter))return std::string("missing");
            return std::string(obs_property_description(health_label));
        };
        bool rate_seen=false;
        for(int trial=0;trial<60&&!rate_seen;++trial){
            if(!render_image(source,current))break;
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
            rate_seen=health_text().find("Recent result rate:")!=std::string::npos;
        }
        const auto active_health=health_text();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        const auto quiet_health=health_text();
        bool quiet_ok=quiet_health.find("No recent inference (source may be idle)")!=std::string::npos
            &&quiet_health.find("Recent result rate:")==std::string::npos;
        std::cout<<"SIGHT_HEALTH_ACTIVE "<<active_health<<"\n";
        std::cout<<"SIGHT_HEALTH_QUIET "<<quiet_health<<"\n";
        if(!rate_seen||(sleep_ms>=6500&&!quiet_ok)){
            std::cerr<<"FAIL: inference health rate/quiet transition\n";
            return 1;
        }
        if (!render_image(source, current) || current.size() != baseline.size()) {
            std::cerr << "FAIL: expired depth readback failed\n";
            return 1;
        }
        size_t changed = 0;
        for (size_t i = 0; i < baseline.size(); ++i)
            changed += baseline[i] != current[i];
        std::cout << "depth_expired_passthrough_diff=" << changed << "\n";
        if (changed > baseline.size() / 100) {
            std::cerr << "FAIL: expired depth retained a stale visualization\n";
            return 1;
        }
        bool recovered = false;
        for (int trial = 0; trial < 40 && !recovered; ++trial) {
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
            if (!render_image(source, current)) break;
            size_t fresh_diff = 0;
            for (size_t i = 0; i < baseline.size(); ++i)
                fresh_diff += current[i] != baseline[i];
            recovered = fresh_diff > required_diff;
        }
        std::cout << "depth_expired_recovers_after_resume=" << recovered << "\n";
        const auto resumed_health=health_text();
        std::cout<<"SIGHT_HEALTH_RESUMED "<<resumed_health<<"\n";
        if (!recovered || resumed_health.find("Recent inference")==std::string::npos) {
            std::cerr << "FAIL: depth stayed in passthrough after inference resumed\n";
            return 1;
        }
    }
    if (std::strcmp(selected_filter, "sight_depth_filter") == 0 &&
        std::getenv("SIGHT_SMOKE_DEPTH_VERIFY_RESIZE")) {
        W += 48;
        H += 28;
        obs_source_set_enabled(filter, false);
        std::vector<uint8_t> resized_reference;
        const bool reference_ok = render_image(source, resized_reference);
        obs_source_set_enabled(filter, true);
        std::vector<uint8_t> resized_filtered;
        const bool filtered_ok = render_image(source, resized_filtered);
        if (!reference_ok || !filtered_ok || resized_reference.empty() ||
            resized_reference.size() != resized_filtered.size()) {
            std::cerr << "FAIL: source resize readback failed\n";
            return 1;
        }
        size_t changed = 0;
        for (size_t i = 0; i < resized_reference.size(); ++i)
            changed += resized_reference[i] != resized_filtered[i];
        std::cout << "depth_resize_immediate_passthrough_diff=" << changed << "\n";
        if (changed > resized_reference.size() / 100) {
            std::cerr << "FAIL: resize retained a mismatched depth visualization\n";
            return 1;
        }
        bool recovered = false;
        for (int trial = 0; trial < 40 && !recovered; ++trial) {
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
            if (!render_image(source, resized_filtered)) break;
            size_t fresh_diff = 0;
            for (size_t i = 0; i < resized_reference.size(); ++i)
                fresh_diff += resized_filtered[i] != resized_reference[i];
            recovered = fresh_diff > resized_reference.size() / 12;
        }
        std::cout << "depth_resize_recovers_at_new_resolution=" << recovered << "\n";
        if (!recovered) {
            std::cerr << "FAIL: resized Depth Camera never resumed actual output\n";
            return 1;
        }
    }
    if (std::strcmp(selected_filter,"sight_u2net_filter")==0 &&
        std::getenv("SIGHT_SMOKE_SHARED_MATTE")) {
        auto* changed=obs_data_create();
        obs_data_set_int(changed,"matte_view",4);
        obs_source_update(filter,changed);
        obs_data_release(changed);
        obs_module_t* mod=nullptr;
        const char* module=std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE");
        const char* data=std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA");
        const char* shader=std::getenv("SIGHT_SMOKE_HYPERSHADER_EFFECT");
        if (!module || !data || !shader || obs_open_module(&mod,module,data)!=MODULE_SUCCESS ||
            !obs_init_module(mod)) return 1;
        auto* opts=obs_data_create();
        obs_data_set_string(opts,"shader_file",shader);
        obs_data_set_bool(opts,"sight_use_tracking",true);
        auto* hyper=obs_source_create("hypershader_filter","Shared matte probe",opts,nullptr);
        obs_data_release(opts);
        if (!hyper) return 1;
        obs_source_filter_add(source,hyper);
        obs_source_inc_showing(source);
        obs_set_output_source(0,source);
        std::vector<uint8_t> painted;
        bool mask=false, cleared=false, recovered=false;
        auto probe=[&](bool expected) {
            for (int i=0;i<100;++i) {
                if (!render_image(source,painted) || painted.size()!=baseline.size()) break;
                size_t marked=0, low=0, high=0, stale=0;
                for (size_t j=0;j<painted.size();j+=4) {
                    if (painted[j+1]==0 && painted[j+2]==255 && painted[j+3]==255) {
                        ++marked;
                        low+=painted[j]<24;
                        high+=painted[j]>230;
                    }
                    stale+=painted[j]==0 && painted[j+1]==255 && painted[j+2]==0 && painted[j+3]==255;
                }
                if (expected ? (marked>W*H*9/10 && low>100 && high>100) : (marked<W*H/100 && stale<W*H/100))
                    return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(55));
            }
            return false;
        };
        mask=probe(true);
        sight_matte_v1 snap{};
        snap.abi_version=SIGHT_MATTE_ABI_V1;
        snap.struct_size=sizeof(snap);
        auto read=[&](int age, uint64_t known) {
            calldata_t args; calldata_init(&args);
            calldata_set_int(&args,"max_age_ms",age);
            calldata_set_int(&args,"known_sequence",known);
            calldata_set_ptr(&args,"snapshot",&snap);
            bool available=false;
            const bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_matte_read_v1",&args);
            const bool returned=calldata_get_bool(&args,"available",&available);
            calldata_free(&args);
            return called && returned && available;
        };
        // The reference is the immutable portrait, so it can be a later
        // inference. Do not race the 750 ms display freshness boundary
        // while independently checking the geometry on the main thread.
        bool reference_read=false;
        for (int i=0;i<20 && !reference_read;++i) {
            reference_read=read(5000,0);
            if (!reference_read) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        bool contract=reference_read && snap.sequence && snap.width==320 && snap.height==320
            && snap.source_width==W && snap.source_height==H;
        bool subject=false;
        if (contract && mask) {
            uint32_t minx=snap.width,miny=snap.height,maxx=0,maxy=0;
            double cx=0,cy=0,n=0;
            for (uint32_t y=0;y<snap.height;++y)
                for (uint32_t x=0;x<snap.width;++x)
                    if (snap.pixels[y*snap.width+x]>=128) {
                        minx=std::min(minx,x); miny=std::min(miny,y);
                        maxx=std::max(maxx,x); maxy=std::max(maxy,y);
                        cx+=x+.5; cy+=y+.5; ++n;
                    }
            if (n>0) {
                const double expected[]={double(minx)/snap.width,double(miny)/snap.height,
                    double(maxx+1)/snap.width,double(maxy+1)/snap.height,
                    cx/n/snap.width,cy/n/snap.height,n/(snap.width*snap.height),1};
                subject=true;
                for (unsigned i=0;i<8;++i) {
                    const size_t offset=(size_t(H/50)*W+(2*i+1)*W/16)*4;
                    const double actual=painted[offset]/255.0;
                    std::cout<<"SIGHT_SUBJECT_COMPONENT "<<i<<" expected="<<expected[i]<<" gpu="<<actual<<"\n";
                    subject=subject && std::abs(actual-expected[i])<.025
                        && painted[offset+1]==255 && painted[offset+2]==0;
                }
            }
        }
        std::cout<<"SIGHT_SUBJECT_GEOMETRY_GPU="<<subject<<"\n";
        snap.abi_version=99;
        contract=contract && !read(750,0);
        snap.abi_version=SIGHT_MATTE_ABI_V1;
        snap.struct_size=sizeof(snap)-1;
        contract=contract && !read(750,0);
        snap.struct_size=sizeof(snap);
        contract=contract && !read(0,0);
        obs_source_set_enabled(filter,false);
        cleared=probe(false);
        obs_source_set_enabled(filter,true);
        recovered=probe(true);
        // Misordered, detached, or ambiguous producers must not leak a matte.
        obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_TOP);
        const bool wrong_order=probe(false);
        obs_source_filter_set_order(source,hyper,OBS_ORDER_MOVE_BOTTOM);
        const bool reordered=probe(true);
        obs_source_filter_remove(source,filter);
        const bool detached=probe(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(950));
        const bool expired=!read(750,0);
        obs_source_filter_add(source,filter);
        obs_source_filter_set_order(source,filter,OBS_ORDER_MOVE_TOP);
        const bool reattached=probe(true);
        std::cout<<"SIGHT_SHARED_MATTE_GPU="<<mask<<" CLEAR="<<cleared<<" RECOVER="<<recovered
            <<" ABI="<<contract<<" ORDER="<<wrong_order<<" REORDERED="<<reordered
            <<" DETACHED="<<detached<<" EXPIRED="<<expired<<" REATTACHED="<<reattached<<"\n";
        obs_set_output_source(0,nullptr);
        obs_source_dec_showing(source);
        smoke_drain_ui();
        obs_source_filter_remove(source,hyper);
        obs_source_release(hyper);
        smoke_drain_ui();
        if (!subject || !mask || !cleared || !recovered || !contract || !wrong_order || !reordered ||
            !detached || !expired || !reattached) {
            obs_source_filter_remove(source,filter); obs_source_release(filter);
            obs_source_release(source); smoke_drain_ui(); obs_shutdown(); return 1;
        }
        if (const char* frame_shader=std::getenv("SIGHT_SMOKE_SUBJECT_FRAME_EFFECT")) {
            auto* opts=obs_data_create();
            obs_data_set_string(opts,"shader_file",frame_shader);
            obs_data_set_bool(opts,"sight_use_tracking",true);
            obs_data_set_double(opts,"param_frame_opacity",1.);
            auto* frame=obs_source_create("hypershader_filter","Subject frame probe",opts,nullptr);
            obs_data_release(opts);
            if (!frame) return 1;
            obs_source_filter_add(source,frame);
            obs_source_inc_showing(source);
            obs_set_output_source(0,source);
            bool drawn=false;
            size_t max_markers=0,max_changed=0;
            std::vector<uint8_t> preview;
            for (int i=0;i<100 && !drawn;++i) {
                if (!render_image(source,preview) || preview.size()!=baseline.size()) break;
                // Exact full-strength default marker color in the crosshair
                // near the independent mask centroid; most pixels untouched.
                size_t changed=0, markers=0;
                for (size_t j=0;j<preview.size();j+=4) {
                    bool differs=false;
                    for (int k=0;k<3;++k)
                        differs|=std::abs(int(preview[j+k])-int(baseline[j+k]))>3;
                    changed+=differs;
                    if (differs && std::abs(int(preview[j])-36)<=4 &&
                        std::abs(int(preview[j+1])-235)<=4 && std::abs(int(preview[j+2])-204)<=4)
                        ++markers;
                }
                max_markers=std::max(max_markers,markers);
                max_changed=std::max(max_changed,changed);
                drawn=markers>20 && changed<W*H/8;
                std::this_thread::sleep_for(std::chrono::milliseconds(55));
            }
            auto* zero=obs_data_create();
            obs_data_set_double(zero,"param_frame_opacity",0.0);
            obs_source_update(frame,zero);
            obs_data_release(zero);
            auto passthrough=[&]() {
                for (int i=0;i<60;++i) {
                    if (!render_image(source,preview) || preview.size()!=baseline.size()) return false;
                    size_t errors=0;
                    for (size_t j=0;j<preview.size();++j)
                        errors+=std::abs(int(preview[j])-int(baseline[j]))>3;
                    if (!errors) return true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(55));
                }
                return false;
            };
            const bool opacity_zero=passthrough();
            auto* restore=obs_data_create();
            obs_data_set_double(restore,"param_frame_opacity",1.0);
            obs_source_update(frame,restore);obs_data_release(restore);
            obs_source_set_enabled(filter,false);
            const bool absent=passthrough();
            std::cout<<"SIGHT_SUBJECT_FRAME_GPU="<<drawn<<" ZERO_MIX="<<opacity_zero
                <<" ABSENT="<<absent<<" MARKER_PIXELS="<<max_markers<<" CHANGED="<<max_changed<<"\n";
            obs_set_output_source(0,nullptr);
            obs_source_dec_showing(source);
            smoke_drain_ui();
            obs_source_filter_remove(source,frame);
            obs_source_release(frame);
            smoke_drain_ui();
            obs_source_set_enabled(filter,true);
            if (!drawn || !opacity_zero || !absent) {
                obs_source_filter_remove(source,filter); obs_source_release(filter);
                obs_source_release(source); smoke_drain_ui(); obs_shutdown(); return 1;
            }
        }

        if (const char* shader_path=std::getenv("SIGHT_SMOKE_DEFOCUS_EFFECT")) {
            auto* settings=obs_data_create();
            obs_data_set_string(settings,"shader_file",shader_path);
            obs_data_set_bool(settings,"sight_use_tracking",true);
            obs_data_set_double(settings,"param_blur_radius",24.0);
            auto* defocus=obs_source_create("hypershader_filter","Sight defocus regression",settings,nullptr);
            obs_data_release(settings);
            if (!defocus) return 1;
            obs_source_filter_add(source,defocus);
            obs_source_inc_showing(source);
            obs_set_output_source(0,source);

            auto check=[&](bool active) {
                for (int i=0;i<80;++i) {
                    if (!render_image(source,painted) || painted.size()!=baseline.size()) return false;
                    size_t foreground=0, foreground_changed=0, background_changed=0, changed=0;
                    uint64_t original_edges=0, blurred_edges=0;
                    bool alpha_ok=true;
                    for (uint32_t y=0;y<H;++y) for (uint32_t x=0;x<W;++x) {
                        const size_t p=(size_t(y)*W+x)*4;
                        const size_t m=size_t(y*snap.height/H)*snap.width+x*snap.width/W;
                        bool differs=false;
                        for (int k=0;k<3;++k)
                            differs|=std::abs(int(painted[p+k])-int(baseline[p+k]))>3;
                        changed+=differs;
                        alpha_ok&=std::abs(int(painted[p+3])-int(baseline[p+3]))<=1;
                        if (snap.pixels[m]>=230) { ++foreground; foreground_changed+=differs; }
                        else if (snap.pixels[m]<=12) {
                            background_changed+=differs;
                            if (x+1<W && snap.pixels[size_t(y*snap.height/H)*snap.width+
                                                     (x+1)*snap.width/W]<=12) {
                                for (int k=0;k<3;++k) {
                                    original_edges+=std::abs(int(baseline[p+k])-int(baseline[p+4+k]));
                                    blurred_edges+=std::abs(int(painted[p+k])-int(painted[p+4+k]));
                                }
                            }
                        }
                    }
                    if (alpha_ok && (active ? foreground>100 && foreground_changed<foreground/20 &&
                                             background_changed>W*H/100 && blurred_edges<original_edges
                                             : changed==0)) return true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(55));
                }
                return false;
            };
            auto radius=[&](double value) {
                auto* update=obs_data_create();
                obs_data_set_double(update,"param_blur_radius",value);
                obs_source_update(defocus,update);
                obs_data_release(update);
            };
            const bool active=check(true);
            radius(0.0);
            const bool zero=check(false);
            radius(24.0);
            const bool restored=check(true);
            obs_source_set_enabled(filter,false);
            const bool absent=check(false);
            obs_source_set_enabled(filter,true);
            const bool resumed=check(true);
            blog(LOG_INFO,"SIGHT_DEFOCUS_D3F0 active=%d zero=%d restored=%d absent=%d resumed=%d",
                 active,zero,restored,absent,resumed);
            obs_set_output_source(0,nullptr);
            obs_source_dec_showing(source);
            smoke_drain_ui();
            obs_source_filter_remove(source,defocus);
            obs_source_release(defocus);
            smoke_drain_ui();
            if (!active || !zero || !restored || !absent || !resumed) {
                obs_source_filter_remove(source,filter); obs_source_release(filter);
                obs_source_release(source); smoke_drain_ui(); obs_shutdown(); return 1;
            }
        }

        if (const char* motion_shader=std::getenv("SIGHT_SMOKE_MOTION_EFFECT")) {
            TemporaryImage replacement;
            const char* original_path=std::getenv("SIGHT_SMOKE_IMAGE_FILE");
            if (!original_path || !replacement.make(W,H)) return 1;
            auto* opts=obs_data_create();
            obs_data_set_string(opts,"shader_file",motion_shader);
            obs_data_set_bool(opts,"sight_use_tracking",true);
            obs_data_set_bool(opts,"param_show_motion",true);
            obs_data_set_double(opts,"param_sensitivity",1.);
            obs_data_set_double(opts,"param_pulse_mix",1.);
            auto* pulse=obs_source_create("hypershader_filter","Motion pulse probe",opts,nullptr);
            obs_data_release(opts);
            if (!pulse) return 1;
            obs_source_filter_add(source,pulse);
            obs_source_inc_showing(source);
            obs_set_output_source(0,source);
            std::vector<uint8_t> preview;
            sight_motion_v1 motion{};
            auto read_motion=[&](int age, bool correct_abi=true) {
                motion.abi_version=correct_abi?SIGHT_MOTION_ABI_V1:99;
                motion.struct_size=sizeof(motion);
                calldata_t args;calldata_init(&args);
                calldata_set_int(&args,"max_age_ms",age);
                calldata_set_ptr(&args,"snapshot",&motion);
                bool available=false;
                const bool called=proc_handler_call(obs_source_get_proc_handler(filter),"sight_motion_read_v1",&args);
                const bool returned=calldata_get_bool(&args,"available",&available);
                calldata_free(&args);
                return called&&returned&&available;
            };
            auto change_image=[&](const char* path){
                auto* opts=obs_data_create();
                obs_data_set_string(opts,"file",path);
                obs_source_update(source,opts);obs_data_release(opts);
            };
            auto probe_energy=[&](bool moving){
                for (int i=0;i<120;++i) {
                    if (!render_image(source,preview) || preview.size()!=baseline.size()) break;
                    const int intensity=preview[0];
                    size_t uniform=0;
                    for (size_t j=0;j<preview.size();j+=4)
                        uniform+=std::abs(int(preview[j])-intensity)<=1 &&
                            std::abs(int(preview[j+1])-intensity)<=1 &&
                            std::abs(int(preview[j+2])-intensity)<=1 && preview[j+3]==255;
                    if (uniform>W*H*99/100 && (moving?intensity>8:intensity<=1)) {
                        std::cout<<"SIGHT_MOTION_ENERGY_GPU level="<<intensity/255.0<<" moving="<<moving<<"\n";
                        return true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(35));
                }
                return false;
            };
            const bool stationary=probe_energy(false);
            const bool abi=!read_motion(750,false) && !read_motion(0);
            change_image(replacement.path.c_str());
            const bool movement=probe_energy(true);
            const bool settled=probe_energy(false);
            // Restore portrait with the real foreground pulse, not debug view.
            auto* normal=obs_data_create();
            obs_data_set_bool(normal,"param_show_motion",false);
            obs_data_set_double(normal,"param_sensitivity",8.);
            obs_source_update(pulse,normal);obs_data_release(normal);
            change_image(original_path);
            bool colored=false;
            size_t changed_pixels=0;
            for (int i=0;i<120;++i) {
                if (!render_image(source,preview) || preview.size()!=baseline.size()) break;
                size_t changed=0,toward_cyan=0;
                for (size_t j=0;j<preview.size();j+=4) {
                    bool differs=false;
                    for (int k=0;k<3;++k)
                        differs|=std::abs(int(preview[j+k])-int(baseline[j+k]))>3;
                    changed+=differs;
                    // The actual preset blends toward (38,217,255).
                    if (differs && preview[j+2]>baseline[j+2]+5 &&
                        std::abs(int(preview[j])-38)<=std::abs(int(baseline[j])-38)+2 &&
                        std::abs(int(preview[j+1])-217)<=std::abs(int(baseline[j+1])-217)+2)
                        ++toward_cyan;
                }
                if (changed>100 && changed<W*H*4/5 && toward_cyan>100) {
                    colored=true;changed_pixels=changed;break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(35));
            }
            obs_source_set_enabled(filter,false);
            bool absent=false;
            for (int i=0;i<60;++i) {
                if (!render_image(source,preview) || preview.size()!=baseline.size()) break;
                size_t errors=0;
                for (size_t j=0;j<preview.size();++j)
                    errors+=std::abs(int(preview[j])-int(baseline[j]))>3;
                if (!errors) {absent=true;break;}
                std::this_thread::sleep_for(std::chrono::milliseconds(35));
            }
            const bool disabled=!read_motion(750);
            std::cout<<"SIGHT_MOTION_PULSE_GPU="<<colored<<" STILL="<<stationary
                <<" MOVING="<<movement<<" SETTLED="<<settled<<" ABI="<<abi
                <<" ABSENT="<<absent<<" DISABLED="<<disabled<<" CHANGED="<<changed_pixels<<"\n";
            obs_set_output_source(0,nullptr);obs_source_dec_showing(source);
            smoke_drain_ui();
            obs_source_filter_remove(source,pulse);obs_source_release(pulse);
            smoke_drain_ui();
            obs_source_set_enabled(filter,true);
            if (!stationary || !movement || !settled || !colored || !absent || !disabled || !abi) {
                obs_source_filter_remove(source,filter);obs_source_release(filter);
                obs_source_release(source);smoke_drain_ui();obs_shutdown();return 1;
            }
        }
    }
    // Optional two-plugin integration: real Sight producer -> versioned OBS
    // snapshot -> native HyperShader filter -> GPU pixels and TTL clear.
    // Runs only when explicitly pointed at the separately built HyperShader.
    if (signal_receiver && producer && std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE")) {
        const char* hsModule = std::getenv("SIGHT_SMOKE_HYPERSHADER_MODULE");
        const char* hsData = std::getenv("SIGHT_SMOKE_HYPERSHADER_DATA");
        const char* hsEffect = std::getenv("SIGHT_SMOKE_HYPERSHADER_EFFECT");
        if (!hsData || !hsEffect) {
            std::cerr << "FAIL: missing HyperShader test assets\n";
            return 1;
        }
        obs_source_set_enabled(filter, false);  // no Sight receiver overlay
        obs_module_t* hyperModule = nullptr;
        if (obs_open_module(&hyperModule, hsModule, hsData) != MODULE_SUCCESS ||
            !obs_init_module(hyperModule)) {
            std::cerr << "FAIL: HyperShader module load\n";
            return 1;
        }
        auto* options = obs_data_create();
        obs_data_set_string(options, "shader_file", hsEffect);
        obs_data_set_bool(options, "sight_use_tracking", true);
        obs_data_set_string(options, "sight_tracking_channel",
                            std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL")
                            ? std::getenv("SIGHT_SMOKE_SIGNAL_CHANNEL") : "face");
        auto* hyper = obs_source_create("hypershader_filter", "Sight shared probe", options, nullptr);
        obs_data_release(options);
        if (!hyper) {
            std::cerr << "FAIL: cannot create HyperShader Sight filter\n";
            return 1;
        }
        obs_source_filter_add(source, hyper);
        obs_source_inc_showing(source);
        obs_set_output_source(0, source);
        std::vector<uint8_t> painted;
        bool shaderPass = false;
        for (int i = 0; i < 110 && !shaderPass; ++i) {
            if (!render_image(source, painted) || painted.size() < size_t(W)*H*4)
                break;
            auto* px = &painted[(size_t(H/2)*W+W/2)*4];
            // The probe shader only writes a full-blue landmark sample when
            // iSightPresent, kind, count AND iSightPoints are bound.
            shaderPass = px[2] > 220 && px[0] > 5 && px[0] < 245 &&
                         px[1] > 5 && px[1] < 245;
            std::this_thread::sleep_for(std::chrono::milliseconds(55));
        }
        std::cout << "HYPERSHADER_SIGHT_LIVE_PIXEL_PASS=" << shaderPass << "\n";
        // Withdraw the producer and let the 750ms snapshot expire. The
        // shader must return to original source, not a frozen landmark.
        obs_source_filter_remove(source, producer);
        obs_source_release(producer);
        producer = nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(850));
        bool expired = false;
        for (int i = 0; i < 40; ++i) {
            if (!render_image(source, painted))
                break;
            const auto* px = &painted[(size_t(H/2)*W+W/2)*4];
            // No-face path is source pass-through; bright-blue probe output
            // would be 255, whereas official hand image center is not.
            if (px[2] < 215) { expired=true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        std::cout << "HYPERSHADER_SIGHT_EXPIRED_CLEAR_PASS=" << expired << "\n";
        if (std::getenv("SIGHT_SMOKE_TEST_NATIVE_SWITCH")) {
            auto* nativeOpts = obs_source_get_settings(hyper);
            obs_data_set_bool(nativeOpts, "sight_use_tracking", false);
            obs_source_update(hyper, nativeOpts);
            obs_data_release(nativeOpts);
            // Video filters receive OBS's deferred update during a later
            // render/tick, not synchronously in obs_source_update().
            for (int tick = 0; tick < 12; ++tick) {
                render_image(source, painted);
                std::this_thread::sleep_for(std::chrono::milliseconds(35));
            }
            auto* updated = obs_source_get_settings(hyper);
            const bool switched = !obs_data_get_bool(updated, "sight_use_tracking");
            obs_data_release(updated);
            std::cout << "HYPERSHADER_SIGHT_NATIVE_SWITCH_APPLIED=" << switched << "\n";
            if (!switched)
                return 1;
        }
        obs_set_output_source(0, nullptr);
        obs_source_dec_showing(source);
        smoke_drain_ui();
        obs_source_filter_remove(source, hyper);
        obs_source_release(hyper);
        smoke_drain_ui();
        if (!shaderPass || !expired)
            return 1;
    }
    if(value_router)
        obs_source_dec_showing(source);
    const double final_gain=routed_gain;
    if(value_router)
        std::cout<<"router_target_gain="<<final_gain
                 <<" changes="<<gain_changes.load()
                 <<" value_min="<<gain_min.load()
                 <<" value_max="<<gain_max.load()<<"\n";
    if(producer && signal_receiver && std::getenv("SIGHT_SMOKE_HOLISTIC_TRIPLE")) {
        // Real worker and all three snapshots must disappear on producer
        // destruction, not linger as a misleading frozen tracking result.
        obs_source_filter_remove(source,producer);
        obs_source_release(producer);
        producer=nullptr;
        smoke_drain_ui();
        // libobs may retain internal references to detached filter sources
        // until shutdown. Their snapshots must still expire by the reader's
        // documented max-age even if destroy() is delayed. Never report
        // indefinite activity from a removed Holistic producer.
        std::this_thread::sleep_for(std::chrono::milliseconds(5250));
        bool empty=true;
        for(const char* channel:{"holistic/face","holistic/body","holistic/hand"}) {
            sight_snapshot_v1 snapshot{};
            snapshot.abi_version=SIGHT_SNAPSHOT_ABI_V1;
            snapshot.struct_size=sizeof(snapshot);
            calldata_t data;calldata_init(&data);
            calldata_set_string(&data,"channel",channel);
            calldata_set_int(&data,"max_age_ms",5000);
            calldata_set_ptr(&data,"snapshot",&snapshot);
            const bool called=proc_handler_call(obs_source_get_proc_handler(filter),
                                                "sight_tracking_read_v1",&data);
            bool available=false;
            const bool returned=calldata_get_bool(&data,"available",&available);
            calldata_free(&data);
            empty &= called && returned && !available;
        }
        std::cout<<"SIGHT_HOLISTIC_EXPIRED_AFTER_REMOVE_PASS="<<(empty?1:0)<<"\n";
        if(!empty) {
            std::cerr<<"FAIL: detached Holistic producer was still readable after max age\n";
            obs_source_filter_remove(source,filter);
            obs_source_release(filter);
            obs_source_release(source);obs_shutdown();return 1;
        }
    }
    obs_source_filter_remove(source, filter);
    obs_source_release(filter);
    if(producer) {
        obs_source_filter_remove(source,producer);
        obs_source_release(producer);
    }
    if(numeric) {
        obs_source_filter_remove(source,numeric);
        obs_source_release(numeric);
    }
    obs_source_release(source);
    smoke_drain_ui();
    obs_shutdown();
    if(value_router) {
        if(router_gate) {
            if(gate_intermediate.load()!=0 ||
               (tracked_gate ? !gate_released_on_loss : gain_changes.load()<5) ||
               std::abs(gain_min.load()-10.)>.001 ||
               std::abs(gain_max.load()-90.)>.001 ||
               (std::abs(final_gain-10.)>.001 && std::abs(final_gain-90.)>.001)) {
                std::cerr<<"FAIL: Threshold gate must write only both endpoints repeatedly; intermediate="
                         <<gate_intermediate.load()<<"\n";
                return 1;
            }
            std::cout<<"PASS: actual OBS threshold gate writes only endpoints\n";
            return 0;
        }
        if(final_gain<=15 || final_gain>=90 ||
           (!std::getenv("SIGHT_SMOKE_ROUTER_SIGNAL") &&
             (gain_changes.load()<5 ||
              gain_max.load()-gain_min.load()<20))) {
            std::cerr<<"FAIL: Value Router did not modulate actual OBS numeric setting\n";
            return 1;
        }
        std::cout<<"PASS: actual OBS effect property updated by Value Router\n";
        return 0;
    }
    if (!expect_passthrough && diff <= required_diff) {
        std::cerr << "FAIL: OBS output did not visibly transform\n";
        return 1;
    }
    if(expect_passthrough)
        std::cout << (privacy_v2?
            "PASS: Privacy V2 real zero-face passthrough\n":
            "PASS: genuine passthrough\n");
    else
        std::cout << "PASS: actual OBS filter changed rendered pixels\n";
    return 0;
}
