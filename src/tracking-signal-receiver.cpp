#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "core/tracking_signal_bus.hpp"
#include "core/tracking_discovery.hpp"
#include "core/sight_snapshot_export.hpp"
#include "core/tracking_overlay.hpp"

namespace {
constexpr const char* Channel = "signal_channel";
constexpr const char* Timeout = "signal_max_age_ms";
constexpr const char* OverlayEdge = "overlay_resolution";

struct Receiver {
    obs_source_t* context = nullptr;
    std::mutex settings_mutex;
    std::string channel;
    unsigned max_age = 750;
    uint32_t edge = 512;

    uint64_t last_sequence = 0;
    uint32_t overlay_width = 0, overlay_height = 0;
    gs_texrender_t* source_render = nullptr;
    gs_texture_t* overlay_texture = nullptr;
    gs_effect_t* shader = nullptr;
    bool announced_first = false;
};

const char* get_name(void*) { return "LLL.sight Tracking Signal Receiver"; }
void get_defaults(obs_data_t* settings) {
    obs_data_set_default_string(settings, Channel, "face");
    obs_data_set_default_int(settings, Timeout, 750);
    obs_data_set_default_int(settings, OverlayEdge, 512);
}
void update(void* data, obs_data_t* settings) {
    auto* f = static_cast<Receiver*>(data);
    const auto* channel = obs_data_get_string(settings, Channel);
    const auto max_age = static_cast<unsigned>(
        std::clamp<int64_t>(obs_data_get_int(settings, Timeout),100,5000));
    const auto edge = static_cast<uint32_t>(
        std::clamp<int64_t>(obs_data_get_int(settings,OverlayEdge),128,1024));
    std::lock_guard lock(f->settings_mutex);
    f->channel = channel ? channel : "";
    f->max_age = max_age;
    f->edge = edge;
    // The sequence/texture remain graphics-thread owned. Changing channels
    // forces a new sequence or transparent passthrough immediately.
}
// A per-source, read-only OBS procedure avoids C++ inter-plugin linkage and
// disappears when this private receiver source is released. The caller owns
// the full versioned output buffer; nothing is retained across the call.
void read_tracking_snapshot_v1(void*, calldata_t* params)
{
    void* buffer = nullptr;
    const char* channel = nullptr;
    long long max_age = 0;
    const bool inputs_ok =
        calldata_get_ptr(params, "snapshot", &buffer) &&
        calldata_get_string(params, "channel", &channel) &&
        calldata_get_int(params, "max_age_ms", &max_age);
    const bool available = inputs_ok && max_age >= 1 && max_age <= 5000 &&
        sight::export_tracking_snapshot(channel, static_cast<unsigned>(max_age),
                                        static_cast<sight_snapshot_v1*>(buffer));
    calldata_set_bool(params, "available", available);
}
void list_tracking_v1(void* ptr,calldata_t* params){
    void* inventory=nullptr;
    if(!calldata_get_ptr(params,"inventory",&inventory)||!inventory)return;
    auto* f=static_cast<Receiver*>(ptr);
    sight::discovery::export_inventory(f->context,static_cast<obs_data_t*>(inventory));
    blog(LOG_DEBUG,"[Sight/Signals] SIGHT_INVENTORY_V1_20261118_C621");
}
void* create(obs_data_t* settings, obs_source_t* context) {
    auto* f = new Receiver;
    f->context = context;
    update(f,settings);
    auto* procedures = obs_source_get_proc_handler(context);
    if (procedures)
        proc_handler_add(procedures,
            "void sight_tracking_read_v1(in string channel, in int max_age_ms, in ptr snapshot, out bool available)",
            read_tracking_snapshot_v1, nullptr);
    if(procedures)proc_handler_add(procedures,
        "void sight_tracking_list_v1(in ptr inventory)",list_tracking_v1,f);
    return f;
}
void destroy(void* data) {
    auto* f = static_cast<Receiver*>(data);
    obs_enter_graphics();
    if(f->source_render) gs_texrender_destroy(f->source_render);
    if(f->overlay_texture) gs_texture_destroy(f->overlay_texture);
    if(f->shader) gs_effect_destroy(f->shader);
    obs_leave_graphics();
    delete f;
}
obs_properties_t* properties(void* data) {
    auto* p=obs_properties_create();
    auto* f=static_cast<Receiver*>(data);
    auto* channel=sight::discovery::add_picker(p,f?f->context:nullptr,
        Channel,"Receive tracking channel","face");
    obs_property_set_long_description(channel,
        "Read the named Face/Body/Hand Motion publisher in this OBS process. "
        "No additional MediaPipe inference. Coordinates are normalized to "
        "the publisher's video; align sources for accurate overlay.");
    obs_properties_add_int_slider(p,Timeout,"Signal expiry (ms)",100,5000,50);
    auto* size=obs_properties_add_list(p,OverlayEdge,"Overlay texture resolution",
                                       OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(size,"256 px",256);
    obs_property_list_add_int(size,"512 px",512);
    obs_property_list_add_int(size,"1024 px",1024);
    return p;
}

void render(void* data, gs_effect_t*) {
    auto* f=static_cast<Receiver*>(data);
    auto* target=obs_filter_get_target(f->context);
    if(!target || !obs_filter_get_parent(f->context)) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    const uint32_t w=obs_source_get_base_width(target);
    const uint32_t h=obs_source_get_base_height(target);
    if(!w||!h) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    std::string channel;
    unsigned max_age=0;
    uint32_t edge=0;
    {
        std::lock_guard lock(f->settings_mutex);
        channel=f->channel;
        max_age=f->max_age;
        edge=f->edge;
    }

    sight::TrackingSignal signal;
    if(!sight::TrackingSignalBus::read(
            channel,std::chrono::milliseconds(max_age),signal) ||
       signal.data.landmarks.empty()) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    const uint32_t longest=std::max(w,h);
    const uint32_t ow=std::max(1u,uint32_t(uint64_t(w)*edge/longest));
    const uint32_t oh=std::max(1u,uint32_t(uint64_t(h)*edge/longest));
    if(signal.sequence != f->last_sequence || !f->overlay_texture ||
       ow!=f->overlay_width || oh!=f->overlay_height) {
        auto overlay=sight::draw_tracking_overlay(
            ow,oh,signal.kind,signal.data);
        if(overlay.empty()) {
            obs_source_skip_video_filter(f->context);
            return;
        }
        if(!f->overlay_texture || ow!=f->overlay_width || oh!=f->overlay_height) {
            if(f->overlay_texture) gs_texture_destroy(f->overlay_texture);
            f->overlay_texture=gs_texture_create(ow,oh,GS_RGBA,1,nullptr,GS_DYNAMIC);
            f->overlay_width=ow;
            f->overlay_height=oh;
        }
        if(!f->overlay_texture) {
            obs_source_skip_video_filter(f->context);
            return;
        }
        gs_texture_set_image(f->overlay_texture,overlay.data(),ow*4,false);
        f->last_sequence=signal.sequence;
        if(!f->announced_first) {
            blog(LOG_INFO,"[Sight/Signals] First shared tracking signal '%s': %zu landmarks",
                 channel.c_str(),signal.data.landmarks.size());
            f->announced_first=true;
        }
    }
    if(!f->shader) {
        auto* path=obs_module_file("sight-tracking-overlay.effect");
        if(path) {
            char* errors=nullptr;
            f->shader=gs_effect_create_from_file(path,&errors);
            if(errors) {
                blog(LOG_ERROR,"[Sight/Signals] Shader: %s",errors);
                bfree(errors);
            }
            bfree(path);
        }
    }
    if(!f->shader) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    if(!f->source_render)
        f->source_render=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(!f->source_render) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_texrender_reset(f->source_render);
    if(!gs_texrender_begin(f->source_render,w,h)) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_projection_push();
    gs_matrix_push();
    gs_matrix_identity();
    gs_ortho(0,float(w),0,float(h),-100,100);
    auto* original_effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if(original_effect)
        while(gs_effect_loop(original_effect,"Draw"))
            obs_source_video_render(target);
    gs_matrix_pop();
    gs_projection_pop();
    gs_texrender_end(f->source_render);
    auto* tex=gs_texrender_get_texture(f->source_render);
    auto* color_param=gs_effect_get_param_by_name(f->shader,"original_frame");
    auto* mask_param=gs_effect_get_param_by_name(f->shader,"tracking_overlay");
    if(!tex||!color_param||!mask_param) {
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_effect_set_texture(color_param,tex);
    gs_effect_set_texture(mask_param,f->overlay_texture);
    while(gs_effect_loop(f->shader,"Draw"))
        gs_draw_sprite(tex,0,w,h);
}
}
struct obs_source_info sight_tracking_receiver_filter_info = {
    .id="sight_tracking_receiver_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=get_name,
    .create=create,
    .destroy=destroy,
    .get_defaults=get_defaults,
    .get_properties=properties,
    .update=update,
    .video_render=render,
};
