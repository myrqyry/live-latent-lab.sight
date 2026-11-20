#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include "core/touch_field_signals.hpp"
#include "core/tracking_discovery.hpp"
#include "core/touch_field_events.hpp"

namespace {
constexpr const char* Channel="signal_channel";
constexpr const char* Age="signal_max_age_ms";
constexpr const char* Radius="touch_radius";
constexpr const char* Strength="touch_strength";
constexpr const char* Waves="touch_waves";
constexpr const char* Speed="touch_speed";
constexpr const char* Pinch="touch_pinch_boost";

struct Settings {
    std::string channel="hand";
    unsigned max_age=750;
    float radius=.19f,strength=.85f,waves=2.5f,speed=.8f,pinch=1.0f;
};
enum class SignalState {
    Active, Missing, Empty, WrongSource, NotRendered
};
struct Diagnostics {
    // Graphics-thread only. Summarize instead of one log per webcam frame.
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    unsigned frames=0,active=0,missing=0,empty=0,wrong_source=0,failed=0;
    unsigned new_snapshots=0,new_ripples=0;
    uint64_t last_sequence=0;
    bool have_sequence=false;
    sight::TouchMotionMeter motion;
};
struct Filter {
    obs_source_t* context=nullptr;
    std::mutex mutex;
    Settings settings;
    gs_texrender_t* source_render=nullptr;
    gs_effect_t* effect=nullptr;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    bool announced=false;
    Diagnostics diagnostics;
    sight::TouchRippleTrail trail;
    std::string live_channel;
};

const char* get_name(void*) {return "LLL.sight Touch Field (Hand Ripples)";}
void defaults(obs_data_t* data) {
    obs_data_set_default_string(data,Channel,"hand");
    obs_data_set_default_int(data,Age,750);
    obs_data_set_default_double(data,Radius,.19);
    obs_data_set_default_double(data,Strength,.85);
    obs_data_set_default_double(data,Waves,2.5);
    obs_data_set_default_double(data,Speed,.8);
    obs_data_set_default_double(data,Pinch,1.0);
}
void update(void* ptr,obs_data_t* data) {
    auto* f=static_cast<Filter*>(ptr);
    Settings s;
    s.channel=obs_data_get_string(data,Channel);
    s.max_age=unsigned(std::clamp<int64_t>(obs_data_get_int(data,Age),100,5000));
    s.radius=float(std::clamp(obs_data_get_double(data,Radius),.03,.65));
    s.strength=float(std::clamp(obs_data_get_double(data,Strength),0.,3.));
    s.waves=float(std::clamp(obs_data_get_double(data,Waves),.5,7.));
    s.speed=float(std::clamp(obs_data_get_double(data,Speed),0.,4.));
    s.pinch=float(std::clamp(obs_data_get_double(data,Pinch),0.,3.));
    std::lock_guard lock(f->mutex);
    f->settings=std::move(s);
}
void* create(obs_data_t* settings,obs_source_t* source) {
    auto* f=new Filter;
    f->context=source;
    update(f,settings);
    obs_enter_graphics();
    char* path=obs_module_file("sight-touch-field.effect");
    if(path) {
        char* errors=nullptr;
        f->effect=gs_effect_create_from_file(path,&errors);
        if(errors) {
            blog(LOG_ERROR,"[Sight/TouchField] Shader: %s",errors);
            bfree(errors);
        }
        bfree(path);
    }
    f->source_render=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    obs_leave_graphics();
    if(!f->effect||!f->source_render)
        blog(LOG_ERROR,"[Sight/TouchField] GPU shader/target unavailable; passthrough");
    return f;
}
void destroy(void* ptr) {
    auto* f=static_cast<Filter*>(ptr);
    obs_enter_graphics();
    if(f->source_render)gs_texrender_destroy(f->source_render);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();
    delete f;
}
obs_properties_t* properties(void* data) {
    auto* p=obs_properties_create();
    auto* f=static_cast<Filter*>(data);
    auto* ch=sight::discovery::add_picker(p,f?f->context:nullptr,Channel,
        "Hand Motion channel","hand",sight::TrackingKind::Hand);
    obs_property_set_long_description(ch,
        "Must match the Hand & Gesture Motion publisher (default 'hand'). "
        "Attach filters to the same untransformed source; no extra AI inference.");
    obs_properties_add_int_slider(p,Age,"Max tracking age (ms)",100,5000,50);
    obs_properties_add_float_slider(p,Radius,"Ripple radius",.03,.65,.01);
    obs_properties_add_float_slider(p,Strength,"Refraction strength",0,3,.05);
    obs_properties_add_float_slider(p,Waves,"Ripple rings",.5,7,.1);
    obs_properties_add_float_slider(p,Speed,"Wave speed",0,4,.1);
    obs_properties_add_float_slider(p,Pinch,"Pinch strength boost",0,3,.1);
    return p;
}
void set_float(gs_effect_t* effect,const char* name,float value) {
    auto* param=gs_effect_get_param_by_name(effect,name);
    if(param)gs_effect_set_float(param,value);
}
void set_point(gs_effect_t* effect,const char* name,const sight::TouchPoint& point) {
    auto* param=gs_effect_get_param_by_name(effect,name);
    if(!param)return;
    vec4 data{};
    vec4_set(&data,point.x,point.y,point.pinch,point.active);
    gs_effect_set_vec4(param,&data);
}
void set_ripple(gs_effect_t* effect,const char* name,const sight::TouchRipple& point) {
    auto* param=gs_effect_get_param_by_name(effect,name);
    if(!param)return;
    vec4 data{};
    vec4_set(&data,point.x,point.y,point.age,point.amplitude);
    gs_effect_set_vec4(param,&data);
}
void sample_diagnostics(Filter* f,SignalState state,
                        const sight::TrackingSignal* signal=nullptr,
                        const sight::TouchFields* fingers=nullptr,
                        bool new_ripple=false) {
    auto& d=f->diagnostics;
    ++d.frames;
    switch(state) {
    case SignalState::Active:
        ++d.active;
        if(signal && (!d.have_sequence || signal->sequence!=d.last_sequence)) {
            ++d.new_snapshots;
            d.last_sequence=signal->sequence;
            d.have_sequence=true;
        }
        if(signal&&fingers) {
            const float aspect=signal->source_height ?
                float(signal->source_width)/float(signal->source_height):1.f;
            d.motion.observe(*fingers,signal->sequence,aspect);
        }
        if(new_ripple)++d.new_ripples;
        break;
    case SignalState::Missing:++d.missing;break;
    case SignalState::Empty:++d.empty;break;
    case SignalState::WrongSource:++d.wrong_source;break;
    case SignalState::NotRendered:++d.failed;break;
    }
    const auto now=std::chrono::steady_clock::now();
    if(now-d.start<std::chrono::seconds(6))return;
    // Only visible with OBS --verbose. Condenses ~180 frames into one
    // useful diagnostic line instead of per-camera-frame log spam.
    blog(LOG_DEBUG,
         "[Sight/TouchField] 6s: rendered=%u/%u new_snapshots=%u "
         "moving_snapshots=%u finger_travel=%.4f new_ripple_batches=%u "
         "no_fresh_signal=%u no_hands=%u source_mismatch=%u gpu_fail=%u",
         d.active,d.frames,d.new_snapshots,d.motion.moved_snapshots,
         d.motion.travel,d.new_ripples,
         d.missing,d.empty,d.wrong_source,d.failed);
    d.frames=d.active=d.missing=d.empty=d.wrong_source=d.failed=0;
    d.new_snapshots=d.new_ripples=0;
    d.motion.clear_interval();
    d.start=now;
}
void render(void* ptr,gs_effect_t*) {
    auto* f=static_cast<Filter*>(ptr);
    auto* target=obs_filter_get_target(f->context);
    if(!target||!obs_filter_get_parent(f->context)||!f->effect||!f->source_render) {
        sample_diagnostics(f,SignalState::NotRendered);
        obs_source_skip_video_filter(f->context);
        return;
    }
    const uint32_t width=obs_source_get_base_width(target);
    const uint32_t height=obs_source_get_base_height(target);
    if(!width||!height) {
        sample_diagnostics(f,SignalState::NotRendered);
        obs_source_skip_video_filter(f->context);
        return;
    }
    Settings s;
    {
        std::lock_guard lock(f->mutex);
        s=f->settings;
    }
    if(f->live_channel!=s.channel) {
        f->trail.reset();
        f->live_channel=s.channel;
    }
    sight::TrackingSignal signal;
    if(!s.strength||!s.radius ||
       !sight::TrackingSignalBus::read(s.channel,
             std::chrono::milliseconds(s.max_age),signal)) {
        f->trail.reset();
        sample_diagnostics(f,SignalState::Missing);
        obs_source_skip_video_filter(f->context);
        return;
    }
    const auto fingers=sight::select_touch_fingertips(signal,width,height);
    if(!fingers.count) {
        f->trail.reset();
        const bool wrong_source=
            signal.source_width!=width||signal.source_height!=height;
        sample_diagnostics(f,wrong_source ? SignalState::WrongSource :
                                        SignalState::Empty);
        obs_source_skip_video_filter(f->context);
        return;
    }
    // Draw the upstream source *once* to a native-resolution GPU texture.
    gs_texrender_reset(f->source_render);
    if(!gs_texrender_begin(f->source_render,width,height)) {
        sample_diagnostics(f,SignalState::NotRendered);
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_projection_push();
    gs_matrix_push();
    gs_matrix_identity();
    gs_ortho(0,float(width),0,float(height),-100,100);
    auto* draw=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if(draw)
        while(gs_effect_loop(draw,"Draw"))
            obs_source_video_render(target);
    gs_matrix_pop();
    gs_projection_pop();
    gs_texrender_end(f->source_render);
    auto* frame=gs_texrender_get_texture(f->source_render);
    auto* image=gs_effect_get_param_by_name(f->effect,"original_frame");
    if(!draw||!frame||!image) {
        sample_diagnostics(f,SignalState::NotRendered);
        obs_source_skip_video_filter(f->context);
        return;
    }
    gs_effect_set_texture(image,frame);
    set_point(f->effect,"fingertip0",fingers.fingers[0]);
    set_point(f->effect,"fingertip1",fingers.fingers[1]);
    set_float(f->effect,"touch_strength",s.strength);
    set_float(f->effect,"touch_radius",s.radius);
    set_float(f->effect,"touch_waves",s.waves);
    set_float(f->effect,"touch_speed",s.speed);
    set_float(f->effect,"touch_pinch_boost",s.pinch);
    set_float(f->effect,"touch_aspect",float(width)/float(height));
    const float elapsed=std::chrono::duration<float>(
         std::chrono::steady_clock::now()-f->started).count();
    set_float(f->effect,"touch_time",std::fmod(elapsed,1024.f));
    const bool emitted=f->trail.consume(fingers,signal.sequence,elapsed,
                                         float(width)/float(height));
    const auto waves=f->trail.active(elapsed,s.pinch);
    for(size_t i=0;i<waves.size();++i) {
        char name[16];
        std::snprintf(name,sizeof(name),"ripple%zu",i);
        set_ripple(f->effect,name,waves[i]);
    }
    while(gs_effect_loop(f->effect,"Draw"))
        gs_draw_sprite(frame,0,width,height);
    sample_diagnostics(f,SignalState::Active,&signal,&fingers,emitted);
    if(!f->announced) {
        blog(LOG_INFO,"[Sight/TouchField] First live hand-driven GPU refraction: %u fingertip(s)",
             fingers.count);
        f->announced=true;
    }
}
}
struct obs_source_info sight_touch_field_filter_info={
    .id="sight_touch_field_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=get_name,
    .create=create,
    .destroy=destroy,
    .get_defaults=defaults,
    .get_properties=properties,
    .update=update,
    .video_render=render,
};
