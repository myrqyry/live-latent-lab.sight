#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/value_modulation.hpp"
#include "core/ocr_processing.hpp"
#include "core/tracking_discovery.hpp"

namespace {
constexpr const char* MODE="input_mode";
constexpr const char* WAVE_TYPE="wave_type";
constexpr const char* FREQ="wave_frequency";
constexpr const char* AMP="wave_amplitude";
constexpr const char* PHASE="wave_phase";
constexpr const char* PULSE="wave_pulse_width";
constexpr const char* STEPS="wave_steps";
constexpr const char* INVERT="wave_invert";
constexpr const char* CHANNEL="signal_channel";
constexpr const char* FEATURE="signal_feature";
constexpr const char* INDEX="signal_index";
constexpr const char* HAND_INDEX="hand_index";
constexpr const char* HAND_GESTURE="hand_gesture";
constexpr const char* AGE="signal_max_age_ms";
constexpr const char* TARGET_SOURCE="target_source";
constexpr const char* TARGET_FILTER="target_filter";
constexpr const char* TARGET_SETTING="target_setting";
constexpr const char* MIN="output_min";
constexpr const char* MAX="output_max";
constexpr const char* RESPONSE="response_mode";
constexpr const char* ACTIVATE="gate_activate";
constexpr const char* RELEASE="gate_release";
constexpr const char* LIMIT="max_updates_per_second";

struct Configuration {
    int mode=0;
    sight::WaveOptions wave;
    std::string channel="face";
    sight::SignalFeature feature=sight::SignalFeature::FaceLandmarkX;
    unsigned index=1,age_ms=750,hand_index=0;
    sight::HandGesture hand_gesture=sight::HandGesture::OpenPalm;
    std::string target_source,target_filter,target_setting;
    std::string ocr_filter,ocr_query;
    int ocr_kind=0,ocr_slot=1;
    double ocr_min=0,ocr_max=100;
    double output_min=0,output_max=100;
    double max_rate=20;
    bool threshold_gate=false;
    double gate_activate=.65, gate_release=.45;
};
// UI-thread-owned lease on an exact filter object and numeric setting.
// Weak references avoid keeping removed targets alive.
struct Baseline {
    obs_weak_source_t* target=nullptr;
    std::string setting;
    obs_property_type type=OBS_PROPERTY_INVALID;
    double original=0,last=0;
    bool had_user_value=false;
    bool written=false;
};
struct Filter {
    obs_source_t* context=nullptr;
    std::mutex mutex;
    Configuration config;
    sight::WaveOscillator oscillator;
    sight::ThresholdGate gate;
    double throttler=0;
    Baseline baseline;
    std::atomic<bool> has_baseline{false};
    bool queued=false, release_requested=false;
    std::optional<double> pending_value;
    uint64_t revision=0,pending_revision=0;
    uint64_t published_id=0;
    unsigned applied=0;
    Filter():oscillator(0x157C68D1u){}
};

obs_source_t* resolve_parent(Filter* f,const std::string& selected) {
    if(!selected.empty()) return obs_get_source_by_name(selected.c_str());
    auto* parent=obs_filter_get_parent(f->context);
    return parent?obs_source_get_ref(parent):nullptr;
}
obs_source_t* resolve_effect(Filter* f,const Configuration& cfg) {
    // Effects are explicit targets. Never silently write the attached
    // source's own settings when a filter hasn't been selected.
    if(cfg.target_filter.empty())return nullptr;
    auto* parent=resolve_parent(f,cfg.target_source);
    if(!parent)return nullptr;
    auto* target=obs_source_get_filter_by_name(parent,cfg.target_filter.c_str());
    obs_source_release(parent);
    if(target==f->context) {
        if(target)obs_source_release(target);
        return nullptr; // Prevent recursive self-mutation
    }
    // Router-to-router cycles are prevented even with multiple instances.
    if(target && std::string(obs_source_get_unversioned_id(target))==
       "sight_value_router_filter") {
        obs_source_release(target);
        return nullptr;
    }
    return target;
}
// All entries are accessed only on the queued UI mutation path. Never
// hold the router configuration mutex across calls into another filter.
using LeaseKey=std::pair<obs_weak_source_t*,std::string>;
std::map<LeaseKey,Filter*> leases;
std::mutex leases_mutex;
double numeric_read(obs_data_t* s,const Baseline& b) {
    return b.type==OBS_PROPERTY_INT?double(obs_data_get_int(s,b.setting.c_str())):
                                  obs_data_get_double(s,b.setting.c_str());
}
bool equal_numeric(double a,double b,obs_property_type type) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<(type==OBS_PROPERTY_INT?.5:1e-4);
}
void numeric_write(obs_data_t* s,const Baseline& b,double value) {
    if(b.type==OBS_PROPERTY_INT)obs_data_set_int(s,b.setting.c_str(),std::llround(value));
    else obs_data_set_double(s,b.setting.c_str(),value);
}
void release_baseline(Filter* f) {
    auto b=std::move(f->baseline);
    f->baseline={};
    if(!b.target)return;
    // Keep ownership during restoration so a synchronous re-entrant
    // callback cannot acquire a competing lease on the same property.
    if(auto* target=obs_weak_source_get_source(b.target)) {
        auto* s=obs_source_get_settings(target);
        if(b.written&&equal_numeric(numeric_read(s,b),b.last,b.type)){
            if(b.had_user_value)numeric_write(s,b,b.original);
            else obs_data_unset_user_value(s,b.setting.c_str());
            obs_source_update(target,nullptr);
        }
        obs_data_release(s);obs_source_release(target);
    }
    {std::lock_guard lock(leases_mutex);leases.erase({b.target,b.setting});}
    obs_weak_source_release(b.target);
    f->has_baseline=false;
}
bool apply_value(Filter* f,const Configuration& cfg,double value) {
    if(!std::isfinite(value)||cfg.target_setting.empty()||cfg.target_setting.size()>128){
        release_baseline(f);return false;
    }
    auto* target=resolve_effect(f,cfg);
    if(!target){release_baseline(f);return false;}
    auto* properties=obs_source_properties(target);
    auto* property=properties?obs_properties_get(properties,cfg.target_setting.c_str()):nullptr;
    auto type=property?obs_property_get_type(property):OBS_PROPERTY_INVALID;
    double clamped=value;
    if(type==OBS_PROPERTY_FLOAT)
        clamped=std::clamp(value,obs_property_float_min(property),obs_property_float_max(property));
    else if(type==OBS_PROPERTY_INT)
        clamped=std::round(std::clamp(value,double(obs_property_int_min(property)),double(obs_property_int_max(property))));
    if(properties)obs_properties_destroy(properties);
    if(type!=OBS_PROPERTY_FLOAT&&type!=OBS_PROPERTY_INT){
        release_baseline(f);obs_source_release(target);return false;
    }
    auto* weak=obs_source_get_weak_source(target);
    auto& b=f->baseline;
    if(b.target&&(b.target!=weak||b.setting!=cfg.target_setting||b.type!=type))release_baseline(f);
    auto* settings=obs_source_get_settings(target);
    if(!b.target){
        LeaseKey key{weak,cfg.target_setting};
        bool acquired;
        {std::lock_guard lock(leases_mutex);acquired=leases.emplace(key,f).second;}
        if(!acquired){
            obs_data_release(settings);obs_weak_source_release(weak);obs_source_release(target);return false;
        }
        b.target=weak;weak=nullptr;b.setting=cfg.target_setting;b.type=type;
        b.original=numeric_read(settings,b);
        b.had_user_value=obs_data_has_user_value(settings,b.setting.c_str());
        if(!std::isfinite(b.original)){
            {std::lock_guard lock(leases_mutex);leases.erase(key);}
            obs_weak_source_release(b.target);b={};
            obs_data_release(settings);obs_source_release(target);return false;
        }
        f->has_baseline=true;
    }
    const double current=numeric_read(settings,b);
    // Compare the actual setting even when the generated output is steady.
    if(b.written&&!equal_numeric(current,b.last,b.type)&&std::isfinite(current)){
        b.original=current;
        b.had_user_value=obs_data_has_user_value(settings,b.setting.c_str());
    }
    if(!equal_numeric(current,clamped,type)){
        numeric_write(settings,b,clamped);
        b.last=clamped;b.written=true;
        obs_source_update(target,nullptr);
        ++f->applied;
        if(cfg.threshold_gate&&f->applied==1)
            blog(LOG_INFO,"SIGHT_AR_THRESHOLD_GATE_20261118_G419");
        if(f->applied==1)
            blog(LOG_INFO,"[Sight/Router] SIGHT_ROUTER_BASELINE_20261118: First numeric effect update: %.3f",clamped);
    }
    obs_data_release(settings);if(weak)obs_weak_source_release(weak);obs_source_release(target);
    return true;
}
struct Dispatch {Filter* filter;obs_source_t* retained;};
void run_dispatch(void* data) {
    std::unique_ptr<Dispatch> task(static_cast<Dispatch*>(data));
    auto* f=task->filter;
    Configuration c;std::optional<double> v;bool release;
    {
        std::lock_guard lock(f->mutex);
        f->queued=false;
        release=f->release_requested;
        f->release_requested=false;
        c=f->config;
        if(f->pending_revision==f->revision)v=f->pending_value;
        f->pending_value.reset();
    }
    if(release||!obs_source_enabled(f->context)||!obs_filter_get_parent(f->context))
        release_baseline(f);
    else if(v)apply_value(f,c,*v);
    // This reference keeps Filter alive until the queued callback ends.
    obs_source_release(task->retained);
}
void dispatch(Filter* f,std::optional<double> value,bool release,uint64_t rev) {
    bool enqueue=false;
    {
        std::lock_guard lock(f->mutex);
        if(rev&&rev!=f->revision)return;
        if(release){f->release_requested=true;f->pending_value.reset();}
        else if(rev==f->revision){f->pending_value=value;f->pending_revision=rev;}
        if(!f->queued){f->queued=true;enqueue=true;}
    }
    if(enqueue){
        if(auto* ref=obs_source_get_ref(f->context))
            obs_queue_task(OBS_TASK_UI,run_dispatch,new Dispatch{f,ref},false);
        else {std::lock_guard lock(f->mutex);f->queued=false;}
    }
}
// OBS desktop saves synchronously on its UI thread. Saved source
// records retain the settings objects until JSON serialization, so this
// works regardless of filter order within a full scene collection save.
// UI dispatch cannot run until that synchronous save returns.
void save(void* data,obs_data_t*) {
    auto* f=static_cast<Filter*>(data);
    // libobs may serialize from a headless main thread as well as the OBS
    // frontend UI thread. This synchronous callback is the serialization
    // boundary; UI-only gating leaves live modulated values in saved JSON.
    {
        std::lock_guard lock(f->mutex);
        f->pending_value.reset();
        ++f->revision;
        f->throttler=1;
    }
    release_baseline(f);
}
void removed(void* data,obs_source_t*) {
    dispatch(static_cast<Filter*>(data),{},true,0);
}
void enabled_changed(void* data,calldata_t* params) {
    if(!calldata_bool(params,"enabled"))dispatch(static_cast<Filter*>(data),{},true,0);
}
void update(void*,obs_data_t*);
void* create(obs_data_t* settings,obs_source_t* context) {
    auto* f=new Filter;
    f->context=context;
    update(f,settings);
    signal_handler_connect(obs_source_get_signal_handler(context),"enable",enabled_changed,f);
    return f;
}
void destroy(void* data) {
    auto* f=static_cast<Filter*>(data);
    signal_handler_disconnect(obs_source_get_signal_handler(f->context),"enable",enabled_changed,f);
    // Normal detachment has already queued restoration and retains the
    // source until it finishes. Shutdown may destroy an attached filter;
    // release its bookkeeping rather than mutating a tearing-down graph.
    if(f->baseline.target){
        {std::lock_guard lock(leases_mutex);leases.erase({f->baseline.target,f->baseline.setting});}
        obs_weak_source_release(f->baseline.target);
    }
    delete f;
}
void defaults(obs_data_t* s) {
    obs_data_set_default_int(s,MODE,0);
    obs_data_set_default_int(s,"ocr_input_kind",0);
    obs_data_set_default_int(s,"ocr_input_slot",1);
    obs_data_set_default_double(s,"ocr_input_min",0);
    obs_data_set_default_double(s,"ocr_input_max",100);
    obs_data_set_default_int(s,WAVE_TYPE,0);
    obs_data_set_default_double(s,FREQ,1);
    obs_data_set_default_double(s,AMP,1);
    obs_data_set_default_double(s,PHASE,0);
    obs_data_set_default_double(s,PULSE,50);
    obs_data_set_default_int(s,STEPS,4);
    obs_data_set_default_bool(s,INVERT,false);
    obs_data_set_default_string(s,CHANNEL,"face");
    obs_data_set_default_int(s,FEATURE,0);
    obs_data_set_default_int(s,INDEX,1);
    obs_data_set_default_int(s,HAND_INDEX,0);
    obs_data_set_default_int(s,HAND_GESTURE,2);
    obs_data_set_default_int(s,AGE,750);
    obs_data_set_default_string(s,TARGET_SOURCE,"");
    obs_data_set_default_string(s,TARGET_FILTER,"");
    obs_data_set_default_string(s,TARGET_SETTING,"");
    obs_data_set_default_double(s,MIN,0);
    obs_data_set_default_double(s,MAX,100);
    obs_data_set_default_double(s,LIMIT,20);
    obs_data_set_default_int(s,RESPONSE,0);
    obs_data_set_default_double(s,ACTIVATE,.65);
    obs_data_set_default_double(s,RELEASE,.45);
}
void update(void* data,obs_data_t* s) {
    auto* f=static_cast<Filter*>(data);
    Configuration cfg;
    cfg.mode=int(std::clamp<int64_t>(obs_data_get_int(s,MODE),0,2));
    cfg.ocr_filter=obs_data_get_string(s,"ocr_input_filter");
    cfg.ocr_query=obs_data_get_string(s,"ocr_input_query");
    cfg.ocr_slot=int(std::clamp<int64_t>(obs_data_get_int(s,"ocr_input_slot"),1,4));
    cfg.ocr_kind=int(std::clamp<int64_t>(obs_data_get_int(s,"ocr_input_kind"),0,2));
    cfg.ocr_min=obs_data_get_double(s,"ocr_input_min");
    cfg.ocr_max=obs_data_get_double(s,"ocr_input_max");
    cfg.wave.waveform=sight::Waveform(std::clamp<int64_t>(obs_data_get_int(s,WAVE_TYPE),0,10));
    cfg.wave.frequency=std::clamp(obs_data_get_double(s,FREQ),.01,10.);
    cfg.wave.amplitude=std::clamp(obs_data_get_double(s,AMP),0.,2.);
    cfg.wave.phase_offset=obs_data_get_double(s,PHASE)/100.;
    cfg.wave.pulse_width=std::clamp(obs_data_get_double(s,PULSE)/100.,.01,.99);
    cfg.wave.steps=unsigned(std::clamp<int64_t>(obs_data_get_int(s,STEPS),2,64));
    cfg.wave.invert=obs_data_get_bool(s,INVERT);
    cfg.channel=obs_data_get_string(s,CHANNEL);
    cfg.feature=sight::SignalFeature(std::clamp<int64_t>(obs_data_get_int(s,FEATURE),0,10));
    cfg.index=unsigned(std::clamp<int64_t>(obs_data_get_int(s,INDEX),0,477));
    cfg.hand_index=unsigned(std::clamp<int64_t>(obs_data_get_int(s,HAND_INDEX),0,1));
    cfg.hand_gesture=sight::HandGesture(std::clamp<int64_t>(
        obs_data_get_int(s,HAND_GESTURE),0,7));
    cfg.age_ms=unsigned(std::clamp<int64_t>(obs_data_get_int(s,AGE),100,5000));
    cfg.target_source=obs_data_get_string(s,TARGET_SOURCE);
    cfg.target_filter=obs_data_get_string(s,TARGET_FILTER);
    cfg.target_setting=obs_data_get_string(s,TARGET_SETTING);
    cfg.output_min=std::clamp(obs_data_get_double(s,MIN),-100000.,100000.);
    cfg.output_max=std::clamp(obs_data_get_double(s,MAX),-100000.,100000.);
    cfg.max_rate=std::clamp(obs_data_get_double(s,LIMIT),1.,60.);
    cfg.threshold_gate=obs_data_get_int(s,RESPONSE)==1;
    cfg.gate_activate=obs_data_get_double(s,ACTIVATE);
    cfg.gate_release=obs_data_get_double(s,RELEASE);
    bool changed_target=false;
    uint64_t revision;
    {
        std::lock_guard lock(f->mutex);
        changed_target=cfg.target_source!=f->config.target_source||
                       cfg.target_filter!=f->config.target_filter||
                       cfg.target_setting!=f->config.target_setting;
        f->gate.reset();
        f->config=std::move(cfg);
        revision=++f->revision;
        f->throttler=1;
    }
    if(changed_target&&f->has_baseline)dispatch(f,{},true,revision);
}
// Explicit same-parent producer. Never resolve an arbitrary global
// source or read text from a concealment-only filter.
std::optional<double> read_ocr_value(Filter* f,const Configuration& c){
    if(c.ocr_filter.empty()||c.ocr_filter.size()>256||c.ocr_query.size()>512)return {};
    auto* parent=obs_filter_get_parent(f->context);
    if(!parent)return {};
    auto* producer=obs_source_get_filter_by_name(parent,c.ocr_filter.c_str());
    if(!producer)return {};
    std::optional<double> result;
    const char* id=obs_source_get_unversioned_id(producer);
    if(id&&std::string_view(id)=="sight_text_obfuscation_filter"&&obs_source_enabled(producer)){
        calldata_t args;calldata_init(&args);calldata_set_int(&args,"max_age_ms",c.age_ms);
        bool available=false;
        auto* handler=obs_source_get_proc_handler(producer);
        calldata_set_int(&args,"slot",c.ocr_slot);
        if(handler&&proc_handler_call(handler,c.ocr_slot==1?"sight_ocr_output_v1":"sight_ocr_region_v1",&args)&&
            calldata_get_bool(&args,"available",&available)&&available){
            const char* raw=calldata_string(&args,"text");
            if(raw){
                std::string_view text(raw);
                if(c.ocr_kind==0){
                    auto n=sight::ocr_numeric(text);
                    if(n&&std::isfinite(c.ocr_min)&&std::isfinite(c.ocr_max)&&c.ocr_max>c.ocr_min)
                        result=std::clamp((*n-c.ocr_min)/(c.ocr_max-c.ocr_min),0.,1.);
                }else result=sight::ocr_match(text,c.ocr_query,c.ocr_kind==2);
            }
        }
        calldata_free(&args);
    }
    obs_source_release(producer);return result;
}
void tick(void* data,float seconds) {
    auto* f=static_cast<Filter*>(data);
    if(!obs_source_enabled(f->context)||!obs_filter_get_parent(f->context)){
        if(f->has_baseline)dispatch(f,{},true,0);
        return;
    }
    std::optional<double> value;
    uint64_t revision=0;
    bool release=false;
    {
        std::unique_lock lock(f->mutex);
        const auto& c=f->config;
        if(c.mode==0)f->oscillator.advance(seconds,c.wave.frequency);
        f->throttler+=std::clamp<double>(seconds,0,.2);
        if(f->throttler < 1./c.max_rate)return;
        f->throttler=0;
        revision=f->revision;
        if(c.output_max<=c.output_min)release=true;
        else {
            std::optional<double> normalized;
            if(c.mode==0)normalized=f->oscillator.sample(c.wave);
            else if(c.mode==1) {
                sight::TrackingSignal signal;
                if(sight::TrackingSignalBus::read(c.channel,std::chrono::milliseconds(c.age_ms),signal))
                    normalized=sight::read_signal_value(signal,c.feature,c.index,c.hand_index,c.hand_gesture);
            }
            else {
                // Calls into another filter occur outside the router mutex.
                const Configuration copied=c;
                lock.unlock();normalized=read_ocr_value(f,copied);lock.lock();
                if(revision!=f->revision)return;
            }
            if(c.threshold_gate)normalized=f->gate.sample(normalized,c.gate_activate,c.gate_release);
            if(normalized&&std::isfinite(*normalized))
                value=sight::range_map(*normalized,c.output_min,c.output_max);
            else release=true;
        }
    }
    // Coalesce on the UI thread rather than acquiring another source's
    // settings or calling obs_source_update from the video tick.
    if(value||(release&&f->has_baseline))dispatch(f,value,release,revision);
}
void render(void* data,gs_effect_t*) {
    auto* f=static_cast<Filter*>(data);
    obs_source_skip_video_filter(f->context);
}
using namespace sight::discovery;
std::string point_label(int feature,unsigned i){
    // Stable documented topology labels, not inferred semantic labels.
    static constexpr const char* hand[]={
        "Wrist","Thumb CMC","Thumb MCP","Thumb IP","Thumb tip",
        "Index MCP","Index PIP","Index DIP","Index tip",
        "Middle MCP","Middle PIP","Middle DIP","Middle tip",
        "Ring MCP","Ring PIP","Ring DIP","Ring tip",
        "Little MCP","Little PIP","Little DIP","Little tip"};
    static constexpr const char* body[]={
        "Nose","Left eye inner","Left eye","Left eye outer",
        "Right eye inner","Right eye","Right eye outer","Left ear","Right ear",
        "Mouth left","Mouth right","Left shoulder","Right shoulder",
        "Left elbow","Right elbow","Left wrist","Right wrist","Left little finger",
        "Right little finger","Left index","Right index","Left thumb","Right thumb",
        "Left hip","Right hip","Left knee","Right knee","Left ankle","Right ankle",
        "Left heel","Right heel","Left foot index","Right foot index"};
    static constexpr const char* coco[]={
        "Nose","Left eye","Right eye","Left ear","Right ear","Left shoulder","Right shoulder",
        "Left elbow","Right elbow","Left wrist","Right wrist","Left hip","Right hip",
        "Left knee","Right knee","Left ankle","Right ankle","Left big toe","Left small toe",
        "Left heel","Right big toe","Right small toe","Right heel"};
    if(feature>=5&&feature<=6&&i<21)return hand[i];
    if(feature>=3&&feature<=4&&i<33)return body[i];
    if(feature>=9&&feature<=10&&i<133){
        if(i<23)return coco[i];
        if(i<91)return "Face point "+std::to_string(i-23);
        return std::string(i<112?"Left hand: ":"Right hand: ")+hand[(i-91)%21];
    }
    if(feature<=1){
        // Only label the unambiguous commonly used dense face anchors.
        switch(i){
            case 1:return "Nose tip";case 10:return "Forehead";
            case 13:return "Upper inner lip";case 14:return "Lower inner lip";
            case 152:return "Chin";case 33:return "Right eye outer corner";
            case 263:return "Left eye outer corner";
            default:break;
        }
    }
    return {};
}
void fill_inputs(Filter* f,obs_properties_t* props,obs_data_t* settings) {
    auto* channel=obs_properties_get(props,CHANNEL);
    auto* feature=obs_properties_get(props,FEATURE);
    auto* index=obs_properties_get(props,INDEX);
    if(!channel||!feature||!index)return;
    if(auto* ocr=obs_properties_get(props,"ocr_input_filter")){
        obs_property_list_clear(ocr);obs_property_list_add_string(ocr,"Choose an OCR filter…","");
        if(auto* parent=obs_filter_get_parent(f->context))
            obs_source_enum_filters(parent,[](obs_source_t*,obs_source_t* filter,void* data){
                const char* id=obs_source_get_unversioned_id(filter);
                if(id&&std::string_view(id)=="sight_text_obfuscation_filter"){
                    const char* name=obs_source_get_name(filter);
                    obs_property_list_add_string(static_cast<obs_property_t*>(data),name,name);
                }
            },ocr);
        retain_missing(ocr,obs_data_get_string(settings,"ocr_input_filter"));
    }
    const auto choices=discover_channels(f->context);
    const std::string selected=obs_data_get_string(settings,CHANNEL);
    const auto age=std::chrono::milliseconds(std::clamp<int64_t>(obs_data_get_int(settings,AGE),100,5000));
    obs_property_list_clear(channel);
    obs_property_list_add_string(channel,"Choose a tracking channel…","");
    for(const auto& [key,c]:choices){
        const auto item=obs_property_list_add_string(channel,label(key,c,age).c_str(),key.c_str());
        if(c.ambiguous)obs_property_list_item_disable(channel,item,true);
    }
    retain_missing(channel,selected);
    sight::TrackingSignal sample;
    const bool fresh=sight::TrackingSignalBus::read(selected,age,sample);
    const auto found=choices.find(selected);
    const bool known=found!=choices.end()&&!found->second.ambiguous;
    const auto kind=known?found->second.kind:sample.kind;
    const bool typed=known||fresh;
    const bool gestures=known?found->second.gestures:true;
    obs_property_list_clear(feature);
    const int chosen=int(obs_data_get_int(settings,FEATURE));
    bool matched=false;
    const auto add=[&](const char* name,int id,sight::TrackingKind required,bool supported=true){
        if((!typed||kind==required)&&supported){
            obs_property_list_add_int(feature,name,id);
            if(chosen==id)matched=true;
        }
    };
    using K=sight::TrackingKind;
    add("Face point X",0,K::Face);add("Face point Y",1,K::Face);
    add("Face expression score",2,K::Face);
    add("Body point X",3,K::Body);add("Body point Y",4,K::Body);
    add("Hand point X",5,K::Hand);add("Hand point Y",6,K::Hand);
    add("Hand pinch (0=open, 1=closed)",7,K::Hand);
    add("Hand gesture confidence",8,K::Hand,gestures);
    add("WholeBody (COCO 133) point X",9,K::WholeBody);
    add("WholeBody (COCO 133) point Y",10,K::WholeBody);
    if(!matched){
        auto item=obs_property_list_add_int(feature,"Saved value (incompatible with channel)",chosen);
        obs_property_list_item_disable(feature,item,true);
    }
    obs_property_list_clear(index);
    const unsigned count=chosen==2?52:chosen<=1?478:chosen<=4?33:
        chosen<=6?21:chosen<=8?0:133;
    const auto saved=obs_data_get_int(settings,INDEX);
    // Semantic labels use the actual model's category order, not an
    // assumed blendshape index mapping. No detection still preserves indices.
    for(unsigned i=0;i<count;++i){
        std::string label;
        if(chosen==2&&fresh&&sample.kind==K::Face&&i<sample.data.blendshape_names.size())
            label=sample.data.blendshape_names[i];
        if(label.empty()&&chosen!=2)label=point_label(chosen,i);
        if(label.empty())label=chosen==2?"Expression ":"Point ";
        else label+=" · ";
        label+=std::to_string(i);
        obs_property_list_add_int(index,label.c_str(),i);
    }
    if((saved<0||saved>=count)&&count){
        auto item=obs_property_list_add_int(index,("Saved index "+std::to_string(saved)+" (unavailable)").c_str(),saved);
        obs_property_list_item_disable(index,item,true);
    }
    if(!count)obs_property_list_add_int(index,"Not used",saved);
    const bool tracking=obs_data_get_int(settings,MODE)==1;
    obs_property_set_visible(index,tracking&&count>0);
    obs_property_set_visible(obs_properties_get(props,HAND_INDEX),tracking&&chosen>=5&&chosen<=8);
    obs_property_set_visible(obs_properties_get(props,HAND_GESTURE),tracking&&chosen==8);
    blog(LOG_DEBUG,"[Sight/Router] SIGHT_PICKER_DISCOVERY_20261118_P741 channels=%zu",choices.size());
}
bool input_changed(void* priv,obs_properties_t* props,obs_property_t*,obs_data_t* settings){
    if(auto* f=static_cast<Filter*>(priv)){fill_inputs(f,props,settings);return true;}
    return false;
}
bool enum_source_cb(void* ctx,obs_source_t* source) {
    auto* field=static_cast<obs_property_t*>(ctx);
    const char* name=obs_source_get_name(source);
    if(name&&*name&&!list_contains(field,name))
        obs_property_list_add_string(field,name,name);
    return true;
}
void enum_filters_cb(obs_source_t*,obs_source_t* filter,void* ptr) {
    auto* list=static_cast<obs_property_t*>(ptr);
    const char* name=obs_source_get_name(filter);
    if(name&&*name&&std::string(obs_source_get_unversioned_id(filter))!=
                        "sight_value_router_filter")
        obs_property_list_add_string(list,name,name);
}
void add_numeric_options(obs_properties_t* props,obs_property_t* list,
                         const std::string& prefix={}) {
    if(!props)return;
    for(auto* prop=obs_properties_first(props);prop;obs_property_next(&prop)){
        const auto type=obs_property_get_type(prop);
        const char* id=obs_property_name(prop);
        const char* description=obs_property_description(prop);
        if(!id||!*id)continue;
        const std::string label=description&&*description?description:id;
        if(type==OBS_PROPERTY_GROUP)
            add_numeric_options(obs_property_group_content(prop),list,prefix+label+" / ");
        else if(type==OBS_PROPERTY_FLOAT||type==OBS_PROPERTY_INT)
            obs_property_list_add_string(list,(prefix+label).c_str(),id);
    }
}
bool use_target_range(obs_properties_t*,obs_property_t*,void* data) {
    auto* f=static_cast<Filter*>(data);
    if(!f)return false;
    auto* settings=obs_source_get_settings(f->context);
    Configuration cfg;
    cfg.target_source=obs_data_get_string(settings,TARGET_SOURCE);
    cfg.target_filter=obs_data_get_string(settings,TARGET_FILTER);
    cfg.target_setting=obs_data_get_string(settings,TARGET_SETTING);
    auto* target=resolve_effect(f,cfg);
    double lo=0,hi=0;
    if(target){
        auto* props=obs_source_properties(target);
        auto* property=props?obs_properties_get(props,cfg.target_setting.c_str()):nullptr;
        if(property&&obs_property_get_type(property)==OBS_PROPERTY_FLOAT){
            lo=obs_property_float_min(property);hi=obs_property_float_max(property);
        } else if(property&&obs_property_get_type(property)==OBS_PROPERTY_INT){
            lo=obs_property_int_min(property);hi=obs_property_int_max(property);
        }
        if(props)obs_properties_destroy(props);
        obs_source_release(target);
    }
    const bool valid=std::isfinite(lo)&&std::isfinite(hi)&&lo<hi&&lo>=-100000&&hi<=100000;
    if(valid){
        obs_data_set_double(settings,MIN,lo);obs_data_set_double(settings,MAX,hi);
        obs_source_update(f->context,settings);
    }
    obs_data_release(settings);
    return valid;
}
void fill_options(Filter* f,obs_properties_t* properties,obs_data_t* selected) {
    auto* source_field=obs_properties_get(properties,TARGET_SOURCE);
    if(source_field){
        obs_property_list_clear(source_field);
        obs_property_list_add_string(source_field,"Attached source","");
        obs_enum_sources(enum_source_cb,source_field);
        obs_enum_scenes(enum_source_cb,source_field);
        retain_missing(source_field,obs_data_get_string(selected,TARGET_SOURCE));
    }
    auto* filter_field=obs_properties_get(properties,TARGET_FILTER);
    auto* setting_field=obs_properties_get(properties,TARGET_SETTING);
    if(!filter_field||!setting_field)return;
    obs_property_list_clear(filter_field);
    obs_property_list_add_string(filter_field,"Choose a filter…","");
    obs_property_list_clear(setting_field);
    obs_property_list_add_string(setting_field,"Choose numeric setting…","");
    auto* root=resolve_parent(f,obs_data_get_string(selected,TARGET_SOURCE));
    if(!root){
        retain_missing(filter_field,obs_data_get_string(selected,TARGET_FILTER));
        retain_missing(setting_field,obs_data_get_string(selected,TARGET_SETTING));
        return;
    }
    obs_source_enum_filters(root,enum_filters_cb,filter_field);
    const char* filter_name=obs_data_get_string(selected,TARGET_FILTER);
    if(filter_name && *filter_name) {
        auto* target=obs_source_get_filter_by_name(root,filter_name);
        if(target&&std::string(obs_source_get_unversioned_id(target))!="sight_value_router_filter") {
            auto* target_props=obs_source_properties(target);
            if(target_props) {
                add_numeric_options(target_props,setting_field);
                obs_properties_destroy(target_props);
            }
        }
        if(target)obs_source_release(target);
    }
    retain_missing(filter_field,obs_data_get_string(selected,TARGET_FILTER));
    retain_missing(setting_field,obs_data_get_string(selected,TARGET_SETTING));
    obs_source_release(root);
}
bool refresh_options(void* priv,obs_properties_t* props,obs_property_t*,
                     obs_data_t* settings) {
    auto* f=static_cast<Filter*>(priv);
    if(!f)return false;
    fill_options(f,props,settings);
    fill_inputs(f,props,settings);
    return true;
}
bool refresh_all(obs_properties_t* props,obs_property_t*,void* data){
    auto* f=static_cast<Filter*>(data);
    if(!f)return false;
    auto* settings=obs_source_get_settings(f->context);
    fill_options(f,props,settings);fill_inputs(f,props,settings);
    obs_data_release(settings);
    return true;
}
bool mode_change(obs_properties_t* props,obs_property_t*,obs_data_t* settings) {
    const auto input=obs_data_get_int(settings,MODE);
    const bool wave=input==0,tracking=input==1,ocr=input==2;
    const bool number=obs_data_get_int(settings,"ocr_input_kind")==0;
    for(const char* key:{"ocr_input_filter","ocr_input_slot","ocr_input_kind","ocr_input_query","ocr_input_min","ocr_input_max"})
        if(auto* p=obs_properties_get(props,key))
            obs_property_set_visible(p,ocr&&(std::string_view(key)=="ocr_input_filter"||
                std::string_view(key)=="ocr_input_kind"||std::string_view(key)=="ocr_input_slot"||(number!=(std::string_view(key)=="ocr_input_query"))));
    for(const char* key:{WAVE_TYPE,FREQ,AMP,PHASE,PULSE,STEPS,INVERT}) {
        auto* p=obs_properties_get(props,key);
        if(p)obs_property_set_visible(p,wave);
    }
    for(const char* key:{CHANNEL,FEATURE,INDEX,HAND_INDEX,HAND_GESTURE,AGE}) {
        auto* p=obs_properties_get(props,key);
        if(p)obs_property_set_visible(p,std::string_view(key)==AGE?!wave:tracking);
    }
    const auto chosen=obs_data_get_int(settings,FEATURE);
    obs_property_set_visible(obs_properties_get(props,INDEX),tracking&&chosen!=7&&chosen!=8);
    obs_property_set_visible(obs_properties_get(props,HAND_INDEX),tracking&&chosen>=5&&chosen<=8);
    obs_property_set_visible(obs_properties_get(props,HAND_GESTURE),tracking&&chosen==8);
    return true;
}
bool response_change(obs_properties_t* props,obs_property_t*,obs_data_t* settings) {
    const bool gate=obs_data_get_int(settings,RESPONSE)==1;
    for(const char* key:{ACTIVATE,RELEASE})
        if(auto* property=obs_properties_get(props,key))
            obs_property_set_visible(property,gate);
    return true;
}
obs_properties_t* properties(void* data) {
    auto* f=static_cast<Filter*>(data);
    auto* p=obs_properties_create();
    auto* mode=obs_properties_add_list(p,MODE,"Input signal",
                        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(mode,"Wave oscillator (no model required)",0);
    obs_property_list_add_int(mode,"Sight tracking channel",1);
    obs_property_list_add_int(mode,"Sight OCR reading",2);
    obs_property_set_modified_callback(mode,mode_change);
    auto* wave=obs_properties_add_list(p,WAVE_TYPE,"Waveform",
                        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(const auto& [label,idx]:std::vector<std::pair<const char*,int>>{
        {"Sine",0},{"Cosine",1},{"Square",2},{"Pulse",3},
        {"Saw up",4},{"Saw down",5},{"Triangle",6},{"Absolute sine",7},
        {"Steps",8},{"Random hold",9},{"Smooth random",10}})
        obs_property_list_add_int(wave,label,idx);
    obs_properties_add_float_slider(p,FREQ,"Frequency (Hz)",.01,10.,.01);
    obs_properties_add_float_slider(p,AMP,"Amplitude",0.,2.,.01);
    obs_properties_add_float_slider(p,PHASE,"Phase offset (%)",0,100,1);
    obs_properties_add_float_slider(p,PULSE,"Pulse/square width (%)",1,99,1);
    obs_properties_add_int_slider(p,STEPS,"Step count",2,64,1);
    obs_properties_add_bool(p,INVERT,"Invert");
    auto* channel=obs_properties_add_list(p,CHANNEL,"Tracking channel",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(channel,"Choose a tracking channel…","");
    auto* feature=obs_properties_add_list(p,FEATURE,"Tracking value",
                        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(const auto& [label,idx]:std::vector<std::pair<const char*,int>>{
        {"Face point X",0},{"Face point Y",1},{"Face expression score",2},
        {"Body point X",3},{"Body point Y",4},
        {"Hand point X",5},{"Hand point Y",6},{"Hand pinch (0=open, 1=closed)",7},
        {"Hand gesture confidence (selected category)",8},
        {"WholeBody (COCO 133) point X",9},{"WholeBody (COCO 133) point Y",10}})
        obs_property_list_add_int(feature,label,idx);
    auto* point=obs_properties_add_list(p,INDEX,"Point / expression",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    if(!f)for(int i=0;i<478;++i)
        obs_property_list_add_int(point,("Point "+std::to_string(i)).c_str(),i);
    auto* hand=obs_properties_add_list(p,HAND_INDEX,"Hand number",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(hand,"First detected hand",0);
    obs_property_list_add_int(hand,"Second detected hand",1);
    auto* gesture=obs_properties_add_list(p,HAND_GESTURE,
        "Gesture category to match",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(const auto& [label,id]:std::vector<std::pair<const char*,int>>{
        {"Closed fist",1},{"Open palm",2},{"Pointing up",3},
        {"Thumb down",4},{"Thumb up",5},{"Victory",6},{"I love you",7}})
        obs_property_list_add_int(gesture,label,id);
    auto* age=obs_properties_add_int_slider(p,AGE,"Signal expiry (ms)",100,5000,50);
    auto* ocr=obs_properties_add_list(p,"ocr_input_filter","OCR filter on attached source",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(ocr,"Choose an OCR filter…","");
    auto* slot=obs_properties_add_list(p,"ocr_input_slot","Recognition region slot",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(int i=1;i<=4;++i)obs_property_list_add_int(slot,("Region "+std::to_string(i)).c_str(),i);
    auto* kind=obs_properties_add_list(p,"ocr_input_kind","OCR value",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(kind,"Single decimal number",0);
    obs_property_list_add_int(kind,"Contains literal text",1);
    obs_property_list_add_int(kind,"Exact text match",2);
    obs_property_set_modified_callback(kind,mode_change);
    obs_properties_add_text(p,"ocr_input_query","Match text (case-sensitive)",OBS_TEXT_DEFAULT);
    obs_properties_add_float(p,"ocr_input_min","Input minimum",-1000000000,1000000000,.01);
    obs_properties_add_float(p,"ocr_input_max","Input maximum",-1000000000,1000000000,.01);
    obs_property_set_long_description(ocr,"Enable recognized text output in the named OCR filter's Original source mode. Use a dedicated recognition region.");

    auto* response=obs_properties_add_list(p,RESPONSE,"Response",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(response,"Continuous mapping",0);
    obs_property_list_add_int(response,"Threshold gate",1);
    obs_property_set_modified_callback(response,response_change);
    auto* activate=obs_properties_add_float_slider(p,ACTIVATE,"Activate at",0,1,.01);
    auto* release=obs_properties_add_float_slider(p,RELEASE,"Release at",0,1,.01);
    obs_property_set_long_description(activate,"Must be greater than Release at.");
    obs_property_set_long_description(release,"Must be less than Activate at. Missing tracking releases an acquired gate to the output minimum.");
    obs_property_set_visible(activate,false);
    obs_property_set_visible(release,false);

    auto* source=obs_properties_add_list(p,TARGET_SOURCE,
        "Target source (blank = attached source)",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(source,"Attached source","");
    obs_enum_sources(enum_source_cb,source);
    auto* filter=obs_properties_add_list(p,TARGET_FILTER,"Target effect/filter",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    auto* setting=obs_properties_add_list(p,TARGET_SETTING,"Numeric setting",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(filter,"Choose a filter…","");
    obs_property_list_add_string(setting,"Choose numeric setting…","");
    if(f){
        obs_property_set_modified_callback2(channel,input_changed,f);
        obs_property_set_modified_callback2(feature,input_changed,f);
        obs_property_set_modified_callback2(age,input_changed,f);
        obs_property_set_modified_callback2(source,refresh_options,f);
        obs_property_set_modified_callback2(filter,refresh_options,f);
        auto* current=obs_source_get_settings(f->context);
        fill_options(f,p,current);
        fill_inputs(f,p,current);
        mode_change(p,mode,current);
        response_change(p,response,current);
        obs_data_release(current);
    }
    obs_properties_add_button2(p,"refresh_io","Refresh inputs and outputs",refresh_all,f);
    obs_property_set_long_description(setting,
        "Choose a numeric control, including controls inside groups. "
        "Changing a filter to an enum or invalid setting is ignored.");
    auto* range=obs_properties_add_button2(p,"use_target_range","Use setting's full range",use_target_range,f);
    obs_property_set_long_description(range,"Copy the selected setting's actual numeric limits. You can narrow the range below.");
    obs_properties_add_float(p,MIN,"Output minimum",-100000,100000,.01);
    obs_properties_add_float(p,MAX,"Output maximum",-100000,100000,.01);
    obs_properties_add_float_slider(p,LIMIT,"Maximum updates/second",1,60,1);
    return p;
}
const char* get_name(void*) {return "LLL.sight Value Router (Signals + Waves)";}
}

struct obs_source_info sight_value_router_filter_info = {
    .id="sight_value_router_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO,
    .get_name=get_name,
    .create=create,
    .destroy=destroy,
    .get_defaults=defaults,
    .get_properties=properties,
    .update=update,
    .video_tick=tick,
    .video_render=render,
    .save=save,
    .filter_remove=removed,
};
