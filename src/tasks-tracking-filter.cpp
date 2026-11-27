#include "common/backend-status.hpp"
#include "common/model-diagnostics.hpp"
#include "common/preferences.hpp"
#include <obs-module.h>
#include <callback/proc.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/tasks_tracker.hpp"
#include "core/tracking_mode.hpp"
#ifdef SIGHT_HAVE_RTMW
#include "core/rtmw_tracker.hpp"
#endif
#include "core/holistic_tracker.hpp"
#include "core/object_detector.hpp"
#include "core/object_tracks.hpp"
#include "core/object_overlay.hpp"
#include "core/holistic_overlay_stability.hpp"
#include "core/holistic_runtime_discovery.hpp"
#include "core/hand_signals.hpp"
#include "core/hand_temporal_smoothing.hpp"
#include "core/face_quality.hpp"
#include "core/hand_frame_dedup.hpp"
#include "core/tracking_overlay.hpp"
#include "core/tracking_signal_bus.hpp"
#include "core/depth_temporal_policy.hpp"

namespace {
constexpr const char *MODEL_PATH="model_path";
constexpr const char *RTMW_DETECTOR="rtmw_person_detector";
constexpr const char *RTMW_BACKEND="rtmw_detector_backend";
constexpr const char *RUNTIME_PATH="mediapipe_runtime";
constexpr const char *CADENCE="inference_cadence";
constexpr const char *SMOOTH="temporal_blend";
constexpr const char *FACE_QUALITY="face_quality_mode";
constexpr const char *EDGE="inference_edge";
constexpr const char *CHANNEL="signal_channel";
constexpr const char *DRAW_OVERLAY="draw_overlay";
constexpr const char *HOLISTIC_OVERLAY_MODE="holistic_overlay_mode";
constexpr const char *STABILITY_MODE="holistic_stability_mode";
constexpr const char *MIN_CONFIDENCE="holistic_min_confidence";
constexpr const char *FADE_LOST="holistic_fade_lost";
constexpr const char *PERSON_MASK="holistic_person_mask";
constexpr const char *OBJECT_SCORE="object_min_score";
constexpr const char *FACE_CHANNEL="signal_channel_face";
constexpr const char *BODY_CHANNEL="signal_channel_body";
constexpr const char *HAND_CHANNEL="signal_channel_hand";

uint32_t input_edge(int64_t value) {
    return value==256||value==512||value==640 ? value:512;
}
unsigned cadence(int64_t value) {
    return value==1||value==2||value==4||value==8 ? value:2;
}
int smooth(int64_t value) {return static_cast<int>(std::clamp<int64_t>(value,0,95));}
sight::HolisticStabilityMode stability_mode(int64_t value) {
    switch(value) {
        case 0: return sight::HolisticStabilityMode::Off;
        case 1: return sight::HolisticStabilityMode::Responsive;
        case 3: return sight::HolisticStabilityMode::Stable;
        default: return sight::HolisticStabilityMode::Balanced;
    }
}
sight::HolisticOverlayMode holistic_mode(int64_t value) {
    switch(value) {
        case 1: return sight::HolisticOverlayMode::Face;
        case 2: return sight::HolisticOverlayMode::Body;
        case 3: return sight::HolisticOverlayMode::Hands;
        default: return sight::HolisticOverlayMode::All;
    }
}

std::string models_dir() {
    char *path=os_get_config_path_ptr(
        "obs-studio/plugin_config/live-latent-lab-sight/models");
    if(!path)return {};
    std::string folder=path;
    bfree(path);
    std::error_code ec;
    std::filesystem::create_directories(folder,ec);
    return !ec && std::filesystem::is_directory(folder)?folder:std::string{};
}
std::string own_library_path() {
    Dl_info info{};
    if(!dladdr(reinterpret_cast<void*>(&own_library_path),&info)||!info.dli_fname)
        return {};
    return (std::filesystem::path(info.dli_fname).parent_path()/"libmediapipe.so").string();
}
std::pair<uint32_t,uint32_t> capture_dimensions(uint32_t w,uint32_t h,uint32_t longest) {
    if(!w||!h)return {0,0};
    const auto dimension=[&](uint32_t v){
        return std::max(1u,static_cast<uint32_t>(
            static_cast<uint64_t>(v)*longest/std::max(w,h)));
    };
    return {dimension(w),dimension(h)};
}
using Mode=sight::TrackingMode;
sight::TrackingKind kind_for(Mode mode) {
    switch(mode) {
        case Mode::Body:return sight::TrackingKind::Body;
        case Mode::Hand:return sight::TrackingKind::Hand;
        case Mode::Rtmw:return sight::TrackingKind::WholeBody;
        default:return sight::TrackingKind::Face;
    }
}
// Native graph controls. Keep values as a single immutable request
// snapshot alongside model path and generation, not mutable inference state.
constexpr const char* DETECTION_CONF="detection_confidence";
constexpr const char* PRESENCE_CONF="presence_confidence";
constexpr const char* TRACKING_CONF="tracking_confidence";
constexpr const char* MAX_HANDS="max_hands";
constexpr const char* BLENDSHAPES="publish_blendshapes";
constexpr const char* HEAD_TRANSFORM="publish_head_transform";
constexpr const char* H_FACE_DETECT="face_detection_confidence";
constexpr const char* H_FACE_SUPPRESS="face_suppression_threshold";
constexpr const char* H_FACE_PRESENCE="face_presence_confidence";
constexpr const char* H_HAND_LANDMARKS="hand_landmark_confidence";
constexpr const char* H_POSE_DETECT="pose_detection_confidence";
constexpr const char* H_POSE_SUPPRESS="pose_suppression_threshold";
constexpr const char* H_POSE_PRESENCE="pose_presence_confidence";
constexpr const char* PERSON_CONF="person_confidence";
constexpr const char* MAX_OBJECTS="max_objects";
struct NativeOptions {
    sight::TasksOptions tasks{};
    sight::HolisticOptions holistic{};
    float person_confidence=.35f;
    int max_objects=16;
    bool operator==(const NativeOptions&) const = default;
};
NativeOptions native_options(obs_data_t* d,Mode mode){
    NativeOptions out;
    auto fraction=[&](const char* key){return float(std::clamp<int64_t>(obs_data_get_int(d,key),0,100))/100.f;};
    if(mode==Mode::Face||mode==Mode::Body||mode==Mode::Hand){
        out.tasks.detection=fraction(DETECTION_CONF);
        out.tasks.presence=fraction(PRESENCE_CONF);
        out.tasks.tracking=fraction(TRACKING_CONF);
        if(mode==Mode::Face){
            out.tasks.blendshapes=obs_data_get_bool(d,BLENDSHAPES);
            out.tasks.head_transform=obs_data_get_bool(d,HEAD_TRANSFORM);
        }
        if(mode==Mode::Hand)out.tasks.max_hands=int(std::clamp<int64_t>(obs_data_get_int(d,MAX_HANDS),1,2));
    }else if(mode==Mode::Holistic){
        out.holistic.face_detection=fraction(H_FACE_DETECT);
        out.holistic.face_suppression=fraction(H_FACE_SUPPRESS);
        out.holistic.face_presence=fraction(H_FACE_PRESENCE);
        out.holistic.hand_landmarks=fraction(H_HAND_LANDMARKS);
        out.holistic.pose_detection=fraction(H_POSE_DETECT);
        out.holistic.pose_suppression=fraction(H_POSE_SUPPRESS);
        out.holistic.pose_presence=fraction(H_POSE_PRESENCE);
        out.holistic.blendshapes=obs_data_get_bool(d,BLENDSHAPES);
    }else if(mode==Mode::Rtmw){
        out.person_confidence=float(std::clamp<int64_t>(obs_data_get_int(d,PERSON_CONF),25,95))/100.f;
    }else if(mode==Mode::Object){
        out.max_objects=int(std::clamp<int64_t>(obs_data_get_int(d,MAX_OBJECTS),1,16));
    }
    return out;
}
void native_defaults(obs_data_t* d,Mode mode){
    if(mode==Mode::Face||mode==Mode::Body||mode==Mode::Hand){
        for(auto key:{DETECTION_CONF,PRESENCE_CONF,TRACKING_CONF})obs_data_set_default_int(d,key,50);
        if(mode==Mode::Hand)obs_data_set_default_int(d,MAX_HANDS,2);
        if(mode==Mode::Face){
            obs_data_set_default_bool(d,BLENDSHAPES,true);
            obs_data_set_default_bool(d,HEAD_TRANSFORM,true);
        }
    }else if(mode==Mode::Holistic){
        for(auto key:{H_FACE_DETECT,H_FACE_PRESENCE,H_HAND_LANDMARKS,H_POSE_DETECT,H_POSE_PRESENCE})obs_data_set_default_int(d,key,50);
        for(auto key:{H_FACE_SUPPRESS,H_POSE_SUPPRESS})obs_data_set_default_int(d,key,30);
        obs_data_set_default_bool(d,BLENDSHAPES,true);
    }else if(mode==Mode::Object)obs_data_set_default_int(d,MAX_OBJECTS,16);
    else if(mode==Mode::Rtmw)obs_data_set_default_int(d,PERSON_CONF,35);
}
void add_native_properties(obs_properties_t* p,Mode mode,const std::string& prefix){
    auto percent=[&](const char* key,const char* label,int min=0,int max=100){
        auto* prop=obs_properties_add_int_slider(p,(prefix+key).c_str(),label,min,max,1);
        obs_property_set_long_description(prop,"Native model threshold. Changing this reloads the inference graph on its worker.");
    };
    auto flags=[&](const char* key,const char* label){
        auto* prop=obs_properties_add_bool(p,(prefix+key).c_str(),label);
        obs_property_set_long_description(prop,"Native model output. Disabling removes this signal from the published tracking result; changing reloads the graph.");
    };
    if(mode==Mode::Face||mode==Mode::Body||mode==Mode::Hand){
        percent(DETECTION_CONF,"Detection confidence (%)");
        percent(PRESENCE_CONF,"Landmark presence confidence (%)");
        percent(TRACKING_CONF,"Tracking confidence (%)");
        if(mode==Mode::Hand)obs_properties_add_int(p,(prefix+MAX_HANDS).c_str(),"Maximum hands",1,2,1);
        if(mode==Mode::Face){
            flags(BLENDSHAPES,"Publish expression scores");
            flags(HEAD_TRANSFORM,"Publish head transform");
        }
    }else if(mode==Mode::Holistic){
        percent(H_FACE_DETECT,"Face detection confidence (%)");
        percent(H_FACE_SUPPRESS,"Face suppression threshold (%)");
        percent(H_FACE_PRESENCE,"Face presence confidence (%)");
        percent(H_HAND_LANDMARKS,"Hand landmark confidence (%)");
        percent(H_POSE_DETECT,"Pose detection confidence (%)");
        percent(H_POSE_SUPPRESS,"Pose suppression threshold (%)");
        percent(H_POSE_PRESENCE,"Pose presence confidence (%)");
        flags(BLENDSHAPES,"Publish expression scores");
    }else if(mode==Mode::Object)obs_properties_add_int(p,(prefix+MAX_OBJECTS).c_str(),"Maximum detected objects",1,16,1);
    else if(mode==Mode::Rtmw)percent(PERSON_CONF,"Person detector confidence (%)",25,95);
}
struct TrackingFilter {
    obs_source_t *context=nullptr;
    std::atomic<bool> holistic{false},objects{false};
    Mode mode=Mode::Face; // protected by input_mutex; coherent with generation
    std::atomic<int> object_score{35};
    std::atomic<bool> logged_object_overlay{false};
    std::array<std::string,3> holistic_channels{};
    const uint64_t publisher=sight::TrackingSignalBus::new_publisher_id();
    std::string channel;
    std::atomic<bool> draw_overlay{true};
    std::atomic<sight::HolisticOverlayMode> holistic_overlay_mode{
        sight::HolisticOverlayMode::All};
    std::atomic<sight::HolisticStabilityMode> holistic_stability_mode{
        sight::HolisticStabilityMode::Balanced};
    std::atomic<int> holistic_min_confidence{55};
    std::atomic<bool> holistic_fade_lost{true};
    std::atomic<bool> holistic_person_mask{false};
    std::atomic<bool> logged_person_mask{false};
    std::atomic<uint64_t> holistic_stability_epoch{1};
    std::atomic<bool> logged_holistic_stability_preview{false};
    bool warned_channel_collision=false;
    std::atomic<bool> logged_holistic_overlay_draw{false};
    std::mutex input_mutex;
    std::condition_variable cv;
    std::thread worker;
    bool stopping=false,loading=false,pending=false;
    std::string requested_model,requested_library,requested_detector;
    sight_backend::Status backend_status;
    int requested_detector_backend=0;
    NativeOptions requested_options{};
    uint64_t generation=0;
    std::atomic<uint64_t> dedup_epoch{0};
    // Inference resolution is deliberately lower than the OBS source.
    // Publish native dimensions with normalized landmarks so consumers
    // can validate that they are reading the correct camera framing.
    uint32_t frame_w=0,frame_h=0;
    uint32_t native_w=0,native_h=0;
    std::vector<uint8_t> frame;
    std::atomic<bool> model_ready{false},wants_frame{false};
    std::atomic<unsigned> capture_cadence{2};
    std::atomic<int> smoothing{35};
    std::atomic<int> face_quality_mode{0};
    std::atomic<uint64_t> face_quality_epoch{0};
    std::chrono::steady_clock::time_point frame_captured{};
    std::atomic<uint32_t> capture_edge{512};
    uint64_t render_index=0;
    uint64_t next_capture=0;
    std::mutex output_mutex;
    std::vector<uint8_t> overlay;
    std::vector<sight::ObjectDetection> object_visible;
    std::chrono::steady_clock::time_point object_observed_at{};
    uint32_t object_source_w=0,object_source_h=0;
    uint32_t overlay_w=0,overlay_h=0;
    bool new_overlay=false;
    std::atomic<bool> reset_overlay{false};
    bool first_result=false;
    gs_texrender_t *capture=nullptr,*full_resolution=nullptr;
    gs_stagesurf_t *staging=nullptr;
    uint32_t stage_w=0,stage_h=0;
    gs_texture_t *overlay_texture=nullptr;
    gs_effect_t *effect=nullptr;
};

void object_snapshot_proc(void* ptr,calldata_t* data) {
    auto* f=static_cast<TrackingFilter*>(ptr);
    if(!f)return;
    std::lock_guard lock(f->output_mutex);
    const bool recent=f->object_observed_at!=std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now()-f->object_observed_at<std::chrono::seconds(2);
    const bool available=f->objects.load()&&recent&&f->model_ready.load();
    std::ostringstream json;
    json<<'[';
    if(available) {
        for(size_t i=0;i<f->object_visible.size();++i) {
            const auto& o=f->object_visible[i];
            if(i)json<<',';
            json<<"{\"id\":"<<o.id<<",\"label\":\"";
            for(unsigned char ch:o.label)
                if(std::isalnum(ch)||ch==' '||ch=='_'||ch=='-')json<<char(ch);
            json<<"\",\"score\":"<<std::setprecision(4)<<o.score
                <<",\"x\":"<<o.x<<",\"y\":"<<o.y
                <<",\"w\":"<<o.w<<",\"h\":"<<o.h<<"}";
        }
    }
    json<<']';
    const std::string output=json.str();
    calldata_set_string(data,"objects_json",output.c_str());
    calldata_set_int(data,"count",available?int64_t(f->object_visible.size()):0);
    calldata_set_int(data,"source_width",f->object_source_w);
    calldata_set_int(data,"source_height",f->object_source_h);
    calldata_set_bool(data,"available",available);
}

void withdraw_channels(TrackingFilter* f) {
    if(f->objects)return;
    if(f->holistic) {
        for(const auto& channel:f->holistic_channels)
            sight::TrackingSignalBus::withdraw(channel,f->publisher);
    } else sight::TrackingSignalBus::withdraw(f->channel,f->publisher);
}
void worker_main(TrackingFilter *f) {
    sight::TasksTracker tracker;
#ifdef SIGHT_HAVE_RTMW
    std::unique_ptr<sight::RtmwTracker> rtmw_tracker;
#endif
    sight::HolisticTracker holistic_tracker;
    sight::ObjectDetector object_detector;
    sight::ObjectTracks object_tracks;
    sight::HolisticOverlayStabilizer holistic_stabilizer;
    uint64_t applied_stability_epoch=0;
    using Clock=std::chrono::steady_clock;
    std::vector<sight::TrackedPoint> previous_points;
    sight::HandSmoothingState previous_hands;
    sight::FaceQuality face_quality;
    uint64_t face_quality_epoch=0;
    // Object IDs use their own temporal confirmation/expiry policy.
    // Do not alter that detector's cadence without an object-ID benchmark.
    sight::TrackingFrameDedup frame_dedup;
    Mode loaded_mode=Mode::None;
    uint64_t dedup_epoch=0, dedup_generation=0;
    uint32_t dedup_native_w=0,dedup_native_h=0;
    unsigned repeated_frame_skips = 0;
    int64_t next_timestamp=0;
    double infer_ms=0;
    unsigned measured=0;
    // Diagnostic counts only; never retain camera frames or landmarks.
    // OBS render cadence isn't a measure of unique camera frames.
    Clock::time_point delivery_window_start{};
    unsigned delivery_samples=0, delivery_with_hands=0;
    double delivery_inference_ms=0;
    for(;;) {
        std::vector<uint8_t> frame;
        uint32_t w=0,h=0,native_w=0,native_h=0;
        uint64_t generation=0;
        std::string model,lib,detector_path;
        int detector_backend=0;
        NativeOptions native{};
        bool load=false;
        Mode mode=Mode::None;
        Clock::time_point captured{};
        {
            std::unique_lock lock(f->input_mutex);
            f->cv.wait(lock,[&]{return f->stopping||f->loading||f->pending;});
            if(f->stopping)return;
            generation=f->generation;mode=f->mode;
            if(f->loading) {
                load=true;
                model=f->requested_model;
                lib=f->requested_library;
                detector_path=f->requested_detector;
                detector_backend=f->requested_detector_backend;
                native=f->requested_options;
                f->loading=false;
                f->pending=false;
            } else {
                frame=std::move(f->frame);
                w=f->frame_w;h=f->frame_h;
                native_w=f->native_w;native_h=f->native_h;
                captured=f->frame_captured;
                f->pending=false;
            }
        }
        const auto kind=kind_for(mode);
        const bool holistic=mode==Mode::Holistic,objects=mode==Mode::Object;
        const bool deduplicate=!objects;
        if(load) {
            if(mode!=loaded_mode || mode==Mode::None||model.empty()){
                tracker.unload();holistic_tracker.unload();object_detector.unload();
#ifdef SIGHT_HAVE_RTMW
                rtmw_tracker.reset();
#endif
                frame_dedup=sight::TrackingFrameDedup(holistic?3u:1u);
                loaded_mode=mode;
                blog(LOG_INFO,"[Sight/Tracking] SIGHT_UNIFIED_MODE_20261118_U653 selected=%s previous model released",sight::tracking_mode_id(mode));
            }
#ifdef SIGHT_HAVE_RTMW
            if(mode==Mode::Rtmw&&!rtmw_tracker)rtmw_tracker=std::make_unique<sight::RtmwTracker>();
#endif
#ifdef SIGHT_HAVE_RTMW
            // Auto recognizes the qualified YOLOX filename. Explicit
            // existing EfficientDet paths remain on their original backend;
            // a failed requested backend is never silently substituted.
            const bool use_yolox=detector_backend==1||
                (detector_backend==0&&std::filesystem::path(detector_path).filename()=="yolox_nano.tflite");
#endif
            const bool success=mode!=Mode::None&&!model.empty()&&(
#ifdef SIGHT_HAVE_RTMW
                kind==sight::TrackingKind::WholeBody
                ? rtmw_tracker->load(model,detector_path,lib,use_yolox?sight::RtmwPersonBackend::YoloX:sight::RtmwPersonBackend::EfficientDet,native.person_confidence) :
#endif
                objects
                ? object_detector.load(lib,model,float(f->object_score.load())/100.f,native.max_objects)
                : holistic
                    ? holistic_tracker.load(lib,model,f->holistic_person_mask.load(),native.holistic)
                    : tracker.load(lib,model,kind,native.tasks));
            // Capture the actual load error before a failed engine is destroyed.
            std::string load_error;
            const char* provider="MediaPipe CPU";
#ifdef SIGHT_HAVE_RTMW
            if(mode==Mode::Rtmw){
                provider=use_yolox?"LiteRT GPU (RTMW + YOLOX)":"LiteRT GPU (RTMW) + MediaPipe CPU detector";
                if(!success&&rtmw_tracker)load_error=rtmw_tracker->last_error();
            } else
#endif
            if(!success)load_error=objects?object_detector.last_error():holistic?holistic_tracker.last_error():tracker.last_error();
            if(!success){
                tracker.unload();holistic_tracker.unload();object_detector.unload();
#ifdef SIGHT_HAVE_RTMW
                // Preserve load diagnostics before destroying the failed GPU engine.
                if(rtmw_tracker&&mode==Mode::Rtmw&&!model.empty())
                    blog(LOG_ERROR,"[Sight/RTMW] Load failed: %s",rtmw_tracker->last_error().c_str());
                rtmw_tracker.reset();
#endif
            }
            object_tracks.reset();
            {
                std::lock_guard output_lock(f->output_mutex);
                f->object_visible.clear();
                f->object_observed_at={};
            }
            previous_points.clear();
            previous_hands = {};
            face_quality.reset();
            holistic_stabilizer.reset();
            applied_stability_epoch=0;
            frame_dedup.reset();
            repeated_frame_skips = 0;
            next_timestamp=0;infer_ms=0;measured=0;
            delivery_window_start={};delivery_samples=0;
            delivery_with_hands=0;delivery_inference_ms=0;
            {
                std::lock_guard lock(f->input_mutex);
                if(generation!=f->generation)continue;
                f->backend_status.complete(model,success,provider,load_error,mode==Mode::None);
                f->model_ready=success;
                f->wants_frame=success;
                if(!success)withdraw_channels(f);
            }
            if(success && (holistic||kind==sight::TrackingKind::Face) && !objects)
                blog(LOG_INFO,"[Sight/Face] FACE_QUALITY_20261118_P742 mode=%d",f->face_quality_mode.load());
            if(success && deduplicate)
                blog(LOG_INFO,"[Sight/Tasks] TRACKING_EXACT_DEDUP_20261118_M731 heartbeat=250ms");
            if(success && kind==sight::TrackingKind::Hand &&
               !holistic && !objects) {
                blog(LOG_INFO,"[Sight/Tasks] HAND_IDENTITY_SMOOTH_ACTIVE_20261104_H642");
                blog(LOG_INFO,"[Sight/Tasks] HAND_RENDER_DEDUP_ACTIVE_20261104_Q247");
            }
#ifdef SIGHT_HAVE_RTMW
            if(kind==sight::TrackingKind::WholeBody) {
                if(success){
                    blog(LOG_INFO,"[Sight/RTMW] SIGHT_RTMW_GPU_TRACKER_20261118_R914 persistent GPU model loaded");
                    blog(LOG_INFO,"[Sight/RTMW] RTMW_DETECTOR_BACKEND_20261118_Y927 detector=%s compiled=%zu",
                        use_yolox?"yolox-gpu":"efficientdet-cpu",rtmw_tracker->compilation_count());
                }
                else if(!model.empty()&&rtmw_tracker)blog(LOG_ERROR,"[Sight/RTMW] Load failed: %s",rtmw_tracker->last_error().c_str());
                continue;
            }
#endif
            if(success && objects)
                blog(LOG_INFO,"[Sight/Object] SIGHT_OBJECT_NATIVE_20261113_W742 MediaPipe 1.2.3 detector ready");
            else if(success && holistic)
                blog(LOG_INFO,"[Sight/Holistic] SIGHT_HOLISTIC_NATIVE_V1_20261112_H153 single-graph video tracker loaded");
            else if(success)
                blog(LOG_INFO,"[Sight/Tasks] Loaded %s video tracker: %s",
                    kind==sight::TrackingKind::Face?"face":
                    kind==sight::TrackingKind::Body?"body":"hand",model.c_str());
            else if(!model.empty())
                blog(LOG_ERROR,"[Sight/Tasks] Tracker load failed: %s",
                    objects?object_detector.last_error().c_str():
                    holistic?holistic_tracker.last_error().c_str():tracker.last_error().c_str());
            continue;
        }
        if(!f->model_ready||w==0||h==0||frame.size()!=size_t(w)*h*4) {
            std::lock_guard lock(f->input_mutex);
            if(generation==f->generation&&f->model_ready)f->wants_frame=true;
            continue;
        }
        const auto before=Clock::now();
        const auto quality_epoch=f->face_quality_epoch.load();
        if(quality_epoch!=face_quality_epoch){
            face_quality.reset();
            if(kind==sight::TrackingKind::Face)previous_points.clear();
            face_quality_epoch=quality_epoch;
        }
        const double sample_time=std::chrono::duration<double>(
            (captured==Clock::time_point{}?before:captured).time_since_epoch()).count();
        const auto quality_mode=static_cast<sight::FaceQualityMode>(f->face_quality_mode.load());
        const auto quality_strength=f->smoothing.load()/100.f;
        const auto policy_epoch=f->dedup_epoch.load();
        if(dedup_epoch!=policy_epoch || dedup_generation!=generation ||
           dedup_native_w!=native_w || dedup_native_h!=native_h) {
            frame_dedup.reset();
            dedup_epoch=policy_epoch;
            dedup_generation=generation;
            dedup_native_w=native_w;
            dedup_native_h=native_h;
        }
        if (deduplicate && !frame_dedup.should_infer(frame,w,h,before)) {
            // OBS may redraw precisely the same camera/filter image more
            // often than the physical camera updates. Avoid a duplicate
            // model run, while still allowing a heartbeat every 250ms.
            ++repeated_frame_skips;
            std::lock_guard lock(f->input_mutex);
            if(generation==f->generation&&f->model_ready)
                f->wants_frame=true;
            continue;
        }
        // MediaPipe VIDEO requires strictly increasing timestamps, even if OBS
        // re-renders the same source several times within one millisecond.
        int64_t now=std::chrono::duration_cast<std::chrono::milliseconds>(
            before.time_since_epoch()).count();
        next_timestamp=std::max(now,next_timestamp+1);
#ifdef SIGHT_HAVE_RTMW
        if(kind==sight::TrackingKind::WholeBody) {
            sight::TrackingData result;
            const bool success=rtmw_tracker->detect(frame.data(),w,h,next_timestamp,result);
            const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
            std::lock_guard lock(f->input_mutex);
            if(generation!=f->generation||!f->model_ready)continue;
            f->backend_status.health.record(success,elapsed,Clock::now());
            if(!success) {
                withdraw_channels(f);
                frame_dedup.reset();
                blog(LOG_WARNING,"[Sight/RTMW] Inference failed: %s",rtmw_tracker->last_error().c_str());
            } else {
                frame_dedup.accept(std::move(frame),w,h,before);
                const bool published=sight::TrackingSignalBus::publish(
                    f->channel,f->publisher,kind,result,native_w,native_h);
                if(!published&&!f->warned_channel_collision) {
                    f->warned_channel_collision=true;
                    blog(LOG_WARNING,"[Sight/RTMW] Channel is already owned or invalid: %s",f->channel.c_str());
                } else if(published) {
                    f->warned_channel_collision=false;
                    if(!f->first_result&&!result.landmarks.empty()) {
                        blog(LOG_INFO,"[Sight/RTMW] First real rtmw tracking output: %zu landmarks, compiled sessions=%zu",
                            result.landmarks.size(),rtmw_tracker->compilation_count());
                        f->first_result=true;
                    }
                }
            }
            if(f->draw_overlay) {
                auto pixels=sight::draw_tracking_overlay(w,h,kind,result);
                std::lock_guard output_lock(f->output_mutex);
                f->overlay=std::move(pixels);
                f->overlay_w=w;f->overlay_h=h;f->new_overlay=true;
            }
            f->wants_frame=true;
            continue;
        }
#endif
        if(objects) {
            std::vector<sight::ObjectDetection> detected;
            const bool success=object_detector.detect(
                frame.data(),w,h,next_timestamp,detected);
            const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
            std::lock_guard input_lock(f->input_mutex);
            if(generation!=f->generation || !f->model_ready)continue;
            f->backend_status.health.record(success,elapsed,Clock::now());
            if(!success) {
                blog(LOG_ERROR,"[Sight/Object] Native detection failed: %s",
                     object_detector.last_error().c_str());
                std::lock_guard output_lock(f->output_mutex);
                f->object_visible.clear();f->object_observed_at={};
                f->overlay.clear();f->new_overlay=false;
                f->reset_overlay=true;
                f->wants_frame=true;
                continue;
            }
            auto visible=object_tracks.step(std::move(detected));
            if(!f->first_result&&!visible.empty()) {
                blog(LOG_INFO,"[Sight/Object] SIGHT_OBJECT_FIRST_REAL_20261113_N392 count=%zu first=%s id=%d",
                     visible.size(),visible.front().label.c_str(),visible.front().id);
                f->first_result=true;
            }
            auto pixels=f->draw_overlay
                ? sight::draw_object_overlay(w,h,visible) : std::vector<uint8_t>{};
            {
                std::lock_guard output_lock(f->output_mutex);
                f->object_visible=std::move(visible);
                f->object_observed_at=Clock::now();
                f->object_source_w=native_w;f->object_source_h=native_h;
                if(f->draw_overlay&&!pixels.empty()) {
                    f->overlay=std::move(pixels);
                    f->overlay_w=w;f->overlay_h=h;
                    f->new_overlay=true;
                }
            }
            f->wants_frame=true;
            continue;
        }
        if(holistic) {
            sight::HolisticSignals result;
            sight::HolisticPersonMask person_mask;
            const bool success=holistic_tracker.detect(
                frame.data(),w,h,next_timestamp,result,
                f->holistic_person_mask ? &person_mask : nullptr);
            const auto elapsed=std::chrono::duration<double,std::milli>(
                Clock::now()-before).count();
            std::lock_guard lock(f->input_mutex);
            if(generation!=f->generation || !f->model_ready)continue;
            f->backend_status.health.record(success,elapsed,Clock::now());
            if(!success) {
                withdraw_channels(f);
                // A failed inference cannot leave a frozen Holistic preview.
                f->reset_overlay=true;
                {
                    std::lock_guard out_lock(f->output_mutex);
                    f->overlay.clear();
                    f->new_overlay=false;
                }
                holistic_stabilizer.reset();
                face_quality.reset();
                frame_dedup.reset();
                blog(LOG_ERROR,"[Sight/Holistic] Inference failed: %s",
                    holistic_tracker.last_error().c_str());
                f->wants_frame=true;
                continue;
            }
            if(quality_mode!=sight::FaceQualityMode::Legacy)
                face_quality.apply(result.face,{frame.data(),w,h,frame.size()},sample_time,
                    quality_mode,quality_strength);
            frame_dedup.accept(std::move(frame),w,h,before);
            std::array<sight::TrackingData,3> payload{
                std::move(result.face),std::move(result.body),
                std::move(result.hand)
            };
            const bool published=sight::TrackingSignalBus::publish_group(
                f->holistic_channels,f->publisher,payload,native_w,native_h);
            if(!published && !f->warned_channel_collision) {
                blog(LOG_WARNING,"[Sight/Holistic] Could not publish all three channels (invalid, duplicate, or another writer owns a channel)");
                f->warned_channel_collision=true;
            } else if(published) {
                f->warned_channel_collision=false;
                if(!f->first_result && !payload[0].landmarks.empty() &&
                   !payload[1].landmarks.empty() && !payload[2].hands.empty()) {
                    blog(LOG_INFO,"[Sight/Holistic] First real holistic tracking output: face=%zu body=%zu hands=%zu elapsed_ms=%.1f",
                        payload[0].landmarks.size(),
                        payload[1].landmarks.size(),
                        payload[2].hands.size(),elapsed);
                    f->first_result=true;
                }
            }
            // One model inference, one local preview: the existing Face,
            // Body and two-hand overlay renderer composites its own pixels.
            // Signal bus publication stays independent of this switch.
            if(f->draw_overlay) {
                const auto epoch=f->holistic_stability_epoch.load();
                if(epoch!=applied_stability_epoch) {
                    holistic_stabilizer.reset();
                    applied_stability_epoch=epoch;
                }
                sight::HolisticStabilitySettings stable_options{
                    f->holistic_stability_mode.load(),
                    float(f->holistic_min_confidence.load())/100.f,
                    f->holistic_fade_lost.load()
                };
                // Publication above uses RAW model data. Only this
                // optional, on-source overlay uses stabilized points.
                const sight::HolisticSignals input{
                    payload[0],payload[1],payload[2]
                };
                const bool use_person_mask=f->holistic_person_mask &&
                    stable_options.mode!=sight::HolisticStabilityMode::Off;
                auto display=holistic_stabilizer.update(input,stable_options,
                    use_person_mask ? &person_mask : nullptr);
                if(use_person_mask && person_mask.valid() &&
                   !f->logged_person_mask.exchange(true))
                    blog(LOG_INFO,"[Sight/Holistic] SIGHT_HOLISTIC_PERSON_MASK_20261113_R862 native owned mask %ux%u",
                        person_mask.width,person_mask.height);
                auto overlay=sight::draw_holistic_overlay(w,h,
                    display.face,display.body,display.hand,
                    f->holistic_overlay_mode.load(),
                    stable_options.mode!=sight::HolisticStabilityMode::Off);
                if(stable_options.mode!=sight::HolisticStabilityMode::Off &&
                   !f->logged_holistic_stability_preview.exchange(true))
                    blog(LOG_INFO,"[Sight/Holistic] SIGHT_HOLISTIC_STABILITY_PREVIEW_20261113_K587 display-only mode=%d confidence=%d fade=%d",
                        int(stable_options.mode),
                        f->holistic_min_confidence.load(),int(stable_options.fade_lost));
                if(!overlay.empty()) {
                    std::lock_guard out_lock(f->output_mutex);
                    f->overlay=std::move(overlay);
                    f->overlay_w=w;f->overlay_h=h;
                    f->new_overlay=true;
                }
            }
            f->wants_frame=true;
            continue;
        }
        sight::TrackingData landmarks;
        const bool success=tracker.detect(frame.data(),w,h,next_timestamp,landmarks);
        const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
        if(!success) {
            frame_dedup.reset();
            face_quality.reset();
            previous_points.clear();
            blog(LOG_ERROR,"[Sight/Tasks] Real tracking failed: %s",tracker.last_error().c_str());
            std::lock_guard lock(f->input_mutex);
            if(generation==f->generation&&f->model_ready){
                f->backend_status.health.record(false,elapsed,Clock::now());
                if(kind==sight::TrackingKind::Face){
                    withdraw_channels(f);
                    f->reset_overlay=true;
                    std::lock_guard out_lock(f->output_mutex);
                    f->overlay.clear();f->new_overlay=false;
                }
                f->wants_frame=true;
            }
            continue;
        }
        {
            std::lock_guard lock(f->input_mutex);
            if(generation!=f->generation||!f->model_ready)continue;
            f->backend_status.health.record(true,elapsed,Clock::now());
            if(kind==sight::TrackingKind::Face && quality_mode!=sight::FaceQualityMode::Legacy)
                face_quality.apply(landmarks,{frame.data(),w,h,frame.size()},sample_time,
                    quality_mode,quality_strength);
            frame_dedup.accept(std::move(frame),w,h,before);
            // Hand results are identity-smoothed below, then hand[0]
            // becomes the backwards-compatible primary alias. Smoothing
            // the alias by array position here would cross-contaminate
            // two hands when MediaPipe changes detection order.
            if(kind!=sight::TrackingKind::Hand &&
               (kind!=sight::TrackingKind::Face||quality_mode==sight::FaceQualityMode::Legacy) &&
               previous_points.size()==landmarks.landmarks.size() &&
               !landmarks.landmarks.empty()) {
                const float weight=f->smoothing/100.f;
                for(size_t i=0;i<landmarks.landmarks.size();++i) {
                    auto &p=landmarks.landmarks[i];
                    p.x=p.x*(1-weight)+previous_points[i].x*weight;
                    p.y=p.y*(1-weight)+previous_points[i].y*weight;
                    p.z=p.z*(1-weight)+previous_points[i].z*weight;
                }
            }
            if (kind==sight::TrackingKind::Hand)
                previous_points.clear();
            else
                previous_points=landmarks.landmarks;
            if(kind==sight::TrackingKind::Hand) {
                // Preserve hand identity even when MediaPipe changes
                // the detection array order. An absent side forgets its
                // prior points rather than blending a newly visible hand
                // with coordinates from the opposite side.
                sight::smooth_hands_by_handedness(
                    landmarks.hands,previous_hands,f->smoothing/100.f);
                landmarks.landmarks=landmarks.hands.empty() ?
                    std::vector<sight::TrackedPoint>{}:landmarks.hands[0].landmarks;
            }
            const bool published=sight::TrackingSignalBus::publish(
                f->channel,f->publisher,kind,landmarks,
                native_w,native_h);
            if(!published && !f->warned_channel_collision) {
                blog(LOG_WARNING,
                    "[Sight/Tasks] Cannot publish channel '%s' (another writer or invalid channel)",
                    f->channel.c_str());
                f->warned_channel_collision=true;
            } else if(published) {
                f->warned_channel_collision=false;
                if(kind==sight::TrackingKind::Hand) {
                    const auto delivered_at=Clock::now();
                    if (delivery_window_start==Clock::time_point{})
                        delivery_window_start=delivered_at;
                    ++delivery_samples;
                    if(!landmarks.hands.empty())
                        ++delivery_with_hands;
                    delivery_inference_ms+=elapsed;
                    const double seconds=std::chrono::duration<double>(
                        delivered_at-delivery_window_start).count();
                    if(seconds>=10.0) {
                        blog(LOG_INFO,
                             "[Sight/Tasks] Hand delivery: %.1f results/s, %u/%u containing hands, %.1fms avg inference over %.1fs; skipped=%u byte-identical rendered images (not unique camera frames)",
                             delivery_samples/seconds,
                             delivery_with_hands,delivery_samples,
                             delivery_inference_ms/delivery_samples,seconds,
                             repeated_frame_skips);
                        repeated_frame_skips=0;
                        delivery_window_start=delivered_at;
                        delivery_samples=0;delivery_with_hands=0;
                        delivery_inference_ms=0;
                    }
                }
            }
            // Model output and visual overlay are independent. An invisible
            // producer still updates the bus for any number of receivers.
            if(f->draw_overlay) {
                auto overlay=sight::draw_tracking_overlay(w,h,kind,landmarks);
                if(!overlay.empty()) {
                    std::lock_guard out_lock(f->output_mutex);
                    f->overlay=std::move(overlay);
                    f->overlay_w=w;f->overlay_h=h;
                    f->new_overlay=true;
                }
            }
            infer_ms+=elapsed;
            if(++measured==10)
                blog(LOG_INFO,"[Sight/Tasks] %.1f ms avg inference (%u frames, %s)",
                    infer_ms/measured,measured,
                    kind==sight::TrackingKind::Face?"face":
                    kind==sight::TrackingKind::Body?"body":"hand");
            if(!f->first_result && !landmarks.landmarks.empty()) {
                blog(LOG_INFO,"[Sight/Tasks] First real %s tracking output: %zu landmarks",
                    kind==sight::TrackingKind::Face?"face":
                    kind==sight::TrackingKind::Body?"body":"hand",
                    landmarks.landmarks.size());
                f->first_result=true;
            }
            f->wants_frame=true;
        }
    }
}

void* create_common(obs_data_t *settings, obs_source_t *source,Mode selected) {
    auto *f=new TrackingFilter;
    const bool holistic=selected==Mode::Holistic,objects=selected==Mode::Object;
    f->context=source;f->holistic=holistic;f->objects=objects;
    f->mode=selected;
    f->requested_options=native_options(settings,f->mode);
    if(objects)f->object_score=int(std::clamp<int64_t>(
        obs_data_get_int(settings,OBJECT_SCORE),25,95));
    const char* channel=obs_data_get_string(settings,CHANNEL);
    f->channel=channel ? channel : "";
    if(holistic) {
        for(size_t i=0;i<3;++i) {
            const char* key= i==0?FACE_CHANNEL:i==1?BODY_CHANNEL:HAND_CHANNEL;
            const char* value=obs_data_get_string(settings,key);
            f->holistic_channels[i]=value?value:"";
        }
        f->channel=f->holistic_channels[0];
    }
    f->draw_overlay=obs_data_get_bool(settings,DRAW_OVERLAY);
    f->holistic_overlay_mode=holistic_mode(
        obs_data_get_int(settings,HOLISTIC_OVERLAY_MODE));
    if(holistic) {
        f->holistic_stability_mode=stability_mode(
            obs_data_get_int(settings,STABILITY_MODE));
        f->holistic_min_confidence=int(std::clamp<int64_t>(
            obs_data_get_int(settings,MIN_CONFIDENCE),0,100));
        f->holistic_fade_lost=obs_data_get_bool(settings,FADE_LOST);
        f->holistic_person_mask=obs_data_get_bool(settings,PERSON_MASK);
    }
    const char *model=obs_data_get_string(settings,MODEL_PATH);
    const char *library=obs_data_get_string(settings,RUNTIME_PATH);
    f->requested_model=model?model:"";
    f->requested_library=library?library:"";
    f->requested_detector=obs_data_get_string(settings,RTMW_DETECTOR);
    f->requested_detector_backend=int(std::clamp<int64_t>(obs_data_get_int(settings,RTMW_BACKEND),0,2));
    f->capture_edge=input_edge(obs_data_get_int(settings,EDGE));
    f->capture_cadence=cadence(obs_data_get_int(settings,CADENCE));
    f->smoothing=smooth(obs_data_get_int(settings,SMOOTH));
    f->face_quality_mode=int(std::clamp<int64_t>(obs_data_get_int(settings,FACE_QUALITY),0,2));
    f->backend_status.request(f->requested_model,f->mode==Mode::None);
    f->loading=!f->requested_model.empty();
    ++f->generation;
    f->worker=std::thread(worker_main,f);
    {
        if(auto* procs=obs_source_get_proc_handler(source))
            proc_handler_add(procs,
                "void sight_object_read_v1(out string objects_json, out int count, "
                "out int source_width, out int source_height, out bool available)",
                object_snapshot_proc,f);
    }
    if(f->loading)f->cv.notify_one();
    return f;
}
void destroy(void *data) {
    auto *f=static_cast<TrackingFilter*>(data);
    if(!f)return;
    {
        std::lock_guard lock(f->input_mutex);
        f->stopping=true;
    }
    f->cv.notify_one();
    if(f->worker.joinable())f->worker.join();
    withdraw_channels(f);
    obs_enter_graphics();
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->full_resolution)gs_texrender_destroy(f->full_resolution);
    if(f->staging)gs_stagesurface_destroy(f->staging);
    if(f->overlay_texture)gs_texture_destroy(f->overlay_texture);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();
    delete f;
}
void update_mode(TrackingFilter* f,obs_data_t *settings,Mode selected) {
    const auto kind=kind_for(selected);
    const auto requested=native_options(settings,selected);
    const bool holistic=selected==Mode::Holistic,objects=selected==Mode::Object;
    const char *model=obs_data_get_string(settings,MODEL_PATH);
    const char *library=obs_data_get_string(settings,RUNTIME_PATH);
    const std::string path=model?model:"",lib=library?library:"";
    const std::string detector_path=obs_data_get_string(settings,RTMW_DETECTOR);
    const int detector_backend=int(std::clamp<int64_t>(obs_data_get_int(settings,RTMW_BACKEND),0,2));
    const char* requested_channel=obs_data_get_string(settings,CHANNEL);
    const std::string channel=requested_channel?requested_channel:"";
    const bool overlay_enabled=obs_data_get_bool(settings,DRAW_OVERLAY);
    const auto mode=holistic_mode(
        obs_data_get_int(settings,HOLISTIC_OVERLAY_MODE));
    // Both updates must run when checkbox and mode change at once.
    const bool was_enabled=f->draw_overlay.exchange(overlay_enabled);
    const auto prior_mode=f->holistic_overlay_mode.exchange(mode);
    const bool overlay_changed=was_enabled!=overlay_enabled||prior_mode!=mode;
    // Standalone Face/Body/Gesture filters do not expose these settings.
    // Do not silently alter their established update/reset behavior.
    bool stability_changed=false;
    bool person_mask_changed=false;
    bool object_score_changed=false;
    if(objects) {
        const int conf=int(std::clamp<int64_t>(
            obs_data_get_int(settings,OBJECT_SCORE),25,95));
        object_score_changed=f->object_score.exchange(conf)!=conf;
    }
    if(holistic) {
        const auto new_mode=stability_mode(
            obs_data_get_int(settings,STABILITY_MODE));
        const int new_confidence=int(std::clamp<int64_t>(
            obs_data_get_int(settings,MIN_CONFIDENCE),0,100));
        const bool new_fade=obs_data_get_bool(settings,FADE_LOST);
        // Always evaluate all settings; one OBS update can change multiple.
        const bool preset_changed=
            f->holistic_stability_mode.exchange(new_mode)!=new_mode;
        const bool confidence_changed=
            f->holistic_min_confidence.exchange(new_confidence)!=new_confidence;
        const bool fade_changed=
            f->holistic_fade_lost.exchange(new_fade)!=new_fade;
        stability_changed=preset_changed||confidence_changed||fade_changed;
        const bool enable_mask=obs_data_get_bool(settings,PERSON_MASK);
        person_mask_changed=f->holistic_person_mask.exchange(enable_mask)!=enable_mask;
    }
    std::array<std::string,3> new_channels{};
    if(holistic) {
        for(size_t i=0;i<3;++i) {
            const char* key=i==0?FACE_CHANNEL:i==1?BODY_CHANNEL:HAND_CHANNEL;
            const char* value=obs_data_get_string(settings,key);
            new_channels[i]=value?value:"";
        }
    }
    f->capture_cadence=cadence(obs_data_get_int(settings,CADENCE));
    const int smoothing=smooth(obs_data_get_int(settings,SMOOTH));
    const bool smoothing_changed=f->smoothing.exchange(smoothing)!=smoothing;
    const int quality=int(std::clamp<int64_t>(obs_data_get_int(settings,FACE_QUALITY),0,2));
    const bool quality_changed=f->face_quality_mode.exchange(quality)!=quality;
    if(quality_changed||smoothing_changed)++f->face_quality_epoch;
    const uint32_t edge=input_edge(obs_data_get_int(settings,EDGE));
    {
        std::lock_guard lock(f->input_mutex);
        // Native graph output layout is fixed at graph creation, so
        // toggling the mask requires reloading the same .task model.
        const bool mode_changed=selected!=f->mode;
        if(mode_changed){
            f->model_ready=false;f->wants_frame=false;
            withdraw_channels(f);
            f->mode=selected;f->holistic=holistic;f->objects=objects;
            f->warned_channel_collision=false;
            f->logged_object_overlay=false;f->logged_holistic_overlay_draw=false;
            f->logged_holistic_stability_preview=false;f->logged_person_mask=false;
        }
        const bool needs_load=mode_changed||requested!=f->requested_options||path!=f->requested_model||
            lib!=f->requested_library||person_mask_changed||object_score_changed||
            (kind==sight::TrackingKind::WholeBody&&(detector_path!=f->requested_detector||detector_backend!=f->requested_detector_backend));
        const bool channels_changed=mode_changed||(!objects && (holistic ? new_channels!=f->holistic_channels : channel!=f->channel));
        if(channels_changed||overlay_changed||stability_changed||
           smoothing_changed||quality_changed||needs_load||edge!=f->capture_edge)
            ++f->dedup_epoch;
        if(channels_changed) {
            ++f->face_quality_epoch;
            if(!mode_changed)withdraw_channels(f);
            if(holistic) {
                f->holistic_channels=std::move(new_channels);
                f->channel=f->holistic_channels[0];
            } else f->channel=channel;
            f->warned_channel_collision=false;
        }
        if(holistic &&
           (stability_changed || overlay_changed || person_mask_changed ||
            edge!=f->capture_edge)) {
            // No previous overlay joint/hand is valid after an on/off
            // transition, preset change, or camera inference-scale change.
            ++f->holistic_stability_epoch;
            f->logged_holistic_stability_preview=false;
            f->logged_person_mask=false;
        }
        if(!needs_load&&edge==f->capture_edge) {
            if(overlay_changed || stability_changed) {
                f->reset_overlay=true;
                std::lock_guard output_lock(f->output_mutex);
                f->overlay.clear();f->new_overlay=false;
            }
            return;
        }
        withdraw_channels(f);
        f->capture_edge=edge;
        if(needs_load || !f->model_ready) {
            f->requested_model=path;
            f->backend_status.request(path,selected==Mode::None);
            f->requested_library=lib;
            f->requested_detector=detector_path;
            f->requested_detector_backend=detector_backend;
            f->requested_options=requested;
            f->loading=true;
            f->model_ready=false;
            f->wants_frame=false;
        } else f->wants_frame=true;
        f->pending=false;
        f->backend_status.health={};
        ++f->generation;
        f->first_result=false;
        f->next_capture=0;
        f->reset_overlay=true;
        std::lock_guard output_lock(f->output_mutex);
        f->overlay.clear();f->new_overlay=false;
        f->object_visible.clear();f->object_observed_at={};
    }
    f->cv.notify_one();
}
void defaults_face(obs_data_t *settings) {
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,
        dir.empty()?"":(dir+"/face_landmarker.task").c_str());
    const auto runtime=sight::tracking_runtime_default(
        std::filesystem::path(own_library_path()).parent_path() /
        "live-latent-lab-sight.so");
    obs_data_set_default_string(settings,RUNTIME_PATH,runtime.c_str());
    obs_data_set_default_int(settings,EDGE,512);
    obs_data_set_default_int(settings,CADENCE,2);
    obs_data_set_default_int(settings,SMOOTH,35);
    obs_data_set_default_int(settings,FACE_QUALITY,1);
    obs_data_set_default_bool(settings,DRAW_OVERLAY,true);
    obs_data_set_default_string(settings,CHANNEL,"face");
    native_defaults(settings,Mode::Face);
}
void defaults_hand(obs_data_t *settings) {
    defaults_face(settings);
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,
        dir.empty()?"":(dir+"/gesture_recognizer.task").c_str());
    obs_data_set_default_int(settings,CADENCE,2);
    obs_data_set_default_string(settings,CHANNEL,"hand");
    native_defaults(settings,Mode::Hand);
}
void defaults_body(obs_data_t *settings) {
    defaults_face(settings);
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,
        dir.empty()?"":(dir+"/pose_landmarker.task").c_str());
    obs_data_set_default_int(settings,CADENCE,4);
    obs_data_set_default_string(settings,CHANNEL,"body");
    native_defaults(settings,Mode::Body);
}
void defaults_objects(obs_data_t *settings) {
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,
        dir.empty()?"":(dir+"/efficientdet_lite0.tflite").c_str());
    const auto runtime=sight::holistic_runtime_default(
        std::filesystem::path(own_library_path()).parent_path() /
        "live-latent-lab-sight.so");
    obs_data_set_default_string(settings,RUNTIME_PATH,runtime.c_str());
    obs_data_set_default_int(settings,EDGE,512);
    obs_data_set_default_int(settings,CADENCE,2);
    obs_data_set_default_int(settings,OBJECT_SCORE,35);
    obs_data_set_default_bool(settings,DRAW_OVERLAY,true);
    native_defaults(settings,Mode::Object);
}
void defaults_holistic(obs_data_t *settings) {
    defaults_face(settings);
    obs_data_set_default_int(settings,FACE_QUALITY,0);
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,
        dir.empty()?"":(dir+"/holistic_landmarker.task").c_str());
    // The separately installed verified 1.2.3 runtime is a sibling of our
    // OBS plugin binary. A fresh filter should not make its user browse for
    // a dependency that is already present. Never select the legacy ABI.
    const auto runtime=sight::holistic_runtime_default(
        std::filesystem::path(own_library_path()).parent_path() /
        "live-latent-lab-sight.so");
    obs_data_set_default_string(settings,RUNTIME_PATH,runtime.c_str());
    obs_data_set_default_bool(settings,DRAW_OVERLAY,false);
    obs_data_set_default_int(settings,STABILITY_MODE,2); // Balanced
    obs_data_set_default_int(settings,MIN_CONFIDENCE,55);
    obs_data_set_default_bool(settings,FADE_LOST,true);
    obs_data_set_default_bool(settings,PERSON_MASK,false);
    obs_data_set_default_string(settings,FACE_CHANNEL,"holistic/face");
    obs_data_set_default_string(settings,BODY_CHANNEL,"holistic/body");
    obs_data_set_default_string(settings,HAND_CHANNEL,"holistic/hand");
    native_defaults(settings,Mode::Holistic);
}
#ifdef SIGHT_HAVE_RTMW
void defaults_rtmw(obs_data_t* settings) {
    defaults_objects(settings);
    const auto dir=models_dir();
    obs_data_set_default_string(settings,MODEL_PATH,dir.empty()?"":(dir+"/rtmw_wholebody.tflite").c_str());
    std::error_code ec;
    const auto yolo=dir+"/yolox_nano.tflite";
    const bool gpu_model=!dir.empty()&&std::filesystem::is_regular_file(yolo,ec)&&!ec;
    obs_data_set_default_string(settings,RTMW_DETECTOR,dir.empty()?"":gpu_model?yolo.c_str():(dir+"/efficientdet_lite0.tflite").c_str());
    obs_data_set_default_int(settings,RTMW_BACKEND,0);
    obs_data_set_default_string(settings,CHANNEL,"wholebody");
    obs_data_set_default_int(settings,CADENCE,1);
    native_defaults(settings,Mode::Rtmw);
}
void add_rtmw_properties(obs_properties_t* p,const std::string& prefix) {
    auto key=[&](const char* k){return prefix+k;};
    const auto dir=sight_preferences::browse(models_dir());
    auto* model=obs_properties_add_path(p,key(MODEL_PATH).c_str(),"RTMW-m WholeBody model",OBS_PATH_FILE,"TFLite model (*.tflite)",dir.c_str());
    sight_diagnostics::add<sight_diagnostics::Format::Tflite>(p,model);
    obs_property_set_long_description(model,"133 COCO WholeBody 2D points. Strict LiteRT GPU acceleration; no CPU fallback.");
    auto* backend=obs_properties_add_list(p,key(RTMW_BACKEND).c_str(),"Person detector backend",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(backend,"Auto (recognized model filename)",0);
    obs_property_list_add_int(backend,"YOLOX-Nano (LiteRT GPU)",1);
    obs_property_list_add_int(backend,"EfficientDet-Lite0 (MediaPipe CPU)",2);
    obs_property_set_long_description(backend,"Auto selects GPU for yolox_nano.tflite and otherwise the MediaPipe detector. Explicit selection supports alternate filenames. No silent fallback.");
    auto* detector=obs_properties_add_path(p,key(RTMW_DETECTOR).c_str(),"Person detector model",OBS_PATH_FILE,"TFLite model (*.tflite)",dir.c_str());
    sight_diagnostics::add<sight_diagnostics::Format::Tflite>(p,detector);
    obs_property_set_long_description(detector,"Use the qualified YOLOX-Nano 416x416 raw-head LiteRT model or a compatible MediaPipe detector. Changing backend/model reloads the tracker.");
    auto* runtime=obs_properties_add_path(p,key(RUNTIME_PATH).c_str(),"MediaPipe 1.2.3 runtime (CPU detector only)",OBS_PATH_FILE,"Shared library (*.so)",sight_preferences::browse(std::filesystem::path(own_library_path()).parent_path().string(),true).c_str());
    sight_diagnostics::add<sight_diagnostics::Format::OptionalMediaPipe>(p,runtime);
    auto* edge=obs_properties_add_list(p,key(EDGE).c_str(),"Tracking input resolution",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(int n:{256,512,640})obs_property_list_add_int(edge,std::to_string(n).c_str(),n);
    auto* cadence=obs_properties_add_list(p,key(CADENCE).c_str(),"Capture no more often than every N OBS frames",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(int n:{1,2,4,8})obs_property_list_add_int(cadence,std::to_string(n).c_str(),n);
    obs_properties_add_text(p,key(CHANNEL).c_str(),"Publish COCO WholeBody channel",OBS_TEXT_DEFAULT);
    obs_properties_add_bool(p,key(DRAW_OVERLAY).c_str(),"Show tracking overlay here");

    add_native_properties(p,Mode::Rtmw,prefix);
}
#endif
void add_properties(obs_properties_t* p,obs_data_t* settings,Mode mode,const std::string& prefix) {
    const bool holistic=mode==Mode::Holistic,objects=mode==Mode::Object;
    const auto kind=kind_for(mode);
    auto key=[&](const char* k){return prefix+k;};
    const auto dir=models_dir();
    std::string default_dir=sight_preferences::browse(dir);
    if(settings) {
        std::error_code ec;
        const std::filesystem::path model_path=obs_data_get_string(settings,key(MODEL_PATH).c_str());
        if(std::filesystem::is_regular_file(model_path,ec))
            default_dir=model_path.parent_path().string();
    }
    auto *model=obs_properties_add_path(p,key(MODEL_PATH).c_str(),
        objects?"MediaPipe Object Detector model":"MediaPipe Tasks .task model",
        OBS_PATH_FILE,objects?"TFLite model (*.tflite)":"MediaPipe task (*.task)",
        default_dir.c_str());
    if(objects) sight_diagnostics::add<sight_diagnostics::Format::Tflite>(p,model);
    else sight_diagnostics::add<sight_diagnostics::Format::Task>(p,model);
    obs_property_set_long_description(model,objects
        ? "Choose a MediaPipe-compatible object detector with TFLite metadata, "
          "such as EfficientDet-Lite0. No model is bundled. Missing or invalid "
          "models leave the source unchanged and never fabricate detections."
        : "Face, Pose or Gesture Recognizer model as appropriate. "
          "A missing or invalid model produces unmodified video, never simulated landmarks.");
    const auto runtime_dir=std::filesystem::path(own_library_path())
                               .parent_path().string();
    auto *runtime=obs_properties_add_path(p,key(RUNTIME_PATH).c_str(),
        (holistic||objects) ? "MediaPipe 1.2.3 runtime (auto-detected)"
                         : "MediaPipe Tasks C runtime",
        OBS_PATH_FILE,"Shared library (*.so)",sight_preferences::browse(runtime_dir,true).c_str());
    if(holistic||objects) sight_diagnostics::add<sight_diagnostics::Format::MediaPipe123>(p,runtime);
    else sight_diagnostics::add<sight_diagnostics::Format::MediaPipe>(p,runtime);
    obs_property_set_long_description(runtime, (holistic||objects)
        ? "Sight automatically selects the verified 1.2.3 runtime installed "
          "beside this plugin. Browse only to override it or if no compatible "
          "runtime was found. The older MediaPipe ABI is not Holistic-compatible."
        : "Face, Body and Gesture trackers support the verified legacy and "
          "official 1.2.3 runtimes.");
    auto *edge=obs_properties_add_list(p,key(EDGE).c_str(),"Tracking input resolution",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(edge,"Fast (256 px)",256);
    obs_property_list_add_int(edge,"Balanced (512 px)",512);
    obs_property_list_add_int(edge,"Detail (640 px)",640);
    auto *cadence_property=obs_properties_add_list(p,key(CADENCE).c_str(),
        "Capture no more often than every N OBS frames",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(int n:{1,2,4,8})obs_property_list_add_int(cadence_property,
        std::to_string(n).c_str(),n);
    if(!holistic&&!objects)
        obs_properties_add_int_slider(p,key(SMOOTH).c_str(),
            "Temporal landmark smoothing (%)",0,95,5);
    if(!objects && (holistic||kind==sight::TrackingKind::Face)) {
        auto* quality=obs_properties_add_list(p,key(FACE_QUALITY).c_str(),"Face tracking quality",
            OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
        obs_property_list_add_int(quality,holistic?"Raw (existing Holistic signals)":"Classic smoothing",0);
        obs_property_list_add_int(quality,"Adaptive (recommended)",1);
        obs_property_list_add_int(quality,"Precision (adaptive + eyelid refinement)",2);
        obs_property_set_long_description(quality,
            "Adaptive reduces stationary jitter while following fast motion. Precision also corrects visible upper lids using the captured image. No new inference model; expression scores and transform unchanged.");
        if(holistic)obs_properties_add_int_slider(p,key(SMOOTH).c_str(),"Face smoothing strength (%)",0,95,5);
    }
    if(objects) {
        obs_properties_add_int_slider(p,key(OBJECT_SCORE).c_str(),
            "Minimum object confidence (%)",25,95,5);
        auto* overlay=obs_properties_add_bool(p,key(DRAW_OVERLAY).c_str(),
            "Show object boxes and persistent IDs here");
        obs_property_set_long_description(overlay,
            "Draw category labels and short-lived stable IDs. A new object "
            "must be seen in two detections. The read-only source procedure "
            "sight_object_read_v1 also exposes JSON metadata. "
            "Changing confidence reloads the inference graph.");
    } else if(holistic) {
        obs_properties_add_text(p,key(FACE_CHANNEL).c_str(),"Publish face channel",OBS_TEXT_DEFAULT);
        obs_properties_add_text(p,key(BODY_CHANNEL).c_str(),"Publish body channel",OBS_TEXT_DEFAULT);
        obs_properties_add_text(p,key(HAND_CHANNEL).c_str(),"Publish hands channel",OBS_TEXT_DEFAULT);
        auto* overlay=obs_properties_add_bool(p,key(DRAW_OVERLAY).c_str(),
            "Show tracking overlay here");
        obs_property_set_long_description(overlay,
            "Draw Holistic face/body/hand landmarks directly on this source "
            "while continuing to publish all three tracking channels.");
        auto* mode=obs_properties_add_list(p,key(HOLISTIC_OVERLAY_MODE).c_str(),
            "Overlay contents",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
        obs_property_list_add_int(mode,"All",0);
        obs_property_list_add_int(mode,"Face",1);
        obs_property_list_add_int(mode,"Body",2);
        obs_property_list_add_int(mode,"Hands (both)",3);
        auto* preset=obs_properties_add_list(p,key(STABILITY_MODE).c_str(),
            "Tracking stability (overlay only)",
            OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
        obs_property_list_add_int(preset,"Off (raw)",0);
        obs_property_list_add_int(preset,"Responsive",1);
        obs_property_list_add_int(preset,"Balanced",2);
        obs_property_list_add_int(preset,"Stable",3);
        obs_property_set_long_description(preset,
            "Confirm new limbs over several model frames and reject sudden "
            "implausible jumps. Changes only this filter's visible overlay; "
            "published tracking channels stay raw.");
        auto* threshold=obs_properties_add_int_slider(p,key(MIN_CONFIDENCE).c_str(),
            "Minimum landmark confidence (%)",0,100,5);
        obs_property_set_long_description(threshold,
            "Filters low-visibility pose joints. Hand landmarks may not "
            "provide confidence, so hand stability also uses repeated "
            "detections and distance to the corresponding body wrist.");
        auto* mask=obs_properties_add_bool(p,key(PERSON_MASK).c_str(),
            "Use person mask to suppress background limbs");
        obs_property_set_long_description(mask,
            "Optional overlay-only person mask from the same Holistic model. "
            "Checks that elbow, wrist, and hand detections fall near a "
            "credible person silhouette. Masks sometimes miss extended "
            "hands, so gating is conservative and absent/unreliable masks "
            "leave the original tracking intact. Turning this on or off "
            "restarts the Holistic inference graph.");
        auto* fade=obs_properties_add_bool(p,key(FADE_LOST).c_str(),
            "Fade temporarily lost joints");
        obs_property_set_long_description(fade,
            "Briefly fade missing landmarks before clearing them. Does not "
            "freeze or alter the published raw tracking signal.");
    } else {
        obs_properties_add_text(p,key(CHANNEL).c_str(),"Publish tracking channel",OBS_TEXT_DEFAULT);
        auto* overlay=obs_properties_add_bool(p,key(DRAW_OVERLAY).c_str(),"Show tracking overlay here");
        obs_property_set_long_description(overlay,
            "Uncheck to publish landmarks and expression signals invisibly. "
            "A separate LLL.sight Tracking Signal Receiver can display this channel. "
            "Channel names must be unique for each producer.");
    }

    add_native_properties(p,mode,prefix);
}
constexpr const char* string_keys[]={MODEL_PATH,RUNTIME_PATH,RTMW_DETECTOR,CHANNEL,FACE_CHANNEL,BODY_CHANNEL,HAND_CHANNEL};
constexpr const char* int_keys[]={CADENCE,SMOOTH,EDGE,FACE_QUALITY,HOLISTIC_OVERLAY_MODE,STABILITY_MODE,MIN_CONFIDENCE,OBJECT_SCORE,RTMW_BACKEND,DETECTION_CONF,PRESENCE_CONF,TRACKING_CONF,MAX_HANDS,H_FACE_DETECT,H_FACE_SUPPRESS,H_FACE_PRESENCE,H_HAND_LANDMARKS,H_POSE_DETECT,H_POSE_SUPPRESS,H_POSE_PRESENCE,PERSON_CONF,MAX_OBJECTS};
constexpr const char* bool_keys[]={DRAW_OVERLAY,FADE_LOST,PERSON_MASK,BLENDSHAPES,HEAD_TRANSFORM};
void mode_defaults(obs_data_t* d,Mode mode){
    switch(mode){
    case Mode::Face:defaults_face(d);break;
    case Mode::Body:defaults_body(d);break;
    case Mode::Hand:defaults_hand(d);break;
    case Mode::Holistic:defaults_holistic(d);break;
    case Mode::Object:defaults_objects(d);break;
#ifdef SIGHT_HAVE_RTMW
    case Mode::Rtmw:defaults_rtmw(d);break;
#endif
    default:break;
    }
}
void unified_defaults(obs_data_t* d){
    obs_data_set_default_string(d,"tracking_mode","face");
    for(auto mode:sight::tracking_modes){
        auto* tmp=obs_data_create();mode_defaults(tmp,mode);
        for(auto k:string_keys)if(obs_data_has_default_value(tmp,k))
            obs_data_set_default_string(d,sight::tracking_mode_key(mode,k).c_str(),obs_data_get_string(tmp,k));
        for(auto k:int_keys)if(obs_data_has_default_value(tmp,k))
            obs_data_set_default_int(d,sight::tracking_mode_key(mode,k).c_str(),obs_data_get_int(tmp,k));
        for(auto k:bool_keys)if(obs_data_has_default_value(tmp,k))
            obs_data_set_default_bool(d,sight::tracking_mode_key(mode,k).c_str(),obs_data_get_bool(tmp,k));
        obs_data_release(tmp);
    }
}
obs_data_t* unified_selection(obs_data_t* d,Mode mode){
    auto* tmp=obs_data_create();
    if(mode==Mode::None)return tmp;
    for(auto k:string_keys)
        obs_data_set_string(tmp,k,obs_data_get_string(d,sight::tracking_mode_key(mode,k).c_str()));
    for(auto k:int_keys)
        obs_data_set_int(tmp,k,obs_data_get_int(d,sight::tracking_mode_key(mode,k).c_str()));
    for(auto k:bool_keys)
        obs_data_set_bool(tmp,k,obs_data_get_bool(d,sight::tracking_mode_key(mode,k).c_str()));
    return tmp;
}
void* create_unified(obs_data_t* d,obs_source_t* source){
    auto mode=sight::parse_tracking_mode(obs_data_get_string(d,"tracking_mode"));
    auto* tmp=unified_selection(d,mode);
    auto* f=create_common(tmp,source,mode);
    obs_data_release(tmp);return f;
}
void update_unified(void* data,obs_data_t* d){
    auto mode=sight::parse_tracking_mode(obs_data_get_string(d,"tracking_mode"));
    auto* tmp=unified_selection(d,mode);update_mode(static_cast<TrackingFilter*>(data),tmp,mode);obs_data_release(tmp);
}
void unified_visibility(obs_properties_t* p,Mode selected){
    for(auto mode:sight::tracking_modes){
        const auto prefix=std::string(sight::tracking_mode_id(mode))+"_";
        for(auto* field=obs_properties_first(p);field;obs_property_next(&field)){
            const std::string key=obs_property_name(field);
            if(key.rfind(prefix,0)==0)obs_property_set_visible(field,mode==selected);
        }
    }
}
bool unified_changed(obs_properties_t* p,obs_property_t*,obs_data_t* d){
    unified_visibility(p,sight::parse_tracking_mode(obs_data_get_string(d,"tracking_mode")));return true;
}
obs_properties_t* unified_properties(void* data){
    auto* p=obs_properties_create();
    auto* selector=obs_properties_add_list(p,"tracking_mode","Tracker",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(selector,"Off (release model)","none");
    auto* settings=data?obs_source_get_settings(static_cast<TrackingFilter*>(data)->context):nullptr;
    for(auto mode:sight::tracking_modes){
        const char* label=mode==Mode::Face?"Face (MediaPipe)":mode==Mode::Body?"Body (MediaPipe)":
          mode==Mode::Hand?"Hands & gestures (MediaPipe)":mode==Mode::Holistic?"Holistic (face, body, hands)":
          mode==Mode::Object?"Objects (MediaPipe)":"WholeBody (RTMW GPU)";
        obs_property_list_add_string(selector,label,sight::tracking_mode_id(mode));
        const auto prefix=std::string(sight::tracking_mode_id(mode))+"_";
#ifdef SIGHT_HAVE_RTMW
        if(mode==Mode::Rtmw){add_rtmw_properties(p,prefix);continue;}
#endif
        add_properties(p,settings,mode,prefix);
    }
    if(settings)obs_data_release(settings);
    auto selected=Mode::Face;
    if(data){
        auto* f=static_cast<TrackingFilter*>(data);
        auto* d=obs_source_get_settings(f->context);
        const char* saved=obs_data_get_string(d,"tracking_mode");
        selected=sight::parse_tracking_mode(saved);
        if(selected==Mode::None&&strcmp(saved,"none")!=0){
            auto n=obs_property_list_add_string(selector,"Saved tracker (unavailable)",saved);
            obs_property_list_item_disable(selector,n,true);
        }
        obs_data_release(d);
    }
    obs_property_set_modified_callback(selector,unified_changed);
    unified_visibility(p,selected);
    sight_backend::add(p,static_cast<TrackingFilter*>(data));
    obs_properties_add_text(p,"unified_note","One model pipeline at a time. Switching releases the previous model and its channels. Add another Tracking filter to run trackers simultaneously.",OBS_TEXT_INFO);
    return p;
}
const char* name_unified(void*){return "LLL.sight Tracking";}
void render(void *data,gs_effect_t*) {
    auto *f=static_cast<TrackingFilter*>(data);
    auto *target=obs_filter_get_target(f->context);
    if(!target||!obs_filter_get_parent(f->context)){
        obs_source_skip_video_filter(f->context);return;
    }
    const auto width=obs_source_get_base_width(target);
    const auto height=obs_source_get_base_height(target);
    if(!width||!height){obs_source_skip_video_filter(f->context);return;}
    if(f->reset_overlay.exchange(false)&&f->overlay_texture){
        gs_texture_destroy(f->overlay_texture);f->overlay_texture=nullptr;
    }
    if(f->draw_overlay && !f->effect) {
        char *path=obs_module_file("sight-tracking-overlay.effect");
        if(path) {
            char *errors=nullptr;
            f->effect=gs_effect_create_from_file(path,&errors);
            if(errors){blog(LOG_ERROR,"[Sight/Tasks] Overlay shader: %s",errors);bfree(errors);}
            bfree(path);
        }
    }
    if(f->draw_overlay && !f->effect){
        obs_source_skip_video_filter(f->context);return;
    }
    const auto [pw,ph]=capture_dimensions(width,height,f->capture_edge);
    ++f->render_index;
    if(f->model_ready&&f->wants_frame&&
       lll::depth_capture_due(f->render_index,f->next_capture)&&
       f->wants_frame.exchange(false)) {
        bool submitted=false;
        if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->staging||f->stage_w!=pw||f->stage_h!=ph) {
            if(f->staging)gs_stagesurface_destroy(f->staging);
            f->staging=gs_stagesurface_create(pw,ph,GS_RGBA);
            f->stage_w=pw;f->stage_h=ph;
        }
        if(f->capture&&f->staging) {
            gs_texrender_reset(f->capture);
            if(gs_texrender_begin(f->capture,pw,ph)) {
                gs_projection_push();gs_matrix_push();gs_matrix_identity();
                gs_ortho(0,float(width),0,float(height),-100,100);
                auto* effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
                if(effect)while(gs_effect_loop(effect,"Draw"))obs_source_video_render(target);
                gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
                auto* tex=gs_texrender_get_texture(f->capture);
                if(tex) {
                    gs_stage_texture(f->staging,tex);
                    uint8_t *pixels=nullptr;uint32_t pitch=0;
                    if(gs_stagesurface_map(f->staging,&pixels,&pitch)) {
                        if(pitch>=pw*4) {
                            std::vector<uint8_t> rgba(size_t(pw)*ph*4);
                            for(uint32_t y=0;y<ph;y++)
                                std::memcpy(rgba.data()+size_t(y)*pw*4,
                                    pixels+size_t(y)*pitch,size_t(pw)*4);
                            std::lock_guard lock(f->input_mutex);
                            if(f->model_ready&&!f->loading) {
                                f->frame_captured=std::chrono::steady_clock::now();
                                f->frame=std::move(rgba);
                                f->frame_w=pw;f->frame_h=ph;
                                f->native_w=width;f->native_h=height;
                                f->pending=true;submitted=true;
                                f->next_capture=lll::next_depth_capture(
                                    f->render_index,f->capture_cadence);
                                f->cv.notify_one();
                            }
                        }
                        gs_stagesurface_unmap(f->staging);
                    }
                }
            }
        }
        if(!submitted&&f->model_ready)f->wants_frame=true;
    }
    {
        std::lock_guard lock(f->output_mutex);
        if(f->new_overlay) {
            if(!f->overlay_texture ||
                gs_texture_get_width(f->overlay_texture)!=f->overlay_w ||
                gs_texture_get_height(f->overlay_texture)!=f->overlay_h) {
                if(f->overlay_texture)gs_texture_destroy(f->overlay_texture);
                f->overlay_texture=gs_texture_create(f->overlay_w,f->overlay_h,
                    GS_RGBA,1,nullptr,GS_DYNAMIC);
            }
            if(f->overlay_texture&&!f->overlay.empty())
                gs_texture_set_image(f->overlay_texture,f->overlay.data(),
                    f->overlay_w*4,false);
            f->new_overlay=false;
        }
    }
    if(!f->draw_overlay || !f->model_ready||!f->overlay_texture){
        obs_source_skip_video_filter(f->context);return;
    }
    if(!f->full_resolution)
        f->full_resolution=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(!f->full_resolution){obs_source_skip_video_filter(f->context);return;}
    gs_texrender_reset(f->full_resolution);
    if(!gs_texrender_begin(f->full_resolution,width,height)) {
        obs_source_skip_video_filter(f->context);return;
    }
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0,float(width),0,float(height),-100,100);
    auto *original=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if(original)while(gs_effect_loop(original,"Draw"))obs_source_video_render(target);
    gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->full_resolution);
    auto *texture=gs_texrender_get_texture(f->full_resolution);
    auto *source_param=gs_effect_get_param_by_name(f->effect,"original_frame");
    auto *overlay_param=gs_effect_get_param_by_name(f->effect,"tracking_overlay");
    if(!texture||!source_param||!overlay_param){
        obs_source_skip_video_filter(f->context);return;
    }
    gs_effect_set_texture(source_param,texture);
    gs_effect_set_texture(overlay_param,f->overlay_texture);
    while(gs_effect_loop(f->effect,"Draw"))gs_draw_sprite(texture,0,width,height);
    if(f->objects && !f->logged_object_overlay.exchange(true))
        blog(LOG_INFO,"[Sight/Object] SIGHT_OBJECT_GPU_OVERLAY_20261113_D668 composite frame drawn");
    if(f->holistic && !f->logged_holistic_overlay_draw.exchange(true))
        blog(LOG_INFO,"[Sight/Holistic] SIGHT_HOLISTIC_GPU_OVERLAY_20261112_J728 composite frame drawn");
}
} // namespace

struct obs_source_info sight_unified_tracking_filter_info = {
    .id="sight_unified_tracking_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=name_unified,.create=create_unified,.destroy=destroy,
    .get_defaults=unified_defaults,.get_properties=unified_properties,
    .update=update_unified,.video_render=render,
};
