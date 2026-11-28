#include "core/tasks_runtime_abi.hpp"
#include "common/preferences.hpp"
#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "fast-face-detector.hpp"
#include "core/privacy_coverage.hpp"
#include "core/semantic_detector.hpp"
#include "core/semantic_tracks.hpp"
#include "core/sight_semantic_v1.h"
#include <sstream>
#include <locale>

namespace {
using Clock=std::chrono::steady_clock;
uint64_t millis() {
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now().time_since_epoch()).count());
}
std::string default_model() {
    char* dir=os_get_config_path_ptr("obs-studio/plugin_config/live-latent-lab-sight/models");
    if(!dir)return {};
    std::string path=std::string(dir)+"/face_landmarker.task";bfree(dir);return path;
}
std::string default_runtime() {
    if(!sight::tasks_runtime_host_supported)return {};
    auto* module=obs_current_module();
    const char* path=module?obs_get_module_binary_path(module):nullptr;
    if(!path)return {};
    return (std::filesystem::u8path(path).parent_path()/"libmediapipe.so").string();
}
struct SemanticSettings {
    bool enabled=false,buffered=false,analysis=false;
    std::string model,classes_spec,threshold_spec;
    unsigned classes=0;
    std::vector<int> ids;
    std::vector<sight::semantic::Threshold> thresholds;
    unsigned hold_ms=180;
    float confidence=.35f,padding=.15f;
    bool valid=true;
    bool operator==(const SemanticSettings&) const=default;
};
SemanticSettings semantic_settings(obs_data_t* s) {
    SemanticSettings v;
    v.enabled=obs_data_get_bool(s,"semantic_enabled");
    v.buffered=obs_data_get_bool(s,"semantic_buffered");
    v.analysis=obs_data_get_bool(s,"semantic_analysis_only");
    v.model=obs_data_get_string(s,"semantic_model");
    v.classes_spec=obs_data_get_string(s,"semantic_classes");
    v.classes=std::clamp<int>(obs_data_get_int(s,"semantic_class_count"),0,1024);
    v.confidence=std::clamp(float(obs_data_get_double(s,"semantic_confidence")),.01f,1.f);
    v.padding=std::clamp(float(obs_data_get_double(s,"semantic_padding")),0.f,1.f);
    v.threshold_spec=obs_data_get_string(s,"semantic_thresholds");
    v.hold_ms=std::clamp<int>(obs_data_get_int(s,"semantic_hold_ms"),0,600);
    v.valid=sight::semantic::parse_ids(v.classes_spec,v.ids)&&
        sight::semantic::parse_thresholds(v.threshold_spec,v.thresholds);
    for(int id:v.ids)if(v.classes&&unsigned(id)>=v.classes)v.valid=false;
    return v;
}
struct PrivacyFilter {
    obs_source_t* context=nullptr;
    std::mutex input_mutex,output_mutex;
    std::condition_variable cv;
    std::thread worker;
    bool stopping=false,pending=false;
    uint64_t generation=1;
    std::vector<uint8_t> frame;
    unsigned fw=0,fh=0;
    std::atomic<bool> model_ready{true},wants_frame{true};
    std::atomic<unsigned> cadence{2},input_edge{512};
    std::atomic<float> block_ratio{.09f};
    std::atomic<int> treatment{0};
    uint64_t frame_index=0,next_capture=0;
    std::vector<sight::privacy::Rect> faces;
    std::vector<sight::semantic::Region> regions;
    // Optional fixed text zones until OCR + model classification is available.
    // Protected by input_mutex. No raw text ever enters the GPU mask path.
    std::vector<sight::privacy::Rect> manual_text_zones;
    std::string manual_text_spec;
    SemanticSettings semantic;
    uint64_t observed_at=0;
    bool valid=false,new_result=false;
    uint64_t result_generation=0,semantic_sequence=0;
    // Candidate and approved are separate full-resolution frame buffers.
    // Never combine detections from one frame with pixels from a later one.
    gs_texrender_t *capture=nullptr,*candidate=nullptr,*approved=nullptr;
    unsigned visible_w=0,visible_h=0;
    bool candidate_pending=false,approved_valid=false,approved_has_faces=false;
    uint64_t candidate_generation=0,approved_generation=0;
    uint64_t approved_at=0,captured_at=0;
    unsigned freeze_left=0;
    bool freeze_spent=false;
    bool stage_ready=false;
    uint64_t stage_generation=0;
    gs_stagesurf_t *stage=nullptr;
    unsigned sw=0,sh=0;
    gs_effect_t *effect=nullptr;
    gs_texture_t *mask=nullptr,*text_mask=nullptr,*slate=nullptr;
    uint64_t text_mask_generation=0;
    std::vector<uint8_t> mask_bytes,text_mask_bytes;
    bool ready_mask=false;
    bool first_success=false,logged_empty_after_face=false;
    bool logged_valid_empty=false;
};
// Same normalized source-space coordinate convention as the object snapshot proc.
void semantic_snapshot(void* data,calldata_t* call) {
    auto* f=static_cast<PrivacyFilter*>(data);
    std::scoped_lock lock(f->input_mutex,f->output_mutex);
    const auto now=millis();
    const bool available=f->semantic.enabled&&f->valid&&f->result_generation==f->generation&&
        f->observed_at&&now>=f->observed_at&&now-f->observed_at<=650;
    std::ostringstream json;json.imbue(std::locale::classic());json<<'[';
    if(available)for(size_t i=0;i<f->regions.size();++i) {
        const auto& r=f->regions[i];if(i)json<<',';
        json<<"{\"id\":"<<r.id<<",\"category\":"<<r.category<<",\"score\":"<<r.score
            <<",\"x\":"<<r.box.x0<<",\"y\":"<<r.box.y0<<",\"w\":"<<r.box.x1-r.box.x0
            <<",\"h\":"<<r.box.y1-r.box.y0<<",\"held\":"<<(r.held?"true":"false")<<'}';
    }
    json<<']';calldata_set_string(call,"regions_json",json.str().c_str());
    calldata_set_int(call,"count",available?int64_t(f->regions.size()):0);
    calldata_set_int(call,"source_width",f->visible_w);calldata_set_int(call,"source_height",f->visible_h);
    calldata_set_bool(call,"available",available);
}
std::atomic<uint64_t> semantic_sequence{0};
void semantic_geometry(void* data,calldata_t* call) {
    auto* f=static_cast<PrivacyFilter*>(data);
    auto* out=static_cast<sight_semantic_v1*>(calldata_ptr(call,"snapshot"));
    calldata_set_bool(call,"available",false);
    const auto age=calldata_int(call,"max_age_ms");
    if(!out||out->abi_version!=SIGHT_SEMANTIC_ABI_V1||out->struct_size!=sizeof(*out)||
       age<1||age>650)return;
    out->sequence=0;out->count=0;
    std::scoped_lock lock(f->input_mutex,f->output_mutex);
    const auto now=millis();
    if(!f->semantic.enabled||!f->semantic.analysis||f->semantic.buffered||
       !f->valid||f->result_generation!=f->generation||!f->observed_at||
       now<f->observed_at||now-f->observed_at>uint64_t(age)||
       f->regions.size()>SIGHT_SEMANTIC_MAX_REGIONS)return;
    out->source_width=f->visible_w;out->source_height=f->visible_h;
    out->sequence=f->semantic_sequence;out->count=uint32_t(f->regions.size());
    for(size_t i=0;i<f->regions.size();++i){
        const auto& r=f->regions[i];
        out->regions[i]={r.box.x0,r.box.y0,r.box.x1,r.box.y1,r.score,
            uint32_t(r.category),uint32_t(r.held),0,r.id};
    }
    calldata_set_bool(call,"available",true);
}
void worker(PrivacyFilter* f) {
    FastFaceDetector detector;
    std::unique_ptr<sight::semantic::Detector> semantic;
    SemanticSettings loaded;
    sight::semantic::Tracker tracks;
    uint64_t track_generation=0;
    bool load_ok=false, attempted=false, logged_failure=false, logged_success=false;
    for(;;) {
        std::vector<uint8_t> input;
        unsigned w=0,h=0;
        uint64_t generation=0,captured_at=0;
        SemanticSettings config;
        {
            std::unique_lock lock(f->input_mutex);
            f->cv.wait(lock,[&]{return f->stopping||f->pending;});
            if(f->stopping)return;
            generation=f->generation;captured_at=f->captured_at;config=f->semantic;
            input=std::move(f->frame);
            w=f->fw;h=f->fh;f->pending=false;
        }
        if(!w||!h||input.size()!=size_t(w)*h*4) {
            f->wants_frame=true;
            continue;
        }
        std::vector<FastFace> faces;
        std::vector<sight::privacy::Rect> boxes;
        std::vector<sight::semantic::Region> regions;
        if(track_generation!=generation){tracks.reset();track_generation=generation;}
        bool valid=false;
        if(config.enabled) {
            try {
                if(!attempted||config.model!=loaded.model||config.classes!=loaded.classes) {
                    attempted=true;loaded=config;logged_failure=false;logged_success=false;
                    semantic=std::make_unique<sight::semantic::Detector>();
                    load_ok=!config.model.empty()&&semantic->load(config.model,config.classes);
                }
                if(config.valid&&load_ok)
                    valid=semantic->detect_regions(input.data(),w,h,config.classes,config.ids,
                        config.confidence,config.padding,config.thresholds,regions);
                if(valid) {
                    regions=tracks.update(std::move(regions),captured_at,config.hold_ms);
                    boxes=sight::semantic::boxes_of(regions);
                }
                if(!valid&&!logged_failure) {
                    blog(LOG_WARNING,"[Sight/Semantic] detector unavailable: %s",
                         !config.valid?"invalid class IDs":(semantic?semantic->error().c_str():"no model"));
                    logged_failure=true;
                }
                if(valid&&!logged_success) {
                    blog(LOG_INFO,"[Sight/Semantic] SIGHT_SEMANTIC_REAL_V1 regions=%zu classes=%u",boxes.size(),semantic->class_count());
                    logged_success=true;logged_failure=false;
                }
            } catch(const std::exception& e) {
                if(!logged_failure)blog(LOG_WARNING,"[Sight/Semantic] %s",e.what());
                logged_failure=true;
            }
        } else {
            semantic.reset();attempted=false;
            // libfacedetection consumes BGR; conversion and inference stay off the render thread.
            std::vector<uint8_t> bgr(size_t(w)*h*3);
            for(size_t i=0;i<size_t(w)*h;++i) {
                bgr[i*3]=input[i*4+2];bgr[i*3+1]=input[i*4+1];bgr[i*3+2]=input[i*4];
            }
            bool inference_ok=false;
            faces=detector.detect(bgr.data(),int(w),int(h),int(w*3),&inference_ok);
            boxes=sight::privacy::coverage(faces,w,h);
            valid=inference_ok&&boxes.size()==faces.size();
        }
        {
            std::lock_guard lock(f->input_mutex);
            if(generation!=f->generation)continue;
        }
        {
            std::lock_guard lock(f->output_mutex);
            f->faces=boxes;
            f->regions=std::move(regions);
            f->valid=valid;
            f->observed_at=captured_at;
            f->result_generation=generation;
            f->semantic_sequence=++semantic_sequence;
            f->new_result=true;
        }
        if(!config.enabled && !f->first_success && !faces.empty()) {
            blog(LOG_INFO,"[Sight/PrivacyV2] PRIVACY_V2_REAL_FACES_20261011 faces=%zu",
                 faces.size());
            f->first_success=true;
        }
        if(!config.enabled && valid && faces.empty() && !f->logged_valid_empty) {
            blog(LOG_INFO,"[Sight/PrivacyV2] PRIVACY_V2_VALID_ZERO_FACES_20261011");
            f->logged_valid_empty=true;
        }
        if(!config.enabled && valid && f->first_success && faces.empty() && !f->logged_empty_after_face) {
            blog(LOG_INFO,"[Sight/PrivacyV2] PRIVACY_V2_NO_FACE_AFTER_APPROVAL_20261011");
            f->logged_empty_after_face=true;
        }
        f->wants_frame=true;
    }
}
void defaults(obs_data_t* s) {
    obs_data_set_default_int(s,"privacy_edge",256);
    obs_data_set_default_int(s,"privacy_cadence",4);
    obs_data_set_default_double(s,"privacy_block_ratio",.09);
    obs_data_set_default_string(s,"privacy_text_zones","");
    obs_data_set_default_bool(s,"semantic_enabled",false);
    obs_data_set_default_bool(s,"semantic_buffered",false);
    obs_data_set_default_bool(s,"semantic_analysis_only",false);
    obs_data_set_default_int(s,"semantic_treatment",0);
    obs_data_set_default_string(s,"semantic_model","");
    obs_data_set_default_string(s,"semantic_classes","");
    obs_data_set_default_int(s,"semantic_class_count",0);
    obs_data_set_default_double(s,"semantic_confidence",.35);
    obs_data_set_default_double(s,"semantic_padding",.15);
    obs_data_set_default_string(s,"semantic_thresholds","");
    obs_data_set_default_int(s,"semantic_hold_ms",180);
}
void* create(obs_data_t* settings,obs_source_t* source) {
    auto* f=new PrivacyFilter;f->context=source;
    f->semantic=semantic_settings(settings);
    f->cadence=std::clamp<int>(obs_data_get_int(settings,"privacy_cadence"),1,8);
    f->input_edge=(obs_data_get_int(settings,"privacy_edge")==256)?256:512;
    f->block_ratio=std::clamp(float(obs_data_get_double(settings,"privacy_block_ratio")),.05f,.2f);
    f->treatment=obs_data_get_bool(settings,"semantic_enabled")?std::clamp<int>(obs_data_get_int(settings,"semantic_treatment"),0,2):0;
    f->manual_text_spec=obs_data_get_string(settings,"privacy_text_zones");
    if(!sight::privacy::parse_text_zones(f->manual_text_spec,f->manual_text_zones)) {
        blog(LOG_WARNING,"[Sight/PrivacyV2] Invalid manual text-zone geometry; none applied");
        f->manual_text_spec.clear();
    }
    f->worker=std::thread(worker,f);
    if(auto* procs=obs_source_get_proc_handler(source)){
        proc_handler_add(procs,"void sight_semantic_read_v1(out string regions_json, out int count, out int source_width, out int source_height, out bool available)",semantic_snapshot,f);
        proc_handler_add(procs,"void sight_semantic_geometry_v1(int max_age_ms, ptr snapshot, out bool available)",semantic_geometry,f);
    }
    return f;
}
void update(void* data,obs_data_t* settings) {
    auto* f=static_cast<PrivacyFilter*>(data);
    if(!f)return;
    const auto edge=(obs_data_get_int(settings,"privacy_edge")==256)?256u:512u;
    const std::string zones=obs_data_get_string(settings,"privacy_text_zones");
    f->cadence=std::clamp<int>(obs_data_get_int(settings,"privacy_cadence"),1,8);
    f->block_ratio=std::clamp(float(obs_data_get_double(settings,"privacy_block_ratio")),.05f,.2f);
    f->treatment=obs_data_get_bool(settings,"semantic_enabled")?std::clamp<int>(obs_data_get_int(settings,"semantic_treatment"),0,2):0;
    auto next_semantic=semantic_settings(settings);
    std::lock_guard lock(f->input_mutex);
    if(next_semantic!=f->semantic) {
        f->semantic=std::move(next_semantic);f->pending=false;++f->generation;f->wants_frame=true;
    }
    if(zones!=f->manual_text_spec) {
        auto next=f->manual_text_zones;
        if(sight::privacy::parse_text_zones(zones,next)) {
            f->manual_text_spec=zones;
            f->manual_text_zones=std::move(next);
            f->pending=false;++f->generation;f->wants_frame=true;
        } else blog(LOG_WARNING,"[Sight/PrivacyV2] Invalid manual text-zone geometry; preserving last valid zones");
    }
    if(edge!=f->input_edge) {
        f->input_edge=edge;f->pending=false;++f->generation;
        f->wants_frame=true;
    }
}
void destroy(void* data) {
    auto* f=static_cast<PrivacyFilter*>(data);
    if(!f)return;
    {
        std::lock_guard lock(f->input_mutex);f->stopping=true;
    }
    f->cv.notify_one();if(f->worker.joinable())f->worker.join();
    obs_enter_graphics();
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->candidate)gs_texrender_destroy(f->candidate);
    if(f->approved)gs_texrender_destroy(f->approved);
    if(f->stage)gs_stagesurface_destroy(f->stage);
    if(f->mask)gs_texture_destroy(f->mask);
    if(f->text_mask)gs_texture_destroy(f->text_mask);
    if(f->slate)gs_texture_destroy(f->slate);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();
    delete f;
}
obs_properties_t* properties(void*) {
    auto* p=obs_properties_create();
    auto* semantic_enabled=obs_properties_add_bool(p,"semantic_enabled","Use custom semantic detector");
    obs_property_set_long_description(semantic_enabled,"Replaces face detection with local raw YOLO ONNX region detection. Experimental; missed detections may pass through.");
    obs_properties_add_path(p,"semantic_model","ONNX detector",OBS_PATH_FILE,"ONNX (*.onnx)",sight_preferences::directory().c_str());
    obs_properties_add_int(p,"semantic_class_count","Model class count (0 = auto)",0,1024,1);
    auto* classes=obs_properties_add_text(p,"semantic_classes","Class IDs (empty = all)",OBS_TEXT_DEFAULT);
    obs_property_set_long_description(classes,"Comma-separated zero-based model-specific category IDs. Verify against the model card.");
    obs_properties_add_float_slider(p,"semantic_confidence","Detection confidence",.01,1.,.01);
    auto* thresholds=obs_properties_add_text(p,"semantic_thresholds","Class confidence overrides",OBS_TEXT_DEFAULT);
    obs_property_set_long_description(thresholds,"Optional ID:confidence pairs, e.g. 1:0.20, 3:0.55. Class selection still applies.");
    obs_properties_add_int_slider(p,"semantic_hold_ms","Region persistence (ms)",0,600,20);
    auto* analysis=obs_properties_add_bool(p,"semantic_analysis_only","Analysis only (shader signals; no concealment)");
    obs_property_set_long_description(analysis,"Passes original live video for downstream shaders. Buffered mode takes precedence.");
    auto* buffered=obs_properties_add_bool(p,"semantic_buffered","Approved frames only (buffered)");
    obs_property_set_long_description(buffered,"Optional semantic mode: display only the exact analyzed frame, or a neutral slate while no fresh result is available. Reduces output frame rate. A successful detector can still miss content; not guaranteed broadcast safety.");
    obs_properties_add_float_slider(p,"semantic_padding","Region padding per side",0.,1.,.05);
    auto* treatment=obs_properties_add_list(p,"semantic_treatment","Region treatment",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(treatment,"Mosaic",0);
    obs_property_list_add_int(treatment,"Opaque cover",1);
    obs_property_list_add_int(treatment,"Soft blur (not secure redaction)",2);
    auto* edge=obs_properties_add_list(p,"privacy_edge","Detection input",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(edge,"Balanced (512 px)",512);
    obs_property_list_add_int(edge,"Fast (256 px)",256);
    auto* cadence=obs_properties_add_list(p,"privacy_cadence","Capture cadence",
        OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    for(int n:{1,2,4,8})obs_property_list_add_int(cadence,std::to_string(n).c_str(),n);
    auto* strength=obs_properties_add_float_slider(p,"privacy_block_ratio",
        "Mosaic cell size",.05,.20,.01);
    auto* manual=obs_properties_add_text(p,"privacy_text_zones",
        "Manual text areas (x0,y0,x1,y1; ...)",OBS_TEXT_DEFAULT);
    obs_property_set_long_description(manual,
        "Optional normalized 0..1 rectangles, semicolon-separated. Conceals these "
        "fixed areas even when no face is detected. For chat overlays or known "
        "on-screen text. Not OCR or OpenAI Privacy Filter inference; text outside "
        "the specified areas is not protected.");
    obs_property_set_long_description(strength,
        "Live camera output runs independently of native face detection. "
        "The latest mask is applied to live frames while fresh. When tracking "
        "falls behind, repeat the last previously masked frame for at most "
        "three output frames, then show the live camera without an old mask. "
        "Never replaces the scene with a black privacy screen. "
        "Missed faces can appear unblurred; not guaranteed anonymity.");
    return p;
}
void render(void* data,gs_effect_t*) {
    auto* f=static_cast<PrivacyFilter*>(data);
    auto* target=obs_filter_get_target(f->context);
    const unsigned w=target?obs_source_get_base_width(target):0;
    const unsigned h=target?obs_source_get_base_height(target):0;
    if(!w||!h){obs_source_skip_video_filter(f->context);return;}
    if(!f->effect) {
        char* path=obs_module_file("sight-privacy-conceal.effect");
        if(path) {
            char* error=nullptr;
            f->effect=gs_effect_create_from_file(path,&error);
            if(error){blog(LOG_ERROR,"[Sight/PrivacyV2] Shader: %s",error);bfree(error);}
            bfree(path);
        }
    }

    if(f->effect)
        if(auto* param=gs_effect_get_param_by_name(f->effect,"semantic_treatment"))
            gs_effect_set_int(param,f->treatment.load());

    uint64_t generation=0;
    bool buffered=false,analysis=false;
    std::vector<sight::privacy::Rect> manual_zones;
    {
        std::lock_guard lock(f->input_mutex);
        generation=f->generation;
        buffered=f->semantic.enabled&&f->semantic.buffered;
        analysis=f->semantic.enabled&&f->semantic.analysis&&!buffered;
        manual_zones=f->manual_text_zones;
        if(f->visible_w!=w||f->visible_h!=h) {
            f->visible_w=w;f->visible_h=h;
            generation=++f->generation;
            f->pending=false;f->stage_ready=false;f->freeze_left=0;f->freeze_spent=false;f->wants_frame=true;
        }
    }
    // Opaque text mask is independent of the face detector's approval/freshness.
    // Regenerate only when the source dimensions or manual zones change.
    if(!f->text_mask)f->text_mask=gs_texture_create(256,144,GS_R8,1,nullptr,GS_DYNAMIC);
    if(f->text_mask && f->text_mask_generation!=generation) {
        sight::privacy::paint_mask(f->text_mask_bytes,256,144,manual_zones);
        gs_texture_set_image(f->text_mask,f->text_mask_bytes.data(),256,false);
        f->text_mask_generation=generation;
    }
    // Changing resolution or inference settings invalidates all old frames.
    if(f->candidate_pending&&f->candidate_generation!=generation)
        f->candidate_pending=false;
    if(f->stage_ready&&f->stage_generation!=generation)f->stage_ready=false;
    if(f->approved_generation!=generation)
        f->approved_valid=false;

    bool has_result=false,valid=false,has_faces=false;
    uint64_t result_generation=0,detected_capture_time=0;
    {
        std::lock_guard lock(f->output_mutex);
        has_result=f->new_result;
        if(has_result) {
            valid=f->valid;
            has_faces=!f->faces.empty();
            result_generation=f->result_generation;
            detected_capture_time=f->observed_at;
            sight::privacy::paint_mask(f->mask_bytes,256,144,f->faces);
            f->new_result=false;
        }
    }
    if(has_result&&result_generation==generation) {
        f->candidate_pending=false;
        const auto now=millis();
        // A successful zero-face result is a normal scene: approve this same
        // camera frame with an empty mask so the stream keeps moving.
        // A genuinely failed inference holds the last approved image briefly.
        bool approved=f->candidate&&f->effect&&f->model_ready&&
            sight::privacy::detection_result_usable(
                valid,now,detected_capture_time);
        if(approved) {
            if(!f->mask)f->mask=gs_texture_create(256,144,GS_R8,1,nullptr,GS_DYNAMIC);
            if(!f->mask||f->mask_bytes.empty())approved=false;
            else {
                gs_texture_set_image(f->mask,f->mask_bytes.data(),256,false);
                f->ready_mask=true;
            }
        }
        if(approved) {
            std::swap(f->approved,f->candidate);
            f->approved_valid=true;
            f->approved_has_faces=has_faces;
            f->approved_generation=generation;
            f->approved_at=now;
            f->freeze_left=0;
            f->freeze_spent=false;
        }
        // A failed candidate does not revoke a previously approved image.
        // A separate hold deadline below ensures we eventually fail closed.
    }

    // Capture an entire native-resolution candidate frame, then derive the
    // detector's reduced RGBA image from that exact same GPU texture.
    // Keep displaying the last approved frame while detection is in flight.
    // Read back the prior staged frame, not one rendered on this callback.
    if(f->stage_ready&&f->stage_generation==generation&&f->stage) {
        uint8_t* pixels=nullptr;uint32_t pitch=0;
        if(gs_stagesurface_map(f->stage,&pixels,&pitch)) {
            if(pitch>=f->sw*4) {
                std::vector<uint8_t> frame(size_t(f->sw)*f->sh*4);
                for(unsigned y=0;y<f->sh;++y)
                    std::memcpy(frame.data()+size_t(y)*f->sw*4,
                                pixels+size_t(y)*pitch,size_t(f->sw)*4);
                std::lock_guard lock(f->input_mutex);
                if(f->model_ready&&f->generation==generation&&!f->stopping) {
                    f->frame=std::move(frame);f->fw=f->sw;f->fh=f->sh;
                    f->captured_at=millis();
                    f->pending=true;
                    f->candidate_pending=true;
                    f->candidate_generation=generation;
                    f->cv.notify_one();
                } else f->wants_frame=true;
            } else f->wants_frame=true;
            gs_stagesurface_unmap(f->stage);
        } else f->wants_frame=true;
        f->stage_ready=false;
    }

    if(!f->stage_ready&&!f->candidate_pending&&f->model_ready&&f->wants_frame&&
       f->frame_index>=f->next_capture&&f->wants_frame.exchange(false)) {
        bool submitted=false;
        const unsigned longest=f->input_edge;
        const unsigned cw=std::max(1u,unsigned(uint64_t(w)*longest/std::max(w,h)));
        const unsigned ch=std::max(1u,unsigned(uint64_t(h)*longest/std::max(w,h)));
        if(!f->candidate)f->candidate=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->stage||f->sw!=cw||f->sh!=ch) {
            if(f->stage)gs_stagesurface_destroy(f->stage);
            f->stage=gs_stagesurface_create(cw,ch,GS_RGBA);
            f->sw=cw;f->sh=ch;
        }
        auto* base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if(f->candidate&&f->capture&&f->stage&&base) {
            gs_texrender_reset(f->candidate);
            if(gs_texrender_begin(f->candidate,w,h)) {
                gs_projection_push();gs_matrix_push();gs_matrix_identity();
                gs_ortho(0,float(w),0,float(h),-100,100);
                while(gs_effect_loop(base,"Draw"))obs_source_video_render(target);
                gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->candidate);
                auto* tex=gs_texrender_get_texture(f->candidate);
                if(tex) {
                    gs_texrender_reset(f->capture);
                    if(gs_texrender_begin(f->capture,cw,ch)) {
                        gs_projection_push();gs_matrix_push();gs_matrix_identity();
                        gs_ortho(0,float(w),0,float(h),-100,100);
                        auto* image=gs_effect_get_param_by_name(base,"image");
                        if(image) {
                            gs_effect_set_texture(image,tex);
                            while(gs_effect_loop(base,"Draw"))gs_draw_sprite(tex,0,w,h);
                        }
                        gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
                        auto* reduced=gs_texrender_get_texture(f->capture);
                        if(reduced) {
                            // Queue GPU readback now, map on a later OBS
                            // callback to avoid a same-frame GPU sync stall.
                            gs_stage_texture(f->stage,reduced);
                            f->stage_ready=true;
                            f->stage_generation=generation;
                            submitted=true;
                            f->next_capture=f->frame_index+f->cadence;
                        }
                    }
                }
            }
        }
        if(!submitted)f->wants_frame=true;
    }
    ++f->frame_index;

    if(analysis){obs_source_skip_video_filter(f->context);return;}

    // Default: always deliver a live frame, at OBS's render rate.
    // A native inference result updates only the GPU face mask.
    const auto now=millis();
    auto* approved_tex=f->approved?gs_texrender_get_texture(f->approved):nullptr;
    const bool same_generation=f->approved_generation==generation;
    const bool has_mask=f->model_ready&&f->approved_valid&&
        f->approved_has_faces&&f->ready_mask&&f->effect&&f->mask&&same_generation;
    const bool mask_fresh=has_mask&&now>=f->approved_at&&
        now-f->approved_at<=650;

    // Opt-in stricter mode: never apply detections to unrelated live pixels.
    // Until the candidate is approved (even a valid empty detection), render
    // an opaque neutral slate, not the input source. The default mode below
    // retains the original live-first, no-blackout policy.
    if(buffered) {
        const bool fresh=f->approved_valid&&same_generation&&f->ready_mask&&f->effect&&
            f->mask&&f->text_mask&&approved_tex&&now>=f->approved_at&&now-f->approved_at<=650;
        if(fresh) {
            auto* image=gs_effect_get_param_by_name(f->effect,"image");
            auto* mask_param=gs_effect_get_param_by_name(f->effect,"privacy_mask");
            auto* text_param=gs_effect_get_param_by_name(f->effect,"privacy_text_mask");
            auto* size_param=gs_effect_get_param_by_name(f->effect,"image_size");
            auto* strength=gs_effect_get_param_by_name(f->effect,"block_px");
            if(image&&mask_param&&text_param&&size_param&&strength) {
                vec2 resolution{};vec2_set(&resolution,float(w),float(h));
                gs_effect_set_texture(image,approved_tex);
                gs_effect_set_texture(mask_param,f->mask);
                gs_effect_set_texture(text_param,f->text_mask);
                gs_effect_set_vec2(size_param,&resolution);
                gs_effect_set_float(strength,std::max(12.f,float(std::min(w,h))*f->block_ratio.load()));
                while(gs_effect_loop(f->effect,"Draw"))gs_draw_sprite(approved_tex,0,w,h);
                return;
            }
        }
        if(!f->slate) {
            const uint8_t pixel[]={24,24,24,255};const uint8_t* pixels=pixel;
            f->slate=gs_texture_create(1,1,GS_RGBA,1,&pixels,0);
        }
        auto* base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if(base&&f->slate)if(auto* image=gs_effect_get_param_by_name(base,"image")) {
            gs_effect_set_texture(image,f->slate);
            while(gs_effect_loop(base,"Draw"))gs_draw_sprite(f->slate,0,w,h);
        }
        // A graphics allocation failure must not fall back to raw pixels.
        return;
    }

    if(mask_fresh) {
        f->freeze_left=0;
        auto* mask_param=gs_effect_get_param_by_name(f->effect,"privacy_mask");
        auto* text_param=gs_effect_get_param_by_name(f->effect,"privacy_text_mask");
        auto* size_param=gs_effect_get_param_by_name(f->effect,"image_size");
        auto* strength=gs_effect_get_param_by_name(f->effect,"block_px");
        if(mask_param&&text_param&&f->text_mask&&size_param&&strength &&
           obs_source_process_filter_begin(f->context,GS_RGBA,
                                           OBS_NO_DIRECT_RENDERING)) {
            vec2 resolution{};vec2_set(&resolution,float(w),float(h));
            gs_effect_set_texture(mask_param,f->mask);
            gs_effect_set_texture(text_param,f->text_mask);
            gs_effect_set_vec2(size_param,&resolution);
            gs_effect_set_float(strength,
                std::max(12.f,float(std::min(w,h))*f->block_ratio.load()));
            obs_source_process_filter_end(f->context,f->effect,w,h);
            return;
        }
    }

    // Fixed text regions are not dependent on a slow/failed face detector.
    // Consciously limited: these are MANUAL zones, not live OCR/model output.
    if(!manual_zones.empty()&&f->effect) {
        if(!f->mask)f->mask=gs_texture_create(256,144,GS_R8,1,nullptr,GS_DYNAMIC);
        if(f->mask) {
            sight::privacy::paint_mask(f->mask_bytes,256,144,{});
            gs_texture_set_image(f->mask,f->mask_bytes.data(),256,false);
            auto* mask_param=gs_effect_get_param_by_name(f->effect,"privacy_mask");
            auto* text_param=gs_effect_get_param_by_name(f->effect,"privacy_text_mask");
            auto* size_param=gs_effect_get_param_by_name(f->effect,"image_size");
            auto* strength=gs_effect_get_param_by_name(f->effect,"block_px");
            if(mask_param&&text_param&&f->text_mask&&size_param&&strength &&
               obs_source_process_filter_begin(f->context,GS_RGBA,OBS_NO_DIRECT_RENDERING)) {
                vec2 resolution{};vec2_set(&resolution,float(w),float(h));
                gs_effect_set_texture(mask_param,f->mask);
                gs_effect_set_texture(text_param,f->text_mask);
                gs_effect_set_vec2(size_param,&resolution);
                gs_effect_set_float(strength,
                    std::max(12.f,float(std::min(w,h))*f->block_ratio.load()));
                obs_source_process_filter_end(f->context,f->effect,w,h);
                return;
            }
        }
    }
    // A short, bounded freeze ONLY when a previously masked frame becomes
    // stale, not on every inference. Never freeze before first success.
    if(!mask_fresh&&has_mask&&approved_tex) {
        if(!f->freeze_spent&&now>=f->approved_at&&
           now-f->approved_at<=1000) {
            f->freeze_left=3;f->freeze_spent=true;
        }
        if(f->freeze_left>0) {
            --f->freeze_left;
            auto* image=gs_effect_get_param_by_name(f->effect,"image");
            auto* mask_param=gs_effect_get_param_by_name(f->effect,"privacy_mask");
            auto* text_param=gs_effect_get_param_by_name(f->effect,"privacy_text_mask");
            auto* size_param=gs_effect_get_param_by_name(f->effect,"image_size");
            auto* strength=gs_effect_get_param_by_name(f->effect,"block_px");
            if(image&&mask_param&&text_param&&f->text_mask&&size_param&&strength) {
                vec2 resolution{};vec2_set(&resolution,float(w),float(h));
                gs_effect_set_texture(image,approved_tex);
                gs_effect_set_texture(mask_param,f->mask);
                gs_effect_set_texture(text_param,f->text_mask);
                gs_effect_set_vec2(size_param,&resolution);
                gs_effect_set_float(strength,
                    std::max(12.f,float(std::min(w,h))*f->block_ratio.load()));
                while(gs_effect_loop(f->effect,"Draw"))
                    gs_draw_sprite(approved_tex,0,w,h);
                return;
            }
        }
    }
    // Detector unavailable, shader failed, or freeze budget exhausted:
    // show the ordinary camera video. Never synthesize a black frame.
    obs_source_skip_video_filter(f->context);
}
const char* name(void*) {return "lll.sight Privacy V2 — Conceal All Faces (Experimental)";}
} // namespace

struct obs_source_info sight_privacy_v2_filter_info={
    .id="sight_privacy_v2_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    // The bounded freeze path draws the last approved masked frame itself
    // (gs_effect_loop + gs_draw_sprite) instead of the process_filter bracket,
    // so the filter declares ownership of its draw path.
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=name,.create=create,.destroy=destroy,
    .get_defaults=defaults,.get_properties=properties,
    .update=update,.video_render=render,
};
