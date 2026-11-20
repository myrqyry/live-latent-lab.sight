#include <obs-module.h>
#include "core/analysis_capture.hpp"
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include "core/dense_motion.hpp"
#include "core/camera_stabilizer.hpp"
namespace {
using Clock=std::chrono::steady_clock;
std::atomic<uint64_t> next_sequence{0};
struct Frame {
    std::vector<uint8_t> rgba;
    unsigned w=0,h=0,sw=0,sh=0;
    uint64_t revision=0;
    Clock::time_point captured{};
};
struct Filter {
    obs_source_t* context=nullptr;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    Frame frame;
    sight_flow_v1 published{};
    sight::CameraCorrection published_correction{},render_correction{};
    Clock::time_point render_correction_time{};
    uint64_t render_correction_revision=0;
    unsigned correction_w=0,correction_h=0;
    Clock::time_point published_time{};
    uint64_t published_revision=0;
    bool pending=false,stopping=false;
    std::atomic<uint64_t> revision{1};
    std::atomic<bool> wants{true};
    std::atomic<int> view{0},edge{320},interval{50},smoothing_ms{500},crop_percent{6};
    gs_texrender_t* capture=nullptr;
    gs_stagesurf_t* stage=nullptr;
    gs_texture_t* texture=nullptr;
    gs_effect_t* effect=nullptr;
    unsigned cw=0,ch=0,sw=0,sh=0;
    uint64_t stage_revision=0,uploaded_sequence=0;
    Clock::time_point captured{},next_capture{};
    bool staged=false,announced=false,stabilization_announced=false;
};
void worker(Filter* f) {
    sight::DenseMotion engine;
    sight::CameraStabilizer stabilizer;
    cv::Mat previous;
    Frame metadata;
    for(;;) {
        Frame frame;
        int mode=0,smooth=500,crop=6;
        {
            std::unique_lock lock(f->mutex);
            f->cv.wait(lock,[&]{return f->stopping||f->pending;});
            if(f->stopping)return;
            frame=std::move(f->frame);f->pending=false;
            mode=f->view.load();smooth=f->smoothing_ms.load();crop=f->crop_percent.load();
        }
        sight_flow_v1 result{};
        cv::Mat gray(int(frame.h),int(frame.w),CV_8U);
        for(size_t i=0;i<gray.total();++i) {
            const auto* p=frame.rgba.data()+i*4;
            gray.data[i]=uint8_t((77u*p[0]+150u*p[1]+29u*p[2]+128u)>>8);
        }
        const auto dt=frame.captured-metadata.captured;
        if(metadata.revision==frame.revision&&metadata.sw==frame.sw&&metadata.sh==frame.sh&&
           dt>Clock::duration::zero()&&dt<=std::chrono::milliseconds(750)&&
           engine.estimate(previous,gray,result)) {
            result.abi_version=SIGHT_FLOW_ABI_V1;result.struct_size=sizeof(result);
            result.source_width=frame.sw;result.source_height=frame.sh;result.sequence=++next_sequence;
            result.interval_seconds=std::chrono::duration<float>(dt).count();
        }
        sight::CameraCorrection correction{};
        if(result.sequence&&mode==3)
            correction=stabilizer.step(result,frame.w,frame.h,smooth/1000.f,crop/100.f);
        else stabilizer.reset();
        previous=std::move(gray);
        metadata.w=frame.w;metadata.h=frame.h;metadata.sw=frame.sw;metadata.sh=frame.sh;
        metadata.revision=frame.revision;metadata.captured=frame.captured;
        {
            std::lock_guard lock(f->mutex);
            if(f->revision==frame.revision) {
                f->published=result;f->published_correction=correction;
                f->published_time=frame.captured;f->published_revision=frame.revision;
            }
            f->wants=true;
        }
    }
}
void defaults(obs_data_t* s) {
    obs_data_set_default_int(s,"flow_view",0);
    obs_data_set_default_int(s,"flow_edge",320);
    obs_data_set_default_int(s,"flow_interval",50);
    obs_data_set_default_int(s,"flow_stabilize_ms",500);
    obs_data_set_default_int(s,"flow_crop_percent",6);
}
void update(void* data,obs_data_t* s) {
    auto* f=static_cast<Filter*>(data);
    const int mode=std::clamp(int(obs_data_get_int(s,"flow_view")),0,3);
    const int smooth=std::clamp(int(obs_data_get_int(s,"flow_stabilize_ms")),50,2000);
    const int crop=std::clamp(int(obs_data_get_int(s,"flow_crop_percent")),0,15);
    std::lock_guard lock(f->mutex);
    int edge=std::clamp(int(obs_data_get_int(s,"flow_edge")),160,640);
    int interval=std::clamp(int(obs_data_get_int(s,"flow_interval")),33,500);
    const int old_edge=f->edge.exchange(edge),old_interval=f->interval.exchange(interval);
    const int old_mode=f->view.exchange(mode);
    const int old_smooth=f->smoothing_ms.exchange(smooth),old_crop=f->crop_percent.exchange(crop);
    if(old_edge!=edge||old_interval!=interval||(old_mode==3)!=(mode==3)||
       (mode==3&&(old_smooth!=smooth||old_crop!=crop)))++f->revision;
}
bool snapshot(Filter* f,sight_flow_v1& out,int age,sight::CameraCorrection* correction=nullptr,
              Clock::time_point* captured=nullptr,bool* busy=nullptr) {
    std::unique_lock lock(f->mutex,std::try_to_lock);
    if(busy)*busy=!lock.owns_lock();
    if(!lock||!obs_source_enabled(f->context)||!f->published.sequence||
       f->published_revision!=f->revision.load()||Clock::now()-f->published_time>std::chrono::milliseconds(age))return false;
    out=f->published;
    if(correction)*correction=f->published_correction;
    if(captured)*captured=f->published_time;
    return true;
}
void read_flow(void* data,calldata_t* args) {
    auto* f=static_cast<Filter*>(data);
    calldata_set_bool(args,"available",false);
    long long age=0;void* ptr=nullptr;
    if(!calldata_get_ptr(args,"snapshot",&ptr)||!ptr||!calldata_get_int(args,"max_age_ms",&age)||age<1||age>1000)return;
    auto* out=static_cast<sight_flow_v1*>(ptr);
    if(out->abi_version!=SIGHT_FLOW_ABI_V1||out->struct_size!=sizeof(*out))return;
    out->sequence=0;
    auto* target=obs_filter_get_target(f->context);
    if(!target||!snapshot(f,*out,int(age))||out->source_width!=obs_source_get_base_width(target)||
       out->source_height!=obs_source_get_base_height(target)) {out->sequence=0;return;}
    calldata_set_bool(args,"available",true);
}
void* create(obs_data_t* s,obs_source_t* context) {
    auto* f=new Filter;f->context=context;update(f,s);
    obs_enter_graphics();
    char* path=obs_module_file("sight-spatial-motion.effect");
    if(path){char* error=nullptr;f->effect=gs_effect_create_from_file(path,&error);
        if(error){blog(LOG_ERROR,"[Sight/Flow] %s",error);bfree(error);}bfree(path);}
    obs_leave_graphics();
    proc_handler_add(obs_source_get_proc_handler(context),"void sight_flow_read_v1(in int max_age_ms, in ptr snapshot, out bool available)",read_flow,f);
    f->worker=std::thread(worker,f);
    return f;
}
void destroy(void* data) {
    auto* f=static_cast<Filter*>(data);
    {std::lock_guard lock(f->mutex);f->stopping=true;}
    f->cv.notify_one();if(f->worker.joinable())f->worker.join();
    obs_enter_graphics();
    if(f->stage)gs_stagesurface_destroy(f->stage);
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->texture)gs_texture_destroy(f->texture);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();delete f;
}
obs_properties_t* properties(void*) {
    auto* p=obs_properties_create();
    auto* v=obs_properties_add_list(p,"flow_view","Output",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(v,"Original source (share motion)",0);
    obs_property_list_add_int(v,"Direction and magnitude",1);
    obs_property_list_add_int(v,"Tracking confidence",2);
    obs_property_list_add_int(v,"Live stabilization (translation)",3);
    obs_property_set_long_description(v,"Stabilization uses recent global translation estimates and a centered crop. No intentional video buffering; rotation, rolling shutter and foreground-dominated motion are not corrected. Use Original source when sharing raw motion to other filters.");
    obs_properties_add_int_slider(p,"flow_edge","Capture longest edge",160,640,32);
    obs_properties_add_int_slider(p,"flow_interval","Capture interval (ms)",33,500,1);
    obs_properties_add_int_slider(p,"flow_stabilize_ms","Stabilization smoothing (ms)",50,2000,50);
    auto* margin=obs_properties_add_int_slider(p,"flow_crop_percent","Stabilization crop per edge (%)",0,15,1);
    obs_property_set_long_description(margin,"Fixed crop reserves room for correction. 0 disables correction and crop.");
    return p;
}
void render(void* data,gs_effect_t*) {
    auto* f=static_cast<Filter*>(data);
    auto* target=obs_filter_get_target(f->context);
    unsigned w=target?obs_source_get_base_width(target):0,h=target?obs_source_get_base_height(target):0;
    if(!w||!h){obs_source_skip_video_filter(f->context);return;}
    const auto now=Clock::now();
    const auto rev=f->revision.load();
    if(f->staged) {
        bool submitted=false;
        if(f->stage_revision==rev&&f->sw==w&&f->sh==h) {
            uint8_t* bytes=nullptr;uint32_t pitch=0;
            if(gs_stagesurface_map(f->stage,&bytes,&pitch)) {
                if(pitch>=f->cw*4) {
                    Frame frame;
                    frame.w=f->cw;frame.h=f->ch;frame.sw=w;frame.sh=h;
                    frame.revision=rev;frame.captured=f->captured;
                    frame.rgba.resize(size_t(f->cw)*f->ch*4);
                    for(unsigned y=0;y<f->ch;++y)
                        std::memcpy(frame.rgba.data()+size_t(y)*f->cw*4,bytes+size_t(y)*pitch,size_t(f->cw)*4);
                    std::unique_lock lock(f->mutex,std::try_to_lock);
                    if(lock&&f->revision==rev&&!f->pending) {
                        f->frame=std::move(frame);f->pending=true;submitted=true;f->cv.notify_one();
                    }
                }
                gs_stagesurface_unmap(f->stage);
            }
        }
        f->staged=false;
        if(!submitted)f->wants=true;
    }
    if(now>=f->next_capture&&f->wants.exchange(false)) {
        const unsigned edge=std::min(unsigned(f->edge.load()),std::max(w,h));
        unsigned cw=std::max(1u,unsigned(uint64_t(w)*edge/std::max(w,h)));
        unsigned ch=std::max(1u,unsigned(uint64_t(h)*edge/std::max(w,h)));
        if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->stage||f->cw!=cw||f->ch!=ch) {
            if(f->stage)gs_stagesurface_destroy(f->stage);
            f->stage=gs_stagesurface_create(cw,ch,GS_RGBA);f->cw=cw;f->ch=ch;
        }
        auto* base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if(cw>=32&&ch>=32&&f->capture&&f->stage&&base) {
            gs_texrender_reset(f->capture);
            if(gs_texrender_begin(f->capture,cw,ch)) {
                gs_projection_push();gs_matrix_push();gs_matrix_identity();
                gs_ortho(0,float(w),0,float(h),-100,100);
                struct vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);
                gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
                sight_draw_analysis_source(target);
                gs_blend_state_pop();
                gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
                if(auto* tex=gs_texrender_get_texture(f->capture)) {
                    gs_stage_texture(f->stage,tex);
                    f->sw=w;f->sh=h;f->staged=true;f->stage_revision=rev;f->captured=now;
                }
            }
        }
        f->next_capture=now+std::chrono::milliseconds(f->interval.load());
        if(!f->staged)f->wants=true;
    }

    sight_flow_v1 result{};
    sight::CameraCorrection correction{};
    Clock::time_point result_time{};
    bool busy=false;
    const bool valid=snapshot(f,result,500,&correction,&result_time,&busy)&&
        result.source_width==w&&result.source_height==h;
    const int mode=f->view.load();
    if(mode==3) {
        const uint64_t current_revision=f->revision.load();
        if(valid&&current_revision==rev) {
            f->render_correction=correction;
            f->render_correction_time=result_time;
            f->render_correction_revision=current_revision;
            f->correction_w=w;f->correction_h=h;
        } else if(!busy||f->render_correction_revision!=current_revision||
                  f->correction_w!=w||f->correction_h!=h||
                  now-f->render_correction_time>std::chrono::milliseconds(500)) {
            f->render_correction={};
        }
        const float margin=f->crop_percent.load()/100.f;
        if(!f->effect||margin==0) {obs_source_skip_video_filter(f->context);return;}
        const auto c=f->render_correction;
        if(c.valid&&!f->stabilization_announced) {
            f->stabilization_announced=true;
            blog(LOG_INFO,"[Sight/Flow] SIGHT_CAUSAL_STABILIZER_20261117: global translation correction");
        }
        if(obs_source_process_filter_begin(f->context,GS_RGBA,OBS_NO_DIRECT_RENDERING)) {
            struct vec2 offset;
            const float limit=margin*.95f;
            vec2_set(&offset,c.valid?std::clamp(c.x,-limit,limit):0.f,
                            c.valid?std::clamp(c.y,-limit,limit):0.f);
            gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"camera_offset"),&offset);
            gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"camera_margin"),margin);
            gs_effect_set_int(gs_effect_get_param_by_name(f->effect,"view_mode"),3);
            obs_source_process_filter_end(f->context,f->effect,w,h);
        }
        return;
    }
    f->render_correction={};
    if(valid&&!f->announced) {
        f->announced=true;
        blog(LOG_INFO,"[Sight/Flow] SIGHT_DENSE_FLOW_20261117: DIS bidirectional grid");
    }
    if(!valid||f->view==0||!f->effect){obs_source_skip_video_filter(f->context);return;}
    if(result.sequence!=f->uploaded_sequence) {
        if(!f->texture)f->texture=gs_texture_create(SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,1,nullptr,GS_DYNAMIC);
        if(f->texture) {
            gs_texture_set_image(f->texture,reinterpret_cast<const uint8_t*>(result.cells),SIGHT_FLOW_WIDTH*16,false);
            f->uploaded_sequence=result.sequence;
        }
    }
    if(!f->texture){obs_source_skip_video_filter(f->context);return;}
    if(obs_source_process_filter_begin(f->context,GS_RGBA,OBS_NO_DIRECT_RENDERING)) {
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"flow_grid"),f->texture);
        // The shader references these in another branch of the same pass;
        // OBS requires initialized numeric parameters even in diagnostics.
        struct vec2 offset;
        vec2_zero(&offset);
        gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"camera_offset"),&offset);
        gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"camera_margin"),0.f);
        gs_effect_set_int(gs_effect_get_param_by_name(f->effect,"view_mode"),mode);
        obs_source_process_filter_end(f->context,f->effect,w,h);
    }
}
const char* name(void*){return "LLL.sight Spatial Motion (DIS)";}
}
struct obs_source_info sight_spatial_motion_filter_info={
    .id="sight_spatial_motion_filter",.type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO,.get_name=name,.create=create,.destroy=destroy,
    .get_defaults=defaults,.get_properties=properties,.update=update,.video_render=render,
};
