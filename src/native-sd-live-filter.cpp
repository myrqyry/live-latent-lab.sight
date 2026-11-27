#include "common/preferences.hpp"
#include <obs-module.h>
#include <graphics/graphics.h>
#include <graphics/vec4.h>
#include <graphics/vec2.h>
#include "core/neural_render_motion.hpp"
#include "core/analysis_capture.hpp"
#include "core/native_sd_frame_geometry.hpp"
#include "core/native_sd_session.hpp"
#include "core/native_sd_temporal.hpp"
#include "engines/stable_diffusion_backend.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#ifdef SIGHT_SD_LIVE_CUDA_GUARD
#include <cuda_runtime_api.h>
#endif

namespace {
using Clock=std::chrono::steady_clock;
using Backend=lll::engines::StableDiffusionBackend;
struct Settings {
    std::string model,tae,device,prompt,negative,weight_type;
    int steps=4,edge=256,sync=0,max_fps=0;
    float denoise=.35f,cfg=1.2f,temporal=0.f;
    int64_t seed=42;
    bool enabled=true,continuous_denoise=false;
    bool operator==(const Settings&)const=default;
};
struct Job {
    std::shared_ptr<const Settings> settings;
    uint64_t generation=0;
    uint32_t source_w=0,source_h=0,w=0,h=0;
    std::vector<uint8_t> rgba;
    Clock::time_point captured{};
};
struct Result {
    uint64_t generation=0,sequence=0;
    uint32_t source_w=0,source_h=0,w=0,h=0;
    Clock::time_point completed{};
    std::vector<uint8_t> rgba;
    std::vector<float> delta;
    sight::sd::TemporalStats temporal;
    std::shared_ptr<const sight::nr::MotionReference> reference;
};
struct Filter {
    explicit Filter(obs_source_t*s):source(s){}
    obs_source_t* source;
    std::mutex mutex;
    std::condition_variable wake;
    std::shared_ptr<const Settings> settings;
    std::optional<Job> pending;
    std::optional<Result> result;
    std::atomic<uint64_t> generation{1},failed_generation{0},drawn_sequence{0};
    std::atomic<bool> stop{false},busy{false};
    std::atomic<double> generated_fps{0.};
    std::thread worker;
    gs_texrender_t* capture=nullptr;
    gs_stagesurf_t* stage=nullptr;
    gs_texture_t* texture=nullptr;
    uint32_t stage_w=0,stage_h=0;
    Result displayed;
    gs_effect_t* motion_effect=nullptr;
    gs_texture_t *residual=nullptr,*reference_texture=nullptr,*flow_texture=nullptr;
    gs_texrender_t* live_capture=nullptr;
    sight::nr::MotionSync motion;
    Clock::time_point next_motion{},next_inference{};
    uint64_t cadence_generation=0;
    std::atomic<bool> motion_drawn{false};
    std::atomic<float> flow_coverage{0.f},history_weight{0.f};
};
// One experimental live model context at a time. Prevent a second filter
// from independently loading another multi-GB model onto the same GPU.

bool stale(Filter*f,uint64_t gen){return f->stop||f->generation!=gen;}
// Check memory on the actual CUDA device (not physical-index NVML).
// This is conservative admission control, not a reservation: other
// applications can allocate VRAM after the check.
void require_device_headroom(const Settings&s,bool cold){
    if(s.device=="cpu")return;
#ifdef SIGHT_SD_LIVE_CUDA_GUARD
    size_t free_bytes=0,total_bytes=0;
    if(cudaSetDevice(0)!=cudaSuccess||
       cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)
        throw std::runtime_error("CUDA0 unavailable; no CPU fallback");
    const size_t minimum=(cold?4500ull:700ull)*1024*1024;
    if(free_bytes<minimum)
        throw std::runtime_error("Insufficient CUDA0 free VRAM; disable other GPU effects and toggle Generate to retry");
#else
    (void)cold;
    throw std::runtime_error("This Sight build has no CUDA live diffusion backend; no CPU fallback");
#endif
}
void worker_loop(Filter*f){
    Backend backend;
    std::unique_lock<std::mutex> slot(sight::sd::session_slot,std::defer_lock);
    std::shared_ptr<const Settings> loaded;
    uint64_t sequence=0,previous_generation=0;
    Clock::time_point previous{};
    sight::sd::TemporalHistory history;
    auto release=[&]{history.reset();backend.shutdown();loaded.reset();if(slot.owns_lock())slot.unlock();};
    for(;;){
        Job job;
        {
            std::unique_lock lock(f->mutex);
            if(!f->wake.wait_for(lock,std::chrono::seconds(2),[&]{return f->stop||f->pending.has_value();})){
                lock.unlock();release();continue;
            }
            if(f->stop)break;
            job=std::move(*f->pending);f->pending.reset();
        }
        try{
            if(stale(f,job.generation)){f->busy=false;continue;}
            const auto&s=*job.settings;
            if(!loaded||loaded->model!=s.model||loaded->tae!=s.tae||loaded->device!=s.device||loaded->weight_type!=s.weight_type){
                release();
                if(!slot.try_lock())throw std::runtime_error("Another native diffusion context is active");
                if(!std::filesystem::is_regular_file(s.model)||
                    (!s.tae.empty()&&!std::filesystem::is_regular_file(s.tae)))
                    throw std::runtime_error("Local model or tiny VAE file unavailable");
                require_device_headroom(s,true);
                Backend::EngineConfig config{};
                config.model_path=s.model;config.taesd_path=s.tae;
                config.use_gpu=s.device!="cpu";config.backend_device=s.device;
                config.weight_type=s.weight_type;
                config.num_threads=2;
                if(!backend.initialize(config))throw std::runtime_error(backend.last_error());
                if(backend.model_version()!="SD 1.x")
                    throw std::runtime_error("This experimental live filter currently accepts SD 1.x models only");
                loaded=job.settings;
                blog(LOG_INFO,"[Sight/SD] SIGHT_SD_WEIGHT_PRECISION_20261126 backend=%s weights=%s",s.device.c_str(),s.weight_type.empty()?"default":s.weight_type.c_str());
            }
            if(stale(f,job.generation)){f->busy=false;continue;}
            require_device_headroom(s,false);
            const sight::sd::FrameGeometry geometry(job.w,job.h,64);
            std::vector<uint8_t> rgb(size_t(job.w)*job.h*3);
            for(size_t i=0;i<rgb.size()/3;++i){
                const unsigned alpha=job.rgba[i*4+3];
                for(unsigned c=0;c<3;++c)
                    rgb[i*3+c]=alpha?uint8_t(std::min(255u,(unsigned(job.rgba[i*4+c])*255+alpha/2)/alpha)):0;
            }
            auto aligned=geometry.pad_rgb(rgb);
            Backend::GenerationParams p{};
            p.width=geometry.width;p.height=geometry.height;p.steps=s.steps;
            p.strength=s.denoise;p.continuous_denoise=s.continuous_denoise;p.cfg_scale=s.cfg;p.seed=s.seed;
            p.prompt=s.prompt;p.negative_prompt=s.negative;
            p.sample_method_name="lcm";p.scheduler_name="lcm";p.init_image=aligned.data();
            backend.set_progress_callback([&](int,int,float){
                if(stale(f,job.generation))backend.cancel_current_generation();
            });
            auto output=backend.generate(p);
            backend.set_progress_callback({});
            if(stale(f,job.generation)){f->busy=false;continue;}
            if(output.width!=geometry.width||output.height!=geometry.height||output.channels!=3)
                throw std::runtime_error("Invalid native image output: "+backend.last_error());
            auto cropped=geometry.crop_rgb(output.data);
            Result result;
            result.generation=job.generation;result.sequence=++sequence;
            result.source_w=job.source_w;result.source_h=job.source_h;
            result.w=job.w;result.h=job.h;result.completed=Clock::now();
            if(s.sync==1){
                // Keep premultiplied source pixels for the correspondence
                // check, but transfer a straight RGB appearance residual.
                auto reference=std::make_shared<sight::nr::MotionReference>();
                reference->sequence=result.sequence;
                cv::Mat captured(int(job.h),int(job.w),CV_8UC4,job.rgba.data());
                const int edge=std::min(320,int(std::max(job.w,job.h)));
                cv::resize(captured,reference->rgba,{std::max(1,int(job.w)*edge/int(std::max(job.w,job.h))),
                    std::max(1,int(job.h)*edge/int(std::max(job.w,job.h)))},0,0,cv::INTER_AREA);
                cv::cvtColor(reference->rgba,reference->gray,cv::COLOR_RGBA2GRAY);
                result.reference=std::move(reference);
                result.delta.resize(size_t(job.w)*job.h*4);
                for(size_t i=0;i<cropped.size()/3;++i)
                    for(unsigned c=0;c<3;++c)
                        result.delta[i*4+c]=(float(cropped[i*3+c])-rgb[i*3+c])/255.f;
                // The model worker owns both raw keyframes; no GPU readback
                // or optical-flow wait is added to the graphics thread.
                try{
                    result.temporal=history.apply(job.generation,captured,result.delta,s.temporal,job.captured);
                }catch(const cv::Exception&){history.reset();}
                if(result.temporal.mean_weight>0.f)
                    blog(LOG_DEBUG,"[Sight/SD] SIGHT_SD_TEMPORAL_20261124_R19 coverage=%.3f weight=%.3f",
                        result.temporal.coverage,result.temporal.mean_weight);
            }else history.reset();
            result.completed=Clock::now();
            result.rgba=std::move(job.rgba);
            for(size_t i=0;i<cropped.size()/3;++i){
                const unsigned alpha=result.rgba[i*4+3];
                for(unsigned c=0;c<3;++c)result.rgba[i*4+c]=uint8_t((unsigned(cropped[i*3+c])*alpha+127)/255);
            }
            double fps=0.;
            if(previous_generation==job.generation){
                const double elapsed=std::chrono::duration<double>(result.completed-previous).count();
                if(elapsed>0)fps=1./elapsed;
            }
            previous=result.completed;previous_generation=job.generation;
            {
                std::lock_guard lock(f->mutex);
                if(!stale(f,job.generation)){
                    f->generated_fps=fps;f->result=std::move(result);
                    blog(LOG_INFO,"[Sight/SD] SIGHT_SD_LIVE_REAL_20261123_Q74 %ux%u aligned=%ux%u sequence=%llu generated_fps=%.3f",
                        job.w,job.h,geometry.width,geometry.height,(unsigned long long)sequence,fps);
                }
            }
        }catch(const std::exception&e){
            backend.set_progress_callback({});
            release();
            if(!stale(f,job.generation)){
                f->failed_generation=job.generation;
                blog(LOG_WARNING,"[Sight/SD] Native live diffusion paused: %s",e.what());
            }
        }
        f->busy=false;
    }
    release();
}
const char* name(void*){return "LLL.sight Live Diffusion (Experimental)";}
void defaults(obs_data_t*s){
    obs_data_set_default_bool(s,"enabled",true);
    obs_data_set_default_string(s,"backend","CUDA0");
    obs_data_set_default_string(s,"weight_precision","default");
    obs_data_set_default_string(s,"prompt","cinematic illustration, detailed lighting");
    obs_data_set_default_int(s,"steps",4);
    obs_data_set_default_int(s,"sync_mode",0);
    obs_data_set_default_int(s,"long_edge",256);
    obs_data_set_default_int(s,"max_generation_fps",0);
    obs_data_set_default_int(s,"seed",42);
    obs_data_set_default_double(s,"denoise",.35);
    obs_data_set_default_double(s,"temporal_stability",0.);
    obs_data_set_default_bool(s,"continuous_denoise",false);
    obs_data_set_default_double(s,"cfg",1.2);
}
void update(void*data,obs_data_t*s){
    auto*f=static_cast<Filter*>(data);
    auto next=std::make_shared<Settings>();
    next->model=obs_data_get_string(s,"model_path");next->tae=obs_data_get_string(s,"taesd_path");
    next->device=obs_data_get_string(s,"backend");if(next->device!="cpu")next->device="CUDA0";
    if(std::string(obs_data_get_string(s,"weight_precision"))=="q8_0")next->weight_type="q8_0";
    next->prompt=obs_data_get_string(s,"prompt");next->negative=obs_data_get_string(s,"negative_prompt");
    next->steps=int(std::clamp<int64_t>(obs_data_get_int(s,"steps"),1,8));
    next->edge=int(std::clamp<int64_t>(obs_data_get_int(s,"long_edge"),128,512));
    next->max_fps=int(std::clamp<int64_t>(obs_data_get_int(s,"max_generation_fps"),0,30));
    next->seed=obs_data_get_int(s,"seed");
    auto bounded=[&](const char*key,double low,double high,double fallback){
        const auto v=obs_data_get_double(s,key);return float(std::isfinite(v)?std::clamp(v,low,high):fallback);
    };
    next->denoise=bounded("denoise",.01,1.,.35);next->cfg=bounded("cfg",1.,10.,1.2);
    next->enabled=obs_data_get_bool(s,"enabled");
    next->continuous_denoise=obs_data_get_bool(s,"continuous_denoise");
    next->sync=obs_data_get_int(s,"sync_mode")==1?1:0;
    next->temporal=bounded("temporal_stability",0.,.9,0.);
    std::lock_guard lock(f->mutex);
    if(f->settings&&*f->settings==*next)return;
    f->settings=std::move(next);++f->generation;f->result.reset();
    f->drawn_sequence=0;f->generated_fps=0.;f->history_weight=0.f;
}
void stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_int(args,"sequence",f->drawn_sequence.load());
    calldata_set_float(args,"generated_fps",f->generated_fps.load());
}
void motion_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_bool(args,"motion_drawn",f->motion_drawn.load());
    calldata_set_float(args,"flow_coverage",f->flow_coverage.load());
}
void temporal_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_float(args,"history_weight",f->history_weight.load());
}
void* create(obs_data_t*s,obs_source_t*source){
    auto f=std::make_unique<Filter>(source);
    update(f.get(),s);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_native_sd_live_stats_v1(out int sequence, out float generated_fps)",stats,f.get());
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_native_sd_live_motion_stats_v1(out bool motion_drawn, out float flow_coverage)",motion_stats,f.get());
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_native_sd_live_temporal_stats_v1(out float history_weight)",temporal_stats,f.get());
    obs_enter_graphics();
    if(char*path=obs_module_file("sight-live-diffusion-motion.effect")){
        char*errors=nullptr;
        f->motion_effect=gs_effect_create_from_file(path,&errors);
        if(errors){blog(LOG_WARNING,"[Sight/SD] Motion shader: %s",errors);bfree(errors);}
        bfree(path);
    }
    obs_leave_graphics();
    if(f->motion_effect)blog(LOG_INFO,"[Sight/SD] SIGHT_SD_MOTION_20261124_E72 ready=1");
    f->worker=std::thread(worker_loop,f.get());return f.release();
}
void destroy(void*data){
    auto*f=static_cast<Filter*>(data);
    f->stop=true;f->wake.notify_all();
    if(f->worker.joinable())f->worker.join();
    obs_enter_graphics();
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->stage)gs_stagesurface_destroy(f->stage);
    if(f->texture)gs_texture_destroy(f->texture);
    if(f->residual)gs_texture_destroy(f->residual);
    if(f->reference_texture)gs_texture_destroy(f->reference_texture);
    if(f->flow_texture)gs_texture_destroy(f->flow_texture);
    if(f->live_capture)gs_texrender_destroy(f->live_capture);
    if(f->motion_effect)gs_effect_destroy(f->motion_effect);
    obs_leave_graphics();delete f;
}
obs_properties_t* properties(void*){
    auto*p=obs_properties_create();
    obs_properties_add_bool(p,"enabled","Generate");
    obs_properties_add_path(p,"model_path","SD 1.x LCM checkpoint",OBS_PATH_FILE,"Models (*.safetensors *.gguf *.ckpt)",sight_preferences::directory().c_str());
    obs_properties_add_path(p,"taesd_path","Tiny autoencoder (optional)",OBS_PATH_FILE,"Safetensors (*.safetensors)",sight_preferences::directory().c_str());
    auto*device=obs_properties_add_list(p,"backend","Native device",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(device,"NVIDIA CUDA 0","CUDA0");
    obs_property_list_add_string(device,"CPU (diagnostic only)","cpu");
    auto*precision=obs_properties_add_list(p,"weight_precision","Model weight precision",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(precision,"Checkpoint default","default");
    obs_property_list_add_string(precision,"Q8_0 (lower model memory)","q8_0");
    obs_property_set_long_description(precision,"Quantize supported weights at load. Reduces model weight memory; may change output and need not be faster. Changing this reloads the local model. The CUDA safety guard is unchanged.");
    obs_properties_add_text(p,"prompt","Prompt",OBS_TEXT_MULTILINE);
    obs_properties_add_text(p,"negative_prompt","Negative prompt",OBS_TEXT_MULTILINE);
    obs_properties_add_int_slider(p,"long_edge","Inference long edge",128,512,32);
    auto*rate=obs_properties_add_int_slider(p,"max_generation_fps","Maximum generated FPS (0 = unlimited)",0,30,1);
    obs_property_set_long_description(rate,"Limit new diffusion keyframes without delaying OBS rendering. Useful for GPU headroom and giving motion compensation time to reproject between keyframes. Stale corrections still expire.");
    auto*sync=obs_properties_add_list(p,"sync_mode","Synchronization",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(sync,"Frame locked",0);
#ifdef SIGHT_HAVE_DENSE_FLOW
    obs_property_list_add_int(sync,"Motion compensated (live detail)",1);
#else
    obs_property_list_add_int(sync,"Motion gated (live detail)",1);
#endif
    obs_property_set_long_description(sync,"Frame locked holds the generated image. Motion compensation transfers the generated appearance to live video, fading uncertain matches; works best at low denoise.");
    auto*temporal=obs_properties_add_float_slider(p,"temporal_stability","Temporal stability",0.,.9,.05);
    obs_property_set_long_description(temporal,"Blend the previous raw diffusion correction into each new keyframe only at matching source pixels. Motion-compensated mode only. Zero disables history.");
    obs_properties_add_int_slider(p,"steps","Requested LCM steps",1,8,1);
    obs_properties_add_float_slider(p,"denoise","Denoise strength",.01,1.,.01);
    auto* continuous=obs_properties_add_bool(p,"continuous_denoise","Continuous denoise (experimental)");
    obs_property_set_long_description(continuous,"Use explicit noise level instead of discrete step selection. Useful for low-step source preservation; noise level is capped at 0.93.");
    auto* guidance=obs_properties_add_float_slider(p,"cfg","Guidance",1.,10.,.1);
    obs_property_set_long_description(guidance,"For LCM, 1.0 uses a single conditional denoiser pass. Higher values use classifier-free guidance and may take nearly twice as much UNet compute. At low continuous denoise, try 1.0 first.");
    obs_properties_add_int(p,"seed","Fixed seed",0,2147483647,1);
    obs_properties_add_text(p,"info","Native output; audio is not delayed. Frame locked may lag video. Model loading and generation may take time.",OBS_TEXT_INFO);
    return p;
}
bool render_capture(gs_texrender_t*&capture,obs_source_t*target,uint32_t cw,uint32_t ch,uint32_t w,uint32_t h){
    if(!capture)capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(!capture)return false;
    gs_texrender_reset(capture);
    if(!gs_texrender_begin(capture,cw,ch))return false;
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0,float(w),0,float(h),-100.f,100.f);
    vec4 clear{};gs_clear(GS_CLEAR_COLOR,&clear,0,0);
    sight_draw_analysis_source(target);
    gs_matrix_pop();gs_projection_pop();gs_texrender_end(capture);
    return gs_texrender_get_texture(capture)!=nullptr;
}
bool capture(Filter*f,obs_source_t*target,Job&job,bool from_live=false){
    if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(!f->stage||f->stage_w!=job.w||f->stage_h!=job.h){
        if(f->stage)gs_stagesurface_destroy(f->stage);
        f->stage=gs_stagesurface_create(job.w,job.h,GS_RGBA);
        f->stage_w=job.w;f->stage_h=job.h;
    }
    if(!f->capture||!f->stage)return false;
    if(from_live){
        auto*tex=f->live_capture?gs_texrender_get_texture(f->live_capture):nullptr;
        auto*effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if(!tex||!effect)return false;
        gs_texrender_reset(f->capture);
        if(!gs_texrender_begin(f->capture,job.w,job.h))return false;
        gs_projection_push();gs_matrix_push();gs_matrix_identity();
        gs_ortho(0,float(job.w),0,float(job.h),-100.f,100.f);
        gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
        gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),tex);
        while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(tex,0,job.w,job.h);
        gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
    }else if(!render_capture(f->capture,target,job.w,job.h,job.source_w,job.source_h))return false;
    auto*tex=gs_texrender_get_texture(f->capture);if(!tex)return false;
    gs_stage_texture(f->stage,tex);
    uint8_t*data=nullptr;uint32_t stride=0;
    if(!gs_stagesurface_map(f->stage,&data,&stride))return false;
    bool ok=stride>=job.w*4;
    if(ok){
        job.rgba.resize(size_t(job.w)*job.h*4);
        for(uint32_t y=0;y<job.h;++y)
            std::memcpy(job.rgba.data()+size_t(y)*job.w*4,data+size_t(y)*stride,size_t(job.w)*4);
    }
    gs_stagesurface_unmap(f->stage);return ok;
}
void render(void*data,gs_effect_t*){
    auto*f=static_cast<Filter*>(data);
    f->motion_drawn=false;f->flow_coverage=0.f;f->history_weight=0.f;
    auto skip=[&]{f->drawn_sequence=0;obs_source_skip_video_filter(f->source);};
    auto*target=obs_filter_get_target(f->source);
    if(!target||!obs_filter_get_parent(f->source)){skip();return;}
    Job job;
    job.source_w=obs_source_get_base_width(target);job.source_h=obs_source_get_base_height(target);
    if(!job.source_w||!job.source_h||job.source_w>8192||job.source_h>8192){skip();return;}
    std::optional<Result> ready;
    {
        std::unique_lock lock(f->mutex,std::try_to_lock);
        if(!lock){skip();return;}
        job.settings=f->settings;job.generation=f->generation;
        if(f->result){ready=std::move(f->result);f->result.reset();}
    }
    if(!job.settings||!job.settings->enabled||job.settings->model.empty()||f->failed_generation==job.generation){
        f->displayed={};
        if(f->texture){gs_texture_destroy(f->texture);f->texture=nullptr;}
        if(f->residual){gs_texture_destroy(f->residual);f->residual=nullptr;}
        if(f->reference_texture){gs_texture_destroy(f->reference_texture);f->reference_texture=nullptr;}
        if(f->flow_texture){gs_texture_destroy(f->flow_texture);f->flow_texture=nullptr;}
        if(f->live_capture){gs_texrender_destroy(f->live_capture);f->live_capture=nullptr;}
        skip();return;
    }
    const auto max_edge=std::max(job.source_w,job.source_h);
    const auto edge=std::min(max_edge,uint32_t(job.settings->edge));
    job.w=uint32_t(uint64_t(job.source_w)*edge/max_edge);
    job.h=uint32_t(uint64_t(job.source_h)*edge/max_edge);
    if(job.w<32||job.h<32){skip();return;}
    auto matches=[&](const Result&r){
        return r.generation==job.generation&&r.source_w==job.source_w&&r.source_h==job.source_h&&r.w==job.w&&r.h==job.h;
    };
    const bool moving=job.settings->sync==1;
    if(moving&&!f->motion_effect){skip();return;}
    if(!moving){
        f->displayed.reference.reset();
        if(f->residual){gs_texture_destroy(f->residual);f->residual=nullptr;}
        if(f->reference_texture){gs_texture_destroy(f->reference_texture);f->reference_texture=nullptr;}
        if(f->flow_texture){gs_texture_destroy(f->flow_texture);f->flow_texture=nullptr;}
        if(f->live_capture){gs_texrender_destroy(f->live_capture);f->live_capture=nullptr;}
    }
    auto upload=[](gs_texture_t*&texture,uint32_t w,uint32_t h,gs_color_format format,const uint8_t*data,uint32_t stride){
        if(!texture||gs_texture_get_width(texture)!=w||gs_texture_get_height(texture)!=h){
            if(texture)gs_texture_destroy(texture);
            texture=gs_texture_create(w,h,format,1,nullptr,GS_DYNAMIC);
        }
        if(!texture)return false;
        gs_texture_set_image(texture,data,stride,false);return true;
    };
    if(ready&&matches(*ready)&&moving&&ready->reference&&!ready->delta.empty()){
        auto&ref=ready->reference->rgba;
        if(upload(f->residual,job.w,job.h,GS_RGBA32F,reinterpret_cast<const uint8_t*>(ready->delta.data()),job.w*4*sizeof(float))&&
           upload(f->reference_texture,ref.cols,ref.rows,GS_RGBA,ref.data,uint32_t(ref.step))){
            f->displayed=std::move(*ready);f->displayed.rgba.clear();f->displayed.delta.clear();
            f->next_motion={};
        }else f->displayed={};
    }else if(ready&&matches(*ready)&&!moving){
        if(!f->texture||gs_texture_get_width(f->texture)!=job.w||gs_texture_get_height(f->texture)!=job.h){
            if(f->texture)gs_texture_destroy(f->texture);
            f->texture=gs_texture_create(job.w,job.h,GS_RGBA,1,nullptr,GS_DYNAMIC);
        }
        if(f->texture){
            gs_texture_set_image(f->texture,ready->rgba.data(),job.w*4,false);
            f->displayed=std::move(*ready);f->displayed.rgba.clear();
        }
    }
    const bool have_live=moving&&render_capture(f->live_capture,target,job.source_w,job.source_h,job.source_w,job.source_h);
    const auto now=Clock::now();
    if(f->cadence_generation!=job.generation){
        f->cadence_generation=job.generation;f->next_inference={};
    }
    bool expected=false;
    const bool inference_due=now>=f->next_inference&&f->busy.compare_exchange_strong(expected,true);
    const bool flow_due=have_live&&matches(f->displayed)&&f->displayed.reference&&now>=f->next_motion&&
        now-f->displayed.completed<=std::chrono::milliseconds(500);
    bool captured=false;
    if(inference_due||flow_due)captured=(!moving||have_live)&&capture(f,target,job,moving);
    if(flow_due){
        f->next_motion=now+std::chrono::milliseconds(33);
        if(captured){
            // The motion worker owns its copy and performs downsampling
            // off the render thread, at most once per motion cadence.
            try{
                cv::Mat rgba(int(job.h),int(job.w),CV_8UC4,job.rgba.data());
                f->motion.submit(f->displayed.reference,rgba.clone(),now);
            }catch(const cv::Exception&){}
        }
    }
    if(inference_due){
        bool queued=false;
        if(captured){
            std::unique_lock lock(f->mutex,std::try_to_lock);
            if(lock&&!stale(f,job.generation)){
                Job pending{job.settings,job.generation,job.source_w,job.source_h,job.w,job.h,std::move(job.rgba),now};
                f->pending=std::move(pending);queued=true;f->wake.notify_one();
                if(job.settings->max_fps>0)
                    f->next_inference=now+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0/job.settings->max_fps));
            }
        }
        if(!queued)f->busy=false;
    }
    if(moving){
        auto*live=have_live?gs_texrender_get_texture(f->live_capture):nullptr;
        if(!live){skip();return;}
        if(matches(f->displayed)&&f->generation==job.generation&&f->displayed.reference&&f->residual&&f->reference_texture&&
           now-f->displayed.completed<=std::chrono::milliseconds(500)){
            sight::nr::MotionResult motion;
            bool flow=false;
            if(f->motion.latest(f->displayed.sequence,motion)&&motion.flow.coverage>0.f){
                std::array<float,SIGHT_FLOW_CELLS*4> grid{};
                for(size_t i=0;i<SIGHT_FLOW_CELLS;++i){
                    const auto&c=motion.flow.cells[i];
                    grid[i*4]=c.dx*c.confidence;grid[i*4+1]=c.dy*c.confidence;grid[i*4+2]=c.confidence;
                }
                flow=upload(f->flow_texture,SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,
                    reinterpret_cast<const uint8_t*>(grid.data()),SIGHT_FLOW_WIDTH*4*sizeof(float));
                if(flow)f->flow_coverage=motion.flow.coverage;
            }
            auto*effect=f->motion_effect;
            gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),live);
            gs_effect_set_texture(gs_effect_get_param_by_name(effect,"correction"),f->residual);
            gs_effect_set_texture(gs_effect_get_param_by_name(effect,"reference_image"),f->reference_texture);
            gs_effect_set_texture(gs_effect_get_param_by_name(effect,"motion_grid"),flow?f->flow_texture:f->reference_texture);
            gs_effect_set_float(gs_effect_get_param_by_name(effect,"flow_enabled"),flow?1.f:0.f);
            vec2 size{};vec2_set(&size,float(f->displayed.reference->rgba.cols),float(f->displayed.reference->rgba.rows));
            gs_effect_set_vec2(gs_effect_get_param_by_name(effect,"proxy_size"),&size);
            while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(live,0,job.source_w,job.source_h);
            f->drawn_sequence=f->displayed.sequence;f->motion_drawn=true;f->history_weight=f->displayed.temporal.mean_weight;
        }else{
            auto*effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
            if(!effect){skip();return;}
            gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),live);
            while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(live,0,job.source_w,job.source_h);
            f->drawn_sequence=0;
        }
        return;
    }
    if(!f->texture||!matches(f->displayed)||f->generation!=job.generation||
        Clock::now()-f->displayed.completed>std::chrono::seconds(job.settings&&job.settings->device=="cpu"?60:5)){
        skip();return;
    }
    auto*effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    if(!effect){skip();return;}
    gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),f->texture);
    while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(f->texture,0,job.source_w,job.source_h);
    f->drawn_sequence=f->displayed.sequence;
}
}
obs_source_info sight_native_sd_live_filter_info={
    .id="sight_native_sd_live_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=name,.create=create,.destroy=destroy,
    .get_defaults=defaults,.get_properties=properties,.update=update,.video_render=render,
};
