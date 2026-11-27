#include <obs-module.h>
#include <callback/proc.h>
#include <graphics/graphics.h>
#include <graphics/vec2.h>
#include <util/bmem.h>
#include "core/neural_render_process.hpp"
#include "core/neural_render_residual.hpp"
#include "core/neural_render_depth.hpp"
#include "core/native_sd_temporal.hpp"
#include "core/analysis_capture.hpp"
#include "core/neural_render_motion.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <utility>

namespace {
using Clock=std::chrono::steady_clock;
struct Request {
    uint64_t generation=0,sequence=0;
    int sync=0,structure=100,clarity=0,shadows=0,residual=0;
    int temporal=0;
    uint32_t w=0,h=0,source_w=0,source_h=0;
    std::string precision;
    Clock::time_point captured;
    Clock::time_point completed;
    std::vector<uint8_t> rgb,alpha;
    float history_weight=0.f;
};
// Large frames may take seconds. Age the completed correction, not the
// capture, so a slow successful inference is actually visible. The capture
// itself is bounded by the worker timeout; no queue of obsolete frames.
int frame_timeout(const Request &r){
    const auto pixels=uint64_t(r.w)*r.h;
    if(pixels>1920ull*1080)return 60000;
    if(pixels>1280ull*720)return 15000;
    if(pixels>640ull*360)return 8000;
    return 3000;
}
auto output_age_limit(const Request &r){
    return std::chrono::milliseconds(frame_timeout(r)==3000?500:std::min(8000,frame_timeout(r)));
}
struct Result {
    Request key;
    std::vector<float> delta;
    std::shared_ptr<const sight::nr::MotionReference> reference;
};
struct Filter {
    obs_source_t *source=nullptr;
    std::string config,packaged_worker;
    std::atomic<bool> stop{false},busy{false};
    std::atomic<uint64_t> generation{1};
    std::atomic<int> edge{320},sync{1};
    // These settings are captured together under input_mutex.
    int structure=100,clarity=0,shadows=0,residual=0;
    int temporal=0;
    std::atomic<uint32_t> drawn_conditioning{0};
    std::atomic<int> drawn_residual{0};
    std::atomic<float> drawn_history_weight{0.f};
    std::atomic<uint64_t> drawn_size{0};
    std::atomic<uint64_t> submitted_sequence{0};
    std::atomic<int> submitted_mode{-1};
    std::atomic<float> strength{1.f},relighting{1.f},surface_detail{1.f},color_strength{1.f},highlight_protection{0.f},edge_aware{0.f},depth_guidance{0.f};
    std::mutex input_mutex,output_mutex;
    std::condition_variable wake;
    bool queued=false;
    std::string precision="w8a8";
    Request input;
    Result output;
    std::thread worker;
    gs_texrender_t *capture=nullptr;
    gs_stagesurf_t *stage=nullptr;
    gs_texture_t *texture=nullptr;
    gs_effect_t *effect=nullptr;
    gs_texrender_t *locked_pending=nullptr,*locked_display=nullptr,*motion_capture=nullptr;
    gs_stagesurf_t *motion_stage=nullptr;
    gs_texture_t *reference_texture=nullptr,*flow_texture=nullptr,*depth_texture=nullptr;
    std::unique_ptr<sight_depth_v1> depth_snapshot;
    std::string depth_owner;
    uint64_t depth_sequence=0;
    bool depth_announced=false;
    std::atomic<bool> depth_drawn{false};
    std::shared_ptr<const sight::nr::MotionReference> reference;
    sight::nr::MotionSync motion;
    Clock::time_point next_motion{},motion_at{};
    uint64_t pending_sequence=0,next_sequence=0,motion_sequence=0;
    uint32_t motion_w=0,motion_h=0;
    float motion_coverage=0.f;
    Request displayed;
    uint32_t stage_w=0,stage_h=0;
};
bool same(const Request&a,const Request&b){
    return a.generation==b.generation&&a.w==b.w&&a.h==b.h&&
        a.source_w==b.source_w&&a.source_h==b.source_h&&a.precision==b.precision&&a.sync==b.sync&&
        a.structure==b.structure&&a.clarity==b.clarity&&a.shadows==b.shadows&&a.residual==b.residual&&a.temporal==b.temporal;
}
std::vector<std::string> arguments(const std::string &config,const std::string &packaged_worker,const Request&r){
    if(config.empty())return {};
    auto *data=obs_data_create_from_json_file(config.c_str());
    if(!data)return {};
    auto read=[&](const char *key){return std::string(obs_data_get_string(data,key));};
    auto python=read("python"),script=read("worker"),repo=read("graph_repo"),
        checkpoint=read("checkpoint"),fused=read("fused_so");
    bool graph=obs_data_get_bool(data,"cuda_graph");
    obs_data_release(data);
    if(script.empty())script=packaged_worker;
    for(const auto *path:{&python,&script,&repo,&checkpoint})
        if(path->empty()||(*path)[0]!='/'||!std::filesystem::exists(*path))return {};
    if(!fused.empty()&&(fused[0]!='/'||!std::filesystem::is_regular_file(fused)))return {};
    // Paths are trusted local plugin configuration, never executable commands
    // or settings embedded in a scene collection.
    std::vector<std::string> args={python,"-B","-c",
        "import os,runpy,sys;os.environ.setdefault('OMP_NUM_THREADS','4');"
        "os.environ.setdefault('MKL_NUM_THREADS','4');"
        "repo,worker=sys.argv[1:3];sys.path[:0]=[repo,os.path.dirname(worker)];"
        "sys.argv=[worker]+sys.argv[3:];runpy.run_path(worker,run_name='__main__')",
        repo,script,"--checkpoint",checkpoint,"--width",std::to_string(r.w),
        "--height",std::to_string(r.h),"--precision",r.precision};
    blog(LOG_INFO,"[Sight/NR] SIGHT_NR_OWNED_WORKER_20261124_E82 packaged=%d",script==packaged_worker);
    // Neutral settings retain the original RGB24 worker command contract.
    auto control=[&](const char*flag,int value,int neutral){
        if(value!=neutral){args.emplace_back(flag);args.push_back(std::to_string(value/100.));}
    };
    control("--structure",r.structure,100);
    control("--input-clarity",r.clarity,0);
    control("--shadow-recovery",r.shadows,0);
    if(r.residual){
        args.emplace_back("--output-format");
        args.emplace_back(r.residual==2?"raw-residual-f32":"residual-f32");
    }
    if(!fused.empty()){args.push_back("--fused-so");args.push_back(fused);}
    if(graph && uint64_t(r.w)*r.h<=1280ull*720)args.push_back("--cuda-graph");
    return args;
}
void worker(Filter*f){
    sight::nr::Process process;
    sight::sd::TemporalHistory history(640);
    Request active,failed;
    bool first=true;
    uint64_t first_residual_generation=0;
    while(!f->stop){
        Request r;
        {
            std::unique_lock lock(f->input_mutex);
            f->wake.wait(lock,[&]{return f->stop||f->queued;});
            if(f->stop)break;
            r=std::move(f->input);f->queued=false;
        }
        auto cancelled=[&]{return f->stop||f->generation.load()!=r.generation;};
        if(cancelled()||same(r,failed)){f->busy=false;continue;}
        if(!same(r,active)){
            history.reset();process.close();active={};
            try {
                auto args=arguments(f->config,f->packaged_worker,r);
                if(args.empty()||!process.start(std::move(args))){
                    blog(LOG_WARNING,"[Sight/NR] local worker configuration missing or invalid; passthrough");
                    failed=r;failed.rgb.clear();failed.alpha.clear();f->busy=false;continue;
                }
            }catch(const std::exception&e){
                blog(LOG_WARNING,"[Sight/NR] local worker setup failed: %s",e.what());
                failed=r;failed.rgb.clear();failed.alpha.clear();f->busy=false;continue;
            }
            active=r;active.rgb.clear();active.alpha.clear();first=true;
        }
        std::vector<uint8_t> rendered;
        const size_t response_size=r.residual?4+size_t(r.w)*r.h*12:0;
        if(!process.exchange(r.rgb,rendered,first?frame_timeout(r)+30000:frame_timeout(r),cancelled,response_size)){
            history.reset();process.close();active={};
            if(!cancelled()){
                failed=r;failed.rgb.clear();failed.alpha.clear();
                blog(LOG_WARNING,"[Sight/NR] worker stopped or timed out; passthrough until settings change");
            }
            f->busy=false;continue;
        }
        if(first) {
            blog(LOG_INFO,"[Sight/NR] SIGHT_NR_REAL_RGB_20261123_C8A4 %ux%u %s",
                r.w,r.h,r.precision.c_str());
            blog(LOG_INFO,"[Sight/NR] SIGHT_NR_CONDITIONING_20261123_90AF structure=%d clarity=%d shadows=%d",
                r.structure,r.clarity,r.shadows);
            first=false;
        }
        if(!cancelled()){
            Result result;
            if(r.residual){
                if(!sight::nr::decode_float_residual(rendered,size_t(r.w)*r.h,result.delta)){
                    history.reset();process.close();active={};failed=r;failed.rgb.clear();failed.alpha.clear();f->busy=false;
                    blog(LOG_WARNING,"[Sight/NR] Invalid NRF1 residual; passthrough until settings change");
                    continue;
                }
                if(first_residual_generation!=r.generation){
                    blog(LOG_INFO,"[Sight/NR] SIGHT_NR_FLOAT_RESIDUAL_20261124_A91 mode=%d %ux%u",r.residual,r.w,r.h);
                    first_residual_generation=r.generation;
                }
            }else{
                result.delta.resize(size_t(r.w)*r.h*4);
                for(size_t i=0;i<size_t(r.w)*r.h;++i){
                    for(int c=0;c<3;++c)
                        result.delta[i*4+c]=(int(rendered[i*3+c])-int(r.rgb[i*3+c]))/255.f;
                    result.delta[i*4+3]=0.f;
                }
            }
            if(r.sync==1 && r.temporal>0 && std::max(r.w,r.h)<=640 && r.alpha.size()==size_t(r.w)*r.h){
                try {
                    cv::Mat rgba;
                    cv::cvtColor(cv::Mat(int(r.h),int(r.w),CV_8UC3,r.rgb.data()),rgba,cv::COLOR_RGB2RGBA);
                    cv::insertChannel(cv::Mat(int(r.h),int(r.w),CV_8UC1,r.alpha.data()),rgba,3);
                    auto stats=history.apply(r.generation,rgba,result.delta,r.temporal/100.f,r.captured);
                    r.history_weight=stats.mean_weight;
                }catch(const cv::Exception&){history.reset();}
            }else history.reset();
            if(r.sync==1) {
                try{result.reference=sight::nr::make_reference(r.sequence,r.rgb,int(r.w),int(r.h));}
                catch(const cv::Exception&){result.reference.reset();}
            }
            r.completed=Clock::now();
            result.key=std::move(r);result.key.rgb.clear();result.key.alpha.clear();
            std::lock_guard lock(f->output_mutex);
            if(!cancelled())f->output=std::move(result);
        }
        f->busy=false;
    }
}
void defaults(obs_data_t*s){
    obs_data_set_default_int(s,"resolution",320);
    obs_data_set_default_string(s,"precision","w8a8");
    obs_data_set_default_double(s,"strength",1.);
    obs_data_set_default_double(s,"relighting",1.);
    obs_data_set_default_double(s,"surface_detail",1.);
    obs_data_set_default_double(s,"color_strength",1.);
    obs_data_set_default_double(s,"highlight_protection",0.);
    obs_data_set_default_int(s,"sync_mode",1);
    obs_data_set_default_double(s,"structure_preservation",1.);
    obs_data_set_default_double(s,"input_clarity",0.);
    obs_data_set_default_double(s,"shadow_recovery",0.);
    obs_data_set_default_int(s,"residual_format",0);
    obs_data_set_default_double(s,"temporal_stability",0.);
    obs_data_set_default_double(s,"edge_aware",0.);
    obs_data_set_default_double(s,"depth_guidance",0.);
}
void update(void*data,obs_data_t*s){
    auto*f=static_cast<Filter*>(data);
    auto edge=obs_data_get_int(s,"resolution");
    std::string p=obs_data_get_string(s,"precision");
    if(p!="fp32")p="w8a8";
    double strength=obs_data_get_double(s,"strength");
    if(!std::isfinite(strength))strength=0.;
    auto slider=[&](const char*name,double maximum,double fallback){
        const auto v=obs_data_get_double(s,name);
        return float(std::isfinite(v)?std::clamp(v,0.,maximum):fallback);
    };
    {
        std::lock_guard lock(f->input_mutex);
        const int next=(edge==0||edge==640||edge==960||edge==1280)?int(edge):320;
        const int sync=int(std::clamp<int64_t>(obs_data_get_int(s,"sync_mode"),0,2));
        const int structure=int(std::lround(slider("structure_preservation",1.,1.)*100));
        const int clarity=int(std::lround(slider("input_clarity",1.,0.)*100));
        const int shadows=int(std::lround(slider("shadow_recovery",1.,0.)*100));
        const int residual=int(std::clamp<int64_t>(obs_data_get_int(s,"residual_format"),0,2));
        const int temporal=int(std::lround(slider("temporal_stability",.9,0.)*100));
        if(f->edge!=next||f->precision!=p||f->sync!=sync||f->structure!=structure||
           f->clarity!=clarity||f->shadows!=shadows||f->residual!=residual||f->temporal!=temporal||((f->strength<=0)!=(strength<=0)))
            {++f->generation;f->drawn_size=0;}
        f->edge=next;f->sync=sync;f->structure=structure;f->clarity=clarity;f->shadows=shadows;
        f->precision=p;f->residual=residual;f->temporal=temporal;
        f->strength=float(std::clamp(strength,0.,5.));
        f->relighting=slider("relighting",5.,1.);
        f->surface_detail=slider("surface_detail",5.,1.);
        f->color_strength=slider("color_strength",5.,1.);
        f->highlight_protection=slider("highlight_protection",1.,0.);
        f->edge_aware=slider("edge_aware",1.,0.);
        f->depth_guidance=slider("depth_guidance",1.,0.);
    }
}
void render_stats(void *data,calldata_t *args){
    auto *f=static_cast<Filter*>(data);
    const auto size=f->drawn_size.load();
    calldata_set_int(args,"width",int64_t(size>>32));
    calldata_set_int(args,"height",int64_t(size&0xffffffffu));
}
void sync_stats(void* data,calldata_t* args){
    auto *f=static_cast<Filter*>(data);
    calldata_set_int(args,"submitted_mode",f->submitted_mode.load());
    calldata_set_int(args,"submitted_sequence",f->submitted_sequence.load());
}
void input_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    const uint32_t packed=f->drawn_conditioning.load();
    calldata_set_int(args,"structure",packed&255u);
    calldata_set_int(args,"clarity",(packed>>8)&255u);
    calldata_set_int(args,"shadows",(packed>>16)&255u);
}
void residual_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_int(args,"format",f->drawn_size?f->drawn_residual.load():-1);
}
void temporal_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_float(args,"history_weight",f->drawn_size?f->drawn_history_weight.load():0.f);
}
void depth_stats(void*data,calldata_t*args){
    auto*f=static_cast<Filter*>(data);
    calldata_set_bool(args,"active",f->drawn_size&&f->depth_drawn.load());
}
void* create(obs_data_t*s,obs_source_t*source){
    auto*f=new Filter;
    f->source=source;
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_stats_v1(out int width, out int height)",
        render_stats,f);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_sync_stats_v1(out int submitted_mode, out int submitted_sequence)",
        sync_stats,f);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_input_stats_v1(out int structure, out int clarity, out int shadows)",
        input_stats,f);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_residual_stats_v1(out int format)",residual_stats,f);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_temporal_stats_v1(out float history_weight)",temporal_stats,f);
    proc_handler_add(obs_source_get_proc_handler(source),
        "void sight_neural_rendering_depth_stats_v1(out bool active)",depth_stats,f);
    if(const char *env=std::getenv("SIGHT_NR_CONFIG"))f->config=env;
    else if(char *path=obs_module_config_path("neural-rendering.json")){
        f->config=path;bfree(path);
    }
    if(char *path=obs_module_file("neural-rendering/worker.py")){
        // OBS may supply a relative module data root (notably staged/local
        // launches). Resolve only the trusted module-owned worker path;
        // explicit config paths still require absolute local paths.
        std::error_code ec;
        const auto absolute=std::filesystem::absolute(path,ec);
        if(!ec)f->packaged_worker=absolute.lexically_normal().string();
        bfree(path);
    }
    update(f,s);
    obs_enter_graphics();
    if(char *path=obs_module_file("sight-neural-rendering.effect")){
        char *errors=nullptr;
        f->effect=gs_effect_create_from_file(path,&errors);
        if(errors){blog(LOG_WARNING,"[Sight/NR] shader: %s",errors);bfree(errors);}
        bfree(path);
    }
    obs_leave_graphics();
    f->worker=std::thread(worker,f);
    return f;
}
void destroy(void*data){
    auto*f=static_cast<Filter*>(data);if(!f)return;
    {std::lock_guard lock(f->input_mutex);f->stop=true;}
    f->wake.notify_all();
    if(f->worker.joinable())f->worker.join();
    obs_enter_graphics();
    if(f->texture)gs_texture_destroy(f->texture);
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->stage)gs_stagesurface_destroy(f->stage);
    if(f->locked_pending)gs_texrender_destroy(f->locked_pending);
    if(f->locked_display)gs_texrender_destroy(f->locked_display);
    if(f->motion_capture)gs_texrender_destroy(f->motion_capture);
    if(f->motion_stage)gs_stagesurface_destroy(f->motion_stage);
    if(f->reference_texture)gs_texture_destroy(f->reference_texture);
    if(f->flow_texture)gs_texture_destroy(f->flow_texture);
    if(f->depth_texture)gs_texture_destroy(f->depth_texture);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();
    delete f;
}
obs_properties_t*properties(void*){
    auto*p=obs_properties_create();
    auto*r=obs_properties_add_list(p,"resolution","Inference long edge",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(r,"320 (Realtime target)",320);
    obs_property_list_add_int(r,"640 (More detail)",640);
    obs_property_list_add_int(r,"960 (High detail)",960);
    obs_property_list_add_int(r,"1280 (Heavy)",1280);
    obs_property_list_add_int(r,"Full source resolution (slow)",0);
    auto*q=obs_properties_add_list(p,"precision","Precision",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(q,"Selective INT8","w8a8");
    obs_property_list_add_string(q,"FP32","fp32");
    auto*residual=obs_properties_add_list(p,"residual_format","Neural correction precision",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(residual,"RGB24 (compatible)",0);
    obs_property_list_add_int(residual,"Float32 (fine correction)",1);
    obs_property_list_add_int(residual,"Float32 (unclipped correction)",2);
    obs_property_set_long_description(residual,"Float modes require the updated local worker. Unclipped retains predictions before display-range clipping; use highlight / shadow protection when amplifying. Changes restart the worker.");
    auto *sync=obs_properties_add_list(p,"sync_mode","Synchronization",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(sync,"Live (lowest latency)",0);
#ifdef SIGHT_HAVE_DENSE_FLOW
    obs_property_list_add_int(sync,"Motion compensated",1);
#else
    obs_property_list_add_int(sync,"Motion gated (no optical flow)",1);
#endif
    obs_property_list_add_int(sync,"Frame locked (exact alignment)",2);
    obs_property_set_long_description(sync,"Motion compensated tracks the captured frame and fades uncertain corrections. Frame locked holds the full-resolution input until its corresponding neural result is ready; video may lag audio.");
    auto*temporal=obs_properties_add_float_slider(p,"temporal_stability","Temporal stability",0.,.9,.05);
    obs_property_set_long_description(temporal,"Motion-compensated mode at inference long edge up to 640. Blend only the preceding raw correction at matching source pixels. Zero disables history. Changes restart the worker.");
    auto*input=obs_properties_create();
    auto*structure=obs_properties_add_float_slider(input,"structure_preservation","Structure preservation",0.,1.,.01);
    obs_property_set_long_description(structure,"Experimental network conditioning. 1 preserves the original conditioning; lower values alter the network's structure input. Changes restart the worker.");
    auto*clarity=obs_properties_add_float_slider(input,"input_clarity","Input clarity",0.,1.,.01);
    obs_property_set_long_description(clarity,"Edge-aware suppression of small noise and compression variations before inference only.");
    auto*shadows=obs_properties_add_float_slider(input,"shadow_recovery","Shadow recovery",0.,1.,.01);
    obs_property_set_long_description(shadows,"Lift low and mid tones for neural inference only. The original source exposure is preserved in compositing.");
    obs_properties_add_group(p,"neural_input","Neural input",OBS_GROUP_NORMAL,input);
    obs_properties_add_float_slider(p,"strength","Strength multiplier",0.,5.,.05);
    auto *light=obs_properties_add_float_slider(p,"relighting","Relighting",0.,5.,.05);
    obs_property_set_long_description(light,"Amplifies broad luminance changes predicted by the neural renderer. 1 is neutral.");
    auto *detail=obs_properties_add_float_slider(p,"surface_detail","Surface detail",0.,5.,.05);
    obs_property_set_long_description(detail,"Amplifies localized luminance changes in the neural residual, not sharpening of the original source. Native inference preserves the finest available neural variations.");
    auto *color=obs_properties_add_float_slider(p,"color_strength","Color",0.,5.,.05);
    obs_property_set_long_description(color,"Amplifies chromatic changes predicted by the neural renderer. 1 is neutral.");
    auto*edge_aware=obs_properties_add_float_slider(p,"edge_aware","Edge-aware upscaling",0.,1.,.05);
    obs_property_set_long_description(edge_aware,"Guide the low-resolution neural correction with full-resolution source color and alpha. Reduces correction bleed at visible boundaries. Zero uses ordinary interpolation; native inference is unchanged. No model restart.");
    auto*depth=obs_properties_add_float_slider(p,"depth_guidance","Depth guidance",0.,1.,.05);
    obs_property_set_long_description(depth,"Reuse a Depth Camera filter immediately upstream in Original source mode. Fresh relative depth guides correction interpolation and relighting separation. Live/motion-compensated only; falls back to RGB guidance if unavailable. No additional model.");
    auto *protect=obs_properties_add_float_slider(p,"highlight_protection","Highlight / shadow protection",0.,1.,.01);
    obs_property_set_long_description(protect,"Softly compresses amplified corrections near RGB limits to reduce clipping without changing their color direction.");
    auto*t=obs_properties_add_text(p,"local_backend","Experimental: local Small31 CUDA worker. No bundled model. Missing backend passes through.",OBS_TEXT_INFO);
    obs_property_text_set_info_type(t,OBS_TEXT_INFO_NORMAL);
    return p;
}
const char* name(void*){return "LLL.sight Neural Rendering (Experimental)";}
bool capture_source(gs_texrender_t *&capture,obs_source_t *target,uint32_t cw,uint32_t ch,uint32_t w,uint32_t h){
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
bool capture_locked(Filter*f,obs_source_t*target,const Request&key){
    if(!capture_source(f->locked_pending,target,key.source_w,key.source_h,key.source_w,key.source_h))return false;
    if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(!f->capture)return false;
    gs_texrender_reset(f->capture);
    if(!gs_texrender_begin(f->capture,key.w,key.h))return false;
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0,float(key.w),0,float(key.h),-100.f,100.f);
    auto*tex=gs_texrender_get_texture(f->locked_pending);
    auto*base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
    if(base){
        gs_effect_set_texture(gs_effect_get_param_by_name(base,"image"),tex);
        while(gs_effect_loop(base,"Draw"))gs_draw_sprite(tex,0,key.w,key.h);
    }
    gs_blend_state_pop();
    gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
    return base&&gs_texrender_get_texture(f->capture);
}
void update_motion(Filter*f,obs_source_t*target,const Request&key){
    if(!f->reference||!f->reference_texture)return;
    const auto now=Clock::now();
    const auto mw=uint32_t(f->reference->rgba.cols),mh=uint32_t(f->reference->rgba.rows);
    if(now>=f->next_motion){
        f->next_motion=now+std::chrono::milliseconds(50);
        if(!f->motion_stage||mw!=f->motion_w||mh!=f->motion_h){
            if(f->motion_stage)gs_stagesurface_destroy(f->motion_stage);
            f->motion_stage=gs_stagesurface_create(mw,mh,GS_RGBA);
            f->motion_w=mw;f->motion_h=mh;
        }
        if(f->motion_stage&&capture_source(f->motion_capture,target,mw,mh,key.source_w,key.source_h)){
            gs_stage_texture(f->motion_stage,gs_texrender_get_texture(f->motion_capture));
            uint8_t *bytes=nullptr;uint32_t pitch=0;
            if(gs_stagesurface_map(f->motion_stage,&bytes,&pitch)){
                if(pitch>=mw*4){
                    cv::Mat frame(int(mh),int(mw),CV_8UC4,bytes,pitch);
                    f->motion.submit(f->reference,frame.clone(),now);
                    f->motion_at=now;
                }
                gs_stagesurface_unmap(f->motion_stage);
            }
        }
    }
    sight::nr::MotionResult motion;
    if(f->motion.latest(f->reference->sequence,motion)){
        if(!f->flow_texture)f->flow_texture=gs_texture_create(SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,1,nullptr,GS_DYNAMIC);
        if(f->flow_texture){
            std::array<float,SIGHT_FLOW_CELLS*4> grid{};
            for(size_t i=0;i<SIGHT_FLOW_CELLS;++i){
                const auto&c=motion.flow.cells[i];
                grid[i*4]=c.dx*c.confidence;grid[i*4+1]=c.dy*c.confidence;grid[i*4+2]=c.confidence;
            }
            gs_texture_set_image(f->flow_texture,reinterpret_cast<const uint8_t*>(grid.data()),SIGHT_FLOW_WIDTH*4*sizeof(float),false);
            f->motion_sequence=motion.sequence;f->motion_coverage=motion.flow.coverage;
        }
    }else{f->motion_sequence=0;f->motion_coverage=0.f;}
}

// Only the direct upstream producer can provide a geometrically compatible
// snapshot. Do not search unrelated sources or silently launch a model.
bool update_depth(Filter*f,obs_source_t*target,const Request&key){
    if(key.sync==2||f->depth_guidance<=0.f||!target||
       std::strcmp(obs_source_get_unversioned_id(target),"sight_depth_filter")!=0||
       !obs_source_enabled(target)){
        f->depth_sequence=0;f->depth_owner.clear();return false;
    }
    const char* uuid=obs_source_get_uuid(target);
    if(!uuid||!*uuid)return false;
    if(f->depth_owner!=uuid){f->depth_owner=uuid;f->depth_sequence=0;}
    if(!f->depth_snapshot)f->depth_snapshot=std::make_unique<sight_depth_v1>();
    auto&s=*f->depth_snapshot;
    if(!sight::nr::read_depth_snapshot(obs_source_get_proc_handler(target),s,key.source_w,key.source_h,f->depth_sequence)){
        f->depth_sequence=0;return false;
    }
    if(!f->depth_texture||gs_texture_get_width(f->depth_texture)!=s.width||gs_texture_get_height(f->depth_texture)!=s.height){
        // The producer can omit pixels for a known sequence. Request a
        // complete snapshot only in the unexpected same-sequence resize case.
        if(f->depth_sequence==s.sequence&&
           !sight::nr::read_depth_snapshot(obs_source_get_proc_handler(target),s,key.source_w,key.source_h,0)){
            f->depth_sequence=0;return false;
        }
        if(f->depth_texture)gs_texture_destroy(f->depth_texture);
        f->depth_texture=gs_texture_create(s.width,s.height,GS_R8,1,nullptr,GS_DYNAMIC);
        f->depth_sequence=0;
    }
    if(!f->depth_texture){f->depth_sequence=0;return false;}
    if(f->depth_sequence!=s.sequence){
        gs_texture_set_image(f->depth_texture,s.pixels,s.width,false);
        f->depth_sequence=s.sequence;
        if(!f->depth_announced){
            blog(LOG_INFO,"[Sight/NR] SIGHT_NR_DEPTH_GUIDE_20261124_D61 active=%ux%u",s.width,s.height);
            f->depth_announced=true;
        }
    }
    return true;
}
void render(void*data,gs_effect_t*){
    auto*f=static_cast<Filter*>(data);
    auto*target=obs_filter_get_target(f->source);
    f->depth_drawn=false;
    auto passthrough=[&]{f->drawn_size=0;obs_source_skip_video_filter(f->source);};
    if(!target||!obs_filter_get_parent(f->source)||!f->effect||f->strength<=0){passthrough();return;}
    uint32_t w=obs_source_get_base_width(target),h=obs_source_get_base_height(target);
    if(!w||!h||w>8192||h>8192){passthrough();return;}
    Request key;
    {
        std::unique_lock lock(f->input_mutex,std::try_to_lock);
        if(!lock.owns_lock()){passthrough();return;}
        const auto edge=uint32_t(f->edge.load());
        key.generation=f->generation;key.precision=f->precision;key.sync=f->sync;
        key.structure=f->structure;key.clarity=f->clarity;key.shadows=f->shadows;key.residual=f->residual;key.temporal=f->temporal;
        key.source_w=w;key.source_h=h;
        const uint32_t limit=edge?std::min(edge,std::max(w,h)):std::max(w,h);
        key.w=uint32_t(uint64_t(w)*limit/std::max(w,h));
        key.h=uint32_t(uint64_t(h)*limit/std::max(w,h));
    }
    if(std::min(key.w,key.h)<33){passthrough();return;}
    // GPU snapshots belong only to the frame-locked path. The worker owns
    // RGB bytes, not these textures, so switching modes can release them.
    if(key.sync!=2){
        if(f->locked_pending){gs_texrender_destroy(f->locked_pending);f->locked_pending=nullptr;}
        if(f->locked_display){gs_texrender_destroy(f->locked_display);f->locked_display=nullptr;}
    }
    if(key.sync!=1){
        f->reference.reset();
        if(f->reference_texture){gs_texture_destroy(f->reference_texture);f->reference_texture=nullptr;}
        if(f->flow_texture){gs_texture_destroy(f->flow_texture);f->flow_texture=nullptr;}
        if(f->motion_capture){gs_texrender_destroy(f->motion_capture);f->motion_capture=nullptr;}
        if(f->motion_stage){gs_stagesurface_destroy(f->motion_stage);f->motion_stage=nullptr;}
        f->motion_at={};f->next_motion={};
    }
    bool available=false;
    const bool capture_due=f->busy.compare_exchange_strong(available,true);
    {
        Result result;
        {
            std::unique_lock lock(f->output_mutex,std::try_to_lock);
            if(lock.owns_lock()&&!f->output.delta.empty())result=std::move(f->output);
        }
        if(!result.delta.empty()&&same(key,result.key)&&Clock::now()-result.key.completed<=output_age_limit(result.key)){
            if(!f->texture||gs_texture_get_width(f->texture)!=key.w||gs_texture_get_height(f->texture)!=key.h){
                if(f->texture)gs_texture_destroy(f->texture);
                f->texture=gs_texture_create(key.w,key.h,GS_RGBA32F,1,nullptr,GS_DYNAMIC);
            }
            if(f->texture&& (key.sync!=2||result.key.sequence==f->pending_sequence)){
                if(key.sync==2)std::swap(f->locked_pending,f->locked_display);
                f->reference=std::move(result.reference);
                f->motion_sequence=0;f->motion_coverage=0.f;f->next_motion={};f->motion_at={};
                if(f->reference){
                    const auto rw=uint32_t(f->reference->rgba.cols),rh=uint32_t(f->reference->rgba.rows);
                    if(!f->reference_texture||gs_texture_get_width(f->reference_texture)!=rw||gs_texture_get_height(f->reference_texture)!=rh){
                        if(f->reference_texture)gs_texture_destroy(f->reference_texture);
                        f->reference_texture=gs_texture_create(rw,rh,GS_RGBA,1,nullptr,GS_DYNAMIC);
                    }
                    if(f->reference_texture)gs_texture_set_image(f->reference_texture,f->reference->rgba.data,
                        uint32_t(f->reference->rgba.step),false);
                }
                gs_texture_set_image(f->texture,reinterpret_cast<const uint8_t*>(result.delta.data()),key.w*4*sizeof(float),false);
                f->displayed=std::move(result.key);
            }
        }
    }
    if(capture_due){
        bool queued=false;
        if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->stage||f->stage_w!=key.w||f->stage_h!=key.h){
            if(f->stage)gs_stagesurface_destroy(f->stage);
            f->stage=gs_stagesurface_create(key.w,key.h,GS_RGBA);
            f->stage_w=key.w;f->stage_h=key.h;
        }
        if(f->stage){
            if(key.sync==2?capture_locked(f,target,key):capture_source(f->capture,target,key.w,key.h,w,h)){
                if(auto*tex=gs_texrender_get_texture(f->capture)){
                    gs_stage_texture(f->stage,tex);
                    uint8_t*bytes=nullptr;uint32_t pitch=0;
                    if(gs_stagesurface_map(f->stage,&bytes,&pitch)){
                        if(pitch>=key.w*4){
                            Request r=key;r.captured=Clock::now();r.sequence=++f->next_sequence;
                            r.rgb.resize(size_t(key.w)*key.h*3);
                            if(key.sync==1&&key.temporal>0&&std::max(key.w,key.h)<=640)
                                r.alpha.resize(size_t(key.w)*key.h);
                            for(uint32_t y=0;y<key.h;++y)
                                for(uint32_t x=0;x<key.w;++x){
                                    const auto i=size_t(y)*key.w+x;
                                    const auto*pixel=bytes+size_t(y)*pitch+x*4;
                                    std::memcpy(r.rgb.data()+i*3,pixel,3);
                                    if(!r.alpha.empty())r.alpha[i]=pixel[3];
                                }
                            std::unique_lock lock(f->input_mutex,std::try_to_lock);
                            if(lock.owns_lock()&&f->generation==key.generation&&!f->stop){
                                f->pending_sequence=r.sequence;
                                f->submitted_sequence=r.sequence;f->submitted_mode=r.sync;
                                f->input=std::move(r);f->queued=true;queued=true;f->wake.notify_one();
                            }
                        }
                        gs_stagesurface_unmap(f->stage);
                    }
                }
            }
        }
        if(!queued)f->busy=false;
    }
    if(!f->texture||!same(key,f->displayed)||f->generation!=key.generation||
       Clock::now()-f->displayed.completed>output_age_limit(f->displayed)){passthrough();return;}
    auto*correction=gs_effect_get_param_by_name(f->effect,"correction");
    auto*strength=gs_effect_get_param_by_name(f->effect,"strength");
    if(!correction||!strength){passthrough();return;}
    vec2 correction_size{};vec2_set(&correction_size,float(key.w),float(key.h));
    gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"correction_size"),&correction_size);
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"relighting"),f->relighting.load());
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"surface_detail"),f->surface_detail.load());
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"color_strength"),f->color_strength.load());
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"highlight_protection"),f->highlight_protection.load());
    gs_effect_set_int(gs_effect_get_param_by_name(f->effect,"sync_mode"),key.sync);
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"edge_aware"),
        (key.w<w||key.h<h)?f->edge_aware.load():0.f);
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"flow_enabled"),0.f);
    // libobs binds all shader uniforms even when the runtime branch is inactive.
    vec2 fallback_size{};vec2_set(&fallback_size,float(key.w),float(key.h));
    gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"proxy_size"),&fallback_size);
    gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"reference_image"),f->texture);
    gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"motion_grid"),f->texture);
    if(key.sync==1){
        update_motion(f,target,key);
        if(!f->reference_texture||!f->motion_capture||Clock::now()-f->motion_at>std::chrono::milliseconds(150)){
            passthrough();return;
        }
        auto*proxy=gs_texrender_get_texture(f->motion_capture);
        if(!proxy){passthrough();return;}
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"reference_image"),f->reference_texture);
        vec2 proxy_size{};vec2_set(&proxy_size,float(f->motion_w),float(f->motion_h));
        gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"proxy_size"),&proxy_size);
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"motion_grid"),f->flow_texture?f->flow_texture:f->reference_texture);
        gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"flow_enabled"),
            f->motion_sequence==f->displayed.sequence&&f->motion_coverage>0.f?1.f:0.f);
    }
    const bool depth_ready=update_depth(f,target,key);
    gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"depth_image"),depth_ready?f->depth_texture:f->texture);
    gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"depth_guidance"),depth_ready?f->depth_guidance.load():0.f);
    if(key.sync==2){
        auto*tex=f->locked_display?gs_texrender_get_texture(f->locked_display):nullptr;
        if(!tex){passthrough();return;}
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"image"),tex);
        gs_effect_set_texture(correction,f->texture);
        gs_effect_set_float(strength,f->strength.load());
        while(gs_effect_loop(f->effect,"Draw"))gs_draw_sprite(tex,0,w,h);
        f->drawn_conditioning=uint32_t(key.structure)|(uint32_t(key.clarity)<<8)|(uint32_t(key.shadows)<<16);
        f->drawn_residual=key.residual;
        f->drawn_history_weight=f->displayed.history_weight;
        f->drawn_size=(uint64_t(key.w)<<32)|key.h;
        return;
    }
    if(obs_source_process_filter_begin(f->source,GS_RGBA,OBS_NO_DIRECT_RENDERING)){
        gs_effect_set_texture(correction,f->texture);
        gs_effect_set_float(strength,f->strength.load());
        obs_source_process_filter_end(f->source,f->effect,w,h);
        f->depth_drawn=depth_ready;
        f->drawn_conditioning=uint32_t(key.structure)|(uint32_t(key.clarity)<<8)|(uint32_t(key.shadows)<<16);
        f->drawn_residual=key.residual;
        f->drawn_history_weight=f->displayed.history_weight;
        f->drawn_size=(uint64_t(key.w)<<32)|key.h;
    }
}
}
struct obs_source_info sight_neural_rendering_filter_info={
    .id="sight_neural_rendering_filter",
    .type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=name,.create=create,.destroy=destroy,.get_defaults=defaults,
    .get_properties=properties,.update=update,.video_render=render,
};
