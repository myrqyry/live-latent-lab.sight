// Real OBS filter instances, shared process/device, independent inputs.
#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <callback/proc.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
constexpr unsigned W=320,H=192;
struct Pattern{gs_texture_t* texture=nullptr;};
static const char* name(void*){return "Concurrent CUDA fixture";}
static unsigned width(void*){return W;}
static unsigned height(void*){return H;}
static void* create(obs_data_t*,obs_source_t*){
    auto* p=new Pattern;
    std::vector<uint8_t> pixels(W*H*4);
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
        size_t i=(y*W+x)*4;pixels[i]=x>60&&x<260?245:35;
        pixels[i+1]=y>45&&y<152?30:180;pixels[i+2]=(x+y)%256;pixels[i+3]=255;
    }
    const auto* bytes=pixels.data();
    obs_enter_graphics();p->texture=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);obs_leave_graphics();
    return p;
}
static void destroy(void* data){
    auto* p=static_cast<Pattern*>(data);
    obs_enter_graphics();if(p->texture)gs_texture_destroy(p->texture);obs_leave_graphics();delete p;
}
static void draw(void* data,gs_effect_t*){
    auto* p=static_cast<Pattern*>(data);if(p->texture)obs_source_draw(p->texture,0,0,W,H,false);
}
struct Case{
    const char* name;const char* id;const char* mode;const char* key;const char* proc;const char* field;
    obs_source_t* source=nullptr;obs_source_t* filter=nullptr;
    int64_t start=0,last=0;
};
static int64_t read_stat(const Case& c,const char* proc,const char* field){
    if(!c.filter)return -1;
    calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(c.filter),proc,&d);
    auto n=calldata_int(&d,field);calldata_free(&d);return ok?n:-1;
}
static int64_t completed(const Case& c){return read_stat(c,c.proc,c.field);}
static void queue_ui(obs_task_t task,void* param,bool){task(param);}
int main(int argc,char** argv){
    if(argc!=10){std::cerr<<"Usage: module data host|device sketch depth cutout hed teed pidinet\n";return 2;}
    const bool gpu=std::string(argv[3])=="device";
    if(!gpu&&std::string(argv[3])!="host")return 2;
    unsigned rounds=180;
    if(const char* value=std::getenv("SIGHT_CONCURRENT_ROUNDS"))
        rounds=unsigned(std::clamp(std::atoi(value),180,1800));
    setenv("SIGHT_ONNX_DEVICE","cuda",1);
    for(auto key:{"SIGHT_DEPTH_CUDA_IO","SIGHT_CUTOUT_CUDA_IO","SIGHT_HED_CUDA_IO","SIGHT_TEED_CUDA_IO","SIGHT_PIDINET_CUDA_IO"})setenv(key,argv[3],1);
    if(!obs_startup("en-US",nullptr,nullptr))return 1;
    obs_set_ui_task_handler(queue_ui);
    obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=v.output_width=W;v.base_height=v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;
    v.colorspace=VIDEO_CS_DEFAULT;v.range=VIDEO_RANGE_DEFAULT;v.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS){obs_shutdown();return 1;}
    obs_module_t* module=nullptr;
    if(obs_open_module(&module,argv[1],argv[2])!=MODULE_SUCCESS||!obs_init_module(module)){obs_shutdown();return 1;}
    obs_source_info info{};info.id="sight_concurrent_pattern";info.type=OBS_SOURCE_TYPE_INPUT;
    info.output_flags=OBS_SOURCE_VIDEO;info.get_name=name;info.create=create;info.destroy=destroy;
    info.get_width=width;info.get_height=height;info.video_render=draw;obs_register_source(&info);
    std::array<Case,6> cases{{
        {"sketch","neural_enhance_filter",nullptr,"model_path",gpu?"sight_sketch_gpu_stats_v1":"sight_sketch_capture_stats_v1",gpu?"uploads":"completed"},
        {"depth","sight_depth_filter",nullptr,"model_path","sight_depth_buffer_stats_v1","uploads"},
        {"cutout","sight_u2net_filter",nullptr,"model_path","sight_cutout_buffer_stats_v1","uploads"},
        {"hed","sight_preprocessors_filter","hed","hed_model_path","sight_preprocessor_hed_stats_v1","hed_completed"},
        {"teed","sight_preprocessors_filter","teed","teed_model_path","sight_preprocessor_teed_stats_v1","teed_completed"},
        {"pidinet","sight_preprocessors_filter","pidinet","pidinet_model_path","sight_preprocessor_pidinet_stats_v1","pidinet_completed"}
    }};
    bool ok=true;
    for(size_t i=0;i<cases.size();++i){
        auto& c=cases[i];c.source=obs_source_create("sight_concurrent_pattern",c.name,nullptr,nullptr);
        auto* s=obs_data_create();obs_data_set_string(s,c.key,argv[4+i]);
        if(c.mode)obs_data_set_string(s,"mode_id",c.mode);
        if(i==0){obs_data_set_bool(s,"cuda_texture",gpu);obs_data_set_int(s,"detail_size",256);}
        if(i==1)obs_data_set_int(s,"inference_edge",252);
        if(i==2){obs_data_set_int(s,"cutout_device",2);obs_data_set_int(s,"cutout_model_family",0);}
        c.filter=obs_source_create(c.id,c.name,s,nullptr);obs_data_release(s);
        if(!c.source||!c.filter){ok=false;break;}
        obs_source_filter_add(c.source,c.filter);
    }
    obs_enter_graphics();auto* target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);obs_leave_graphics();
    auto cycle=[&]{
        obs_enter_graphics();
        for(auto& c:cases){
            gs_texrender_reset(target);
            if(!gs_texrender_begin(target,W,H)){obs_leave_graphics();return false;}
            gs_projection_push();gs_matrix_push();gs_matrix_identity();gs_ortho(0,float(W),0,float(H),-100,100);
            obs_source_video_render(c.source);
            gs_matrix_pop();gs_projection_pop();gs_texrender_end(target);
        }
        obs_leave_graphics();return true;
    };
    if(ok&&target){
        auto until=Clock::now()+std::chrono::seconds(30);
        bool warm=false;
        while(Clock::now()<until){
            if(!cycle())break;
            warm=true;for(auto& c:cases)if(completed(c)<4)warm=false;
            if(warm)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
        ok=warm;
        if(warm){
            for(auto& c:cases)c.start=completed(c);
            std::vector<double> times;auto begin=Clock::now();unsigned missed=0;
            for(unsigned i=0;i<rounds;++i){
                auto start=Clock::now();if(!cycle()){ok=false;break;}
                times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
                auto deadline=begin+std::chrono::milliseconds((i+1)*1000/30);
                if(Clock::now()>deadline)++missed;
                else std::this_thread::sleep_until(deadline);
            }
            double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
            for(auto& c:cases){
                c.last=completed(c);ok=ok&&c.last-c.start>=int64_t(rounds/9);
                if(gpu&&std::string(c.name)=="sketch"){
                    if(const char* minimum=std::getenv("SIGHT_CONCURRENT_MIN_SKETCH"))
                        ok=ok&&c.last-c.start>=std::atoi(minimum);
                }
                std::cout<<"SIGHT_CONCURRENT_EFFECT "<<c.name<<" completed="<<c.last-c.start<<" rate="<<(c.last-c.start)/elapsed<<"\n";
            }
            if(gpu)ok=ok&&read_stat(cases[0],"sight_sketch_gpu_stats_v1","failures")==0&&
                read_stat(cases[0],"sight_sketch_gpu_stats_v1","readbacks")==0;
            for(size_t i=3;i<cases.size();++i)
                ok=ok&&read_stat(cases[i],"sight_preprocessor_stats_v1","failed")==0&&
                    read_stat(cases[i],"sight_preprocessor_buffer_stats_v1","uploads")>=cases[i].last-2;
            if(!times.empty()){
                std::sort(times.begin(),times.end());
                std::cout<<"SIGHT_CONCURRENT "<<argv[3]<<" rounds="<<times.size()<<" median_ms="<<times[times.size()/2]<<" p95_ms="<<times[times.size()*95/100]<<" max_ms="<<times.back()<<" deadline_misses="<<missed<<" elapsed="<<elapsed<<" pass="<<ok<<std::endl;
            }
        }else{
            for(auto& c:cases)std::cerr<<"Warmup "<<c.name<<" completed="<<completed(c)<<"\n";
        }
    }else ok=false;
    obs_enter_graphics();if(target)gs_texrender_destroy(target);obs_leave_graphics();
    for(auto& c:cases){if(c.source&&c.filter)obs_source_filter_remove(c.source,c.filter);if(c.filter)obs_source_release(c.filter);if(c.source)obs_source_release(c.source);}
    obs_shutdown();return ok?0:1;
}
