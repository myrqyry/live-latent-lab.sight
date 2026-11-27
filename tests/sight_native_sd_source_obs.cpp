// Verify the optional native generator registers inside actual libobs.
// Optional local model-backed acceptance; no scenes or OBS installation writes.
#include <obs.h>
#include <obs-module.h>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <graphics/graphics.h>
#include <callback/proc.h>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <filesystem>
#include <fstream>
#include <set>

namespace {
struct Stats {int64_t frames=0,feedback=0,guided=0;bool running=false,loaded=false;std::string status;};
Stats stats(obs_source_t*s){
    Stats out;calldata_t args;calldata_init(&args);
    if(proc_handler_call(obs_source_get_proc_handler(s),"sight_generator_stats_v1",&args)){
        out.frames=calldata_int(&args,"frames");out.feedback=calldata_int(&args,"feedback");
        out.guided=calldata_int(&args,"guided");out.running=calldata_bool(&args,"running");
        out.loaded=calldata_bool(&args,"loaded");const char*status=calldata_string(&args,"status");if(status)out.status=status;
    }
    calldata_free(&args);return out;
}
struct Preview {int64_t draws=0;float phase=1;bool available=false;};
Preview preview(obs_source_t*s){
    Preview out;calldata_t args;calldata_init(&args);
    if(proc_handler_call(obs_source_get_proc_handler(s),"sight_generator_preview_stats_v1",&args)){
        out.draws=calldata_int(&args,"draws");out.phase=float(calldata_float(&args,"phase"));out.available=calldata_bool(&args,"available");
    }
    calldata_free(&args);return out;
}
std::pair<int64_t,float> flow_stats(obs_source_t*s){
    calldata_t args;calldata_init(&args);
    proc_handler_call(obs_source_get_proc_handler(s),"sight_generator_preview_flow_stats_v1",&args);
    auto out=std::pair{calldata_int(&args,"draws"),float(calldata_float(&args,"coverage"))};
    calldata_free(&args);return out;
}
struct ExportDir {
    std::filesystem::path path;
    bool create(){
        auto pattern=(std::filesystem::temp_directory_path()/"sight-recursive-acceptance.XXXXXX").string();
        if(!mkdtemp(pattern.data()))return false;
        path=pattern;return true;
    }
    ~ExportDir(){if(!path.empty()){std::error_code ec;std::filesystem::remove_all(path,ec);}}
};
bool read_ppm(const std::filesystem::path&path,unsigned size,std::vector<uint8_t>&rgb){
    std::ifstream f(path,std::ios::binary);
    std::string magic;unsigned w=0,h=0,max=0;
    if(!(f>>magic>>w>>h>>max)||magic!="P6"||w!=size||h!=size||max!=255||f.get()!='\n')return false;
    rgb.resize(size_t(w)*h*3);
    return bool(f.read(reinterpret_cast<char*>(rgb.data()),rgb.size()))&&f.peek()==EOF;
}
bool verify_exports(const ExportDir&dir,unsigned size,int count,const std::vector<uint8_t>&last){
    if(last.size()!=size_t(size)*size*4)return false;
    std::set<int> indices;std::vector<std::vector<uint8_t>> frames;
    std::vector<uint8_t> final_rgb;
    for(const auto&entry:std::filesystem::directory_iterator(dir.path)){
        auto name=entry.path().stem().string();auto sep=name.rfind('_');
        if(entry.path().extension()!=".ppm"||sep==std::string::npos)return false;
        int index=-1;try{index=std::stoi(name.substr(sep+1));}catch(...){return false;}
        std::vector<uint8_t> rgb;
        if(index<0||index>=count||!indices.insert(index).second||!read_ppm(entry.path(),size,rgb))return false;
        if(index==count-1)final_rgb=rgb;
        if(std::find(frames.begin(),frames.end(),rgb)!=frames.end())return false;
        frames.push_back(std::move(rgb));
    }
    bool exact=final_rgb.size()==size_t(size)*size*3;
    if(exact)for(size_t i=0;i<final_rgb.size()/3;++i){
        for(unsigned c=0;c<3;++c)exact&=final_rgb[i*3+c]==last[i*4+c];
        exact&=last[i*4+3]==255;
    }
    std::printf("exported_frames=%zu distinct_exports=%zu final_pixels_exact=%d\n",indices.size(),frames.size(),int(exact));
    return exact&&indices.size()==size_t(count)&&frames.size()==size_t(count);
}
bool command(obs_source_t*s,const char*name){
    calldata_t args;calldata_init(&args);bool ok=proc_handler_call(obs_source_get_proc_handler(s),name,&args);
    calldata_free(&args);return ok;
}
bool draw(obs_source_t*s,std::vector<uint8_t>&out){
    const unsigned W=obs_source_get_width(s),H=obs_source_get_height(s);
    if(!W||!H||W>1024||H>1024)return false;
    obs_enter_graphics();
    auto*stage=gs_stagesurface_create(W,H,GS_RGBA);
    auto*target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    bool ok=false;
    if(stage&&target&&gs_texrender_begin(target,W,H)){
        gs_projection_push();gs_matrix_push();gs_matrix_identity();gs_ortho(0,float(obs_source_get_width(s)),0,float(obs_source_get_height(s)),-100,100);
        obs_source_video_render(s);
        gs_matrix_pop();gs_projection_pop();gs_texrender_end(target);
        gs_stage_texture(stage,gs_texrender_get_texture(target));
        uint8_t*data=nullptr;uint32_t stride=0;
        if(gs_stagesurface_map(stage,&data,&stride)){
            if(stride<W*4){gs_stagesurface_unmap(stage);gs_stagesurface_destroy(stage);gs_texrender_destroy(target);obs_leave_graphics();return false;}
            out.resize(W*H*4);
            for(unsigned y=0;y<H;++y)std::memcpy(out.data()+y*W*4,data+y*stride,W*4);
            gs_stagesurface_unmap(stage);ok=true;
        }
    }
    if(stage)gs_stagesurface_destroy(stage);
    if(target)gs_texrender_destroy(target);
    obs_leave_graphics();return ok;
}
}
int main(int argc, char** argv) {
    if (argc != 3 || !obs_startup("en-US", nullptr, nullptr)) return 2;
    obs_video_info video{};
    video.graphics_module = "libobs-opengl";
    video.fps_num = 30; video.fps_den = 1;
    video.base_width = 128; video.base_height = 128;
    video.output_width = 128; video.output_height = 128;
    video.output_format = VIDEO_FORMAT_RGBA;
    video.scale_type = OBS_SCALE_DISABLE;
    if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) { obs_shutdown(); return 3; }

    obs_module_t* module = nullptr;
    if (obs_open_module(&module, argv[1], argv[2]) != MODULE_SUCCESS ||
        !obs_init_module(module)) { obs_shutdown(); return 4; }

    const char* type = nullptr;
    size_t index = 0;
    bool registered = false;
    while (obs_enum_input_types(index++, &type)) {
        if (type && std::strcmp(type, "ai_image_generator") == 0) registered = true;
    }
    if (!registered) {
        std::fprintf(stderr, "Native Sight generator is not registered\n");
        obs_shutdown(); return 5;
    }
    obs_properties_t* props = obs_get_source_properties("ai_image_generator");
    if (!props) { obs_shutdown(); return 6; }
    for (const char* key : {"model_path", "diffusion_model_path", "high_noise_diffusion_model_path",
                            "vae_path", "taesd_path", "clip_l_path", "llm_path", "lora_path", "prompt",
                            "sampler_name", "scheduler_name", "cache_mode", "distilled_guidance",
                            "feedback_guide","controlnet_path","guide_depth_model","continuous_denoise",
                            "rotation_deg","seed_stride","refresh_interval","stop_generation","motion_schedule","prompt_schedule","preview_ms","frame_limit","exposure_strength"}) {
        if (!obs_properties_get(props, key)) {
            std::fprintf(stderr, "Missing OBS generator property: %s\n", key);
            obs_properties_destroy(props); obs_shutdown(); return 7;
        }
    }
    obs_properties_destroy(props);

    obs_source_t* source = obs_source_create_private(
        "ai_image_generator", "Sight native generator test", nullptr);
    if (!source) { obs_shutdown(); return 8; }
    obs_data_t* settings = obs_source_get_settings(source);
    const bool valid_defaults = settings &&
        obs_data_get_int(settings, "width") == 512 &&
        obs_data_get_int(settings, "height") == 512 &&
        obs_data_get_int(settings, "seed") == -1&&
        obs_data_get_int(settings,"frame_limit")==0&&obs_data_get_double(settings,"exposure_strength")==0.;
    bool passed=valid_defaults&&preview(source).available;
    obs_data_set_string(settings,"device","cpu");
    obs_data_set_bool(settings,"endless_mode",true);
    obs_data_set_string(settings,"feedback_guide","canny");
    obs_source_update(source,settings);
    passed &= command(source,"sight_generator_start_v1");
    for(int i=0;i<100&&stats(source).running;++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto rejected=stats(source);
    passed &= !rejected.running&&!rejected.loaded&&!rejected.frames&&rejected.status.find("matching local ControlNet")!=std::string::npos;
    std::printf("missing_control_rejected=%d\n",int(passed));
    obs_data_set_string(settings,"feedback_guide","off");
    obs_data_set_string(settings,"motion_schedule","0: zoom=0");
    obs_source_update(source,settings);
    passed &= command(source,"sight_generator_start_v1");
    for(int i=0;i<100&&stats(source).running;++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto invalid=stats(source);
    const bool schedule_rejected=!invalid.running&&!invalid.loaded&&!invalid.frames&&invalid.status.find("Invalid schedule line")!=std::string::npos;
    passed &= schedule_rejected;
    std::printf("invalid_schedule_rejected=%d\n",int(schedule_rejected));
    obs_data_set_string(settings,"motion_schedule","");
    if(const char*model=std::getenv("SIGHT_SD_FEEDBACK_MODEL")){
        obs_data_set_string(settings,"model_path",model);
        obs_data_set_string(settings,"device",std::getenv("SIGHT_SD_FEEDBACK_DEVICE")?std::getenv("SIGHT_SD_FEEDBACK_DEVICE"):"cpu");
        if(const char*tae=std::getenv("SIGHT_SD_FEEDBACK_TAE"))obs_data_set_string(settings,"taesd_path",tae);
        obs_data_set_string(settings,"prompt","illustration of a forest cottage beside a river, detailed landscape");
        obs_data_set_string(settings,"sampler_name","lcm");obs_data_set_string(settings,"scheduler_name","lcm");
        const int size=std::getenv("SIGHT_SD_FEEDBACK_SIZE")?std::clamp(std::atoi(std::getenv("SIGHT_SD_FEEDBACK_SIZE")),128,512)/64*64:128;
        obs_data_set_int(settings,"width",size);obs_data_set_int(settings,"height",size);
        obs_data_set_int(settings,"steps",1);obs_data_set_int(settings,"seed",42);
        obs_data_set_int(settings,"seed_stride",1);obs_data_set_double(settings,"cfg_scale",1.);
        obs_data_set_double(settings,"denoising_strength",.35);obs_data_set_double(settings,"pan_x",2.);
        obs_data_set_bool(settings,"continuous_denoise",true);
        const char*control=std::getenv("SIGHT_SD_FEEDBACK_CONTROL");
        const char*guide=std::getenv("SIGHT_SD_FEEDBACK_GUIDE");
        obs_data_set_string(settings,"feedback_guide",control?(guide?guide:"canny"):"off");
        if(const char*depth=std::getenv("SIGHT_SD_FEEDBACK_DEPTH_MODEL"))
            obs_data_set_string(settings,"guide_depth_model",depth);
        const bool refresh=std::getenv("SIGHT_SD_FEEDBACK_REFRESH")!=nullptr;
        if(refresh)obs_data_set_int(settings,"refresh_interval",2);
        if(control)obs_data_set_string(settings,"controlnet_path",control);
        const bool flow=std::getenv("SIGHT_SD_FEEDBACK_PREVIEW_FLOW")!=nullptr;
        const bool smooth=flow||std::getenv("SIGHT_SD_FEEDBACK_PREVIEW")!=nullptr;
        if(flow){
            obs_data_set_bool(settings,"preview_flow",true);
            obs_data_set_double(settings,"denoising_strength",.12);
            obs_data_set_int(settings,"seed_stride",0);
        }
        if(smooth)obs_data_set_int(settings,"preview_ms",400);
        const int target=std::getenv("SIGHT_SD_FEEDBACK_FRAMES")?
            std::clamp(std::atoi(std::getenv("SIGHT_SD_FEEDBACK_FRAMES")),3,30):3;
        const bool exposure=std::getenv("SIGHT_SD_FEEDBACK_EXPOSURE")!=nullptr;
        const bool finite=std::getenv("SIGHT_SD_FEEDBACK_FINITE")!=nullptr;
        if(exposure)obs_data_set_double(settings,"exposure_strength",1.);
        ExportDir exports;
        if(finite){
            if(!exports.create()){obs_data_release(settings);obs_source_release(source);obs_shutdown();return 9;}
            obs_data_set_int(settings,"frame_limit",target);
            obs_data_set_bool(settings,"save_every_frame",true);
            obs_data_set_string(settings,"output_dir",exports.path.c_str());
        }
        if(std::getenv("SIGHT_SD_FEEDBACK_SCHEDULE")){
            obs_data_set_string(settings,"motion_schedule","0: zoom=1, pan_x=0, denoise=.2\n2: zoom=1.02, pan_x=4, denoise=.4");
            obs_data_set_string(settings,"prompt_schedule","1: illustration of an ocean coast, waves and clouds");
        }
        obs_source_update(source,settings);
        passed &= command(source,"sight_generator_start_v1");
        passed &= obs_source_get_width(source)==unsigned(size);
        std::vector<uint8_t> first,latest;
        int64_t seen=0;
        bool changed=false;
        auto valid_rgb=[size](const std::vector<uint8_t>&pixels){
            if(pixels.size()!=size_t(size)*size*4)return false;
            unsigned low=255,high=0;
            for(size_t i=0;i<pixels.size();i+=4)for(unsigned c=0;c<3;++c){
                low=std::min(low,unsigned(pixels[i+c]));high=std::max(high,unsigned(pixels[i+c]));
            }
            return high>low+5;
        };
        auto checksum=[](const std::vector<uint8_t>&pixels){
            uint64_t hash=14695981039346656037ull;
            for(size_t i=0;i<pixels.size();i+=4)for(unsigned c=0;c<3;++c){hash^=pixels[i+c];hash*=1099511628211ull;}
            return hash;
        };
        bool midpoint=false,settled=false;uint64_t mid_hash=0,settled_hash=0;
        int64_t mid_frame=0;int64_t distinct=0;uint64_t last_hash=0;
        for(int i=0;i<4500;++i){
            const auto current=stats(source);
            if(current.frames>0&&(smooth||current.frames>seen)){
                if(!draw(source,latest)||!valid_rgb(latest))break;
                auto hash=checksum(latest);
                auto p=preview(source);
                if(current.frames>seen){
                    seen=current.frames;
                    std::printf("rendered_frame=%lld rgb_hash=%016llx\n",(long long)seen,(unsigned long long)hash);
                    if(hash!=last_hash)++distinct;
                    last_hash=hash;
                    if(first.empty())first=latest;
                    else if(hash!=checksum(first))changed=true;
                }
                if(smooth&&p.draws>0&&p.phase>.25f&&p.phase<.75f){
                    midpoint=true;mid_hash=hash;mid_frame=current.frames;
                }
                if(smooth&&midpoint&&mid_frame==current.frames&&p.phase>=1.f&&hash!=mid_hash){
                    settled=true;settled_hash=hash;
                }
            }
            if((!finite&&current.frames>=target&&changed&&(!smooth||(midpoint&&settled)))||!current.running)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto generated=stats(source);
        const bool output_changed=changed&&valid_rgb(first)&&valid_rgb(latest);
        passed &= generated.frames>=target&&generated.feedback>=(refresh?1:target-1)&&output_changed&&(!control||generated.guided>=(refresh?1:2))&&(!refresh||generated.feedback<generated.frames-1);
        if(finite){
            const bool complete=generated.frames==target&&!generated.running&&generated.status=="Sequence complete";
            std::printf("finite_sequence_complete=%d frames=%lld\n",int(complete),(long long)generated.frames);
            passed &= complete;
        }else passed &= command(source,"sight_generator_stop_v1");
        if(exposure){
            calldata_t args;calldata_init(&args);
            bool ok=proc_handler_call(obs_source_get_proc_handler(source),"sight_generator_exposure_stats_v1",&args);
            auto count=calldata_int(&args,"frames");calldata_free(&args);
            std::printf("exposure_adjusted_frames=%lld\n",(long long)count);
            int64_t anchors=1;
            if(refresh)anchors+=(generated.frames-1)/2;
            if(std::getenv("SIGHT_SD_FEEDBACK_SCHEDULE")&&generated.frames>1)++anchors;
            passed &= ok&&count>0&&count<=generated.frames-anchors;
        }
        const auto stopped=stats(source);
        for(int i=0;i<1000&&stats(source).loaded;++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto after=stats(source);
        const bool stopped_cleanly=!after.loaded&&!after.running&&after.frames==stopped.frames;
        if(smooth){
            bool final_draw=draw(source,latest)&&valid_rgb(latest)&&preview(source).phase==1.f;
            std::printf("preview_midpoint=%d settled=%d draws=%lld stop_exact=%d mid_hash=%016llx settled_hash=%016llx\n",
                int(midpoint),int(settled),(long long)preview(source).draws,int(final_draw),(unsigned long long)mid_hash,(unsigned long long)settled_hash);
            // A finite GPU sequence can publish the next keyframe before
            // the previous interpolation settles. Stop intentionally snaps
            // to its final raw image. Verify that endpoint byte-for-byte
            // against the actual worker export instead of requiring an
            // intermediate endpoint before the next generated frame.
            passed &= midpoint&&(finite||settled)&&final_draw;
        }
        if(flow){
            auto [draws,coverage]=flow_stats(source);
            std::printf("preview_flow_draws=%lld stop_coverage=%.3f\n",(long long)draws,coverage);
            passed &= draws>0&&coverage==0.f;
        }
        std::printf("distinct_sampled_frames=%lld target=%d\n",(long long)distinct,target);
        if(finite){
            const bool final_draw=draw(source,latest)&&valid_rgb(latest)&&preview(source).phase==1.f;
            passed &= final_draw&&verify_exports(exports,unsigned(size),target,latest);
        }else if(!smooth)passed &= distinct>=target;
        passed &= stopped_cleanly;
        std::printf("model_backed_frames=%lld recursive=%lld guided=%lld output_changed=%d stop_released=%d\n",
            (long long)generated.frames,(long long)generated.feedback,(long long)generated.guided,int(output_changed),int(stopped_cleanly));
    }
    if (settings) obs_data_release(settings);
    obs_source_release(source);
    obs_shutdown();
    if (!passed) return 9;
    std::puts("SIGHT_NATIVE_SD_GENERATOR_SOURCE_PASS=1");
    return 0;
}
